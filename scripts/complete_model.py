"""Complete the primary saved fit using nominal FE mechanics, then assess it.

No fitting or held-response selection occurs. Build files are new artifacts;
the earlier geometric and separate calibration experiments are preserved.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fetune.fe_completion import ElasticContext,energy_dictionary,project_operators,energy_coefficients,constrained_energy_coefficients,physical_rows,relative as rel
from fetune.fe_completion import modal_response_weight,forced_coefficients,constrained_forced_coefficients,forced_objective,forced_objective_full
from fetune.spatial_completion import physical_realization,latent_nullspace
from fetune.mechanical import MechanicalModel,score,modal_resolvent
from fetune.io import json_write,digest
from fetune.completion_diagnostics import column_spectrum,trial_feasibility,physical_diagnostics
import os
CFG=json.loads(Path(os.environ.get('FETUNE_COMPLETION_CONFIG',ROOT/'configs/primary_fe_completion.json')).read_text())
OUT=ROOT/CFG['output'];FE=ROOT/CFG['nominal_FE']


def spectrum(T):
    return column_spectrum(T,CFG['rank_relative_tolerance'])


def dictionary_signature(K,M,A,represented,count):
    """Invalidate cached FE fields when any mechanical input or rule changes."""
    sha=hashlib.sha256()
    def add(a):
        a=np.ascontiguousarray(a)
        sha.update(str((a.shape,a.dtype.str)).encode());sha.update(a.tobytes())
    for operator in (K,M):
        matrix=sp.csr_matrix(operator);matrix.sort_indices()
        add(np.asarray(matrix.shape));add(matrix.indptr);add(matrix.indices);add(matrix.data)
    add(A);add(represented)
    sha.update(json.dumps(dict(count=count,eigen_tolerance=CFG['eigen_tolerance'],
        cluster_tolerance=CFG['cluster_tolerance'],definition_version=4),sort_keys=True).encode())
    return sha.hexdigest()


def complete(name,K,M,A,W,omega,Gamma,represented,B,S,Pi=None):
    start=time.perf_counter();h,r=W.shape;n=K.shape[0];d=min(r,len(represented)-h)
    if d<r-spectrum(W)['rank']:raise ValueError('insufficient admissible hidden directions')
    dictionary=OUT/(name+'_dictionary.npz')
    signature=dictionary_signature(K,M,A,represented,d)
    cached=np.load(dictionary) if dictionary.exists() else None
    cache_valid=cached is not None and 'input_signature' in cached and str(cached['input_signature'])==signature
    if cache_valid:
        z=cached;E,N=z['E'],z['N'];ops=[z[x] for x in ['KP','MP','HP']]
        checks=json.loads((OUT/(name+'_dictionary.json')).read_text())
        assert np.array_equal(z['A'],A)
    else:
        print(name,'full-FE constrained eigenfields',n,d,flush=True)
        context=ElasticContext(K,A)
        E,N,checks=energy_dictionary(K,M,A,d,CFG['eigen_tolerance'],CFG['cluster_tolerance'],represented,context=context)
        print(name,'projecting full-FE operators',flush=True)
        ops,solve_residual=project_operators(K,M,E,N,context=context);checks['energy_solve_residual']=solve_residual
        checks['input_signature']=signature
        np.savez_compressed(dictionary,E=E,N=N,A=A,omega=omega,KP=ops[0],MP=ops[1],HP=ops[2],
            definition_version=4,input_signature=signature)
        json_write(OUT/(name+'_dictionary.json'),checks)
    feasibility=trial_feasibility(A,E,N,W,represented)
    forced=CFG.get('criterion')=='forced'
    print(name,'solving FE coefficient criterion','J_F' if forced else 'J_0',flush=True)
    if forced:
        Z,criterion=forced_coefficients(W,omega,ops,Pi)
    else:
     try:
        Z,criterion=energy_coefficients(W,omega,ops,CFG['eta'])
     except la.LinAlgError:
        if not cache_valid:raise
        # Cached small blocks cannot evaluate the original residual form.
        # Reattach nominal FE solves only if a cancellation diagnostic needs it.
        context=ElasticContext(K,A)
        ops,_=project_operators(K,M,E,N,context=context)
        Z,criterion=energy_coefficients(W,omega,ops,CFG['eta'])
        criterion['cached_blocks_reprojected_for_stability']=True
    base=E@W;full=base+N@Z;T=full[represented];T0=base[represented]
    ts=spectrum(T);base_sv=la.svdvals(T0);floor=CFG['minimum_singular_value_relative_to_baseline_max']*base_sv[0]
    context=ElasticContext(K,A)
    Z_LS=Z.copy()
    ls_check=physical_diagnostics(K,M,A,full,W,omega,Gamma,represented,floor,solve=context.solve,N=N)
    baseline_path=ROOT/'artifacts/injective_revision/baseline/fe_primary_revision'/(name+'_completion.npz')
    baseline=np.load(baseline_path)
    for key,value in [('W',W),('omega',omega),('Gamma',Gamma),('A_full',A)]:
        assert np.array_equal(baseline[key],value),key
    assert baseline['N_full'].shape==N.shape
    family_singular=la.svdvals(baseline['N_full'].T@(M@N))
    family=dict(lift_relative=rel(E,baseline['E_full']),
        hidden_cross_mass_singular_values=family_singular,
        hidden_span_max_singular_deviation=float(np.max(abs(family_singular-1.))))
    assert family['lift_relative']<1e-10 and family['hidden_span_max_singular_deviation']<1e-8
    feasibility['archived_trial_family_equivalence']=family
    baseline_check=physical_diagnostics(K,M,A,baseline['T_full'],W,omega,Gamma,represented,floor,solve=context.solve)
    warm=N.T@(M@(baseline['T_full']-base))
    family_error=rel(base+N@warm,baseline['T_full'])
    assert family_error<1e-8,('physical affine family changed',family_error)
    criterion['baseline_representability_relative']=family_error
    if forced:
        criterion['J_F_full_LS']=forced_objective_full(K,M,full,omega,Pi,context.solve)
    if ts['minimum']<floor and forced:
        print(name,'enforcing finite singular-value margin under J_F',floor,flush=True)
        Z,constrained=forced_margin(name,W,omega,ops,Pi,E,N,represented,floor)
        criterion['constrained']=constrained
        full=base+N@Z;T=full[represented];ts=spectrum(T)
    elif ts['minimum']<floor:
        print(name,'enforcing finite singular-value margin',floor,flush=True)
        rule=CFG['rank_solver'];D=np.c_[E[represented],N[represented]]
        Z,constrained=constrained_energy_coefficients(W,omega,ops,D,CFG['eta'],
            floor*rule['margin_guard'],warm,rule['max_iterations'],rule['tolerance'],
            acceptance_margin=floor,feasible_starts=(Z_LS,warm),
            adaptive_penalty=rule.get('adaptive_penalty',False))
        criterion['constrained']=constrained
        full=base+N@Z;T=full[represented];ts=spectrum(T)
    report=dict(config=CFG,rank_W=spectrum(W)['rank'],hidden_fields=N.shape[1],
        rank_and_condition=ts,baseline_singular_values=base_sv,required_sigma_min=float(floor),
        criterion=criterion,dictionary=checks,
        anchor=rel(A@full,W),full_FE_dimension=n,represented_dimension=len(represented))
    diagnostics=physical_diagnostics(K,M,A,full,W,omega,Gamma,represented,floor,solve=context.solve,N=N)
    if forced:
        report['forced']=dict(J_F_accepted=forced_objective_full(K,M,full,omega,Pi,context.solve),
            J_F_accepted_blocks=forced_objective(W,Z,omega,ops,Pi),
            J_F_equality_only=criterion['J_F_full_LS'],
            equality_only_sigma_min=float(la.svdvals((base+N@Z_LS)[represented])[-1]),
            J0_equivalent_accepted=forced_objective(W,Z,omega,ops,np.diag(omega**-2.)),
            modal_weight=PI_INFO.get(name,{}),
            residual_energy_share=float(np.sum(omega**2*np.diag(Pi))),
            interpretation='J_F = sum over the test of twice the strain energy of the static deflection under the F=ma residual, divided by twice the fitted modal strain energy')
    report.update(trial_feasibility=feasibility,physical_diagnostics=diagnostics,
                  equality_only=ls_check,regularized_baseline=baseline_check)
    report['objective_reference']=dict(J_LS_numerical=ls_check['objective_J'],
        J_accepted=diagnostics['objective_J'],gap=diagnostics['objective_J']-ls_check['objective_J'],
        full_vs_projected_LS_relative=(abs(criterion['J']-ls_check['objective_J'])/max(ls_check['objective_J'],1e-300)
            if 'J' in criterion else abs(criterion['J_F']-criterion['J_F_full_LS'])/max(criterion['J_F_full_LS'],1e-300)),
        certified_lower_bound=False,
        reason='numerically solved equality-only reference; no interval arithmetic or certified roundoff bound')
    json_write(OUT/(name+'_precheck.json'),report)
    print(name,'rank',ts['rank'],'condition',ts['condition'],'sigma',ts['minimum'],'floor',floor,flush=True)
    assert ts['rank']==r and ts['minimum']>=floor and ts['condition']<CFG['maximum_completed_condition'],(ts['rank'],ts['minimum'],floor,ts['condition'])
    assert diagnostics['port_relative']<CFG['port_relative_tolerance']
    assert diagnostics['port_worst_column_relative']<CFG['port_relative_tolerance']
    assert diagnostics['left_inverse_frobenius']<CFG['left_inverse_tolerance']
    selected,row_check=physical_rows(T)
    phy=physical_realization(T,Gamma,omega,rows=selected);ps=spectrum(phy['P'])
    assert ps['condition']<CFG['maximum_selected_condition']
    inv=la.solve(phy['P'],np.eye(r));full_expansion=la.solve(phy['P'].T,full.T).T
    gm=full.T@(M@full);gk=full.T@(K@full)
    coefficients=np.vstack([W,Z]);dual_gram=coefficients.T@ops[2]@coefficients
    projected_residual=float(np.mean(np.diag(gk)/omega**2-2*np.diag(gm)+omega**2*np.diag(dual_gram)))
    residual_squared=2*diagnostics['objective_J']/r
    criterion['full_vs_projected_residual_relative']=abs(projected_residual-residual_squared)/max(residual_squared,1e-300)
    criterion['final_energy_residual']=float(np.sqrt(max(0.,residual_squared)))
    criterion['final_objective']=residual_squared+CFG['eta']*float(np.trace(gm))/r
    mass_prior=inv.T@gm@inv;stiffness_prior=inv.T@gk@inv
    Af=sp.csr_matrix(A);MY=sla.splu(sp.csc_matrix(M)).solve(A.T);GA=A@MY
    mass_residual=np.eye(r)-W.T@la.solve(GA,W,assume_a='pos');me=la.eigvalsh(mass_residual)
    D=np.column_stack([E[represented],N[represented]])
    ds=la.svdvals(D);coef=np.vstack([W,Z]);cs=la.svdvals(coef)
    report.update(P=ps,row_selection=row_check,represented_dictionary_minimum=float(ds[-1]),
        coefficient_minimum=float(cs[-1]),dictionary_lower_bound=float(ds[-1]*cs[-1]),
        mass_gram=rel(gm,np.eye(r)),stiffness_gram=rel(gk,np.diag(omega**2)),
        mass_gram_eigenvalues=la.eigvalsh(gm),
        mass_prior_physical_disagreement=rel(mass_prior,phy['M']),
        stiffness_prior_physical_disagreement=rel(stiffness_prior,phy['K']),
        mass_anchor_feasibility=dict(minimum=float(me[0]),maximum=float(me[-1]),rank=int(np.sum(abs(me)>1e-10)),
            available_hidden=n-h,feasible=bool(me[0]>=-1e-10 and np.sum(abs(me)>1e-10)<=n-h)),
        seconds=time.perf_counter()-start)
    # Selected factors and all coordinate maps are frozen before observations are opened.
    np.savez_compressed(OUT/(name+'_completion.npz'),T=T,T0=T0,T_full=full,E_full=E,N_full=N,Z=Z,
        W=W,A_full=A,omega=omega,Gamma=Gamma,represented_rows=represented,Z_LS=Z_LS,
        full_expansion=full_expansion,force=phy['expansion'].T@B,response=S@phy['expansion'],
        prior_mass_gram=gm,prior_stiffness_gram=gk,prior_M=mass_prior,prior_K=stiffness_prior,
        delta_M=phy['M']-mass_prior,delta_K=phy['K']-stiffness_prior,
        **({'Pi':Pi} if Pi is not None else {}),**phy)
    MechanicalModel(T,omega,Gamma).save(OUT/(name+'_model.npz'))
    json_write(OUT/(name+'_construction.json'),report)
    np.savetxt(OUT/(name+'_all_modes.csv'),np.column_stack([np.arange(1,r+1),omega/(2*np.pi),
        diagnostics['J_per_mode'],diagnostics['epsilon_per_mode'],diagnostics['port_column_absolute'],
        diagnostics['port_column_relative']]),delimiter=',',
        header='modal_index,frequency_Hz,J_j,epsilon_j,port_absolute,port_relative',comments='')
    return report


PI_INFO={}


def forced_margin(name,W,omega,ops,Pi,E,N,represented,floor):
    """Constrained J_F from one deterministic start: the equality-only J_F map with
    its represented singular values raised to the margin."""
    rule=CFG['rank_solver'];fc=CFG['forced'];D=np.c_[E[represented],N[represented]];h=W.shape[0]
    Z_LS,_=forced_coefficients(W,omega,ops,Pi)
    Rq=la.qr(D,mode='raw',check_finite=False)[1];RN=Rq[:,h:];b=Rq[:,:h]@W
    u,s,vh=la.svd(b+RN@Z_LS,full_matrices=False)
    start=la.lstsq(RN,(u*np.maximum(s,1.02*floor))@vh-b)[0]
    Z,info=constrained_forced_coefficients(W,omega,ops,Pi,D,floor*rule['margin_guard'],start,
        rule['max_iterations'],rule['tolerance'],acceptance_margin=floor,feasible_starts=(start,),
        rho_scale=fc['rho_scale'],inner_tolerance=fc['inner_tolerance'],log_every=100)
    info['start']='equality-only J_F map with represented singular values raised to 1.02 delta'
    info['start_J_F']=forced_objective(W,start,omega,ops,Pi)
    return Z,info


def force_spectra(record):
    """Welch force spectral matrix of a training record, G_ij = mean(F_i conj F_j) per Hz."""
    cache=OUT/(record+'_force_spectra.npz')
    if cache.exists():
        z=np.load(cache);return z['frequency'],z['G_ff']
    import h5py
    from fetune import data
    from fetune.response_spectra import spectral_matrix
    fc=CFG['forced'];w=fc['welch']
    with h5py.File(data.DATA/fc['raw'],'r') as handle:
        group=handle['measurement/'+record]
        assert list(group.attrs['force_channels'])==[59,60] and list(group.attrs['excitation_positions'])==[6,8]
        force=group['force'][0]
    f,G,info=spectral_matrix(force,w['rate_hz'],w['nperseg'],w['noverlap'],w['window'])
    band=(f>=fc['band_hz'][0])&(f<=fc['band_hz'][1])
    np.savez_compressed(cache,frequency=f[band],G_ff=G[band])
    json_write(cache.with_suffix('.json'),dict(record=record,estimator=info,units='N^2/Hz',convention='G_ij = mean(F_i conj F_j)'))
    return f[band],G[band]


def nominal_spring():
    """Assemble the declared prior without accessing observations or truth."""
    cfg=CFG['spring_nominal'];D=np.array(cfg['incidence'],dtype=float)
    m=np.array(cfg['mass_kg']);k=np.array(cfg['stiffness_N_per_m']);c=np.array(cfg['damping_Ns_per_m'])
    assert D.shape==(4,3) and np.all(m>0) and np.all(k>0) and np.all(c>0)
    M=np.diag(m);K=D.T@np.diag(k)@D;C=D.T@np.diag(c)@D
    for matrix in [M,C,K]:la.cholesky(matrix)
    return dict(M=M,C=C,K=K,D=D,mass=m,stiffness=k,damping=c)


def construct(names=('spring','laboratory')):
    OUT.mkdir(exist_ok=True,parents=True)
    json_write(OUT/'declaration.json',dict(config=CFG,sources={str(p.relative_to(ROOT)):digest(p) for p in
        [ROOT/CFG['primary_fit'],ROOT/CFG['primary_factors'],FE/'M_nominal.npz',FE/'K_nominal.npz',FE/'dof_mapping.npz',ROOT/'configs/primary_fe_completion.json',ROOT/'src/fetune/fe_completion.py']}))
    if 'spring' in names:
        z=np.load(ROOT/'artifacts/robust_revision/prior_spring/spring_completion.npz')
        prior=nominal_spring();np.savez_compressed(OUT/'spring_nominal.npz',**prior)
        assert np.array_equal(z['A'],np.eye(3)[[0,2]])
        Pi=None
        if CFG.get('criterion')=='forced':
            from fetune.three_mass import configuration,frequency_grid
            f=frequency_grid(configuration())
            Pi,PI_INFO['spring']=modal_response_weight(z['omega'],z['Gamma'],z['W'][[0]],f,np.ones((len(f),1,1)))
        complete('spring',sp.csc_matrix(prior['K']),sp.csc_matrix(prior['M']),z['A'],z['W'],z['omega'],z['Gamma'],
            np.arange(3),np.eye(3)[:,[0]],np.eye(3)[[2]],Pi)
    if 'laboratory' in names:
        z=np.load(ROOT/CFG['primary_factors']);mapping=np.load(FE/'dof_mapping.npz')
        represented=mapping['represented_rows'];K=sp.load_npz(FE/'K_nominal.npz');M=sp.load_npz(FE/'M_nominal.npz')
        A=np.zeros((len(z['A']),K.shape[0]));A[:,represented]=z['A']
        B=z['A'][-2:].T;S=z['A'][:-2]
        Pi=None
        if CFG.get('criterion')=='forced':
            f,G=force_spectra(CFG['forced']['record'])
            Pi,PI_INFO['laboratory']=modal_response_weight(z['omega'],z['Gamma'],z['W'][-2:],f,G)
        complete('laboratory',K,M,A,z['W'],z['omega'],z['Gamma'],represented,B,S,Pi)


def verify(name,frequency,S,B,baseline=None):
    z=np.load(OUT/(name+'_completion.npz'));T=z['T'];P=z['P'];E=z['expansion']
    M,C,K=(z[x] for x in ['M','C','K']);r=len(P);Br=E.T@B;Sr=S@E
    # These congruent products avoid explicitly multiplying a squared-condition
    # mass inverse. They are checked against the saved physical equations below.
    DK=la.solve(P.T,(P*z['omega']**2).T).T
    DC=la.solve(P.T,(P@z['Gamma']).T).T;DB=P@(T.T@B)
    Ar=np.block([[np.zeros((r,r)),np.eye(r)],[-DK,-DC]])
    Bs=np.vstack([np.zeros_like(Br),DB]);Cu=np.c_[Sr,np.zeros_like(Sr)]
    Cv=np.c_[np.zeros_like(Sr),Sr];Ca=-Sr@np.c_[DK,DC];Da=Sr@DB
    fitted=MechanicalModel(T,z['omega'],z['Gamma']).evaluate(frequency,S,B)
    physical=[];identity_by_frequency=[]
    for j,f in enumerate(frequency):
        s=2j*np.pi*f;u=Sr@la.solve(s*s*M+s*C+K,Br)
        physical.append(s*s*u);identity_by_frequency.append(rel(s*s*u,fitted[j]))
    state_check={}
    for j in np.linspace(0,len(frequency)-1,min(5,len(frequency)),dtype=int):
        s=2j*np.pi*frequency[j];wref=float(max(z['omega']))
        scaling=np.r_[np.ones(r),np.full(r,wref)]
        Ah=Ar*scaling[None,:]/scaling[:,None]/wref
        Bh=Bs/scaling[:,None]/wref
        response=scaling[:,None]*la.solve(s/wref*np.eye(2*r)-Ah,Bh)
        state_check[str(float(frequency[j]))]=dict(displacement=rel(Cu@response,fitted[j]/s**2),
            velocity=rel(Cv@response,fitted[j]/s),acceleration=rel(Ca@response+Da,fitted[j]))
    report=dict(physical_frf=rel(np.array(physical),fitted),
        feedthrough=rel(Da,(S@T)@(T.T@B)),
        force=rel(P.T@Br,T.T@B),output=rel(Sr@P,S@T),
        expansion=rel(E@P,T),selection=rel(E[z['rows']],np.eye(r)),
        congruences={key:rel(P.T@matrix@P,target) for key,matrix,target in
            [('M',M,np.eye(r)),('C',C,z['Gamma']),('K',K,np.diag(z['omega']**2))]},
        state_operator_consistency=dict(stiffness=rel(M@DK,K),damping=rel(M@DC,C),force=rel(M@DB,Br)),
        mechanical={key:dict(symmetry=rel(matrix,matrix.T),min_eigenvalue=float(la.eigvalsh(matrix)[0]))
            for key,matrix in [('M',M),('C',C),('K',K)]},state_space=state_check)
    if baseline is not None:report['registered_frf']=rel(fitted,baseline)
    poles,phi=modal_resolvent(z['omega'],z['Gamma'])
    report['damped_mode_mapping']=rel(T@phi,z['E_full'][z['represented_rows']]@(z['W']@phi)+z['N_full'][z['represented_rows']]@(z['Z']@phi))
    np.savez_compressed(OUT/(name+'_state_space.npz'),A=Ar,B=Bs,Cu=Cu,Cv=Cv,Ca=Ca,Da=Da,Br=Br,Sr=Sr,
        frequency=frequency,physical_frf=np.array(physical),normalized_frf=fitted,frequency_relative_error=identity_by_frequency)
    assert report['physical_frf']<1e-7 and report['feedthrough']<1e-7
    assert all(v['min_eigenvalue']>0 for v in report['mechanical'].values())
    assert max(v for checks in state_check.values() for v in checks.values())<1e-6
    return report


def realize(names=('spring','laboratory')):
    """Update physical row selection and operators for frozen accepted T only."""
    for name in names:
        path=OUT/(name+'_completion.npz');z=dict(np.load(path));T=z['T'];r=T.shape[1]
        rows,check=physical_rows(T);phy=physical_realization(T,z['Gamma'],z['omega'],rows=rows)
        if name=='spring':B=np.eye(3)[:,[0]];S=np.eye(3)[[2]]
        else:
            prior=np.load(ROOT/CFG['primary_factors']);B=prior['A'][-2:].T;S=prior['A'][:-2]
        z.update(phy);inv=la.solve(phy['P'],np.eye(r))
        z['full_expansion']=la.solve(phy['P'].T,z['T_full'].T).T
        z['force']=phy['expansion'].T@B;z['response']=S@phy['expansion']
        z['prior_M']=inv.T@z['prior_mass_gram']@inv;z['prior_K']=inv.T@z['prior_stiffness_gram']@inv
        z['delta_M']=z['M']-z['prior_M'];z['delta_K']=z['K']-z['prior_K']
        np.savez_compressed(path,**z)
        report=json.loads((OUT/(name+'_construction.json')).read_text());report['P']=spectrum(phy['P']);report['row_selection']=check
        report['mass_prior_physical_disagreement']=rel(z['prior_M'],z['M'])
        report['stiffness_prior_physical_disagreement']=rel(z['prior_K'],z['K'])
        json_write(OUT/(name+'_construction.json'),report)


def assess_laboratory():
    # No held observations are read before the FE construction is accepted and saved.
    obs=np.load(ROOT/'artifacts/model/laboratory_expanded/observations.npz');saved=np.load(ROOT/'artifacts/robust_revision/selected_predictions.npz')
    z=np.load(OUT/'laboratory_completion.npz');f=obs['frequency'];fit=obs['fit_rows'];held=obs['held_rows'];S=obs['S_all'];B=obs['B']
    pred=MechanicalModel(z['T'],z['omega'],z['Gamma']).evaluate(f,S,B)
    report=verify('laboratory',f[np.linspace(0,len(f)-1,41,dtype=int)],S[fit],B)
    report['registered_all_lines']=rel(pred[:,fit],saved['prediction'][:,fit])
    report['scores']={key:score(pred[:,fit],obs[key][:,fit],saved['scale'][fit]) for key in ['test_96','test_101','test_103']}
    report['excluded']=score(pred[:,held],obs['test_103'][:,held],saved['scale'][held])
    np.savez_compressed(OUT/'laboratory_predictions.npz',frequency=f,prediction=pred,scale=saved['scale'],channels=obs['channels'])
    json_write(OUT/'laboratory_assessment.json',report)
    print('Primary FE assessment',report['physical_frf'],report['scores']['test_103']['complex'],'excluded',report['excluded']['complex'],flush=True)


def assess_spring():
    # Construction and its rank acceptance precede every use of generating data.
    reference=np.load(ROOT/'artifacts/three_mass/reference.npz');saved_fit=np.load(ROOT/'artifacts/three_mass/noisy_a1e-05_prediction.npz');predicted=saved_fit['prediction']
    z=np.load(OUT/'spring_completion.npz');f=reference['frequency']
    ids=np.linspace(0,len(f)-1,41,dtype=int)
    S=np.eye(3)[[2,1]];B=np.eye(3)[:,[0]]
    report=verify('spring',f[ids],S,B)
    from fetune.three_mass import exact_coordinates,configuration,matrices
    Tstar,_,_=exact_coordinates(configuration());M,C,K=matrices(configuration())
    nominal=np.load(OUT/'spring_nominal.npz')
    def forward(M,C,K):
        return np.array([s*s*S@la.solve(s*s*M+s*C+K,B) for s in 2j*np.pi*f])
    truth=forward(M,C,K);prior=forward(nominal['M'],nominal['C'],nominal['K'])
    identified=MechanicalModel(z['T'],z['omega'],z['Gamma']).evaluate(f,S,B)
    baseline=MechanicalModel(z['T0'],z['omega'],z['Gamma']).evaluate(f,S,B)
    observations=np.load(ROOT/'artifacts/three_mass/noisy_observations.npz')
    report['registered_frf']=rel(identified[:,[0]],predicted)
    report['registered_completion']=rel(identified[:,[0]],baseline[:,[0]])
    assert report['registered_frf']<1e-12 and report['registered_completion']<1e-12
    assert rel(truth[:,[0]],reference['truth'])<1e-12
    # One training-only scale for both transfers: even assessment normalization
    # does not use the unmeasured reference response.
    scale=np.repeat(saved_fit['scale'],2,axis=0)
    assessment={name:{label:score(values[:,[i]],truth[:,[i]],scale[[i]])
        for i,label in enumerate(['endpoint','middle'])}
        for name,values in [('nominal',prior),('identified',identified)]}
    report.update(hidden_coefficients=z['T'][1],hidden_reference=Tstar[1],
        coefficient_discrepancy=rel(z['T'],Tstar),physical_discrepancies={key:rel(z[key],value) for key,value in [('M',M),('C',C),('K',K)]})
    report.update(response_assessment=assessment,assessment_scale=scale,
        training_error=score(identified[:,[0]],observations['training'],saved_fit['scale']),
        independent_noisy_error=score(identified[:,[0]],reference['assessment'],saved_fit['scale']),
        nominal_discrepancies={key:rel(nominal[key],value) for key,value in [('M',M),('C',C),('K',K)]},
        compatible_to_nominal={key:rel(z[key],nominal[key]) for key in ['M','C','K']},
        prior_mass_residual_eigenvalues=la.eigvalsh(np.eye(3)-z['W'].T@la.solve(z['A_full']@la.solve(nominal['M'],z['A_full'].T),z['W'])),
        physical_order=[1,2,3],response_order=[3,2],force_order=[1],
        assessment_role='Reference responses are assessment-only. Both transfers use the saved endpoint training RMS for reporting; no unmeasured response enters normalization or any model choice.')
    np.savez_compressed(OUT/'spring_predictions.npz',frequency=f,truth=truth,nominal=prior,identified=identified,
        anchor_only=baseline,training=observations['training'],assessment_scale=scale,S=S,B=B)
    V,_,_,_=latent_nullspace(z['W']);v=V[:,0]
    if (z['T']@v)[1]<0:v=-v
    np.savez_compressed(OUT/'spring_illustration.npz',v0=v,before=z['T0']@v,after=z['T']@v,
        singular_before=la.svdvals(z['T0']),singular_after=la.svdvals(z['T']),reference=Tstar)
    json_write(OUT/'spring_assessment.json',report)
    print('Spring assessment',report['registered_frf'],'physical',report['physical_frf'],
        'nominal/identified endpoint',assessment['nominal']['endpoint']['complex'],assessment['identified']['endpoint']['complex'],
        'middle',assessment['nominal']['middle']['complex'],assessment['identified']['middle']['complex'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['construct','realize','assess','all'])
    p.add_argument('--case',choices=['spring','laboratory','both'],default='both')
    p.add_argument('--output',type=Path,help='Separate completion destination; defaults to the configured accepted directory')
    args=p.parse_args()
    if args.output is not None:
        OUT=(ROOT/args.output).resolve();CFG=dict(CFG,output=str(OUT))
    names=('spring','laboratory') if args.case=='both' else (args.case,)
    if args.command in ['construct','all']:construct(names)
    if args.command=='realize':realize(names)
    if args.command in ['assess','all']:
        if 'spring' in names:assess_spring()
        if 'laboratory' in names:assess_laboratory()
