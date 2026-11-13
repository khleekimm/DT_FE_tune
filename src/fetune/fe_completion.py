"""Anchor-preserving completion selected by nominal FE mechanics.

All operations use the admissible full FE coordinate space. A restriction is
used only for the published displacement map and physical-row selection.
Neither measured response arrays nor reference modes enter this module.
"""
import hashlib
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def relative(a, b):
    return float(la.norm(a-b)/max(la.norm(b), 1e-300))


def sparse_signature(matrix):
    """Content signature, including sparsity and coordinate order."""
    matrix=sp.csc_matrix(matrix,copy=True);matrix.sort_indices()
    sha=hashlib.sha256()
    for array in (np.asarray(matrix.shape),matrix.indptr,matrix.indices,matrix.data):
        sha.update(str((array.shape,array.dtype.str)).encode())
        sha.update(np.ascontiguousarray(array).tobytes())
    return sha.hexdigest()


class ElasticSolver:
    """One equilibrated sparse factorization; two refinement steps per solve."""
    def __init__(self,K):
        self.K=sp.csc_matrix(K,copy=True)
        diagonal=self.K.diagonal()
        if np.any(diagonal<=0) or not np.all(np.isfinite(diagonal)):
            raise ValueError('positive finite stiffness diagonal required')
        self.scale=1/np.sqrt(diagonal)
        self.factor=sla.splu((sp.diags(self.scale)@self.K@sp.diags(self.scale)).tocsc())
        self.calls=0;self.right_hand_sides=0

    def raw(self,rhs):
        scale=self.scale if rhs.ndim==1 else self.scale[:,None]
        return scale*self.factor.solve(scale*rhs)

    def __call__(self,rhs):
        rhs=np.asarray(rhs);self.calls+=1
        self.right_hand_sides+=1 if rhs.ndim==1 else rhs.shape[1]
        x=self.raw(rhs)
        for _ in range(2):x+=self.raw(rhs-self.K@x)
        return x


def elastic_solver(K):
    return ElasticSolver(K)


class ElasticContext:
    """Reusable K factorization and A-dependent minimum-energy extension.

    Own copies prevent caller mutation from silently invalidating a factor.
    A changed K requires a new context. set_registration changes only A, G, E.
    R, diag(M)[R], count and cutoff define selection; full M defines subsequent
    normalization/projections. W, omega and Gamma are not basis dependencies.
    """
    def __init__(self,K,A=None):
        self.solve=elastic_solver(K);self.K=self.solve.K
        self.stiffness_signature=sparse_signature(self.K)
        self.A=None
        if A is not None:self.set_registration(A)

    def set_registration(self,A):
        A=sp.csr_matrix(A,copy=True);signature=sparse_signature(A)
        if self.A is not None and signature==self.registration_signature:return
        if A.shape[1]!=self.K.shape[0]:raise ValueError('registration dimension mismatch')
        X=self.solve(A.T.toarray());G=A@X
        self.constraint_factor=la.cho_factor((G+G.T)*.5)
        E=la.cho_solve(self.constraint_factor,X.T).T
        # A small right correction removes constraint roundoff, as in the baseline.
        E=la.solve((A@E).T,E.T).T
        self.A=A;self.G=G;self.E=E;self.registration_signature=signature

    def check(self,K,A=None):
        if sparse_signature(K)!=self.stiffness_signature:
            raise ValueError('stiffness changed: construct a new ElasticContext')
        if A is not None:self.set_registration(A)

    def project(self,v):
        return v-self.E@(self.A@v)

    def inverse(self,rhs):
        return self.project(self.solve(rhs))


def physical_rows(T,threshold=1.05,max_exchanges=400):
    """Pivoted QR followed by determinant-increasing row exchanges.

    A coefficient E[i,j] of magnitude > 1 replaces selected row j by row i.
    This increases |det(P)| by that magnitude. Only T enters the selection.
    """
    n,r=T.shape
    if n==r:return np.arange(r),dict(exchanges=0,maximum_expansion_entry=1.)
    _,_,piv=la.qr(T.T,mode='economic',pivoting=True);rows=piv[:r].copy()
    expansion=la.solve(T[rows].T,T.T).T
    for iteration in range(max_exchanges):
        i,j=np.unravel_index(abs(expansion).argmax(),expansion.shape);v=expansion[i,j]
        if abs(v)<=threshold:break
        row=expansion[i].copy();row[j]-=1
        expansion-=np.outer(expansion[:,j]/v,row);rows[j]=i
    return np.sort(rows),dict(exchanges=iteration,maximum_expansion_entry=float(abs(expansion).max()),threshold=threshold)


def _cluster_end(values,count,tolerance):
    """The tolerance belongs to stiffness eigenvalues, never their reciprocals."""
    while count<len(values) and abs(values[count]-values[count-1])<=tolerance*max(1.,abs(values[count])):
        count+=1
    return count


def _partial_fields(op,count,available,eig_tol,cluster_tol,generalized=None):
    n=op.shape[0];wanted=min(count+8,n-1);requests=[]
    v0=np.sin(np.arange(1,n+1)*np.sqrt(2.))
    while True:
        requests.append(wanted)
        try:
            if generalized is None:
                inverse_values,vectors=sla.eigsh(op,k=wanted,which='LA',tol=eig_tol,
                    ncv=min(n,2*wanted+1),v0=v0)
                positive=inverse_values>0
                values=1/inverse_values[positive];vectors=vectors[:,positive]
            else:
                K,M,project=generalized
                values,vectors=sla.eigsh(K,k=wanted,M=M,sigma=0.,OPinv=op,which='LM',
                    tol=eig_tol,ncv=min(n,2*wanted+1),v0=project(v0))
        except sla.ArpackNoConvergence as error:
            raise RuntimeError(f'hidden-field eigensolve did not converge: '
                f'{len(error.eigenvalues)}/{wanted} pairs; no partial basis accepted') from error
        order=np.argsort(values);values=values[order];vectors=vectors[:,order]
        # The exact positive rank is available; numerical null eigenvalues beyond
        # it are not admissible mechanical fields.
        values=values[:available];vectors=vectors[:,:available]
        end=_cluster_end(values,count,cluster_tol)
        if len(values)>=count and (end<len(values) or end==available):
            return values,vectors,end,requests
        if wanted==n-1:
            raise RuntimeError('cannot certify the retention boundary with available eigenpairs')
        wanted=min(n-1,max(wanted+8,end+8))


def _mass_normalize(fields,M):
    gram=fields.T@(M@fields);gram=(gram+gram.T)*.5
    eigenvalues=la.eigvalsh(gram)
    threshold=np.finfo(float).eps*len(gram)*max(abs(eigenvalues[-1]),1e-300)
    if eigenvalues[0]<=threshold:
        # Reveal rank on the retained Gram only. Losing a field can violate the
        # modal rank requirement, so fail with a diagnosis instead of dropping it.
        singular=la.svdvals(gram)
        rank=int(np.count_nonzero(singular>threshold))
        raise la.LinAlgError(f'retained mass Gram is numerically rank deficient '
            f'({rank}/{len(gram)}; minimum eigenvalue {eigenvalues[0]:.3e}); '
            'review recovered fields, metric and eigensolver accuracy')
    try:
        L=la.cholesky(gram,lower=True)
        normalized=la.solve_triangular(L,fields.T,lower=True,check_finite=False).T
        method='retained Gram Cholesky and triangular solve'
    except la.LinAlgError:
        # A full-rank small spectral factor is a fallback, without a ridge or
        # truncation. It spans exactly the same retained fields.
        eigenvalues,U=la.eigh(gram)
        if eigenvalues[0]<=threshold:raise
        normalized=(fields@U)/np.sqrt(eigenvalues)[None,:]
        method='full-rank retained Gram spectral factor'
    return normalized,dict(method=method,condition=float(eigenvalues[-1]/eigenvalues[0]),
                           minimum_eigenvalue=float(eigenvalues[0]),rank=len(gram))


def energy_dictionary(K,M,A,count,eig_tol=1e-10,cluster_tol=1e-8,represented=None,
                      *,context=None,batch_size=32,stage=None,dense_threshold=100):
    """Fixed low-stiffness port-null space, normalized only after selection.

    The displayed path uses diag(M)[represented], not a condensed mass.
    Full fields are recovered with the same projected K inverse. No static
    expansion, condensed stiffness, or FE-space null-space projector is formed.
    The small full-space path retains the original dense generalized problem.
    """
    K=sp.csc_matrix(K);M=sp.csc_matrix(M);A=sp.csr_matrix(A)
    n=K.shape[0];h=A.shape[0];requested=count
    if not 0<count<=n-h:raise ValueError('insufficient hidden FE dimension')
    if batch_size<1:raise ValueError('batch_size must be positive')
    mark=stage or (lambda name:None)
    if context is None:
        mark('factorization_and_extension');context=ElasticContext(K,A)
    else:context.check(K,A)
    E=context.E;report={}
    displayed=represented is not None and n>=dense_threshold
    mark('hidden_selection')
    if n<dense_threshold:
        U=la.null_space(A.toarray())
        values,vec=la.eigh(U.T@(K@U),U.T@(M@U))
        count=_cluster_end(values,count,cluster_tol);N=U@vec[:,:count]
        mark('recovery_and_normalization')
    elif displayed:
        rows=np.asarray(represented,dtype=int);nr=len(rows)
        if len(np.unique(rows))!=nr or np.any(rows<0) or np.any(rows>=n):
            raise ValueError('represented rows must be distinct FE coordinates')
        omitted=np.setdiff1d(np.arange(n),rows)
        if A[:,omitted].count_nonzero():
            raise ValueError('static displayed selection requires registration on displayed coordinates')
        if count>nr-h:raise ValueError('insufficient statically admissible hidden dimension')
        metric=M.diagonal()[rows]
        if np.any(metric<=0):raise ValueError('displayed mass metric must be positive')
        root=np.sqrt(metric)
        def action(v):
            v=np.asarray(v);rhs=np.zeros((n,)+v.shape[1:])
            scale=root if v.ndim==1 else root[:,None]
            rhs[rows]=scale*v
            return scale*context.inverse(rhs)[rows]
        op=sla.LinearOperator((nr,nr),matvec=action,matmat=action,dtype=float)
        values,vectors,count,requests=_partial_fields(op,count,nr-h,eig_tol,cluster_tol)
        report['eigensolver_requests']=requests
        mark('recovery_and_normalization')
        N=np.empty((n,count));eigen_residuals=[]
        for start in range(0,count,batch_size):
            stop=min(start+batch_size,count);y=vectors[:,start:stop]
            rhs=np.zeros((n,stop-start));rhs[rows]=root[:,None]*y
            fields=context.inverse(rhs)*values[None,start:stop]
            N[:,start:stop]=fields
            # H y / mu - y equals D R n_tilde - y, so no extra inverse
            # applications are needed for relative eigenpair residuals.
            eigen_residuals.extend(la.norm(root[:,None]*fields[rows]-y,axis=0))
        report.update(selection_eigenpair_relative_max=float(max(eigen_residuals)),
                      recovery_displayed_relative=relative(N[rows],vectors[:,:count]/root[:,None]),
                      recovery_anchor=float(la.norm(A@N)))
    else:
        op=sla.LinearOperator(K.shape,matvec=context.inverse,matmat=context.inverse,dtype=float)
        values,N,count,requests=_partial_fields(op,count,n-h,eig_tol,cluster_tol,
                                                generalized=(K,M,context.project))
        N=N[:,:count];report['eigensolver_requests']=requests
        mark('recovery_and_normalization')
    selection_values=values[:count].copy()
    report['cutoff']=dict(requested=requested,retained=count,last=float(values[count-1]),
        next=float(values[count]) if count<len(values) else None,
        tolerance=cluster_tol,units='stiffness-selection eigenvalues in SI',
        gap=float(values[count]-values[count-1]) if count<len(values) else None)
    N=context.project(N)
    N,normalization=_mass_normalize(N,M)
    # Keep the inexpensive Ritz rotation: reproducible stiffness ordering and
    # a diagonal hidden stiffness block are useful for diagnostics/comparisons.
    # It changes coordinates within the fixed span, never the selected space.
    values,U=la.eigh(N.T@(K@N));N=N@U
    pivots=np.argmax(abs(N),axis=0);N*=np.sign(N[pivots,np.arange(count)])
    KN=K@N
    if displayed:
        field_mass=N[rows].T@(metric[:,None]*N[rows])
        reduced_action=la.solve(field_mass,N.T@KN,assume_a='pos')
        residual=KN.copy();residual[rows]-=(metric[:,None]*N[rows])@reduced_action
        report.update(omitted_equilibrium=float(la.norm(KN[omitted])/max(la.norm(KN),1e-300)),
            represented_ordering_metric='diagonal of supplied full FE mass on represented translations',
            constrained_eigen_residual_scope='invariance of selected span under displayed-metric constrained stiffness')
    else:residual=KN-(M@N)*values
    reaction=la.solve((A@A.T).toarray(),A@residual,assume_a='pos')
    report.update(hidden_fields=count,eigenvalues=values,selection_eigenvalues=selection_values,
        sign_pivots=pivots,normalization=normalization,
        anchor_extension=relative(A@E,np.eye(h)),hidden_anchor=float(la.norm(A@N)),
        hidden_mass_gram=relative(N.T@(M@N),np.eye(count)),
        constrained_eigen_residual=float(la.norm(residual-A.T@reaction)/max(la.norm(KN),1e-300)),
        extension_stationarity=relative(K@E,A.T@la.cho_solve(context.constraint_factor,np.eye(h))),
        stiffness_factorizations=1,elastic_solve_calls=context.solve.calls)
    return E,N,report


class ProjectedOperators(list):
    """Three small blocks, with optional residual-form rescue on their source."""
    def __init__(self,blocks,residual_form=None):
        super().__init__(blocks);self.residual_form=residual_form


def project_operators(K,M,E,N,*,context=None,batch_size=32):
    """Project all cross blocks, using one factor and bounded RHS batches."""
    K=sp.csc_matrix(K);M=sp.csc_matrix(M)
    if context is None:context=ElasticContext(K)
    else:context.check(K)
    D=np.column_stack([E,N]);MD=M@D
    KP=D.T@(K@D);MP=D.T@MD;HP=np.empty_like(MP);error=0.
    for start in range(0,D.shape[1],batch_size):
        stop=min(start+batch_size,D.shape[1]);rhs=MD[:,start:stop]
        dual=context.solve(rhs);HP[:,start:stop]=MD.T@dual
        error+=la.norm(K@dual-rhs)**2
    solve_residual=float(np.sqrt(error)/max(la.norm(MD),1e-300))
    def residual_form(omega,eta):
        # Slow, stable path used only if cancellation spoils the polynomial.
        basis=np.column_stack([E,N]);mass=M@basis
        residual=(K@basis)-omega**2*mass
        gram=np.empty((basis.shape[1],basis.shape[1]))
        for start in range(0,basis.shape[1],batch_size):
            stop=min(start+batch_size,basis.shape[1])
            gram[:,start:stop]=residual.T@context.solve(residual[:,start:stop])/omega**2
        gram+=eta*MP
        return (gram+gram.T)*.5
    return ProjectedOperators([(v+v.T)*.5 for v in (KP,MP,HP)],residual_form),solve_residual


def _column_quadratic(operators,omega,eta):
    KP,MP,HP=operators
    L=KP/omega**2+(eta-2)*MP+omega**2*HP
    return (L+L.T)*.5


def _coefficient_factor(L,h,operators,omega,eta):
    """Diagnose cancellation without computing every coefficient eigenspectrum."""
    Q=L[h:,h:];fallback=False
    try:
        factor=la.cho_factor(Q,check_finite=False)
        rcond,info=la.lapack.dpocon(factor[0],la.norm(Q,1),uplo='U')
        if info:raise la.LinAlgError('condition estimation failed')
        KP,MP,HP=operators
        scale=la.norm(KP[h:,h:],1)/omega**2+abs(eta-2)*la.norm(MP[h:,h:],1)+omega**2*la.norm(HP[h:,h:],1)
        risk=32*np.finfo(float).eps*scale/max(rcond*la.norm(Q,1),1e-300)
        # Bound relative solve sensitivity, not just subtraction cancellation.
        # The 1e-5 safeguard is conservative relative to the nominal laboratory
        # indicator (about 1.1e-7); it is a diagnostic, not an error certificate.
        if risk>1e-5:raise la.LinAlgError('expanded quadratic loses relative positive-definiteness accuracy')
    except la.LinAlgError as error:
        residual_form=getattr(operators,'residual_form',None)
        if residual_form is None:
            raise la.LinAlgError('coefficient quadratic requires residual-form assembly; '
                'reproject with the nominal FE context instead of regularizing the cached blocks') from error
        L=residual_form(omega,eta);Q=L[h:,h:]
        factor=la.cho_factor(Q,check_finite=False)
        rcond,info=la.lapack.dpocon(factor[0],la.norm(Q,1),uplo='U')
        if info:raise la.LinAlgError('residual-form condition estimation failed')
        fallback=True
    return L,factor,float(1/max(rcond,1e-300)),fallback


def energy_coefficients(W,omega,operators,eta=1e-2):
    """Equality-only FE residual minimizer, retaining every fitted column.

    Objective mean_j (||K t_j-w_j^2 M t_j||^2_(K^-1)/w_j^2
                      + eta*t_j^T M t_j), t_j=E W_j+N Z_j.
    The ridge is a declared physical-mass metric, not mass normalization.
    """
    KP,MP,HP=operators;h=W.shape[0];d=len(KP)-h;r=W.shape[1]
    if eta<0 or np.any(np.asarray(omega)<=0):raise ValueError('nonnegative eta and positive frequencies required')
    if not all(np.all(np.isfinite(x)) for x in [W,omega,*operators]):raise ValueError('nonfinite coefficient inputs')
    Z=np.empty((d,r));normal_residual=[];conditions=[];objective=[];residual=[];fallbacks=[];nullspaces={}
    for j,w in enumerate(omega):
        try:
            L,factor,condition,fallback=_coefficient_factor(_column_quadratic(operators,w,eta),h,operators,w,eta)
            rhs=-L[h:,:h]@W[:,j]
            Z[:,j]=la.cho_solve(factor,rhs,check_finite=False)
        except la.LinAlgError:
            # An exact nominal resonance may give zero-cost directions. Solve
            # the PSD least-squares system, retaining its kernel for refinement.
            residual_form=getattr(operators,'residual_form',None)
            if eta!=0 or residual_form is None:raise
            L=residual_form(w,0.);Q=L[h:,h:];rhs=-L[h:,:h]@W[:,j]
            values,vectors=la.eigh(Q)
            cutoff=64*np.finfo(float).eps*max(d,1)*max(la.norm(Q,2),1e-300)
            if values[0]<-cutoff:raise la.LinAlgError('indefinite residual Gram')
            positive=values>cutoff;null=vectors[:,~positive]
            if la.norm(null.T@rhs)>1e-9*max(la.norm(rhs),1.):
                raise la.LinAlgError('unresolved null-space linear term; residual least squares needed')
            Z[:,j]=vectors[:,positive]@((vectors[:,positive].T@rhs)/values[positive])
            if null.shape[1]:nullspaces[str(j)]=null
            condition=float(values[-1]/values[positive][0]) if np.any(positive) else 1.
            fallback=True
        if fallback:fallbacks.append(j)
        Q=L[h:,h:];rhs=-L[h:,:h]@W[:,j]
        normal_residual.append(la.norm(Q@Z[:,j]-rhs)/max(la.norm(rhs),1e-300))
        conditions.append(condition)
        x=np.r_[W[:,j],Z[:,j]]
        objective.append(x@L@x);residual.append(x@(L-eta*MP)@x)
    return Z,dict(eta=eta,objective=float(np.mean(objective)),
        energy_residual=float(np.sqrt(max(0.,np.mean(residual)))),
        maximum_linear_residual=float(max(normal_residual)),
        maximum_coefficient_condition_estimate_1norm=float(max(conditions)),
        condition_diagnostic='LAPACK POCON 1-norm estimate from existing Cholesky factor; not baseline spectral condition',
        residual_form_columns=fallbacks,minimum_residual_quadratic=float(min(residual)),
        coefficient_norm=float(la.norm(Z)),zero_cost_directions=nullspaces,
        J=float(np.sum(residual)/2),unique_in_retained_affine_family=not bool(nullspaces))


def constrained_energy_coefficients(W,omega,operators,dictionary,eta,margin,Z0,
                                    max_iterations=600,tolerance=2e-7,*,factor_cache_bytes=512*1024**2,
                                    acceptance_margin=None,feasible_starts=(),adaptive_penalty=False):
    """ADMM for the FE quadratic criterion and a finite singular-value floor.

    QR compresses the represented map without changing its singular values.
    The affine coefficient step preserves anchors exactly. The spectral step
    projects onto sigma_min >= margin. The constraint is nonconvex; this is a
    deterministic local solve, with its residuals and termination archived.
    """
    KP,MP,HP=operators;h,r=W.shape;d=Z0.shape[0]
    if dictionary.shape[0]<r or dictionary.shape[1]<r:
        raise ValueError('represented trial space has fewer than r directions')
    acceptance_margin=margin*(1-1e-6) if acceptance_margin is None else acceptance_margin
    if not 0<acceptance_margin<=margin:raise ValueError('invalid independence margins')
    if not all(np.all(np.isfinite(x)) for x in [W,omega,dictionary,Z0,*operators]):
        raise ValueError('nonfinite constrained coefficient inputs')
    # Raw QR avoids forming Q: only the compact triangular factor is needed.
    R=la.qr(dictionary,mode='raw',check_finite=False)[1]
    base=R[:,:h]@W;RN=R[:,h:];G=RN.T@RN
    rhs=[];constant=0.;traces=[];stable_blocks={}
    for j,w in enumerate(omega):
        try:L,_,_,fallback=_coefficient_factor(_column_quadratic(operators,w,eta),h,operators,w,eta)
        except la.LinAlgError:
            if eta!=0 or getattr(operators,'residual_form',None) is None:raise
            L=operators.residual_form(w,0.);fallback=True
        if fallback:stable_blocks[j]=L[h:,h:].copy()
        traces.append(np.trace(L[h:,h:])/d)
        rhs.append(-L[h:,:h]@W[:,j]);constant+=W[:,j]@L[:h,:h]@W[:,j]
    rhs=np.array(rhs).T
    KNN,MNN,HNN=(x[h:,h:] for x in operators);squared=omega**2
    def hessian(j):
        if j in stable_blocks:return stable_blocks[j]
        Q=KNN/squared[j]+(eta-2)*MNN+squared[j]*HNN
        return (Q+Q.T)*.5
    def cost(Z):
        HZ=(KNN@Z)/squared[None,:]+(eta-2)*(MNN@Z)+(HNN@Z)*squared[None,:]
        for j,H in stable_blocks.items():HZ[:,j]=H@Z[:,j]
        return float((constant+np.sum(Z*HZ)-2*np.sum(Z*rhs))/r)
    def projection(Y):
        u,s,vh=la.svd(Y,full_matrices=False,check_finite=False)
        return (u*np.maximum(s,margin))@vh
    rho=float(np.median(traces)/max(np.trace(G)/d,1e-30))
    rho=max(rho*100,1e-6)
    Z=Z0.copy();Y=projection(base+RN@Z);dual=np.zeros_like(Y);history=[]
    best=None;best_cost=np.inf;best_iteration=None;feasible_count=0
    def retain(candidate,iteration):
        nonlocal best,best_cost,best_iteration,feasible_count
        value=cost(candidate)
        if value<best_cost:
            singular=la.svdvals(base+RN@candidate,check_finite=False)
            if len(singular)==r and singular[-1]>=acceptance_margin:
                best=candidate.copy();best_cost=value;best_iteration=iteration;feasible_count+=1
    retain(Z,0)
    for i,candidate in enumerate(feasible_starts):retain(candidate,-i-1)
    # A fixed prefix remains cached even below the all-column memory budget;
    # unlike cyclic LRU this still yields hits during every sequential sweep.
    capacity=max(0,min(r,int(factor_cache_bytes)//(d*d*8)))
    factors={};factorizations=0;cache_hits=0;penalty_changes=0;termination='iteration budget'
    for iteration in range(max_iterations):
        target=RN.T@(Y-dual-base)
        for j in range(r):
            if j in factors:
                factor=factors[j];cache_hits+=1
            else:
                factor=la.cho_factor(hessian(j)+rho*G,check_finite=False)
                factorizations+=1
                if j<capacity:factors[j]=factor
            Z[:,j]=la.cho_solve(factor,rhs[:,j]+rho*target[:,j],check_finite=False)
        X=base+RN@Z;old=Y;Y=projection(X+dual);dual+=X-Y
        primal=float(la.norm(X-Y));dr=float(rho*la.norm(RN.T@(Y-old)))
        scale=max(la.norm(X),1.)
        retain(Z,iteration+1)
        if iteration%10==0:history.append(dict(iteration=iteration,objective=cost(Z),primal=primal/scale,dual=dr,rho=rho))
        if iteration%100==0:print('completion iteration',iteration,'primal',primal/scale,'dual',dr/max(la.norm(rhs),1.),flush=True)
        if primal/scale<tolerance and dr/max(la.norm(rhs),1.)<tolerance:
            termination='primal and dual residual';break
        if iteration%25==24:
            # Residual balancing changes solver penalty only, never the prior.
            if primal/scale>max(10*tolerance,1e-5):
                rho*=2;dual/=2;factors.clear();penalty_changes+=1
            elif adaptive_penalty and dr/max(la.norm(rhs),1.)>10*max(primal/scale,tolerance*.01):
                rho/=2;dual*=2;factors.clear();penalty_changes+=1
    last_cost=cost(Z);last_min=float(la.svdvals(X)[-1]);last_scale=max(la.norm(X),1.)
    if best is None:raise RuntimeError('no feasible physical completion found; no map accepted')
    Z=best;X=base+RN@Z;s=la.svdvals(X)
    return Z,dict(termination=termination,iterations=iteration+1,history=history,
        objective=cost(Z),primal_relative=primal/last_scale,
        dual_relative=dr/max(la.norm(rhs),1.),minimum_singular_value=float(s[-1]),
        requested_margin=margin,penalty=rho,
        coefficient_factorizations=factorizations,factor_cache_hits=cache_hits,
        factor_cache_capacity=capacity,factor_cache_bytes=capacity*d*d*8,
        factor_cache_budget_bytes=int(factor_cache_bytes),penalty_changes=penalty_changes,
        residual_form_columns=sorted(stable_blocks),
        accepted_iteration=best_iteration,feasible_improvements=feasible_count,
        acceptance_margin=acceptance_margin,accepted_J=cost(Z)*r/2,
        final_iterate_objective=last_cost,final_iterate_minimum=last_min,
        stopping_residuals_refer_to='last iteration; accepted map is best physically feasible candidate',
        converged=termination=='primal and dual residual',
        stationary_optimality_certified=False)


# ---------------------------------------------------------------------------
# Equation-of-motion completion under the measured forces (criterion J_F).
#
# With q(s)=Delta(s)^-1 W_i^T f(s) the response of the fitted model to the
# measured forces, and the fitted damping force M0 T Gamma qdot, the nominal
# F=ma residual of u=Tq under the load carried by the modes, M0 T T^T B f, is
#     r(s) = (K0 T - M0 T Omega^2) q(s).
# J_F(T) = tr(G^T K0^-1 G Pi) / tr(Omega^2 Pi),  G = K0 T - M0 T Omega^2,
# Pi = Re sum_l q_l q_l^H dnu.  Pi = Omega^-2 recovers 2 J_0 / r.
# ---------------------------------------------------------------------------

def modal_response_weight(omega,Gamma,Wi,frequency,force_spectra,chunk=256):
    """Pi = Re sum_l X_l G_l X_l^H dnu with X_l = Delta(s_l)^-1 Wi^T.

    force_spectra[l] is the one-sided force spectral matrix with
    G_ij = mean(F_i conj(F_j)) per Hz, so X G X^H is the modal response
    spectral matrix. The frequency grid must be uniform.
    """
    from .mechanical import modal_resolvent
    omega=np.asarray(omega,float);frequency=np.asarray(frequency,float)
    Wi=np.atleast_2d(np.asarray(Wi,float));G=np.asarray(force_spectra)
    r=len(omega);p=Wi.shape[0]
    if Wi.shape[1]!=r or G.shape!=(len(frequency),p,p):raise ValueError('shape mismatch')
    step=np.diff(frequency)
    if not np.allclose(step,step[0],rtol=1e-9,atol=0):raise ValueError('uniform frequency grid required')
    lam,phi=modal_resolvent(omega,Gamma);Y=phi.T@Wi.T
    Pi=np.zeros((r,r))
    for start in range(0,len(frequency),chunk):
        s=2j*np.pi*frequency[start:start+chunk]
        X=np.einsum('ak,lkp->lap',phi,Y[None,:,:]/(s[:,None,None]-lam[None,:,None]),optimize=True)
        Pi+=np.einsum('lap,lpq,lbq->ab',X,G[start:start+chunk],X.conj(),optimize=True).real
    Pi*=step[0];Pi=(Pi+Pi.T)*.5
    values=la.eigvalsh(Pi)
    if values[0]<-1e-10*values[-1]:raise ArithmeticError('modal response weight is not positive semidefinite')
    return Pi,dict(spacing_hz=float(step[0]),lines=len(frequency),
        trace=float(np.trace(Pi)),minimum_eigenvalue=float(values[0]),maximum_eigenvalue=float(values[-1]),
        potential=float(np.sum(omega**2*np.diag(Pi))))


def _forced_terms(omega,Pi):
    O=np.diag(omega**2)
    return Pi,O@Pi+Pi@O,O@Pi@O,float(np.trace(O@Pi))


def forced_objective(W,Z,omega,operators,Pi):
    """J_F from the three projected blocks for T = E W + N Z."""
    KP,MP,HP=operators;X=np.vstack([W,Z]);P,S,U,scale=_forced_terms(omega,Pi)
    value=np.sum((KP@X)*(X@P))-np.sum((MP@X)*(X@S))+np.sum((HP@X)*(X@U))
    return float(value/scale)


def forced_objective_full(K,M,T,omega,Pi,solve):
    """J_F evaluated on the full FE residual with a K0 solve (no projection)."""
    G=K@T-(M@T)*omega[None,:]**2;Y=solve(G)
    return float(np.sum((G.T@Y)*Pi)/np.sum(omega**2*np.diag(Pi)))


class _ForcedSystem:
    """Hessian Z -> Kzz Z Pi - Mzz Z S + Hzz Z U (+ rho G Z), block-Jacobi PCG."""
    def __init__(self,W,omega,operators,Pi,h):
        KP,MP,HP=operators;self.h=h;self.W=W;self.omega=omega
        self.P,self.S,self.U,self.scale=_forced_terms(omega,Pi)
        self.Kzz,self.Mzz,self.Hzz=(x[h:,h:] for x in operators)
        Kzw,Mzw,Hzw=(x[h:,:h] for x in operators)
        self.rhs=-(Kzw@W@self.P-Mzw@W@self.S+Hzw@W@self.U)
        Kww,Mww,Hww=(x[:h,:h] for x in operators)
        self.constant=float(np.sum((Kww@W)*(W@self.P))-np.sum((Mww@W)*(W@self.S))+np.sum((Hww@W)*(W@self.U)))
        self.factors=None;self.rho=None;self.G=None
    def apply(self,Z):
        out=self.Kzz@Z@self.P-self.Mzz@Z@self.S+self.Hzz@Z@self.U
        if self.rho:out+=self.rho*(self.G@Z)
        return out
    def cost(self,Z):
        HZ=self.Kzz@Z@self.P-self.Mzz@Z@self.S+self.Hzz@Z@self.U
        return float((self.constant+np.sum(Z*HZ)-2*np.sum(Z*self.rhs))/self.scale)
    def factor(self,rho=0.,G=None):
        """Explicit inverses of the column blocks; one batched product per PCG step."""
        self.rho=rho;self.G=G;d=self.Kzz.shape[0];w2=self.omega**2;r=len(self.omega)
        self.inverse=np.empty((r,d,d));I=np.eye(d)
        for j in range(r):
            Q=self.P[j,j]*(self.Kzz-2*w2[j]*self.Mzz+w2[j]**2*self.Hzz)
            if rho:Q=Q+rho*G
            Q=(Q+Q.T)*.5
            try:f=la.cho_factor(Q,check_finite=False)
            except la.LinAlgError:
                f=la.cho_factor(Q+1e-12*np.trace(Q)/d*I,check_finite=False)
            self.inverse[j]=la.cho_solve(f,I,check_finite=False)
        self.factors=True
    def precondition(self,R):
        return np.einsum('jab,bj->aj',self.inverse,R,optimize=True)
    def solve(self,b,Z0,tolerance=1e-10,max_iterations=5000):
        Z=Z0.copy();R=b-self.apply(Z);Y=self.precondition(R);D=Y.copy()
        ry=np.sum(R*Y);bnorm=max(la.norm(b),1e-300)
        for k in range(max_iterations):
            if la.norm(R)/bnorm<tolerance:return Z,dict(iterations=k,relative_residual=float(la.norm(R)/bnorm))
            AD=self.apply(D);alpha=ry/np.sum(D*AD)
            Z+=alpha*D;R-=alpha*AD;Y=self.precondition(R);ry_new=np.sum(R*Y)
            D=Y+(ry_new/ry)*D;ry=ry_new
        return Z,dict(iterations=max_iterations,relative_residual=float(la.norm(R)/bnorm),converged=False)


def forced_coefficients(W,omega,operators,Pi,Z0=None,tolerance=1e-11,max_iterations=5000):
    """Equality-only J_F minimizer over Z for T = E W + N Z (no margin)."""
    h,r=W.shape;system=_ForcedSystem(W,omega,operators,Pi,h);system.factor()
    Z0=np.zeros((len(operators[0])-h,r)) if Z0 is None else Z0
    Z,info=system.solve(system.rhs,Z0,tolerance,max_iterations)
    stationarity=float(la.norm(system.apply(Z)-system.rhs)/max(la.norm(system.rhs),1e-300))
    return Z,dict(info,J_F=system.cost(Z),stationarity_relative=stationarity,
        J_F_from_blocks=forced_objective(W,Z,omega,operators,Pi),
        coefficient_norm=float(la.norm(Z)))


def constrained_forced_coefficients(W,omega,operators,Pi,dictionary,margin,Z0,
                                    max_iterations=600,tolerance=2e-7,*,acceptance_margin=None,
                                    feasible_starts=(),inner_tolerance=1e-9,log_every=50,rho_scale=100.):
    """ADMM for J_F with the singular-value floor sigma_r(RT) >= margin.

    Same splitting as constrained_energy_coefficients: an exact affine step that
    keeps AT=W, then a spectral projection onto sigma_min >= margin. The
    coefficient step is the coupled J_F system, solved by warm-started PCG.
    """
    h,r=W.shape;d=Z0.shape[0]
    acceptance_margin=margin*(1-1e-6) if acceptance_margin is None else acceptance_margin
    R=la.qr(dictionary,mode='raw',check_finite=False)[1]
    base=R[:,:h]@W;RN=R[:,h:];G=RN.T@RN
    system=_ForcedSystem(W,omega,operators,Pi,h)
    def projection(Y):
        u,s,vh=la.svd(Y,full_matrices=False,check_finite=False)
        return (u*np.maximum(s,margin))@vh
    diag_scale=np.median([system.P[j,j]*np.trace(system.Kzz-2*omega[j]**2*system.Mzz+omega[j]**4*system.Hzz)/d for j in range(r)])
    rho=max(rho_scale*diag_scale/max(np.trace(G)/d,1e-30),1e-30)
    system.factor(rho,G)
    Z=Z0.copy();Y=projection(base+RN@Z);dual=np.zeros_like(Y);history=[]
    best=None;best_cost=np.inf;best_iteration=None;feasible_count=0;inner=[]
    rhs_norm=max(la.norm(system.rhs),1e-300)
    def retain(candidate,iteration):
        nonlocal best,best_cost,best_iteration,feasible_count
        value=system.cost(candidate)
        if value<best_cost:
            singular=la.svdvals(base+RN@candidate,check_finite=False)
            if singular[-1]>=acceptance_margin:
                best=candidate.copy();best_cost=value;best_iteration=iteration;feasible_count+=1
    retain(Z,0)
    for i,candidate in enumerate(feasible_starts):retain(candidate,-i-1)
    termination='iteration budget';penalty_changes=0
    for iteration in range(max_iterations):
        target=RN.T@(Y-dual-base)
        Z,info=system.solve(system.rhs+rho*target,Z,inner_tolerance);inner.append(info['iterations'])
        X=base+RN@Z;old=Y;Y=projection(X+dual);dual+=X-Y
        primal=float(la.norm(X-Y));scale=max(la.norm(X),1.)
        dr=float(rho*la.norm(RN.T@(Y-old)))
        retain(Z,iteration+1)
        if iteration%10==0:history.append(dict(iteration=iteration,objective=system.cost(Z),primal=primal/scale,dual=dr/rhs_norm,rho=rho,inner=info['iterations']))
        if iteration%log_every==0:print('J_F completion',iteration,'J_F',system.cost(Z),'primal',primal/scale,'dual',dr/rhs_norm,'pcg',info['iterations'],flush=True)
        if primal/scale<tolerance and dr/rhs_norm<tolerance:
            termination='primal and dual residual';break
        if iteration%25==24:
            if primal/scale>max(10*tolerance,1e-5):
                rho*=2;dual/=2;system.factor(rho,G);penalty_changes+=1
            elif dr/rhs_norm>10*max(primal/scale,tolerance*.01):
                rho/=2;dual*=2;system.factor(rho,G);penalty_changes+=1
    if best is None:raise RuntimeError('no feasible completion found; no map accepted')
    last=system.cost(Z);Zb=best;s=la.svdvals(base+RN@Zb)
    return Zb,dict(termination=termination,iterations=iteration+1,history=history,
        J_F=system.cost(Zb),primal_relative=primal/scale,dual_relative=dr/rhs_norm,
        minimum_singular_value=float(s[-1]),requested_margin=margin,acceptance_margin=acceptance_margin,
        penalty=rho,penalty_changes=penalty_changes,accepted_iteration=best_iteration,
        feasible_improvements=feasible_count,final_iterate_J_F=last,
        inner_pcg_iterations=dict(total=int(np.sum(inner)),maximum=int(np.max(inner))),
        converged=termination=='primal and dual residual',stationary_optimality_certified=False)
