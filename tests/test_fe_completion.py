"""Scientific invariants of the post-fit FE completion and physical outputs."""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from fetune.fe_completion import energy_dictionary,project_operators,energy_coefficients,constrained_energy_coefficients,physical_rows
from fetune.spatial_completion import physical_realization
from unittest.mock import patch
from fetune.fe_completion import ElasticContext, _mass_normalize, _partial_fields
from fetune.completion_diagnostics import column_spectrum,physical_diagnostics


def test_fe_completion_preserves_anchors_and_basis_orientation():
    K=sp.csc_matrix([[4.,-1,0,0],[-1,3,-1,0],[0,-1,3,-1],[0,0,-1,4]])
    M=sp.diags([1.,2.,1.,2.],format='csc');A=np.eye(4)[[0]]
    W=np.array([[.3,-.2]]);omega=np.array([1.1,1.7])
    E,N,checks=energy_dictionary(K,M,A,2)
    ops,_=project_operators(K,M,E,N);Z,_=energy_coefficients(W,omega,ops)
    theta=.37;O=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    ops2,_=project_operators(K,M,E,N@O);Z2,_=energy_coefficients(W,omega,ops2)
    T=E@W+N@Z
    assert np.allclose(A@T,W,atol=1e-13)
    assert np.allclose(T,E@W+N@O@Z2,atol=1e-12)
    assert checks['hidden_mass_gram']<1e-12


def test_finite_floor_restores_a_missing_direction():
    E=np.array([[1.],[0.]]);N=np.array([[0.],[1.]])
    W=np.array([[1.,0.]]);omega=np.ones(2)
    ops,_=project_operators(sp.eye(2),sp.eye(2),E,N)
    Z,unconstrained=energy_coefficients(W,omega,ops)
    assert la.svdvals(E@W+N@Z)[-1]==0
    Z,check=constrained_energy_coefficients(W,omega,ops,np.eye(2),.01,.2,Z,
        max_iterations=100,tolerance=1e-9)
    T=E@W+N@Z
    assert np.array_equal(T[0],W[0])
    assert la.svdvals(T)[-1]>.2*(1-1e-6)
    assert check['primal_relative']<1e-8


def test_physical_force_outputs_and_feedthrough_with_full_expansion():
    T=np.array([[1.,.1],[.4,.7],[.2,-.3],[.8,.2]])
    Gamma=np.array([[.2,.03],[.03,.3]]);omega=np.array([2.,3.])
    rows,_=physical_rows(T);p=physical_realization(T,Gamma,omega,rows)
    E=p['expansion'];B=np.eye(4)[:,[0,3]];S=np.eye(4)[[1,2]]
    Br=E.T@B;Sr=S@E;M,C,K=[p[k] for k in ['M','C','K']]
    for frequency in [.5,1.,2.]:
        s=2j*np.pi*frequency
        un= S@T@la.solve(s*s*np.eye(2)+s*Gamma+np.diag(omega**2),T.T@B)
        up=Sr@la.solve(s*s*M+s*C+K,Br)
        for power in [0,1,2]:assert np.allclose(s**power*up,s**power*un,rtol=1e-12,atol=1e-12)
    assert np.allclose(Sr@la.solve(M,Br),S@T@T.T@B,atol=1e-12)


def test_nominal_full_mass_projection_is_not_a_principal_submatrix():
    K=np.array([[4.,1.,-1.],[1.,3.,.5],[-1.,.5,5.]])
    M=np.array([[2.,.1,0],[.1,1.,.2],[0,.2,3.]])
    keep=np.array([0,2]);omitted=np.array([1]);H=np.zeros((3,2));H[keep]=np.eye(2)
    H[omitted]=-la.solve(K[np.ix_(omitted,omitted)],K[np.ix_(omitted,keep)])
    assert np.allclose((K@H)[omitted],0)
    assert not np.allclose(H.T@M@H,M[np.ix_(keep,keep)])


def test_projected_recovery_matches_explicit_static_selection_with_interpolated_ports():
    rng=np.random.default_rng(20261112);n=12
    F=rng.normal(size=(n,n));K=F.T@F+3*np.eye(n)
    F=rng.normal(size=(n,n));M=F.T@F+np.eye(n)
    rows=np.array([0,2,3,5,6,8,9,11]);omitted=np.setdiff1d(np.arange(n),rows)
    A=np.zeros((2,n));A[0,[0,3]]=[.3,.7];A[1,[5,9]]=[-.4,.6]
    context=ElasticContext(K,A)
    E,N,check=energy_dictionary(K,M,A,3,represented=rows,context=context,dense_threshold=0,batch_size=2)
    H=np.zeros((n,len(rows)));H[rows]=np.eye(len(rows))
    H[omitted]=-la.solve(K[np.ix_(omitted,omitted)],K[np.ix_(omitted,rows)])
    U=la.null_space(A[:,rows]);metric=np.diag(M)[rows]
    vals,V=la.eigh(U.T@H.T@K@H@U,U.T@(metric[:,None]*U))
    expected,_=_mass_normalize(H@U@V[:,:3],M)
    assert np.allclose(la.svdvals(expected.T@M@N),1,atol=3e-13)
    assert np.allclose(check['selection_eigenvalues'],vals[:3],rtol=2e-13)
    assert np.allclose(A@E,np.eye(2),atol=1e-14)
    assert np.allclose((K@N)[omitted],0,atol=1e-13)
    assert check['recovery_displayed_relative']<1e-12
    assert check['selection_eigenpair_relative_max']<1e-12
    assert check['constrained_eigen_residual']<1e-12
    ops,_=project_operators(K,M,E,N,context=context,batch_size=2)
    B=np.c_[E,N]
    assert np.allclose(ops[2],(M@B).T@la.solve(K,M@B),atol=1e-13)
    # Full mass and displayed selection metric genuinely differ in this fixture.
    assert not np.allclose(N[rows].T@(metric[:,None]*N[rows]),np.eye(3))


def test_retention_boundary_extends_request_in_stiffness_units():
    import scipy.sparse.linalg as sla
    # The cluster is wider than the initial eight-pair lookahead. Its large
    # absolute stiffness makes a reciprocal-unit tolerance incorrect.
    values=np.r_[1e6,np.full(15,2e6),np.arange(3,33)*1e6]
    calls=[]
    def eigenpairs(op,k,**kwargs):
        calls.append(k)
        return 1/values[:k],np.eye(len(values))[:,:k]
    op=sla.aslinearoperator(np.diag(1/values))
    with patch('fetune.fe_completion.sla.eigsh',side_effect=eigenpairs):
        _,_,retained,requests=_partial_fields(op,3,len(values)-1,1e-10,1e-8)
    assert retained==16 and len(requests)>1 and requests==calls


def test_eigensolver_nonconvergence_does_not_accept_partial_basis():
    import scipy.sparse.linalg as sla
    failure=sla.ArpackNoConvergence('test',np.array([1.]),np.ones((20,1)))
    with patch('fetune.fe_completion.sla.eigsh',side_effect=failure):
        try:_partial_fields(sla.aslinearoperator(np.eye(20)),3,19,1e-10,1e-8)
        except RuntimeError as error:assert 'no partial basis accepted' in str(error)
        else:raise AssertionError('partial basis was accepted')


def test_context_reuses_factor_but_rejects_changed_stiffness():
    K=np.diag([2.,3.,4.]);A=np.array([[.25,.75,0.]])
    ctx=ElasticContext(K,A);factor=ctx.solve.factor
    other=np.array([[0.,.5,.5]])
    ctx.check(K,other)
    assert ctx.solve.factor is factor and np.allclose(other@ctx.E,1.)
    K[0,0]=5.
    try:ctx.check(K)
    except ValueError as error:assert 'stiffness changed' in str(error)
    else:raise AssertionError('stale factor accepted')


def test_rank_deficient_mass_gram_is_not_regularized_or_truncated():
    try:_mass_normalize(np.ones((4,2)),np.eye(4))
    except la.LinAlgError as error:assert 'rank deficient (1/2' in str(error)
    else:raise AssertionError('dependent hidden fields silently retained')


def test_resonant_polynomial_cancellation_uses_original_residual():
    K=np.diag([1.,2.]);M=np.eye(2);A=np.array([[0.,1.]])
    ctx=ElasticContext(K,A);N=np.array([[1.],[0.]])
    ops,_=project_operators(K,M,ctx.E,N,context=ctx)
    Z,check=energy_coefficients(np.array([[.2]]),np.array([1.]),ops,eta=1e-18)
    assert check['residual_form_columns']==[0]
    assert np.array_equal(Z,np.zeros((1,1)))
    assert np.isclose(check['objective'],.02)


def test_bounded_factor_cache_preserves_constrained_iteration():
    E=np.array([[1.],[0.]]);N=np.array([[0.],[1.]])
    W=np.array([[1.,0.]]);omega=np.ones(2)
    ops,_=project_operators(np.eye(2),np.eye(2),E,N)
    Z=np.zeros((1,2))
    args=(W,omega,ops,np.eye(2),.01,.2,Z)
    full,fc=constrained_energy_coefficients(*args,max_iterations=10,tolerance=1e-16)
    small,sc=constrained_energy_coefficients(*args,max_iterations=10,tolerance=1e-16,factor_cache_bytes=8)
    assert np.array_equal(full,small)
    assert fc['coefficient_factorizations']==2 and fc['factor_cache_hits']==2*(fc['iterations']-1)
    assert sc['coefficient_factorizations']==sc['iterations']+1 and sc['factor_cache_bytes']==8


def test_wide_matrix_and_collectively_dependent_columns_fail_independence():
    wide=column_spectrum(np.array([[1.,0.,1.],[0.,1.,1.]]))
    assert len(wide['singular_values'])==3 and wide['minimum']==0
    assert not wide['full_column_rank']
    dependent=np.array([[1.,0.,1.],[0.,1.,1.],[1.,1.,2.]])
    assert np.all(la.norm(dependent,axis=0)>0)
    assert column_spectrum(dependent)['rank']==2
    try:constrained_energy_coefficients(np.eye(2,3),np.ones(3),
        [np.eye(3)]*3,np.eye(2,3),0.,.1,np.zeros((1,3)))
    except ValueError as error:assert 'fewer than r' in str(error)
    else:raise AssertionError('wide auxiliary map incorrectly certified')


def test_zero_ridge_resonance_retains_zero_cost_directions():
    E=np.array([[1.],[0.]]);N=np.array([[0.],[1.]])
    W=np.array([[1.,0.]]);omega=np.ones(2)
    ops,_=project_operators(np.eye(2),np.eye(2),E,N)
    Z,check=energy_coefficients(W,omega,ops,0.)
    assert set(check['zero_cost_directions'])=={'0','1'}
    assert check['J']==0 and not check['unique_in_retained_affine_family']
    warm=np.array([[0.,.3]])
    out,result=constrained_energy_coefficients(W,omega,ops,np.eye(2),0.,.2,Z,
        max_iterations=10,acceptance_margin=.19,feasible_starts=[warm])
    assert la.svdvals(E@W+N@out)[-1]>=.19
    assert result['objective']==0


def test_lower_cost_infeasible_iterate_cannot_replace_feasible_completion():
    E=np.array([[1.],[0.]]);N=np.array([[0.],[1.]])
    W=np.array([[1.,0.]]);omega=np.ones(2)
    ops,_=project_operators(np.diag([1.,2.]),np.eye(2),E,N)
    warm=np.array([[0.,.3]])
    out,status=constrained_energy_coefficients(W,omega,ops,np.eye(2),0.,.2,
        np.zeros((1,2)),max_iterations=1,acceptance_margin=.2,feasible_starts=[warm])
    assert np.array_equal(out,warm)
    assert status['final_iterate_minimum']<.2
    assert status['final_iterate_objective']<status['objective']
    assert status['accepted_iteration']==-1


def test_zero_ridge_completion_covaries_with_nonorthogonal_hidden_basis():
    K=np.array([[4.,-1.,0.],[-1.,3.,-1.],[0.,-1.,2.]])
    M=np.diag([1.,2.,1.]);A=np.eye(3)[[0]]
    E,N,_=energy_dictionary(K,M,A,2)
    W=np.array([[.3,-.2,.1]]);omega=np.array([1.1,1.7,2.1])
    F=np.array([[2.,.7],[0.,.3]])
    ops,_=project_operators(K,M,E,N)
    changed,_=project_operators(K,M,E,N@F)
    Z,_=energy_coefficients(W,omega,ops,0.)
    Z2,_=energy_coefficients(W,omega,changed,0.)
    assert np.allclose(N@Z,N@F@Z2,atol=1e-12)
    # One affine minimization update must use RN.T RN, not an identity metric.
    kwargs=dict(max_iterations=1,acceptance_margin=.00001)
    out,_=constrained_energy_coefficients(W,omega,ops,np.c_[E,N],0.,.00002,Z,**kwargs)
    out2,_=constrained_energy_coefficients(W,omega,changed,np.c_[E,N@F],0.,.00002,Z2,**kwargs)
    assert np.allclose(N@out,N@F@out2,atol=1e-11)


def test_full_force_residual_bound_and_factored_realization():
    K=np.array([[3.,-1.,0.],[-1.,3.,-1.],[0.,-1.,2.]])
    M=np.diag([1.,2.,1.]);T=np.array([[1.,.2],[.4,1.],[.1,-.3]])
    A=np.eye(3)[[0]];w=np.array([1.,2.]);G=np.array([[.3,.1],[.1,.5]])
    report=physical_diagnostics(K,M,A,T,A@T,w,G,np.arange(3),.1)
    g=K@T-(M@T)*w[None,:]**2
    B=la.solve_triangular(la.cholesky(K,lower=True),g,lower=True)/w[None,:]
    assert np.isclose(report['objective_J'],la.norm(B)**2/2)
    assert la.norm(B,2)<=np.sqrt(2*report['objective_J'])*(1+1e-14)
    assert max(report['realization'][k] for k in ['mass','damping','stiffness','KT_MT_Omega'])<1e-14
    assert report['port_invisible']['dimension']==1
    assert report['port_invisible']['represented_minimum']>=.1
    assert report['left_inverse_frobenius']<1e-14


def test_forced_criterion_reduces_to_undamped_and_matches_full_residual():
    from fetune.fe_completion import (forced_coefficients, forced_objective, forced_objective_full,
                                      modal_response_weight, elastic_solver)
    K=sp.csc_matrix([[4.,-1,0,0],[-1,3,-1,0],[0,-1,3,-1],[0,0,-1,4]])
    M=sp.diags([1.,2.,1.,2.],format='csc');A=np.eye(4)[[0]]
    W=np.array([[.3,-.2]]);omega=np.array([1.1,1.7])
    E,N,_=energy_dictionary(K,M,A,2);ops,_=project_operators(K,M,E,N)
    # Pi = Omega^-2 is the undamped per-mode criterion.
    Z0,_=energy_coefficients(W,omega,ops,eta=0.)
    Zf,info=forced_coefficients(W,omega,ops,np.diag(omega**-2.))
    assert np.allclose(Zf,Z0,rtol=1e-9,atol=1e-12) and info['stationarity_relative']<1e-9
    # A forced weight from a damped model: projected and full-FE J_F agree,
    # and the minimizer is stationary (Theorem 1(ii)).
    Gamma=np.array([[.05,.01],[.01,.08]]);f=np.linspace(.05,.4,300)
    Pi,_=modal_response_weight(omega,Gamma,W,f,np.ones((len(f),1,1)))
    Z,info=forced_coefficients(W,omega,ops,Pi)
    T=E@W+N@Z
    assert abs(forced_objective(W,Z,omega,ops,Pi)-forced_objective_full(K,M,T,omega,Pi,elastic_solver(K)))<1e-10
    for scale in [1e-3,-1e-3]:
        assert forced_objective(W,Z+scale*np.ones_like(Z),omega,ops,Pi)>=info['J_F']-1e-14
