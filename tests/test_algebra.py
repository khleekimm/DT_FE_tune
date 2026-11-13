"""Independent finite-dimensional identities used in the revised exposition."""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from fetune.fe_completion import elastic_solver, energy_dictionary, project_operators, energy_coefficients


def system():
    rng=np.random.default_rng(111126)
    x=rng.normal(size=(12,12));K=x.T@x+2*np.eye(12)
    x=rng.normal(size=(12,12));M=x.T@x+np.eye(12)
    A=np.zeros((3,12));A[:,:8]=rng.normal(size=(3,8))
    return rng,K,M,A


def test_weighted_minimum_norm_and_affine_energy_decomposition():
    rng,K,M,A=system();w=rng.normal(size=3)
    E,N,_=energy_dictionary(K,M,A,4)
    eig,U=la.eigh(K);root=(U*np.sqrt(eig))@U.T
    inverse_root=(U/eig**.5)@U.T;AK=A@inverse_root
    y0=AK.T@la.solve(AK@AK.T,w,assume_a='pos')
    t0=E@w
    np.testing.assert_allclose(root@t0,y0,rtol=1e-12,atol=1e-12)
    # Use the entire nullspace, not only the retained hidden fields.
    d=la.null_space(A)@rng.normal(size=9)
    np.testing.assert_allclose(t0@K@d,0,atol=1e-12)
    np.testing.assert_allclose((t0+d)@K@(t0+d),t0@K@t0+d@K@d,rtol=1e-12)
    np.testing.assert_allclose(E.T@K@N,0,atol=1e-12)
    W=rng.normal(size=(3,5));Z=rng.normal(size=(4,5));T=E@W+N@Z
    np.testing.assert_allclose(A@T,W,atol=1e-12)
    np.testing.assert_allclose(np.trace(T.T@K@T),
        np.trace(W.T@E.T@K@E@W)+np.trace(Z.T@N.T@K@N@Z),rtol=1e-12)


def test_static_lift_is_block_square_not_principal_deletion():
    rng,K,M,A=system();dd=np.arange(8);oo=np.arange(8,12)
    solve=elastic_solver(sp.csc_matrix(K[np.ix_(oo,oo)]))
    coupling=solve(K[np.ix_(oo,dd)])
    L=np.vstack([np.eye(8),-coupling]);Kd=L.T@K@L
    schur=K[:8,:8]-K[:8,8:]@coupling
    np.testing.assert_allclose(Kd,schur,rtol=1e-12,atol=1e-12)
    ud=rng.normal(size=8);uo=rng.normal(size=4);u=np.r_[ud,uo]
    remainder=uo+coupling@ud
    np.testing.assert_allclose(u@K@u,ud@schur@ud+remainder@K[8:,8:]@remainder,rtol=1e-12)
    np.testing.assert_allclose((K@L)[8:],0,atol=1e-12)
    assert la.norm(L.T@M@L-M[:8,:8])>.1


def test_nullspace_symmetric_eigenproblem_matches_sparse_constrained_inverse():
    rng,K,M,A=system();solve=elastic_solver(K)
    E,_,_=energy_dictionary(K,M,A,4)
    L=np.vstack([np.eye(8),-la.solve(K[8:,8:],K[8:,:8],assume_a='pos')])
    Kd=L.T@K@L;Gd=np.diag(np.diag(M)[:8]);UA=la.null_space(A@L)
    KN=UA.T@Kd@UA;GN=UA.T@Gd@UA
    xi,V=la.eigh(KN,GN)
    g,U=la.eigh(GN);inverse_root=(U/g**.5)@U.T
    values,Y=la.eigh(inverse_root@KN@inverse_root)
    np.testing.assert_allclose(values,xi,rtol=1e-12)
    fields=UA@inverse_root@Y
    np.testing.assert_allclose(fields.T@Gd@fields,np.eye(5),atol=1e-12)
    tau=rng.normal(size=5);field=fields@tau
    np.testing.assert_allclose(field@Kd@field/(field@Gd@field),
        (xi*tau**2).sum()/(tau**2).sum(),rtol=1e-12)
    inverse=solve(np.eye(12));inverse-=E@(A@inverse)
    np.testing.assert_allclose(inverse[:8,:8],UA@la.solve(KN,UA.T,assume_a='pos'),rtol=1e-12,atol=1e-12)
    root=np.sqrt(np.diag(Gd));operator=root[:,None]*inverse[:8,:8]*root[None,:]
    positive=la.eigvalsh(operator)[-5:][::-1]
    np.testing.assert_allclose(1/positive,xi,rtol=1e-12)
    # The selection metric remains the diagonal full-mass restriction.
    assert la.norm(Gd-L.T@M@L)>.1


def test_completing_square_matches_implemented_coefficient_solve():
    rng,K,M,A=system();E,N,_=energy_dictionary(K,M,A,4)
    ops,_=project_operators(K,M,E,N);W=rng.normal(size=(3,5))
    omega=np.array([.4,.7,1.,1.3,1.8]);eta=.01
    Z,_=energy_coefficients(W,omega,ops,eta)
    for j,w in enumerate(omega):
        b=E@W[:,j];D=K-w*w*M
        Q=D.T@la.solve(K,D,assume_a='pos')/w**2+eta*M
        restricted=N.T@Q@N;c=N.T@Q@b;star=Z[:,j]
        np.testing.assert_allclose(restricted@star,-c,rtol=1e-11,atol=1e-11)
        other=star+rng.normal(size=4)
        cost=lambda z:(b+N@z)@Q@(b+N@z)
        np.testing.assert_allclose(cost(other),cost(star)+(other-star)@restricted@(other-star),rtol=1e-12)


def test_backward_error_equals_smallest_symmetric_stiffness_correction():
    # Proposition: min |||dK||| over symmetric dK with (K0+dK-w^2 M0)t=0 equals
    # ||(K0-w^2M0)t||_{K0^-1}/||t||_{K0}, attained by the explicit construction.
    rng,K,M,A=system();t=rng.normal(size=12);w=.9
    eig,U=la.eigh(K);root=(U*np.sqrt(eig))@U.T;inverse_root=(U/eig**.5)@U.T
    g=(K-w*w*M)@t
    epsilon=np.sqrt(g@la.solve(K,g))/np.sqrt(t@K@t)
    tt=root@t;gt=inverse_root@g
    x=tt/la.norm(tt);rho=gt/la.norm(tt);alpha=x@rho;perp=rho-alpha*x;beta=la.norm(perp);y=perp/beta
    E=-alpha*np.outer(x,x)-beta*(np.outer(x,y)+np.outer(y,x))+alpha*np.outer(y,y)
    dK=root@E@root
    np.testing.assert_allclose(dK,dK.T,atol=1e-12)
    np.testing.assert_allclose((K+dK-w*w*M)@t,0,atol=1e-9)
    np.testing.assert_allclose(la.norm(inverse_root@dK@inverse_root,2),epsilon,rtol=1e-12)
    # No symmetric perturbation can do better: random feasible ones are larger.
    for _ in range(20):
        S=rng.normal(size=(12,12));S=S+S.T
        # Project S onto {dK symmetric: dK t = -g} by a symmetric rank-two correction.
        r=-g-S@t;u=t/la.norm(t)
        corr=np.outer(r,u)+np.outer(u,r)-(u@r)*np.outer(u,u)
        feasible=S+corr/la.norm(t)
        np.testing.assert_allclose(feasible@t,-g,atol=1e-9)
        assert la.norm(inverse_root@feasible@inverse_root,2)>=epsilon*(1-1e-12)
    # The nominal norm decomposes as the weighted squared backward error plus the ridge.
    eta=.01;Q=(K-w*w*M)@la.solve(K,K-w*w*M)/w**2+eta*M
    np.testing.assert_allclose(t@Q@t,(t@K@t)/w**2*epsilon**2+eta*t@M@t,rtol=1e-12)


def test_full_nullspace_completion_is_the_Q_minimal_interpolant():
    # Theorem (iii): when N spans ker A the two-step solution equals
    # Q^-1 A^T (A Q^-1 A^T)^-1 w and does not depend on the lift.
    rng,K,M,A=system();w=rng.normal(size=3);omega=1.3;eta=.01
    E,N,_=energy_dictionary(K,M,A,9)
    assert N.shape[1]==9 and la.norm(A@N)<1e-10
    Q=(K-omega**2*M)@la.solve(K,K-omega**2*M)/omega**2+eta*M
    z=-la.solve(N.T@Q@N,N.T@Q@E@w);t=E@w+N@z
    direct=la.solve(Q,A.T)@la.solve(A@la.solve(Q,A.T),w)
    np.testing.assert_allclose(t,direct,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(A@t,w,atol=1e-12)
    np.testing.assert_allclose(N.T@Q@t,0,atol=1e-10)
    # Another particular solution gives the same completed column.
    other=E@w+N@rng.normal(size=9)
    z2=-la.solve(N.T@Q@N,N.T@Q@other);np.testing.assert_allclose(other+N@z2,direct,rtol=1e-10,atol=1e-12)


def test_realization_identities_including_new_loads_and_outputs():
    rng=np.random.default_rng(2026);n,r,h=9,4,3
    T=rng.normal(size=(n,r));omega=np.array([.5,.9,1.4,2.1]);L0=rng.normal(size=(r,r));Gamma=L0@L0.T+.1*np.eye(r)
    _,_,piv=la.qr(T.T,pivoting=True);Pi=np.eye(n)[np.sort(piv[:r])];P=Pi@T;L=T@la.solve(P,np.eye(r))
    M=la.solve(P.T,la.solve(P,np.eye(r)).T).T if False else la.inv(P).T@la.inv(P)
    C=la.inv(P).T@Gamma@la.inv(P);Kp=la.inv(P).T@np.diag(omega**2)@la.inv(P)
    np.testing.assert_allclose(Pi@L,np.eye(r),atol=1e-12)
    Snew=rng.normal(size=(2,n));Bnew=rng.normal(size=(n,3))
    for s in [.3j,1.1j,2.4j]:
        physical=s*s*Snew@L@la.solve(s*s*M+s*C+Kp,L.T@Bnew)
        latent=s*s*Snew@T@la.solve(s*s*np.eye(r)+s*Gamma+np.diag(omega**2),T.T@Bnew)
        np.testing.assert_allclose(physical,latent,rtol=1e-10)
    np.testing.assert_allclose(Snew@L@la.solve(M,L.T@Bnew),Snew@T@T.T@Bnew,rtol=1e-10)
    np.testing.assert_allclose(np.linalg.cond(M),np.linalg.cond(P)**2,rtol=1e-9)
    M0=rng.normal(size=(n,n));M0=M0@M0.T+np.eye(n)
    np.testing.assert_allclose(M-L.T@M0@L,la.inv(P).T@(np.eye(r)-T.T@M0@T)@la.inv(P),rtol=1e-9)


def test_fe_space_realization_generalized_inverse_for_any_left_inverse():
    # Theorem (realization): for any left inverse G of T, R = T Delta^-1 T^T is a
    # generalized inverse of D = G^T Delta G, the latent and FE-space equations
    # agree on range T, and the selected-row choice G = P^-1 Pi is a special case.
    rng=np.random.default_rng(2027);n,r=9,4
    T=rng.normal(size=(n,r));omega=np.array([.5,.9,1.4,2.1]);L0=rng.normal(size=(r,r));Gamma=L0@L0.T+.1*np.eye(r)
    M0=rng.normal(size=(n,n));M0=M0@M0.T+np.eye(n)
    _,_,piv=la.qr(T.T,pivoting=True);Pi=np.eye(n)[np.sort(piv[:r])];P=Pi@T
    for G in [la.solve(T.T@M0@T,T.T@M0),la.solve(P,Pi),la.pinv(T)]:
        np.testing.assert_allclose(G@T,np.eye(r),atol=1e-10)
        PG=T@G;np.testing.assert_allclose(PG@PG,PG,atol=1e-10)
        Mm=G.T@G;Cm=G.T@Gamma@G;Km=G.T@np.diag(omega**2)@G
        assert np.linalg.matrix_rank(Mm)==r and la.eigvalsh(Mm).min()>-1e-12
        for s in [.3j,1.1j,2.4j]:
            D=s*s*Mm+s*Cm+Km;R=T@la.solve(s*s*np.eye(r)+s*Gamma+np.diag(omega**2),T.T)
            np.testing.assert_allclose(D@R@D,D,rtol=1e-9,atol=1e-12)
            np.testing.assert_allclose(R@D@R,R,rtol=1e-9,atol=1e-12)
        # the FE-space equation on range T is the latent equation
        q=rng.normal(size=r);dq=rng.normal(size=r);f=rng.normal(size=3);B=rng.normal(size=(n,3))
        ddq=T.T@B@f-Gamma@dq-np.diag(omega**2)@q   # latent acceleration: T^T B f
        lhs=Mm@(T@ddq)+Cm@(T@dq)+Km@(T@q)
        np.testing.assert_allclose(lhs,PG.T@B@f,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(Mm-PG.T@M0@PG,G.T@(np.eye(r)-T.T@M0@T)@G,rtol=1e-9,atol=1e-10)
    G=la.solve(P,Pi);np.testing.assert_allclose(G.T@G,Pi.T@la.inv(P).T@la.inv(P)@Pi,rtol=1e-9)
