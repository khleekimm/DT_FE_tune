"""Independent physical checks for an injective, fixed-order completion."""
import numpy as np
import scipy.linalg as la
from .fe_completion import elastic_solver, relative


def column_spectrum(T, rtol=1e-10):
    """Account for ALL column directions, including implicit zeros in wide maps."""
    T=np.asarray(T);r=T.shape[1]
    s=la.svdvals(T);s=np.pad(s,(0,r-len(s)))
    rank=int(np.count_nonzero(s>rtol*s[0])) if len(s) and s[0]>0 else 0
    return dict(singular_values=s,rank=rank,columns=r,rows=T.shape[0],
                minimum=float(s[-1]),maximum=float(s[0]),
                condition=float(s[0]/s[-1]) if s[-1]>0 else None,
                full_column_rank=rank==r)


def trial_feasibility(A,E,N,W,represented):
    n=E.shape[0];h,r=W.shape;d=N.shape[1]
    if n<r or len(represented)<r or h+d<r:
        raise ValueError('fundamental dimensional obstruction to r independent directions')
    D=np.c_[E,N];AD=A@D
    rank_D=int(np.linalg.matrix_rank(D));rank_AD=int(np.linalg.matrix_rank(AD))
    rank_W=int(np.linalg.matrix_rank(W,tol=1e-10*la.svdvals(W)[0]))
    port_residual=relative(AD@la.lstsq(AD,W)[0],W)
    represented_s=column_spectrum(D[represented])
    if rank_D!=h+d or rank_AD!=h or port_residual>1e-10:
        raise ValueError('trial dictionary independence or exact port feasibility failed')
    if d<r-rank_W:raise ValueError('insufficient port-invisible trial directions')
    if represented_s['rank']<r:raise ValueError('represented trial map cannot retain r directions')
    return dict(n=n,r=r,h=h,d=d,trial_dimension=rank_D,rank_AD=rank_AD,
                rank_W=rank_W,invisible_directions_required=r-rank_W,
                available_invisible_directions=d,port_feasibility_relative=port_residual,
                represented_dictionary=represented_s,
                algebraically_feasible=True,finite_margin_requires_separate_check=True)


def physical_diagnostics(K,M,A,T,W,omega,Gamma,represented,delta,*,solve=None,N=None):
    """Evaluate the full FE force residual, never just its trial-space projection.

    The left inverse is stored as (RT)^dagger, followed implicitly by R;
    realization checks use r by r products, not n by n realized operators.
    """
    n,r=T.shape
    if W.shape[1]!=r or len(omega)!=r or Gamma.shape!=(r,r):
        raise ValueError('fixed fitted coordinate count changed')
    if solve is None:solve=elastic_solver(K)
    RT=T[represented];spec=column_spectrum(RT)
    mismatch=A@T-W
    column_port=la.norm(mismatch,axis=0)
    column_relative=column_port/np.maximum(la.norm(W,axis=0),1e-300)
    per_mode=np.empty(r);epsilon=np.empty(r);stiffness=np.empty(r);mass=np.empty(r)
    gradients=np.empty((N.shape[1],r)) if N is not None else None
    solve_error=0.;residual_size=0.
    for start in range(0,r,32):
        end=min(start+32,r);t=T[:,start:end];w=omega[start:end]
        kt=K@t;mt=M@t;g=kt-mt*w[None,:]**2;kg=solve(g)
        imbalance=np.sum(g*kg,axis=0)
        if np.any(imbalance<-1e-12):raise ArithmeticError('negative compliance residual')
        imbalance=np.maximum(imbalance,0.)
        per_mode[start:end]=imbalance/(2*w*w)
        stiffness[start:end]=np.sum(t*kt,axis=0)
        mass[start:end]=np.sum(t*mt,axis=0)
        epsilon[start:end]=np.sqrt(imbalance/stiffness[start:end])
        solve_error+=la.norm(K@kg-g)**2;residual_size+=la.norm(g)**2
        if N is not None:gradients[:,start:end]=N.T@(g/w[None,:]**2-M@kg)
    report=dict(objective_J=float(per_mode.sum()),dynamic_RMS=float(np.sqrt(2*per_mode.mean())),
        J_per_mode=per_mode,epsilon_per_mode=epsilon,omega=omega,frequency_hz=omega/(2*np.pi),
        modal_indices=np.arange(1,r+1),worst_J_mode=int(per_mode.argmax()+1),
        worst_J=float(per_mode.max()),worst_epsilon_mode=int(epsilon.argmax()+1),
        worst_epsilon=float(epsilon.max()),stiffness_per_mode=stiffness,mass_per_mode=mass,
        total_nominal_mass=float(mass.sum()),port_relative=relative(A@T,W),
        port_absolute=float(la.norm(mismatch)),port_column_absolute=column_port,
        port_column_relative=column_relative,port_worst_column_absolute=float(column_port.max()),
        port_worst_column_relative=float(column_relative.max()),spectrum=spec,delta=float(delta),
        margin_feasible=bool(spec['full_column_rank'] and spec['minimum']>=delta),
        elastic_solve_relative=float(np.sqrt(solve_error/max(residual_size,1e-300))),
        objective_source='full (K T - M T Omega^2), sparse compliance solves, half sum over all r columns')
    if N is not None:
        report['coefficient_gradient_frobenius']=float(la.norm(gradients))
        report['coefficient_gradient_max_column']=float(la.norm(gradients,axis=0).max())
    if spec['full_column_rank']:
        u,s,vh=la.svd(RT,full_matrices=False)
        left=(vh.T/s)@u.T  # no truncation: all r singular directions survive
        L=left@RT;I=np.eye(r);O=np.diag(omega**2)
        report['left_inverse_frobenius']=float(la.norm(L-I))
        report['left_inverse_spectral']=float(la.norm(L-I,2))
        report['realization']=dict(mass=relative(L.T@L,I),
            damping=relative(L.T@Gamma@L,Gamma),stiffness=relative(L.T@O@L,O),
            KT_MT_Omega=relative(left.T@(O@L),left.T@(L@O)),state_embedding_rank=2*r)
        ker=la.null_space(W,rcond=1e-10)
        report['port_invisible']=dict(dimension=ker.shape[1],
            port_absolute=float(la.norm(A@(T@ker))),
            represented_minimum=float(la.svdvals(RT@ker)[-1]) if ker.shape[1] else None)
    return report
