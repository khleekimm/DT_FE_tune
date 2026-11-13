"""Post-fit geometric completion, independent of response/reference data.

No fitted factor is truncated. Numerical rank selects the dimension of an
added nullspace correction, not a replacement for W. Coordinate ordering is
part of the declared prior; projector canonicalization removes arbitrary
rotations returned by eigensolvers and SVDs.
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def canonical_basis(basis, count=None, tie=1e-12):
    """Canonical orthonormal basis for a subspace, from projected axes.

    Work in the supplied orthonormal basis, without forming its n-by-n
    projector. Greedy pivoting maximizes the remaining projector diagonal.
    All signs have a positive selected coordinate. Ties use the first index.
    """
    b=np.asarray(basis,float)
    count=b.shape[1] if count is None else count
    if count == 0:return np.empty((len(b),0)),np.empty(0,dtype=int)
    residual=b.copy(); columns=[]; pivots=[]
    for _ in range(count):
        diagonal=np.sum(residual*residual,axis=1)
        top=float(diagonal.max())
        pivot=int(np.flatnonzero(diagonal >= top-tie*max(1.,top))[0])
        if top < 1e-20:raise ValueError('exhausted canonical subspace')
        direction=residual[pivot].copy()
        direction/=la.norm(direction)
        # Twice reorthogonalize the coefficient vector, protecting large bases.
        if columns:
            c=np.column_stack(columns)
            for _ in range(2):direction-=c@(c.T@direction)
            direction/=la.norm(direction)
        if b[pivot]@direction<0:direction=-direction
        columns.append(direction);pivots.append(pivot)
        residual-=np.outer(residual@direction,direction)
    return b@np.column_stack(columns),np.asarray(pivots,int)


def latent_nullspace(W, rtol=1e-10):
    _,singular,vh=la.svd(W,full_matrices=True)
    k=int(np.sum(singular>rtol*singular[0]))
    v0,pivots=canonical_basis(vh[k:].T)
    return v0,singular,k,pivots


def transport_nullspace(W, reference, rtol=1e-10, chart_floor=1e-6):
    """Polar transport of an archived nullspace reference, without pivot resets.

    Rank and nullity must be fixed. A chart below chart_floor is rejected;
    callers must explicitly archive a new overlapping reference at the last
    accepted state and transport both paired coefficient representations.
    The saved W is never truncated or replaced.
    """
    _,singular,vh=la.svd(W,full_matrices=False)
    k=int(np.sum(singular>rtol*singular[0]))
    if reference.shape != (W.shape[1],W.shape[1]-k):
        raise ValueError('nullity changed: reference chart cannot be continued')
    projected=reference-vh[:k].T@(vh[:k]@reference)
    gram=(reference.T@projected+projected.T@reference)/2
    values,vectors=la.eigh(gram)
    if len(values) and values[0]<=chart_floor:
        raise ValueError('nullspace reference chart requires an explicit transition')
    basis=projected@(vectors@np.diag(values**-.5)@vectors.T)
    return basis,dict(rank=k,chart_min=float(values[0]) if len(values) else 1.,
        orthogonality=float(la.norm(basis.T@basis-np.eye(len(values)))),
        null_residual=float(la.norm(W@basis)/max(la.norm(W),1e-300)))


def hidden_fields(Q,A,metric,count,cluster_rtol=1e-8,tol=1e-10):
    """Lowest Q-energy fields in ker A, orthonormal in diag(metric).

    A sparse constrained inverse supplies a symmetric shift-invert operator.
    This requires Q positive on ker A (true for the two declared geometries).
    The laboratory support locations are not treated as fixed boundaries.
    """
    Q=sp.csc_matrix(Q);A=np.asarray(A,float);g=np.asarray(metric,float)
    n=len(g);h=A.shape[0];root=np.sqrt(g)
    if count>n-h or np.any(g<=0):raise ValueError('insufficient admissible space')
    if n<100:
        null=la.null_space(A/root[None,:])
        vals,vec=la.eigh((null/root[:,None]).T@(Q@(null/root[:,None])))
        vectors=null@vec
    else:
        sparse_a=sp.csc_matrix(A)
        lu=sla.splu(sp.bmat([[Q,sparse_a.T],[sparse_a,None]],format='csc'))
        def inverse(v):
            v=np.asarray(v)
            rhs=np.r_[root*v,np.zeros(h)]
            return root*lu.solve(rhs)[:n]
        op=sla.LinearOperator((n,n),matvec=inverse,dtype=float)
        wanted=min(count+8,n-h-1)
        while True:
            invvals,vectors=sla.eigsh(op,k=wanted,which='LA',tol=tol,
                v0=np.sin(np.arange(1,n+1)*np.sqrt(2.)),ncv=min(n,2*wanted+1))
            vals=1/invvals;order=np.argsort(vals);vals=vals[order];vectors=vectors[:,order]
            if wanted>=n-h-1 or abs(vals[count]-vals[count-1])>cluster_rtol*max(1.,abs(vals[count])):break
            wanted=min(wanted+8,n-h-1)
    groups=[];canonical=[];pivots=[];j=0
    while j<count:
        end=j+1
        while end<len(vals) and abs(vals[end]-vals[j])<=cluster_rtol*max(1.,abs(vals[j])):end+=1
        block,piv=canonical_basis(vectors[:,j:end])
        take=min(end,count)-j;canonical.append(block[:,:take]);pivots.extend(piv[:take])
        groups.append([j,end]);j=end
    fields=np.column_stack(canonical)/root[:,None]
    # Roundoff-only projection. A is unchanged and no W is truncated.
    fields-=A.T@la.solve(A@A.T,A@fields,assume_a='pos')
    gram=(root[:,None]*fields).T@(root[:,None]*fields)
    ev,uu=la.eigh(gram)
    fields=fields@(uu@np.diag(ev**-.5)@uu.T)
    energies=np.diag(fields.T@(Q@fields))
    return fields,dict(eigenvalues=vals[:count],energies=energies,
        canonical_pivots=pivots,clusters=groups,
        constraint_norm=float(la.norm(A@fields)),
        metric_orthogonality=float(la.norm(fields.T@(g[:,None]*fields)-np.eye(count))))


def physical_realization(T,Gamma,omega,rows=None):
    """Selected physical coordinates, using solves and frozen row indices."""
    n,r=T.shape
    if rows is None:
        if n==r:rows=np.arange(r)
        else:
            _,_,piv=la.qr(T.T,mode='economic',pivoting=True)
            rows=np.sort(piv[:r])
    P=T[rows]
    inverse=la.solve(P,np.eye(r))
    expansion=la.solve(P.T,T.T).T
    return dict(rows=np.asarray(rows),P=P,expansion=expansion,
        M=inverse.T@inverse,C=inverse.T@Gamma@inverse,
        K=inverse.T@np.diag(omega**2)@inverse)


def calibrate_spatial_coefficients(kernel,target,scale,Ecal,fixed,Q,weight,constraint=None):
    """Real convex calibration of T=fixed+Ecal@U with full-map roughness.

    kernel[f,:,i] is the fixed normalized displacement response per force.
    Minimize mean(abs((U@kernel-target)/scale)^2) + weight*tr(T^T Q T).
    The caller supplies an invariant independent block in fixed to guarantee
    rank for every U. Returning the normal system permits independent
    objective/optimality checks without numerical differentiation.
    """
    n,r,p=kernel.shape;m=target.shape[1]
    normal=np.zeros((m*r,m*r));rhs=np.empty((m,r));factor=1/(n*m*p)
    grams=[(kernel[:,:,i].conj().T@kernel[:,:,i]).real for i in range(p)]
    for j in range(m):
        normal[j*r:(j+1)*r,j*r:(j+1)*r]=factor*sum(grams[i]/scale[j,i]**2 for i in range(p))
        rhs[j]=factor*sum((kernel[:,:,i].conj().T@target[:,j,i]).real/scale[j,i]**2 for i in range(p))
    normal+=weight*np.kron(Ecal.T@(Q@Ecal),np.eye(r))
    rhs-=weight*(Ecal.T@(Q@fixed))
    # Solve the augmented equations, not their squared-condition normal system.
    # Keep the small calibration normal operator only for independent diagnostics
    # and the existing public return interface (the 31-output solver never forms it).
    from .structured_fit import conditional_solve
    dictionary=np.c_[Ecal,fixed]
    full_q=dictionary.T@(Q@dictionary)
    factors=np.vstack([np.zeros((m,r)),np.eye(r)])
    matrix,diagnostic=conditional_solve(kernel,target,scale,full_q,factors,
        np.arange(m),weight,constraint)
    if diagnostic['stationarity_relative']>1e-9:
        raise ArithmeticError('calibration augmented stationarity was not attained')
    solution=matrix.ravel();multiplier=np.empty(0)
    if constraint is not None:
        directions,values=constraint
        D=np.kron(np.eye(m),directions.T)
        multiplier=la.lstsq(D.T,rhs.ravel()-normal@solution)[0]
    return solution.reshape(m,r),normal,rhs,multiplier
