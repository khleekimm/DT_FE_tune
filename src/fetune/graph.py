"""Geometry-only discrete operators used as the spatial prior.

Nothing here reads a material law, a support condition or any measured
response; these are quadrature objects built from node positions and element
connectivity alone.
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def graph_operators(nodes, quads):
    """Geometry-only Laplacian and tributary area for a connected quad mesh.

    Integrate the surface gradient energy with linear triangles, averaging the
    two diagonals of each quad. This passes the affine patch test on distorted
    planar grids; a heuristic edge-length weighting need not. L is
    dimensionless and the returned tributary areas have m^2 units. No boundary displacement is fixed.
    Couplings occur only between vertices sharing an element.
    """
    triangles=np.concatenate([quads[:,ix] for ix in [[0,1,2],[0,2,3],[0,1,3],[1,2,3]]])
    xyz=np.asarray(nodes)[triangles]
    twice_area=la.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)
    if np.any(twice_area<=1e-16):raise ValueError('degenerate geometric triangle')
    indices=[];ends=[];values=[]
    for k in range(3):
        a,b=(k+1)%3,(k+2)%3
        cot=np.einsum('ei,ei->e',xyz[:,a]-xyz[:,k],xyz[:,b]-xyz[:,k])/twice_area
        indices.append(triangles[:,a]);ends.append(triangles[:,b]);values.append(.25*cot)
    i,j,weight=np.concatenate(indices),np.concatenate(ends),np.concatenate(values)
    n = len(nodes)
    L = sp.coo_matrix((np.r_[weight, weight, -weight, -weight],
        (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n)).tocsr()
    q = np.bincount(triangles.ravel(), weights=np.repeat(twice_area/12, 3), minlength=n)
    return L, q


def connectivity_extension(L, area, sampling, order=1, ridge=1e-6,length_scale_m=None):
    """Minimum roughness interpolation with sampling @ extension = I.

    Input sampling is scalar nodal interpolation. Order one minimizes an edge
    gradient; order two minimizes area-weighted graph curvature. Ridge and
    roughness are priors only. Unobserved output directions are not supplied by
    this scalar construction.
    """
    sampling = sp.csr_matrix(sampling)
    if order not in (1, 2):
        raise ValueError('order is one or two')
    base = L if order == 1 else L@sp.diags(1/area)@L
    if length_scale_m is None:
        norm = np.mean(base.diagonal())/np.mean(area)
        penalty = base + ridge*norm*sp.diags(area)
    else:
        if length_scale_m<=0:raise ValueError('positive physical interpolation length required')
        penalty = base + sp.diags(area)/length_scale_m**(2*order)
    # The constrained solve remains well conditioned when the roughness has
    # a nearly free constant component. Forming Q^-1 S^T followed by a Schur
    # inverse loses digits in precisely that useful long-length limit.
    count=sampling.shape[0]
    kkt=sp.bmat([[penalty,sampling.T],[sampling,None]],format='csc')
    rhs=np.vstack([np.zeros((L.shape[0],count)),np.eye(count)])
    extension=sla.splu(kkt).solve(rhs)[:L.shape[0]]
    return extension, dict(order=order, ridge=ridge if length_scale_m is None else None,
        length_scale_m=length_scale_m,
        interpolation_residual=float(la.norm(sampling@extension-np.eye(count))),
        sampling_gram_condition=float(np.linalg.cond((sampling@sampling.T).toarray())),
        role='geometry prior; no support condition or material law')
