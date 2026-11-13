"""Real conditional least squares for disjoint mechanical port factors.

QR compresses frequency equations before a right-preconditioned augmented
LSMR solve. The roughness square root retains coupling and cross terms.
There is no dense (number_of_ports * order)^2 normal matrix.
"""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from .mechanical import modal_resolvent


def optimizer_coordinates(model, E):
    W = la.lstsq(E, model.Z)[0]
    omega, gamma = model.omega.copy(), model.Gamma.copy()
    if model.export_permutation is not None:
        inv = np.argsort(model.export_permutation)
        sign = model.export_sign
        W = (W * sign)[:, inv]
        omega = omega[inv]
        gamma = (gamma * sign[:, None] * sign[None, :])[np.ix_(inv, inv)]
    return W, omega, gamma


def kernels(s, omega, gamma, port):
    """s^2 Delta(s)^-1 port.T, retaining the direct resolvent fallback."""
    try:
        poles, phi = modal_resolvent(omega, gamma)
        return s[:, None, None]**2 * np.einsum(
            'aj,fj,ji->fai', phi, 1/(s[:, None]-poles), phi.T@port.T,
            optimize=True)
    except ArithmeticError:
        return np.asarray([v*v*la.solve(v*v*np.eye(len(omega)) +
            v*gamma+np.diag(omega**2), port.T) for v in s])


def conditional_solve(kernel, target, scale, q, fixed, indices, eta,
                      constraint=None, tolerance=2e-12):
    """Minimize data residual + eta tr(W.T q W), changing selected rows.

    kernel has shape (frequency, coordinate, fixed_port). Target's middle
    index is the variable port. Constraints are U @ directions = values.
    q and fixed include every anchor, including the rows being solved.
    """
    started = time.perf_counter()
    nf, r, p = kernel.shape
    m = target.shape[1]
    indices = np.asarray(indices)
    others = np.setdiff1d(np.arange(len(q)), indices)
    if constraint is None:
        basis = np.eye(r); particular = np.zeros((m, r))
    else:
        directions, values = constraint
        basis = la.null_space(directions.T)
        particular = values @ la.pinv(directions)
    d = basis.shape[1]
    rootval, rootvec = la.eigh((q+q.T)/2)
    if rootval[0] < -1e-10*max(rootval[-1], 1.):
        raise ValueError('roughness matrix is not positive semidefinite')
    root = np.sqrt(eta*np.maximum(rootval, 0))[:, None]*rootvec.T
    coupling = root[:, indices]
    fixed_penalty = root[:, others]@fixed[others] + coupling@particular
    # Orthogonal compression acts on real and imaginary equations together.
    designs, targets, spectra = [], [], []
    for i in range(p):
        complex_design = kernel[:, :, i] @ basis
        data = target[:, :, i] - kernel[:, :, i] @ particular.T
        real_design = np.vstack([complex_design.real, complex_design.imag])
        orth, triangular = la.qr(real_design, mode='economic', check_finite=False)
        compressed = orth.T @ np.vstack([data.real, data.imag])
        designs.append(triangular)
        targets.append(compressed.T)
    weights = 1/(scale*np.sqrt(target.size))
    # Each QR factor includes the diagonal part of the coupled penalty.
    preconditioners = []
    for j in range(m):
        data_design = np.vstack([weights[j, i]*designs[i] for i in range(p)])
        spectrum = la.svdvals(data_design)
        spectra.append(spectrum)
        augmented = np.vstack([data_design, la.norm(coupling[:, j])*np.eye(d)])
        preconditioners.append(la.qr(augmented, mode='r')[0][:d])
    def inverse(v, transpose=False):
        return np.asarray([la.solve_triangular(preconditioners[j], v[j],
            trans='T' if transpose else 'N', check_finite=False) for j in range(m)])
    data_size = p*m*d
    def forward(v):
        u = inverse(v.reshape(m, d))
        data = [(u@designs[i].T)*weights[:, i, None] for i in range(p)]
        return np.r_[np.asarray(data).ravel(), (coupling@u).ravel()]
    def backward(v):
        data = v[:data_size].reshape(p, m, d)
        u = sum((data[i]*weights[:, i, None])@designs[i] for i in range(p))
        u += coupling.T@v[data_size:].reshape(len(q), d)
        return inverse(u, True).ravel()
    # Only the free latent directions enter the penalty; orthogonal constrained
    # directions contribute a constant to this reduced least-squares problem.
    rhs = np.r_[np.asarray([targets[i]*weights[:, i, None]
                            for i in range(p)]).ravel(),
                (-fixed_penalty@basis).ravel()]
    operator = sla.LinearOperator((len(rhs), m*d), matvec=forward,
                                  rmatvec=backward, dtype=float)
    result = sla.lsmr(operator, rhs, atol=tolerance, btol=tolerance,
                      conlim=1e12, maxiter=500)
    u = inverse(result[0].reshape(m, d))
    solution = particular + u@basis.T
    residual = forward(result[0])-rhs
    stationarity = backward(residual)
    diagnostic = dict(seconds=time.perf_counter()-started, iterations=result[2],
        termination=int(result[1]), preconditioned_condition_estimate=float(result[6]),
        stationarity_relative=float(la.norm(stationarity)/max(la.norm(backward(rhs)), 1e-300)),
        stationarity_absolute=float(la.norm(stationarity, np.inf)),
        data_rank=[int(np.sum(v > 1e-10*v[0])) for v in spectra],
        data_sigma_min=[float(v[-1]) for v in spectra],
        data_sigma_max=[float(v[0]) for v in spectra],
        data_condition=[float(v[0]/v[-1]) for v in spectra],
        coefficient_norm=float(la.norm(solution)), free_parameters=m*d)
    return solution, diagnostic


def balance_scale(W, q, m):
    """Minimize the same penalty along the common reciprocal row-scale orbit."""
    a = float(np.sum(W[:m]*(q[:m, :m]@W[:m])))
    b = float(np.sum(W[m:]*(q[m:, m:]@W[m:])))
    cross = float(2*np.sum(W[:m]*(q[:m, m:]@W[m:])))
    if a <= 0 or b <= 0:
        return W.copy(), dict(applied=False, a=a, b=b, cross=cross)
    beta = (a/b)**.25
    result = W.copy(); result[:m] /= beta; result[m:] *= beta
    return result, dict(applied=True, beta=beta, a=a, b=b, cross=cross,
                       before=a+b+cross, after=2*np.sqrt(a*b)+cross)
