import unittest
import numpy as np
import scipy.linalg as la
from fetune.structured_fit import conditional_solve, balance_scale
from fetune.spatial_completion import transport_nullspace, latent_nullspace


class StructuredSolveTests(unittest.TestCase):
    def test_coupled_complex_solve_and_constraints(self):
        rng=np.random.default_rng(81);f,r,m,p=24,7,3,2
        kernel=rng.normal(size=(f,r,p))+1j*rng.normal(size=(f,r,p))
        y=rng.normal(size=(f,m,p))+1j*rng.normal(size=(f,m,p))
        scale=np.exp(rng.normal(size=(m,p)));W=rng.normal(size=(m+p,r))
        root=rng.normal(size=(m+p,m+p));q=root.T@root;eta=.03
        directions=la.qr(rng.normal(size=(r,2)),mode='economic')[0]
        values=rng.normal(size=(m,2))
        actual,report=conditional_solve(kernel,y,scale,q,W,np.arange(m),eta,(directions,values))
        # Independent full augmented real least-squares nullspace construction.
        blocks=[];targets=[]
        for j in range(m):
            a=(kernel/scale[j][None,None,:]).transpose(0,2,1).reshape(-1,r)/np.sqrt(y.size)
            b=(y[:,j]/scale[j]).ravel()/np.sqrt(y.size)
            block=np.zeros((2*f*p,m*r));block[:,j*r:(j+1)*r]=np.vstack([a.real,a.imag])
            blocks.append(block);targets.append(np.r_[b.real,b.imag])
        blocks.append(np.sqrt(eta)*np.kron(root[:,:m],np.eye(r)))
        targets.append((-np.sqrt(eta)*root[:,m:]@W[m:]).ravel())
        A=np.vstack(blocks);b=np.concatenate(targets)
        D=np.kron(np.eye(m),directions.T);part=la.lstsq(D,values.ravel())[0];N=la.null_space(D)
        expected=part+N@la.lstsq(A@N,b-A@part)[0]
        self.assertLess(la.norm(actual.ravel()-expected),1e-9)
        self.assertLess(report['stationarity_relative'],1e-10)
        self.assertLess(la.norm(actual@directions-values),1e-12)

    def test_transport_continuity_and_coordinate_covariance(self):
        rng=np.random.default_rng(82);W=rng.normal(size=(3,8));ref=latent_nullspace(W)[0]
        direction=rng.normal(size=W.shape)
        v,report=transport_nullspace(W+1e-7*direction,ref)
        self.assertLess(la.norm(v-ref),1e-6)
        self.assertLess(report['null_residual'],1e-13)
        permutation=np.eye(8)[:,[4,2,0,5,1,7,6,3]];permutation[:,0]*=-1
        changed,_=transport_nullspace((W+1e-7*direction)@permutation,permutation.T@ref)
        self.assertLess(la.norm(changed-permutation.T@v),1e-12)

    def test_scale_balance_preserves_transfer_factors(self):
        rng=np.random.default_rng(83);W=rng.normal(size=(5,8));a=rng.normal(size=(5,5));q=a.T@a
        balanced,info=balance_scale(W,q,3)
        self.assertLess(la.norm(balanced[:3]@balanced[3:].T-W[:3]@W[3:].T),1e-12)
        self.assertLessEqual(info['after'],info['before'])
