"""An independent least-squares check of calibrated full-rank mechanics."""
import unittest
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from fetune.mechanical import completion,MechanicalModel
from fetune.spatial_completion import latent_nullspace,hidden_fields,calibrate_spatial_coefficients,physical_realization


class SpatialCalibrationTest(unittest.TestCase):
    def test_calibration_against_augmented_real_least_squares(self):
        rng=np.random.default_rng(471);n,r,m=9,4,2
        Q=sp.diags([-np.ones(n-1),2*np.ones(n),-np.ones(n-1)],[-1,0,1],format='csc')
        Af=np.eye(n)[[0,8]];Sc=np.eye(n)[[3,5]];A=np.vstack([Af,Sc]);B=Af[[0]].T;S=Af[[1]]
        E,_=completion(Q,A);W=rng.normal(size=(2,r));V,_,k,_=latent_nullspace(W)
        N,_=hidden_fields(Q,A,np.ones(n),r-k)
        fixed=E[:,:2]@W+N@V.T;Ec=E[:,2:]
        omega=np.array([2.,3.,4.,5.]);Gamma=np.diag([.1,.2,.3,.4])
        freq=np.linspace(.1,1.2,31);kernel=np.array([s*s*la.solve(s*s*np.eye(r)+s*Gamma+np.diag(omega**2),W[[0]].T) for s in 2j*np.pi*freq])
        true=rng.normal(size=(m,r));target=np.einsum('jr,fri->fji',true,kernel)+.003*(rng.normal(size=(31,m,1))+1j*rng.normal(size=(31,m,1)))
        scale=np.array([[1.3],[.7]]);weight=.02
        U,normal,rhs,_=calibrate_spatial_coefficients(kernel,target,scale,Ec,fixed,Q,weight)
        # Independent overdetermined formulation: real/imag observations plus
        # an explicit square root of the physical roughness operator.
        designs=[];targets=[]
        for j in range(m):
            design=kernel[:,:,0]/scale[j,0]/np.sqrt(31*m)
            y=target[:,j,0]/scale[j,0]/np.sqrt(31*m)
            designs.append(np.vstack([design.real,design.imag]));targets.append(np.r_[y.real,y.imag])
        root=la.cholesky(Q.toarray())
        design=np.vstack([la.block_diag(*designs),np.sqrt(weight)*np.kron(root@Ec,np.eye(r))])
        target_real=np.r_[np.concatenate(targets),-np.sqrt(weight)*(root@fixed).ravel()]
        independently=la.lstsq(design,target_real)[0].reshape(m,r)
        np.testing.assert_allclose(U,independently,rtol=1e-11,atol=1e-11)
        directions=la.qr(rng.normal(size=(r,2)),mode='economic')[0]
        values=true@directions
        U,_,_,_=calibrate_spatial_coefficients(kernel,target,scale,Ec,fixed,Q,weight,(directions,values))
        null=la.null_space(directions.T);particular=values@directions.T
        expansion=np.kron(np.eye(m),null)
        reduced=la.lstsq(design@expansion,target_real-design@particular.ravel())[0]
        independently=(particular.ravel()+expansion@reduced).reshape(m,r)
        np.testing.assert_allclose(U,independently,rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(U@directions,values,atol=1e-12)
        t=fixed+Ec@U;before=fixed+Ec@rng.normal(size=U.shape)
        self.assertEqual(np.linalg.matrix_rank(t),r);self.assertEqual(np.linalg.matrix_rank(before),r)
        np.testing.assert_allclose(Af@t,W,atol=1e-13);np.testing.assert_allclose(Sc@t,U,atol=1e-13)
        np.testing.assert_allclose(MechanicalModel(t,omega,Gamma).evaluate(freq,S,B),MechanicalModel(before,omega,Gamma).evaluate(freq,S,B),atol=1e-12)
        phy=physical_realization(t,Gamma,omega);P=phy['P'];R=phy['expansion']
        for key,v in [('M',np.eye(r)),('C',Gamma),('K',np.diag(omega**2))]:
            np.testing.assert_allclose(P.T@phy[key]@P,v,rtol=1e-11,atol=1e-11);la.cholesky(phy[key])
        physical=np.array([s*s*(S@R)@la.solve(s*s*phy['M']+s*phy['C']+phy['K'],R.T@B) for s in 2j*np.pi*freq])
        np.testing.assert_allclose(physical,MechanicalModel(t,omega,Gamma).evaluate(freq,S,B),rtol=1e-10,atol=1e-10)


if __name__=='__main__':unittest.main()
