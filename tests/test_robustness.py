"""Regression coverage for numerical risks identified by the manuscript audit."""
import unittest
import numpy as np
import scipy.linalg as la
from test_mechanical import system
from fetune.mechanical import MechanicalModel, Objective, modal_resolvent

class Robustness(unittest.TestCase):
    def test_rank_one_map_can_have_many_observable_temporal_modes(self):
        model=MechanicalModel(np.ones((1,3)),np.array([2.,4.,6.]),np.eye(3)*.2)
        A,B,C,D=model.state_space(np.ones((1,1)),np.ones((1,1)))
        reachable=np.column_stack([np.linalg.matrix_power(A,k)@B for k in range(6)])
        observable=np.vstack([C@np.linalg.matrix_power(A,k) for k in range(6)])
        self.assertEqual(np.linalg.matrix_rank(model.Z),1)
        self.assertEqual(np.linalg.matrix_rank(reachable),6)
        self.assertEqual(np.linalg.matrix_rank(observable),6)

    def test_critical_and_repeated_poles_use_resolvent_fallback(self):
        for omega,gamma in [(np.array([2.,3.]),np.diag([4.,.3])),
                            (np.array([2.,2.]),np.eye(2)*.3)]:
            model=MechanicalModel(np.eye(2),omega,gamma)
            with self.assertRaises(ArithmeticError):modal_resolvent(omega,gamma)
            for f in [0.,.1,.4,2.]:
                s=2j*np.pi*f
                expected=s*s*la.inv(s*s*np.eye(2)+s*gamma+np.diag(omega**2))
                np.testing.assert_allclose(model.evaluate([f],np.eye(2),np.eye(2))[0],expected,atol=1e-12)

    def test_real_poles_and_accelerance_limits(self):
        model=MechanicalModel(np.eye(2),np.array([2.,3.]),np.diag([6.,10.]))
        poles=model.all_poles()
        self.assertEqual(np.sum(abs(poles.imag)<1e-12),4)
        self.assertTrue(np.all(poles.real<0))
        self.assertEqual(len(model.poles()),0)
        np.testing.assert_allclose(model.evaluate([0],np.eye(2),np.eye(2)),0,atol=0)
        np.testing.assert_allclose(model.evaluate([1e10],np.eye(2),np.eye(2))[0],np.eye(2),atol=1e-8)

    def test_close_pole_and_fallback_directional_derivatives(self):
        rng=np.random.default_rng(567);f=np.linspace(.1,2,31)
        y=rng.normal(size=(len(f),2,2))+1j*rng.normal(size=(len(f),2,2))
        for omega,gamma in [(np.array([2.,2.+5e-6]),np.eye(2)*.3),
                            (np.array([2.,3.]),np.diag([4.,.3]))]:
            obj=Objective(f,y,np.eye(2),np.eye(2),np.eye(2),np.eye(2),2,1)
            x=obj.pack(np.eye(2),omega,gamma);_,g=obj(x);d=rng.normal(size=len(x));d/=la.norm(d)
            errors=[]
            for h in [1e-3,1e-4,1e-5,1e-6]:
                numerical=(obj(x+h*d)[0]-obj(x-h*d)[0])/(2*h)
                errors.append(abs(numerical-g@d)/max(abs(g@d),1e-10))
            self.assertLess(min(errors),1e-5)

    def test_energy_and_reversed_rectangular_ports(self):
        f,S,B,E,Q,W,w,gamma,rng=system(seed=83,m=3,p=2)
        model=MechanicalModel(W,w,gamma)
        np.testing.assert_allclose(model.evaluate(f,S,B),model.evaluate(f,B.T,S.T).transpose(0,2,1),rtol=1e-10,atol=1e-10)
        q=rng.normal(size=len(w));v=rng.normal(size=len(w));force=rng.normal(size=2)
        acc=-gamma@v-w*w*q+W.T@B@force
        np.testing.assert_allclose(v@acc+q@(w*w*v),-v@gamma@v+force@(B.T@W@v),rtol=1e-10,atol=1e-10)

    def test_export_permutation_preserves_transfer_but_not_bandwidth(self):
        r=4;E=np.eye(r);f=np.linspace(1,3,8);S=B=E
        obj=Objective(f,np.ones((8,r,r),complex),S,B,E,E,r,1)
        L=np.eye(r)*.4;L[np.arange(1,r),np.arange(r-1)]=.1
        gamma=obj.ref*(L@L.T+1e-7*E);w=np.array([2.,8.,4.,6.]);W=np.eye(r)
        x=obj.pack(W,w,gamma);model,_=obj.model(x)
        np.testing.assert_allclose(model.evaluate(f,S,B),obj.transfer(x),rtol=1e-10,atol=1e-10)
        ii,jj=np.indices((r,r));self.assertGreater(la.norm(model.Gamma[abs(ii-jj)>1]),0)
        inv=np.argsort(model.export_permutation);sgn=model.export_sign
        recovered=(model.Gamma*sgn[:,None]*sgn[None])[np.ix_(inv,inv)]
        np.testing.assert_allclose(recovered,gamma,atol=1e-12)

    def test_zero_regularizer_and_scale_invariance(self):
        f,S,B,E,Q,W,w,gamma,rng=system(seed=84)
        y=rng.normal(size=(len(f),S.shape[0],B.shape[1]))+1j*rng.normal(size=(len(f),S.shape[0],B.shape[1]))
        obj=Objective(f,y,S,B,E,np.zeros_like(Q),len(w),len(w)-1)
        x=obj.pack(W,w,gamma);value,g=obj(x)
        self.assertTrue(np.isfinite(value) and np.all(np.isfinite(g)))
        self.assertEqual(la.norm(obj.q),0)
        for gain in [1e-3,1e3]:
            scaled=Objective(f,y*gain,S,B,E,np.zeros_like(Q),len(w),len(w)-1)
            xx=scaled.pack(W*np.sqrt(gain),w,gamma)
            self.assertAlmostEqual(scaled(xx)[0],value,places=7)

if __name__=='__main__':unittest.main()
