"""Independent reference and spatial-prior regression checks (no refitting)."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import scipy.linalg as la
from fetune.three_mass import configuration,geometry,matrices,exact_coordinates,physical_frf,scalar_frf
from fetune.mechanical import MechanicalModel,Objective,completion


class ThreeMassPrior(unittest.TestCase):
    def test_grounded_completion_and_conditional_minimum(self):
        cfg=configuration();g,_=geometry(cfg['geometry'])
        assert np.allclose(g['E'],[[1,0],[.5,.5],[0,1]])
        assert np.allclose(g['Qtilde'],[[.75,-.25],[-.25,.75]])
        rng=np.random.default_rng(301)
        for w in rng.normal(size=(8,2)):
            t=g['E']@w
            for delta in [-3.,-.5,.5,3.]:
                pert=t+np.array([0,delta,0])
                assert pert@g['Q']@pert>t@g['Q']@t
        # The geometric constructor has no physical-parameter argument.
        changed=dict(cfg);changed['parameters']={'mu_kg':3,'k_N_per_m':700,'c_N_s_per_m':7}
        assert np.array_equal(geometry(changed['geometry'])[0]['Q'],g['Q'])


    def test_independent_scalar_normalization_and_spatial_ambiguity(self):
        cfg=configuration();g,_=geometry(cfg['geometry']);M,C,K=matrices(cfg)
        T,omega,Gamma=exact_coordinates(cfg)
        assert np.allclose(T.T@M@T,np.eye(3),atol=1e-14)
        assert np.allclose(T.T@K@T,np.diag(omega**2))
        assert np.allclose(T.T@C@T,Gamma,atol=1e-14)
        f=np.array([.1,3.8,3.852,5.,7.11,9.3,12.,120.])
        expected=physical_frf(f,cfg)
        assert la.norm(scalar_frf(2j*np.pi*f,cfg)[:,None,None]-expected)/la.norm(expected)<1e-12
        W=g['A']@T;TE=g['E']@W
        assert np.linalg.matrix_rank(TE)==2
        assert not np.allclose(TE.T@M@TE,np.eye(3))
        for beta in [.15,1.,7.]:
            Wb=np.diag([beta,1/beta])@W
            actual=MechanicalModel(g['E']@Wb,omega,Gamma).evaluate(f,g['S'],g['B'])
            assert la.norm(actual-expected)/la.norm(expected)<1e-12
            assert np.isclose(np.trace(Wb.T@g['Qtilde']@Wb),.75*(beta**2+beta**-2))


    def test_anchor_order_and_explicit_pack(self):
        cfg=configuration();g,_=geometry(cfg['geometry']);T,omega,Gamma=exact_coordinates(cfg)
        f=np.linspace(.1,12,121);y=physical_frf(f,cfg)
        obj=Objective(f,y,g['S'],g['B'],g['E'],g['Q'],3,2,1e-5)
        W=g['A']@T;x=obj.pack(W,omega,Gamma)
        assert x.shape==(15,)
        assert np.allclose(x[:6],(W/obj.zscale).ravel(order='C'))
        assert list(zip(*obj.tri))==[(0,0),(1,0),(1,1),(2,0),(2,1),(2,2)]
        unpacked=obj.unpack(x)
        assert np.allclose(unpacked[0],W)
        assert np.allclose(unpacked[1]*obj.ref,omega)
        assert np.allclose(unpacked[2]*obj.ref,Gamma)
        # Response-first ordering is a consistent alternative, not silently I.
        A=np.vstack([g['S'],g['B'].T]);E,_=completion(g['Q'],A)
        other=Objective(f,y,g['S'],g['B'],E,g['Q'],3,2,1e-5)
        assert np.allclose(other.transfer(other.pack(A@T,omega,Gamma)),obj.transfer(x))


    def test_regularizer_directional_gradient_and_bias(self):
        cfg=configuration();g,_=geometry(cfg['geometry']);T,omega,Gamma=exact_coordinates(cfg)
        f=np.linspace(.1,12,201);truth=physical_frf(f,cfg)
        obj=Objective(f,truth,g['S'],g['B'],g['E'],g['Q'],3,2,1e-5)
        zero=Objective(f,truth,g['S'],g['B'],g['E'],g['Q'],3,2,0)
        x=obj.pack(g['A']@T,omega,Gamma)
        shrunk=x.copy();shrunk[:6]*=(1-1e-7)
        assert obj(shrunk)[0]<obj(x)[0]  # Exact interpolation is biased by positive penalty.
        direction=np.arange(1,16,dtype=float);direction/=la.norm(direction)
        analytical=(obj(x)[1]-zero(x)[1])@direction
        errors=[]
        for step in np.logspace(-3,-6,7):
            def reg(t):return obj(t)[0]-zero(t)[0]
            fd=(reg(x+step*direction)-reg(x-step*direction))/(2*step)
            errors.append(abs(fd-analytical)/abs(analytical))
        assert min(errors)<1e-6

if __name__=="__main__":unittest.main()
