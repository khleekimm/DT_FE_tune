"""Three-mass reference algebra and geometry; no separate estimator."""
from pathlib import Path
import json
import numpy as np
import scipy.linalg as la
from .mechanical import completion

CONFIG = Path(__file__).resolve().parents[2]/'configs/three_mass.json'

def configuration():
    return json.loads(CONFIG.read_text())

def frequency_grid(cfg):
    g=cfg['frequency_hz']
    return g['start']+g['step']*np.arange(round((g['stop']-g['start'])/g['step'])+1)

def geometry(settings):
    """Accept geometry only: no mass, spring, damper or generating matrix."""
    pos=np.asarray(settings['positions'],float)
    assert len(pos)==5 and np.allclose(np.diff(pos),np.diff(pos)[0])
    assert settings['fixed_wall_indices']==[0,4]
    incidence=np.zeros((4,5))
    for j in range(4):incidence[j,j:j+2]=[-1,1]
    D=incidence[:,1:-1]  # zero boundary values; dimensionless first differences
    Q=D.T@D
    B=np.eye(3)[:,[settings['force_coordinate_one_based']-1]]
    S=np.eye(3)[[settings['response_coordinate_one_based']-1]]
    A=np.vstack([B.T,S])
    E,info=completion(Q,A,rigid=None)
    QE=E.T@Q@E
    return dict(A=A,B=B,S=S,D=D,Q=Q,E=E,QE=QE,Qtilde=QE/la.norm(QE,2)),info

def matrices(cfg):
    p=cfg['parameters'];mu,k,c=(p[a] for a in ['mu_kg','k_N_per_m','c_N_s_per_m'])
    H=np.array([[2.,-1,0],[-1,2,-1],[0,-1,2]])
    return mu*np.eye(3),c*H,k*H

def physical_frf(f,cfg):
    M,C,K=matrices(cfg);g,_=geometry(cfg['geometry'])
    return np.array([s*s*g['S']@la.solve(s*s*M+s*C+K,g['B']) for s in 2j*np.pi*f])

def scalar_frf(s,cfg):
    a=cfg['parameters'];mu,k,c=(a[x] for x in ['mu_kg','k_N_per_m','c_N_s_per_m'])
    p=mu*s*s+2*c*s+2*k;g=c*s+k
    return s*s*g*g/(p*(p*p-2*g*g))

def exact_coordinates(cfg):
    p=cfg['parameters'];mu,k,c=(p[x] for x in ['mu_kg','k_N_per_m','c_N_s_per_m'])
    T=np.array([[.5,1/np.sqrt(2),.5],[1/np.sqrt(2),0,-1/np.sqrt(2)],[.5,-1/np.sqrt(2),.5]])/np.sqrt(mu)
    kappa=np.array([2-np.sqrt(2),2,2+np.sqrt(2)])
    return T,np.sqrt(k/mu*kappa),np.diag(c/mu*kappa)

def noisy_acquisition(truth,eta,seed):
    rng=np.random.default_rng(seed)
    variance=eta**2*(.5*np.mean(abs(truth)**2,axis=0)+.5*abs(truth)**2)
    return truth+np.sqrt(variance/2)*(rng.standard_normal(truth.shape)+1j*rng.standard_normal(truth.shape))

def optimizer_coordinates(model,A):
    """Undo the production export's simultaneous frequency/sign permutation."""
    W=A@model.Z
    inv=np.argsort(model.export_permutation)
    sign=model.export_sign
    return (W*sign)[:,inv],model.omega[inv],(model.Gamma*sign[:,None]*sign[None,:])[np.ix_(inv,inv)]

def condensation(s,cfg):
    """Exact hidden-coordinate weight: u_2(s)=rho(s)*(u_1(s)+u_3(s))."""
    p=cfg['parameters'];mu,k,c=(p[x] for x in ['mu_kg','k_N_per_m','c_N_s_per_m'])
    return (c*s+k)/(mu*s*s+2*c*s+2*k)

def undamped_condensation(omega,cfg):
    """Undamped eigenvector ratio where defined; NaN denotes the 0=0 mode."""
    p=cfg['parameters']
    denominator=2*p['k_N_per_m']-p['mu_kg']*np.asarray(omega,float)**2
    return np.divide(p['k_N_per_m'],denominator,out=np.full_like(denominator,np.nan),
                     where=abs(denominator)>1e-9*p['k_N_per_m'])

def latent_modes(omega,Gamma):
    """Quadratic eigenpairs of s^2 I + s Gamma + Omega^2, positive-imaginary half."""
    r=len(omega)
    companion=np.block([[np.zeros((r,r)),np.eye(r)],[-np.diag(omega**2),-Gamma]])
    lam,vec=la.eig(companion)
    keep=np.argsort(lam.imag)[r:]
    lam,v=lam[keep],vec[:r,keep]
    order=np.argsort(abs(lam.imag))
    lam,v=lam[order],v[:,order]
    return lam,v/la.norm(v,axis=0)

def mac(a,b):
    """Modal assurance criterion between two real or complex column sets."""
    a=np.asarray(a);b=np.asarray(b)
    num=abs(a.conj().T@b)**2
    return num/np.outer(np.sum(abs(a)**2,0),np.sum(abs(b)**2,0)).real
