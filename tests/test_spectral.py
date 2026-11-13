"""Independent convention, scaling and correlated-input propagation checks."""
import numpy as np
from scipy.signal import csd, get_window, welch
from fetune.response_spectra import spectral_matrix, propagate_spectra, psd_level


def test_welch_matrix_matches_scipy_including_dc_and_nyquist():
    rng=np.random.default_rng(1111)
    first=rng.normal(size=8192)
    second=.8*np.roll(first,3)+.2*rng.normal(size=len(first))
    x=np.column_stack([first,second]);rate=1024.;length=1024
    frequency,G,info=spectral_matrix(x,rate,length,length//2)
    kwargs=dict(fs=rate,window=get_window('hann',length),nperseg=length,
        noverlap=length//2,nfft=length,detrend='constant',scaling='density',average='mean')
    _,opposite=csd(x[:,1],x[:,0],**kwargs)
    _,powers=welch(x,axis=0,**kwargs)
    np.testing.assert_allclose(G[:,0,1],opposite,rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(np.diagonal(G,axis1=1,axis2=2).real,powers,rtol=1e-12,atol=1e-15)
    assert np.linalg.norm(G[:,0,1].imag)>0.1*np.linalg.norm(G[:,0,1])
    np.testing.assert_allclose(G,G.conj().swapaxes(1,2),atol=1e-15)
    assert np.linalg.eigvalsh(G).min()>0
    assert info['segment_count']==15 and info['enbw_hz']==1.5


def test_correlated_synthetic_forces_complex_transfer_and_cross_term():
    # One complete periodic record gives exact spectral propagation without
    # finite-window leakage. Phase is deliberately substantive in both inputs.
    rng=np.random.default_rng(1112);length=8192;rate=1024.
    f1=rng.normal(size=length)
    f2=.75*np.roll(f1,7)+.25*rng.normal(size=length)
    f=np.column_stack([f1,f2]);nu=np.fft.rfftfreq(length,1/rate)
    H=np.empty((len(nu),2,2),complex)
    H[:,0,0]=np.exp(-2j*np.pi*nu*2/rate)
    H[:,0,1]=.7*np.exp(-2j*np.pi*nu*11/rate)
    H[:,1,0]=-.3*np.exp(-2j*np.pi*nu*4/rate)
    H[:,1,1]=1.2*np.exp(-2j*np.pi*nu*5/rate)
    output=np.fft.irfft(np.einsum('foi,fi->fo',H,np.fft.rfft(f,axis=0)),n=length,axis=0)
    _,Gf,_=spectral_matrix(f,rate,length,0,window='boxcar')
    _,Ga,_=spectral_matrix(output,rate,length,0,window='boxcar')
    pred=propagate_spectra(H,Gf)
    np.testing.assert_allclose(pred,Ga,rtol=1e-11,atol=1e-14)
    scalar=(abs(H[:,:,0])**2*Gf[:,0,0,None].real
        +abs(H[:,:,1])**2*Gf[:,1,1,None].real
        +2*np.real(H[:,:,0]*Gf[:,0,1,None]*H[:,:,1].conj()))
    np.testing.assert_allclose(scalar,np.diagonal(pred,axis1=1,axis2=2).real,atol=1e-14)
    wrong=propagate_spectra(H,Gf.conj())
    independent=Gf.copy();independent[:,0,1]=0;independent[:,1,0]=0
    assert np.linalg.norm(wrong-Ga)/np.linalg.norm(Ga)>.2
    assert np.linalg.norm(propagate_spectra(H,independent)-Ga)/np.linalg.norm(Ga)>.2
    assert np.linalg.eigvalsh(pred).min()>-1e-13


def test_psd_power_units_and_decibel_conversion():
    np.testing.assert_allclose(psd_level(np.array([.01,1.,100.])),[-20.,0.,20.])
    np.testing.assert_allclose(psd_level(np.array([2.,20.]),2.),[0.,10.])
    for values in [np.array([-1.]),np.array([1j]),np.array([np.nan])]:
        try:psd_level(values)
        except ValueError:pass
        else:raise AssertionError('invalid spectral power accepted')
    rng=np.random.default_rng(1113);x=rng.normal(size=(4096,2))
    nu,G,_=spectral_matrix(x,256.,len(x),0,'boxcar')
    # Sum discrete bin energies, including one-sided endpoint normalization.
    power=np.diagonal(G,axis1=1,axis2=2).real.sum(0)*(nu[1]-nu[0])
    np.testing.assert_allclose(power,np.var(x,axis=0),rtol=1e-13)
