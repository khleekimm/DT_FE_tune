"""Figure 10 spectra from measured record 103 and the unchanged completed fit.

A cached extract of four measured channels permits reproduction without the
external full campaign archive. Use --refresh-record to re-extract and compare.
No fitting, gain adjustment, phase interpolation or spectral noise fit occurs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import h5py
import numpy as np
import scipy
from scipy.signal import csd, get_window, welch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fetune import data as source
from fetune.io import digest, json_write
from fetune.mechanical import MechanicalModel
from fetune.response_spectra import spectral_matrix, propagate_spectra, psd_level
OUT=ROOT/'artifacts/spectral_assessment'
MODEL=ROOT/'artifacts/fe_primary_revision/laboratory_completion.npz'
OBS=ROOT/'artifacts/model/laboratory_expanded/observations.npz'
SETTINGS=ROOT/'artifacts/preserved/spectra/e3_support.json'


def extract(refresh=False):
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/'record_103.npz'
    if cache.exists() and not refresh:
        return dict(np.load(cache))
    raw=source.DATA/'raw/august_2026/ship_mockup_timeseries_2026_08.h5'
    with h5py.File(raw,'r') as handle:
        group=handle['measurement/test_103']
        assert np.array_equal(group.attrs['force_channels'],[59,60])
        assert np.array_equal(group.attrs['excitation_positions'],[6,8])
        assert group.attrs['n_samples']==491520 and group.attrs['duration_s']==60
        arrays=dict(force=group['force'][0],acceleration=np.column_stack([
            group['acceleration/hull'][0,:,3],group['acceleration/added'][0,:,7]]),
            channels=np.array([8,40]),rate=np.array(8192.))
        metadata=dict(raw=str(raw),record='measurement/test_103',
            datasets={'P6':'force[0,:,0]','P8':'force[0,:,1]',
                      'ACC8':'acceleration/hull[0,:,3]','ACC40':'acceleration/added[0,:,7]'},
            units={'force':'N','acceleration':'m s^-2'},
            source_csv=str(group.attrs['source_csv']),datetime=str(group.attrs['datetime']),
            samples=491520,rate_hz=8192.,dtype=str(arrays['force'].dtype),
            array_sha256={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                          for k,v in arrays.items()},
            extraction='Exact recorded values, no sign, gain, delay or unit correction.')
    if cache.exists():
        previous=np.load(cache)
        for key,value in arrays.items():
            assert np.array_equal(previous[key],value),key
    else:
        np.savez_compressed(cache,**arrays)
    metadata['cache_sha256']=digest(cache)
    json_write(OUT/'record_103.json',metadata)
    return arrays


def main(refresh=False):
    record=extract(refresh);rate=float(record['rate'])
    archived=json.loads(SETTINGS.read_text())['settings']['hann16']
    nperseg=round(rate*archived['duration_s']);noverlap=round(nperseg*archived['overlap'])
    samples=np.column_stack([record['force'],record['acceleration']])
    frequency,G,info=spectral_matrix(samples,rate,nperseg,noverlap,archived['window'])
    band=(frequency>=360)&(frequency<=1000)
    frequency=frequency[band];G=G[band]
    Gff=G[:,:2,:2];Gaa=G[:,2:,2:]
    z=np.load(MODEL);obs=np.load(OBS)
    rows=[list(obs['channels']).index(c) for c in record['channels']]
    H=MechanicalModel(z['T'],z['omega'],z['Gamma']).evaluate(frequency,obs['S_all'][rows],obs['B'])
    predicted=propagate_spectra(H,Gff)
    measured_psd=np.diagonal(Gaa,axis1=1,axis2=2).real
    predicted_psd=np.diagonal(predicted,axis1=1,axis2=2).real
    # Independent SciPy convention and normalization checks on these records.
    settings=dict(fs=rate,window=get_window('hann',nperseg,fftbins=True),nperseg=nperseg,
        noverlap=noverlap,nfft=nperseg,detrend='constant',return_onesided=True,
        scaling='density',average='mean')
    _,cross=csd(record['force'][:,1].astype(float),record['force'][:,0].astype(float),**settings)
    cross_error=np.linalg.norm(Gff[:,0,1]-cross[band])/np.linalg.norm(cross[band])
    _,scipy_psd=welch(record['acceleration'].astype(float),axis=0,**settings)
    scipy_error=np.linalg.norm(measured_psd-scipy_psd[band])/np.linalg.norm(measured_psd)
    # A scalar expansion must reproduce the matrix product, including G_68.
    diagonal_terms=abs(H[:,:,0])**2*Gff[:,0,0,None].real+abs(H[:,:,1])**2*Gff[:,1,1,None].real
    cross_terms=2*np.real(H[:,:,0]*Gff[:,0,1,None]*H[:,:,1].conj())
    expansion_error=np.linalg.norm(predicted_psd-diagonal_terms-cross_terms)/np.linalg.norm(predicted_psd)
    def matrix_check(a):
        return dict(hermitian_relative=float(np.linalg.norm(a-a.conj().swapaxes(1,2))/np.linalg.norm(a)),
            minimum_eigenvalue=float(np.linalg.eigvalsh(a).min()),
            diagonal_imaginary_max=float(abs(np.diagonal(a,axis1=1,axis2=2).imag).max()))
    checks={name:matrix_check(a) for name,a in [('force',Gff),('measured',Gaa),('predicted',predicted)]}
    for check in checks.values():
        assert check['hermitian_relative']<1e-12
        assert check['minimum_eigenvalue']>=-1e-12
    assert np.all(measured_psd>0) and np.all(predicted_psd>0)  # no display floor needed
    assert max(cross_error,scipy_error)<1e-12 and expansion_error<1e-12
    measured_db=psd_level(measured_psd);predicted_db=psd_level(predicted_psd)
    rms_measured=np.sqrt(np.trapezoid(measured_psd,frequency,axis=0))
    rms_predicted=np.sqrt(np.trapezoid(predicted_psd,frequency,axis=0))
    np.savez_compressed(OUT/'spectra.npz',frequency=frequency,H=H,G_ff=Gff,
        G_aa_measured=Gaa,G_aa_predicted=predicted,measured_psd=measured_psd,
        predicted_psd=predicted_psd,measured_db=measured_db,predicted_db=predicted_db,
        diagonal_terms=diagonal_terms,cross_terms=cross_terms,channels=record['channels'],
        rms_measured=rms_measured,rms_predicted=rms_predicted,G_ref=np.array(1.))
    columns=np.column_stack([frequency,measured_psd,predicted_psd,measured_db,predicted_db,
        Gff[:,0,0].real,Gff[:,1,1].real,Gff[:,0,1].real,Gff[:,0,1].imag,
        H[:,0,0].real,H[:,0,0].imag,H[:,0,1].real,H[:,0,1].imag,
        H[:,1,0].real,H[:,1,0].imag,H[:,1,1].real,H[:,1,1].imag])
    np.savetxt(OUT/'spectra.csv',columns,delimiter=',',fmt='%.16e',header=','.join([
        'frequency_Hz','measured_ACC8_PSD','measured_ACC40_PSD','predicted_ACC8_PSD','predicted_ACC40_PSD',
        'measured_ACC8_dB','measured_ACC40_dB','predicted_ACC8_dB','predicted_ACC40_dB',
        'G66_N2_per_Hz','G88_N2_per_Hz','G68_real_N2_per_Hz','G68_imag_N2_per_Hz',
        'H_ACC8_P6_real','H_ACC8_P6_imag','H_ACC8_P8_real','H_ACC8_P8_imag',
        'H_ACC40_P6_real','H_ACC40_P6_imag','H_ACC40_P8_real','H_ACC40_P8_imag']),comments='')
    report=dict(record='test_103',channels=[8,40],forces=['P6','P8'],band_hz=[360.,1000.],
        estimator=info,archived_settings_source=str(SETTINGS.relative_to(ROOT))+' / settings / hann16',
        convention='G_ij = Fourier(force_i) conjugate(Fourier(force_j)); SciPy csd(force_j,force_i)',
        units={'G_ff':'N^2/Hz','G_aa':'(m s^-2)^2/Hz','H':'m s^-2/N'},
        decibels='10 log10(G_aa / G_ref)',G_ref='1 (m s^-2)^2/Hz',display_floor=None,
        model_evaluation='Complex accelerance evaluated directly at all 10241 Welch frequencies; no extra acceleration factor.',
        finite_resolution='Same Hann windows, mean removal, segments and averaging for both measured force and acceleration spectra. Plug-in H G_ff H*; windowing and rapidly varying H do not commute. No pointwise equality near narrow resonances is implied; neither curve is additionally smoothed.',
        rms_integration='trapezoid, inclusive 360--1000 Hz, identical 0.0625 Hz grid; no display floor',
        rms_measured=rms_measured,rms_predicted=rms_predicted,
        relative_cross_term_band_power=np.trapezoid(cross_terms,frequency,axis=0)/np.trapezoid(predicted_psd,frequency,axis=0),
        checks=dict(matrices=checks,scipy_csd_relative=float(cross_error),scipy_welch_relative=float(scipy_error),
                    scalar_expansion_relative=float(expansion_error)),
        sources={str(p.relative_to(ROOT)):digest(p) for p in [MODEL,OBS,SETTINGS,OUT/'record_103.npz',Path(__file__),ROOT/'src/fetune/response_spectra.py']},
        versions=dict(numpy=np.__version__,scipy=scipy.__version__))
    json_write(OUT/'assessment.json',report)
    tex=[]
    for j,c in enumerate(record['channels']):
        for name,values in [('Measured',rms_measured),('Predicted',rms_predicted)]:
            tex.append(r'\newcommand{\Spectral'+name+('Eight' if c==8 else 'Forty')+'}{'+f'{values[j]:.4f}'+'}')
    (ROOT/'paper/generated/spectral_numbers.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(dict(estimator=info,rms_measured=rms_measured.tolist(),rms_predicted=rms_predicted.tolist(),checks=report['checks']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-record',action='store_true')
    main(parser.parse_args().refresh_record)
