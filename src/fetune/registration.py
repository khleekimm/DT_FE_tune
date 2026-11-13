"""Single physical port registry, shared by assembly, validation and figures.

Coordinates are not fitted. Photo-derived locations carry their source uncertainty.
The sign of a recorded engineering channel is not established by an empty Direction
field: the declared outward/+z convention remains an explicit apparatus assumption.
"""
from copy import deepcopy
import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from . import data as D, geometry as G


XI = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)


def _shape(xi, eta):
    """Bilinear quadrilateral shape functions and their local derivatives."""
    N = 0.25 * (1 + XI[:, 0] * xi) * (1 + XI[:, 1] * eta)
    dN = 0.25 * np.stack([XI[:, 0] * (1 + XI[:, 1] * eta), XI[:, 1] * (1 + XI[:, 0] * xi)])  # (2,4)
    return N, dN

FIT_CHANNELS=(33,34,36,37,39)
HOLD_CHANNELS=(35,38,40)


def registry():
    channels, xyz=D.accelerometers();drives=D.drive_points()
    ports=[]
    note='data/raw/august_2026/import_metadata/2026_08_test/data_inf_mockup_2026_08.md, section 3; HDF5 coordinates/added'
    for c in range(33,41):
        survey=xyz[np.flatnonzero(channels==c)[0]];p,_=G.snap(survey)
        ports.append(dict(id=f'A{c-32}',channel=c,kind='acceleration',position_m=p.tolist(),
          photo_coordinate_m=survey.tolist(),surface_projection_m=(p-survey).tolist(),
          surface_projection_role='project approximate photo location to the declared shell surface; not a measured rigid-arm offset',
          direction=[0.,-1.,0.],units='m/s^2',surface='side',
          status='calibration' if c in FIT_CHANNELS else 'spatial_validation',
          position_uncertainty_m=.05,position_uncertainty_kind='approximate source estimate; not a statistical confidence interval',
          provenance=note,registration='channel identity documented; location photo-derived',
          polarity='outward normal assumption; raw export Direction field is blank'))
    for i,p in enumerate(('P6','P8')):
        ports.append(dict(id=p,channel=59+i,kind='force',position_m=drives[p].tolist(),
          direction=[0.,0.,1.],units='N',surface='foundation_top',status='calibration',
          foundation='F1' if p=='P6' else 'F2',deck_offset_m=G.ZFDN-G.ZDECK,
          shell_midsurface_offset_m=0.,attachment_offset_status='source point is on modeled top flange; no measured transducer/fixture eccentricity supplied',
          provenance='curated/drive_points.json; E3 record exciter1/2_xyz and force_channels; forcing_positions.jpg',
          numbering='unified P labels coincide with E3 ex labels; the separate E2 sweep drawing uses different numbers',
          e3_identity='ex6' if p=='P6' else 'ex8',e2_sweep_position=2 if p=='P6' else 9,
          registration='exciter 1 / ch59 = P6; exciter 2 / ch60 = P8',
          polarity='+z assumption; source CSV Direction field is blank'))
    return dict(frame='x axial, y transverse, z as curated geometry; SI',ports=ports,
        excluded=[dict(channel=c,reason='physical channel-to-coordinate permutation unresolved') for c in range(1,33)],
        training_records=['test_96','test_101'],validation_records=['test_103'],
        operator='H_a(s)=s^2 S (s^2 M+s C+K)^-1 B; s=+i 2*pi*f',
        time_transform='DFT exp(-i omega t); H1=S_af S_ff^-1 with S_af=E[A F^H]',
        calibration='engineering units in raw exports; sensor sensitivity certificates and polarity trace unavailable')


def interpolate_port(mesh,port):
    p=np.asarray(port['position_m']);d=np.asarray(port['direction']);X=mesh.nodes[mesh.quads]
    if not np.isclose(np.linalg.norm(d),1):raise ValueError('direction must be unit length')
    if port['surface']=='foundation_top':
        mask=(mesh.sub==G.SUBSTRUCTURES.index('foundation')) & (np.max(abs(X[:,:,2]-G.ZFDN),axis=1)<1e-7)
    elif port['surface']=='side':
        mask=(mesh.sub==G.SUBSTRUCTURES.index('side')) & (X[:,:,1].mean(1)*p[1]>0)
    elif port['surface']=='bilge':
        mask=mesh.sub==G.SUBSTRUCTURES.index('bilge')
    elif port['surface']=='deck':
        mask=mesh.sub==G.SUBSTRUCTURES.index('deck')
    else:raise ValueError('unregistered surface')
    candidates=np.flatnonzero(mask)
    candidates=candidates[np.argsort(np.linalg.norm(X[candidates].mean(1)-p,axis=1))[:24]]
    choices=[]
    for e in candidates:
        fun=lambda q:_shape(*q)[0]@X[e]-p
        sol=least_squares(fun,[0.,0.],bounds=(-1,1),ftol=1e-13,xtol=1e-13,gtol=1e-13)
        choices.append((np.linalg.norm(sol.fun),int(e),sol.x))
    if not choices:raise ValueError(f'no supporting elements for {port["id"]}')
    distance,e,uv=min(choices,key=lambda a:a[0]);N=_shape(*uv)[0];base=N@X[e];offset=p-base
    if port.get('nominal_mesh_projection',False):
        # Curved nominal surface to its faceted mesh; no invented rigid arm.
        if distance>.003:raise ValueError(f'{port["id"]} mesh projection exceeds 3 mm: {distance}')
        port['mesh_projection_m']=offset.tolist()
        port['position_m']=base.tolist()
        p=base;offset=np.zeros(3)
    elif distance>1e-5:raise ValueError(f'{port["id"]} is {distance:g} m off its registered surface')
    # Rigid arm to physical point: d.(u+theta cross offset), dual moment offset cross d.
    vec=np.r_[d,np.cross(offset,d)]
    row=sp.coo_matrix(((N[:,None]*vec).ravel(),(np.zeros(24,int),(mesh.quads[e,:,None]*6+np.arange(6)).ravel())),shape=(1,len(mesh.nodes)*6)).tocsr()
    nodal=row.toarray().reshape(-1,6)
    force=nodal[:,:3].sum(0);moment=np.cross(mesh.nodes,nodal[:,:3]).sum(0)+nodal[:,3:].sum(0)
    info=dict(element=e,element_nodes=mesh.quads[e].tolist(),natural_coordinate=uv.tolist(),shape_weights=N.tolist(),
      interpolated_position_m=base.tolist(),offset_m=offset.tolist(),resultant=force.tolist(),
      moment_about_origin=moment.tolist(),resultant_error=float(np.linalg.norm(force-d)),
      moment_error=float(np.linalg.norm(moment-np.cross(p,d))))
    return row,info


def operators(mesh,registration):
    registration=deepcopy(registration);S=[];B=[];channels=[]
    for p in registration['ports']:
        row,p['interpolation']=interpolate_port(mesh,p)
        if p['kind']=='force':B.append(row.T)
        else:S.append(row);channels.append(p['channel'])
    return sp.vstack(S).tocsr(),sp.hstack(B).tocsr(),np.array(channels),registration


def expanded_registry(fit_channels, held_channels):
    """Unflagged channels under explicit nominal mapping/normal assumptions.

    Original registry remains available for replaying the five-response study.
    This expansion does not certify the inherited channel permutation.
    """
    import json
    geometry=json.loads((D.DATA/'curated/geometry.json').read_text())
    old=registry(); added={p['channel']:p for p in old['ports'] if p['kind']=='acceleration'}
    ports=[];excluded=[]
    for c in geometry['acceleration_channels']:
        ch=c['channel']
        if c['fixture_or_mapping_concern']:
            excluded.append(dict(channel=ch,reason='source fixture or mapping concern'));continue
        if ch>=33:
            p=deepcopy(added[ch])
        else:
            xyz=np.array(c['xyz'],float); x,y,z=xyz
            if c['group']=='deck':
                position=xyz;direction=np.array([0.,0.,1.]);surface='deck'
            elif z>=0:
                position=np.array([x,np.sign(y)*G.R,z]);direction=np.array([0.,np.sign(y),0.]);surface='side'
            else:
                yz=np.array([y,z])/np.hypot(y,z)
                position=np.r_[x,G.R*yz];direction=np.r_[0.,yz];surface='bilge'
            p=dict(id=f'C{ch}',channel=ch,kind='acceleration',position_m=position.tolist(),
                photo_coordinate_m=xyz.tolist(),surface_projection_m=(position-xyz).tolist(),
                direction=direction.tolist(),surface=surface,units='m/s^2',nominal_mesh_projection=True,
                registration='stored nominal channel order; physical permutation unverified',
                polarity='outward hull normal / +z deck assumption; export Direction blank',
                provenance='curated/geometry.json and August source notes section 3')
        p['status']='calibration' if ch in fit_channels else 'spatial_validation'
        assert ch in fit_channels or ch in held_channels
        ports.append(p)
    ports += [p for p in old['ports'] if p['kind']=='force']
    return dict(frame=old['frame'],ports=ports,excluded=excluded,
        mapping_scope='nominal geometry for C1--C32, photo-based A1--A8; no physical permutation certificate',
        training_records=['test_96'],monitor_records=['test_101'],development_records=['test_103'])

