from pyuvdata import UVData
from pathlib import Path
import numpy as np

"""
beam_paths:
    0: 'airy'
diameter: 14.7
ref_freq: 150000000.0
spectral_index: -2
telescope_location: (-30.72152777777791, 21.428305555555557, 1073.0000000093132)
telescope_name: end_to_end_example

beam_paths:
  0: hera.uvbeam
  1:
    type: airy
    diameter: 16
  2:
    type: gaussian
    sigma: 0.03
  3: airy
diameter: 12
spline_interp_opts:
        kx: 4
        ky: 4
freq_interp_kind: 'cubic'
telescope_location: (-30.72152777777791, 21.428305555555557, 1073.0000000093132)
telescope_name: BLLITE
"""


def from_uvd(uvd: UVData,
             yaml_path: str | Path,
             same_beam: bool = False,
             telescope_name: str = None,
             type_: str = 'airy',
             spectral_index: int = -2,
             ref_freq=None):
    """All beams have the same type. Might change in the future.
    If no ref freq is given then the average of freq_array is used.
    If telescope name is None then it will try to find the info in uvd, else it will be 'unknown'"""
    # beams
    beams = []
    if same_beam:
        avg_diam = np.average(uvd.antenna_diameters)
        beams.append({'id': 0, 'type': type_, 'diameter': avg_diam})
    else:
        for i, diam in enumerate(uvd.antenna_diameters):
            beams.append({'id': i, 'type': type_, 'diameter': diam})

    # ref_freq
    if ref_freq is None:
        ref_freq = np.average(uvd.freq_array)

    #telescope name
    if not telescope_name:
        telescope_name = uvd.telescope_name
    if not telescope_name:
        telescope_name = uvd.object_name
    if not telescope_name:
        telescope_name = 'unknown'

    with open(yaml_path, "w") as cfg:
        s = 'beam_paths:\n'
        for b in beams:
            s += f'    {b["id"]}:\n'
            s += f'        type: {b["type"]}\n'
            s += f'        diameter: {b["diameter"]}\n'
        s += f'spectral_index: {spectral_index}\n'
        s += f'ref_freq: {ref_freq}\n'
        s += f'telescope_name: {telescope_name}\n'
        s += f'telescope_location: {uvd.telescope_location_lat_lon_alt_degrees}\n'
        cfg.write(s)
