# Code for making the array_layout.csv file
import pandas as pd
import hera_sim
import numpy as np
from pathlib import Path

from pyuvdata import UVData


def hex_array_csv(csv_path: str|Path):
    """Uses hera_sim.antpos.hex_array"""
    fields = ["Name", "Number", "BeamID", "E", "N", "U"]
    antpos = hera_sim.antpos.hex_array(2, split_core=False, outriggers=0)
    ant_to_beam = np.zeros(len(antpos), dtype=int)
    rows = list(
        [f"ANT{ant}", ant, ant_to_beam[ant], e, n, u]
        for ant, (e, n, u) in antpos.items()
    )

    #Write the array to a csv file.
    df = pd.DataFrame(rows, columns=fields)
    df.to_csv(csv_path, index=False, sep=' ')


def from_uvd(uvd: UVData, csv_path: str|Path, same_beam=False):
    """Note the 0 for BeamID for all antennas if same_beam"""

    fields = ["Name", "Number", "BeamID", "E", "N", "U"]
    names = uvd.antenna_names
    enus = uvd.get_ENU_antpos()
    ant_to_beam = np.zeros(len(names), dtype=int) if same_beam else enus[1]
    rows = list(
        [name, num, beam_id, e, n, u]
        for name, num, beam_id, (e,n,u) in zip(names, enus[1], ant_to_beam, enus[0])
    )

    #Write the array to a csv file.
    df = pd.DataFrame(rows, columns=fields)
    df.to_csv(csv_path, index=False, sep=' ')
        
#hex_array_csv('hex_array.csv')


