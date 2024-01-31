import sys
import os
import glob
import numpy as np
import re
from pathlib import Path
from ase import Atoms
from ase.units import Bohr, Ha
from ase.io import read, write
from ase.parallel import paropen, parprint
from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.external import static_polarizability


def anisotropy(eig):
    eig.sort()
    norm = np.sqrt(2 * np.sum(eig**2))
    fa = np.sqrt(
        (eig[2] - eig[1])**2 +
        (eig[2] - eig[0])**2 +
        (eig[1] - eig[0])**2) / norm
    return fa


def alpha_tensor(atoms, charge, base_dir, fname):
    """ Calculate polarizability tensors in GPAW.

    - Input
    atoms: atoms object
    charge: net charge on molecule
    fname: file name for output file

    - Returns [Ang**3]
    alpha_cc: polarizability tensor
    cvol: cavity volume
    """
    atoms = Cluster(atoms)
    atoms.minimal_box(4.0, h=0.2)
    atoms.calc = GPAW(
    mode='fd',
    xc='PBE', charge=charge,
    txt=f'{base_dir}/output/alpha_gas/{fname}.gpaw_out')
    atoms.get_potential_energy()

    alpha_cc = static_polarizability(atoms, strength=0.01) * Bohr * Ha
    return alpha_cc


# calculate polarizability for all files in folder
base_dir = '/home/fr/fr_fr/fr_ez1021/sims/amino_acids_publish'
files = glob.glob(f'{base_dir}/structures/gasphase/*.xyz')

# run array job
task_id = int(sys.argv[1])
fname = Path(files[task_id]).stem

# set total charge based on filename
if re.search('.+anion', fname):
    charge = -1
elif re.search('.+cation', fname):
    charge = 1
else:
    charge = 0

atoms = Atoms(read(f'{base_dir}/structures/gasphase/{fname}.xyz'))

log = paropen(f'{base_dir}/results/alpha_gas.dat', 'a')
alpha_cc = alpha_tensor(
    atoms, charge=charge, base_dir=base_dir, fname=fname)

# get eigenvalues and eigenvectors
w, v = np.linalg.eig(alpha_cc)
axx = w[0]
ayy = w[1]
azz = w[2]

# fractional anisotropy
fa = anisotropy(w)

# isotropic polarizability
alpha = w.sum()/3

print(f'{fname} {charge} {alpha:.4f} {axx:.4f} {ayy:.4f} {azz:.4f} {fa:.4f}', file=log)
log.close()
