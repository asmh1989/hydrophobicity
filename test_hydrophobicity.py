import logging

import numpy as np
import pytest
import math

from electrostatic import run_electrosatatic
from find_pocket import find_pocket, layer_grids
from hydrophobicity import run_hydro, cal_grids_hydro
from mol_surface import sa_surface
from pdb_io import read_pdb, to_pdb

logger = logging.getLogger(__name__)


pdb_6fs6 = "data/6FS6-mono_noe4z.pdb"
pdb_4ey5 = 'data/4ey5_clean.pdb'
pdb_6fyi = "data/6fyi_clean.pdb"
pdb = pdb_6fyi


def test_6fs6_run_hydro_rust():
    pdb = pdb_6fs6
    logging.info("test_6fs6_run_hydro_rust...")
    h = run_hydro(pdb, dir='test', n=100)
    hyo = h[:, 3]
    logger.info("hyo max = %s, min = %s, m = %s, len = %s",
                np.max(hyo), np.min(hyo), np.mean(hyo), hyo.shape[0])
    assert(math.isclose(80.74128537450976, np.max(hyo), rel_tol=1e-6))
    assert(math.isclose(-228.6843365052476, np.min(hyo), rel_tol=1e-6))
    assert(math.isclose(-76.97541431222143, np.mean(hyo), rel_tol=1e-6))
    assert(6460 == hyo.shape[0])

    logging.info("test_6fs6_run_hydro_rust done ...")


def test_run_hydro_rust():
    logging.info("test_run_hydro_rust...")
    h = run_hydro(pdb, dir='test', n=100)


def test_4ey5_run_hydro_rust():
    pdb = pdb_4ey5
    logging.info("test_4ey5_run_hydro_rust...")
    h = run_hydro(pdb, dir='test', n=100)
    hyo = h[:, 3]
    logger.info("hyo max = %s, min = %s, m = %s, len = %s",
                np.max(hyo), np.min(hyo), np.mean(hyo), hyo.shape[0])
    assert(math.isclose(76.50174143608243, np.max(hyo), rel_tol=1e-6))
    assert(math.isclose(-197.84444459826696, np.min(hyo), rel_tol=1e-6))
    assert(math.isclose(-75.44844935474565, np.mean(hyo), rel_tol=1e-6))
    assert(14389 == hyo.shape[0])

    logging.info("test_4ey5_run_hydro_rust done ...")


def test_6fs6_run_hydro_python():
    pdb = pdb_6fs6
    logging.info("test_run_hydro_python...")
    h = run_hydro(pdb, dir='test', n=100, enable_ext=False)
    hyo = h[:, 3]

    logger.info("hyo max = %s, min = %s, m = %s, len = %s",
                np.max(hyo), np.min(hyo), np.mean(hyo), hyo.shape[0])
    logging.info("test_run_hydro_python done ...")


def test_run_electro():
    logging.info("test_run_electro...")
    c, e, r = read_pdb('data/6FS6-mono_noe4z.pdb')
    grids = find_pocket(c, e)
    elecs = run_electrosatatic(grids, c, e, r)
    to_pdb(grids, elecs, filename="test/test.pdb")


def test_cutom_hydro():
    logging.info("test_cutom_hydro...")
    c = np.array([
        [5.80400e+00, 7.71280e+01, 3.75770e+01],
        [1.15920e+01, 8.69370e+01, 3.19960e+01],
        [3.04400e+00, 9.48200e+01, 5.90210e+01],
        [7.69800e+00, 1.04841e+02, 4.05320e+01],
        [1.57460e+01, 9.82030e+01, 5.00190e+01],
        [1.27360e+01, 1.04393e+02, 6.32570e+01],
        [1.98680e+01, 9.97520e+01, 6.71680e+01],
        [1.08790e+01, 9.28970e+01, 6.84030e+01],
        [2.43770e+01, 7.28520e+01, 4.72010e+01],
        [7.02600e+00, 7.53090e+01, 3.73980e+01],
        [1.25710e+01, 8.66980e+01, 3.00440e+01],
        [3.09400e+00, 9.30790e+01, 6.03660e+01],
        [5.77400e+00, 1.03860e+02, 4.01630e+01],
        [1.70960e+01, 9.74770e+01, 5.15840e+01],
        [1.45720e+01, 1.03200e+02, 6.34270e+01],
        [2.07020e+01, 1.01729e+02, 6.67280e+01],
        [9.60500e+00, 9.36190e+01, 7.00330e+01],
        [2.47620e+01, 7.29910e+01, 4.50380e+01],
        [9.03600e+00, 8.09360e+01, 3.48480e+01],
        [5.00000e-03, 9.38070e+01, 3.97600e+01],
        [4.39500e+00, 8.90320e+01, 3.82840e+01],
        [1.14030e+01, 1.00312e+02, 3.92750e+01],
        [1.86520e+01, 9.02370e+01, 3.51820e+01],
        [1.99960e+01, 8.57930e+01, 3.65880e+01],
        [8.24200e+00, 8.60170e+01, 5.20920e+01],
        [5.43500e+00, 1.05824e+02, 5.37510e+01],
        [1.44180e+01, 9.72540e+01, 4.74530e+01],
        [2.50970e+01, 1.13595e+02, 5.18200e+01],
        [1.79550e+01, 1.06014e+02, 6.69910e+01],
        [1.95970e+01, 9.96520e+01, 5.07750e+01],
        [3.03660e+01, 8.34110e+01, 5.82150e+01],
        [2.87640e+01, 9.12670e+01, 4.82120e+01],
        [2.63990e+01, 8.39690e+01, 5.99980e+01],
        [1.90280e+01, 9.09220e+01, 6.35440e+01],
        [1.72860e+01, 8.86570e+01, 7.04010e+01],
        [7.12100e+00, 8.89970e+01, 7.06170e+01],
        [8.21100e+00, 7.78960e+01, 5.15780e+01],
        [2.46210e+01, 8.73660e+01, 4.91020e+01],
        [3.20860e+01, 8.06230e+01, 3.66420e+01],
        [7.41100e+00, 8.22890e+01, 3.54450e+01],
        [-2.53000e-01, 9.23460e+01, 4.13810e+01],
        [5.37900e+00, 8.75000e+01, 3.70590e+01],
        [1.06620e+01, 1.01987e+02, 3.80660e+01],
        [1.97990e+01, 9.19120e+01, 3.60070e+01],
        [2.03290e+01, 8.62490e+01, 3.44630e+01],
        [6.66900e+00, 8.61080e+01, 5.36150e+01],
        [6.10700e+00, 1.07107e+02, 5.54030e+01],
        [1.26860e+01, 9.73600e+01, 4.61060e+01],
        [2.51710e+01, 1.13492e+02, 5.40130e+01],
        [1.70600e+01, 1.04244e+02, 6.79330e+01],
        [1.99590e+01, 9.75000e+01, 5.05250e+01],
        [3.07600e+01, 8.14850e+01, 5.72390e+01],
        [3.07140e+01, 9.03740e+01, 4.86900e+01],
        [2.58450e+01, 8.19220e+01, 6.05740e+01],
        [1.94380e+01, 8.96860e+01, 6.53080e+01],
        [1.83800e+01, 9.02540e+01, 7.14330e+01],
        [6.30500e+00, 8.74020e+01, 6.93420e+01],
        [7.13200e+00, 7.70040e+01, 5.32690e+01],
        [2.36800e+01, 8.53870e+01, 4.89620e+01],
        [3.09930e+01, 7.93490e+01, 3.80640e+01],
        [1.58600e+00, 9.70140e+01, 3.88130e+01],
        [6.20600e+00, 9.69940e+01, 3.08260e+01],
        [6.55100e+00, 8.39770e+01, 3.19110e+01],
        [2.05300e+01, 8.99400e+01, 3.80640e+01],
        [1.74480e+01, 1.15868e+02, 4.62290e+01],
        [2.92900e+01, 1.09602e+02, 4.80050e+01],
        [1.24330e+01, 1.03662e+02, 7.07970e+01],
        [2.21510e+01, 9.61260e+01, 4.91850e+01],
        [2.44550e+01, 9.91140e+01, 4.41590e+01],
        [3.44570e+01, 1.04502e+02, 4.74770e+01],
        [2.48460e+01, 1.14000e+02, 5.72050e+01],
        [2.77310e+01, 1.05822e+02, 6.51290e+01],
        [1.90110e+01, 8.68850e+01, 6.52190e+01],
        [4.49600e+00, 8.48850e+01, 3.82510e+01],
        [1.20850e+01, 1.01124e+02, 6.68980e+01],
        [4.97400e+00, 1.04582e+02, 4.85950e+01],
        [1.71910e+01, 1.02904e+02, 3.82380e+01],
        [2.06710e+01, 1.04095e+02, 6.44010e+01],
        [2.03670e+01, 8.55330e+01, 4.68310e+01],
        [3.00350e+01, 8.73280e+01, 4.86200e+01],
        [1.88810e+01, 9.12420e+01, 6.74910e+01],
        [6.34600e+00, 8.81260e+01, 6.24560e+01],
        [7.51000e+00, 8.14790e+01, 5.39010e+01],
        [2.08090e+01, 7.75330e+01, 5.66710e+01],
        [8.67000e+00, 7.34130e+01, 4.91670e+01],
        [2.74160e+01, 7.66170e+01, 4.81380e+01],
        [3.40930e+01, 8.78960e+01, 4.53060e+01],
        [6.15200e+00, 8.47160e+01, 3.66770e+01],
        [1.38470e+01, 1.02163e+02, 6.58660e+01],
        [5.81800e+00, 1.04189e+02, 5.06870e+01],
        [1.59590e+01, 1.01894e+02, 3.65920e+01],
        [1.90750e+01, 1.02461e+02, 6.45870e+01],
        [2.09390e+01, 8.39690e+01, 4.52590e+01],
        [2.80940e+01, 8.85510e+01, 4.86470e+01],
        [1.75960e+01, 9.23060e+01, 6.90580e+01],
        [4.23800e+00, 8.75430e+01, 6.31330e+01],
        [7.23200e+00, 8.31000e+01, 5.23020e+01],
        [2.12700e+01, 7.53210e+01, 5.62670e+01],
        [6.48500e+00, 7.35580e+01, 4.85030e+01],
        [2.61910e+01, 7.47670e+01, 4.75810e+01],
        [3.43370e+01, 8.56260e+01, 4.51470e+01],
        [1.81800e+01, 9.60300e+01, 5.04480e+01],
        [1.63620e+01, 9.84730e+01, 4.80930e+01]
    ])

    e = np.array([
        "N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "N", "CA", "C", "O", "CB", "CG",
        "OD1", "OD2", "N", "CA", "C", "O", "CB", "N", "CA", "C", "O", "CB", "CG", "CD", "OE1",
        "OE2", "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "N", "CA", "C", "O", "CB", "CG",
        "CD1", "CD2", "N", "CA", "C", "O", "CB", "CG1", "CG2", "N", "CA", "C", "O", "CB",
        "CG2", "OG1", "N", "CA", "C", "O", "CB", "CG1", "CG2", "N", "CA", "C", "O", "CB", "CG",
        "CD", "NE", "CZ", "NH1", "NH2", "N", "CA", "C", "O", "N", "CA", "C", "O", "N", "CA",
        "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "N", "CA", "C", "O", "CB", "CG",
                  "CD1", "CD2", "N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "N",
                  "CA", "C", "O", "N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "N", "CA", "C", "O",
                  "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "N", "CA", "C", "O", "CB", "CG", "CD1",
                  "CD2", "N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "N", "CA", "C", "O", "CB",
                  "CG2", "OG1", "N", "CA", "C", "O", "CB", "CG", "CD", "N", "CA", "C", "O", "N", "CA",
                  "C", "O", "N", "CA", "C", "O", "CB", "CG", "CD", "N", "CA", "C", "O", "CB", "CG1",
                  "CG2", "N", "CA", "C", "O", "CB", "OG", "N", "CA", "C", "O", "CB", "N", "CA", "C", "O",
                  "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "CA", "C", "O", "CB", "CG", "CD1",
                  "CD2", "N", "CA", "C", "O", "N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "N", "CA",
                  "C", "O", "CB", "CG", "CD", "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1",
                  "CE2", "CZ", "N", "CA", "C", "O", "CB", "N", "CA", "C", "O", "CB", "CG", "CD", "OE1",
                  "OE2", "N", "CA", "C", "O", "CB", "CG", "CD", "N", "CA", "C", "O", "CB", "CG", "CD",
                  "N", "CA", "C", "O", "CB", "CG", "SD", "CE", "N", "CA", "C", "O", "N", "CA", "C", "O",
                  "CB", "CG", "CD", "N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ"])
    r = np.array([
        "GLU", "GLU", "GLU", "GLU", "GLU", "GLU", "GLU", "GLU", "GLU", "ASP", "ASP", "ASP",
        "ASP", "ASP", "ASP", "ASP", "ASP", "ALA", "ALA", "ALA", "ALA", "ALA", "GLU", "GLU",
        "GLU", "GLU", "GLU", "GLU", "GLU", "GLU", "GLU", "LEU", "LEU", "LEU", "LEU", "LEU",
        "LEU", "LEU", "LEU", "LEU", "LEU", "LEU", "LEU", "LEU", "LEU", "LEU", "LEU", "VAL",
        "VAL", "VAL", "VAL", "VAL", "VAL", "VAL", "THR", "THR", "THR", "THR", "THR", "THR",
        "THR", "VAL", "VAL", "VAL", "VAL", "VAL", "VAL", "VAL", "ARG", "ARG", "ARG", "ARG",
        "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "GLY", "GLY", "GLY", "GLY", "GLY",
        "GLY", "GLY", "GLY", "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "ARG",
        "ARG", "ARG", "LEU", "LEU", "LEU", "LEU", "LEU", "LEU", "LEU", "LEU", "ARG", "ARG",
        "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "GLY", "GLY", "GLY",
        "GLY", "ILE", "ILE", "ILE", "ILE", "ILE", "ILE", "ILE", "ILE", "ARG", "ARG", "ARG",
        "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "LEU", "LEU", "LEU", "LEU",
        "LEU", "LEU", "LEU", "LEU", "LYS", "LYS", "LYS", "LYS", "LYS", "LYS", "LYS", "LYS",
        "LYS", "THR", "THR", "THR", "THR", "THR", "THR", "THR", "PRO", "PRO", "PRO", "PRO",
        "PRO", "PRO", "PRO", "GLY", "GLY", "GLY", "GLY", "GLY", "GLY", "GLY", "GLY", "PRO",
        "PRO", "PRO", "PRO", "PRO", "PRO", "PRO", "VAL", "VAL", "VAL", "VAL", "VAL", "VAL",
        "VAL", "SER", "SER", "SER", "SER", "SER", "SER", "ALA", "ALA", "ALA", "ALA", "ALA",
        "PHE", "PHE", "PHE", "PHE", "PHE", "PHE", "PHE", "PHE", "PHE", "PHE", "PHE", "LEU",
        "LEU", "LEU", "LEU", "LEU", "LEU", "LEU", "LEU", "GLY", "GLY", "GLY", "GLY", "ILE",
        "ILE", "ILE", "ILE", "ILE", "ILE", "ILE", "ILE", "PRO", "PRO", "PRO", "PRO", "PRO",
        "PRO", "PRO", "PHE", "PHE", "PHE", "PHE", "PHE", "PHE", "PHE", "PHE", "PHE", "PHE",
        "PHE", "ALA", "ALA", "ALA", "ALA", "ALA", "GLU", "GLU", "GLU", "GLU", "GLU", "GLU",
        "GLU", "GLU", "GLU", "PRO", "PRO", "PRO", "PRO", "PRO", "PRO", "PRO", "PRO", "PRO",
        "PRO", "PRO", "PRO", "PRO", "PRO", "MET", "MET", "MET", "MET", "MET", "MET", "MET",
        "MET", "GLY", "GLY", "GLY", "GLY", "PRO", "PRO", "PRO", "PRO", "PRO", "PRO", "PRO",
        "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "ARG", "ARG",
    ])

    n = 100
    sa = sa_surface(c, e, n=n, pr=1.4)
    logger.info("sa = %s", sa.shape)
    layered_grids = layer_grids(c, e, n=n, pr=20)
    logger.info("layered_grids = %s", layered_grids.shape)
    hyo = cal_grids_hydro(layered_grids, c, e, r, sa, n=n)

    grid_coors = layered_grids[:, :3]
    result = np.insert(grid_coors, 3, hyo, axis=1)

    print("hyo ", result)

    print(result.tolist(),  file=open('e.json', 'w'))
