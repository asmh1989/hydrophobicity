import pytest
import numpy as np

from mol_surface import connolly_surface, sa_surface

from pdb_io import to_xyz, read_pdb
from find_pocket import find_pocket, gen_grid, pas_search_for_pocket, sas_search_del

import logging

logger = logging.getLogger(__name__)

pdb_6f6s = 'data/6FS6-mono_noe4z.pdb'
pdb_4ey5 = 'data/4ey5_clean.pdb'

pdb = pdb_6f6s

n = 40
c, e, r = read_pdb(pdb_6f6s)


def test_sa_surface_rust(p=pdb):
    print('parse : ', p)
    c, e, r = read_pdb(p)
    dots = sa_surface(c, e, n=n, pr=1.4)
    to_xyz(dots, filename='test/{}-{}-cs_rust.xyz'.format(p.replace('/', '_'), n))


def test_sa_surface_python(p=pdb):
    c, e, r = read_pdb(p)
    dots = sa_surface(c, e, n=n, pr=1.4, enable_ext=False)
    to_xyz(dots, filename='test/{}-{}-cs_rust.xyz'.format(p.replace('/', '_'), n))


def test_connolly_surface_rust(p=pdb):
    c, e, r = read_pdb(p)
    dots = connolly_surface(c, e, n=n, pr=1.4)
    to_xyz(dots, filename='test/{}-{}-cs_rust.xyz'.format(p.replace('/', '_'), n))


def test_connolly_surface_python(p=pdb):
    c, e, r = read_pdb(p)
    dots = connolly_surface(c, e, n=n, pr=1.4, enable_ext=False)
    to_xyz(dots, filename='test/{}-{}-cs_python.xyz'.format(p.replace('/', '_'), n))


def test_connolly_surface():
    coor = np.array([[0, 0, 0], [0, 0, 2.7]])
    elements = np.array(['C', 'O'])
    dots = connolly_surface(coor, elements, n=40, pr=1.4)
    logging.info(dots.shape)
    to_xyz(dots, filename='test/test_connolly_surface.xyz')


def test_find_pockets_rust(p=pdb):
    c, e, r = read_pdb(p)
    grids = find_pocket(c, e, n, 20, enable_ext=True)
    print("grids = ", grids.shape)


def test_find_pockets_python(p=pdb):
    c, e, r = read_pdb(p)
    grids = find_pocket(c, e, n,  20, enable_ext=False)
    print("grids = ", grids.shape)


def test_custom_data():
    c = np.array([[5.80400e+00, 7.71280e+01, 3.75770e+01],
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
                  [4.39500e+00, 8.90320e+01, 3.82840e+01]])
    e = np.array(["N",
                  "CA",
                  "C",
                  "O",
                  "CB",
                  "CG",
                  "CD",
                  "OE1",
                  "OE2",
                  "N",
                  "CA",
                  "C",
                  "O",
                  "CB",
                  "CG",
                  "OD1",
                  "OD2",
                  "N",
                  "CA",
                  "C",
                  "O",
                  "CB",
                  "N",
                  "CA",
                  "C",
                  "O",
                  "CB",
                  "CG",
                  "CD",
                  "OE1",
                  "OE2",
                  "N",
                  "CA",
                  "C",
                  "O",
                  "CB",
                  "CG",
                  "CD1",
                  "CD2",
                  "N",
                  "CA",
                  "C",
                  "O",
                  "CB",
                  "CG",
                  "CD1",
                  "CD2",
                  "N",
                  "CA",
                  "C",
                  "O",
                  "CB",
                  "CG1",
                  "CG2",
                  "N",
                  "CA",
                  "C",
                  "O",
                  "CB",
                  "CG2",
                  "OG1",
                  "N",
                  "CA",
                  "C",
                  "O",
                  "CB",
                  "CG1",
                  "CG2",
                  "N",
                  "CA",
                  "C",
                  "O",
                  "CB",
                  "CG",
                  "CD",
                  "NE",
                  "CZ",
                  "NH1",
                  "NH2",
                  "N",
                  "CA",
                  "C",
                  "O",
                  "N",
                  "CA",
                  "C",
                  "O",
                  "N",
                  "CA",
                  "C",
                  "O",
                  "CB",
                  "CG",
                  "CD",
                  "NE",
                  "CZ",
                  "NH1",
                  "NH2",
                  "N",
                  "CA",
                  "C",
                  "O",
                  "CB",
                  "CG"])
    dot = sa_surface(c, e, n=100, pr=20)
    print('shape = ', dot.shape)
    pocket_grids = gen_grid(c, n=1)
    print('shape = ', pocket_grids.shape)
    pocket_grids = sas_search_del(c, e, pocket_grids, pr=1.4)
    print('shape = ', pocket_grids.shape)
    pocket_grids = pas_search_for_pocket(pocket_grids, dot, n=n, pr=20)
    print('shape = ', pocket_grids.shape)

    grid = find_pocket(c, e, 100, 20)
    print('shape = ', grid.shape)
