import pytest
import numpy as np

from mol_surface import connolly_surface, sa_surface

from pdb_io import to_xyz, read_pdb

import logging

logger = logging.getLogger(__name__)

pdb_6f6s = 'data/6FS6-mono_noe4z.pdb'
pdb_4ey5 = 'data/4ey5_clean.pdb'

pdb = pdb_4ey5


def test_sa_surface_rust(p=pdb):
    print('parse : ', p)
    c, e, r = read_pdb(p)
    dots = sa_surface(c, e, n=100, pr=1.4)
    to_xyz(dots, filename='test/{}-{}-cs_rust.xyz'.format(p.replace('/', '_'), n))


def test_sa_surface_python(p=pdb):
    c, e, r = read_pdb(p)
    dots = sa_surface(c, e, n=100, pr=1.4, enable_ext=False)
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


c, e, r = read_pdb(pdb)
n = 100
