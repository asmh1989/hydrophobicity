import pytest
import numpy as np

from mol_surface import connolly_surface2, connolly_surface

from pdb_io import to_xyz, read_pdb

import logging

logger = logging.getLogger(__name__)


def test_sa_surface():
    coor = np.array([[0, 0, 0], [0, 0, 2.7]])
    elements = np.array(['C', 'O'])
    dots = connolly_surface(coor, elements, n=100, pr=1.4)
    to_xyz(dots, filename='test/test_sa_surface_100.xyz')
    # assert dots.shape[0] == 73


def test_sa_surface_rust():
    c, e, r = read_pdb('data/6FS6-mono_noe4z.pdb')
    n = 200
    dots = connolly_surface(c, e, n=n, pr=1.4)
    to_xyz(dots, filename='test/6f6s-{}-cs_rust.xyz'.format(n))


def test_sa_surface_python():
    c, e, r = read_pdb('data/6FS6-mono_noe4z.pdb')
    dots = connolly_surface2(c, e, n=50, pr=1.4)
    to_xyz(dots, filename='test/6f6s-100-cs_python.xyz')


def test_connolly_surface():
    coor = np.array([[0, 0, 0], [0, 0, 2.7]])
    elements = np.array(['C', 'O'])
    dots = connolly_surface(coor, elements, n=40, pr=1.4)
    logging.info(dots.shape)
    to_xyz(dots, filename='test/test_connolly_surface.xyz')
