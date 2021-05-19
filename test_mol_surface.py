import pytest
import numpy as np

from mol_surface import sa_surface_vec, connolly_surface

from pdb_io import to_xyz

import logging

logger = logging.getLogger(__name__)


def test_sa_surface_vec():
    coor = np.array([[0, 0, 0], [0, 0, 2.7]])
    elements = np.array(['C', 'O'])
    dots = sa_surface_vec(coor, elements, n=40, pr=0)
    to_xyz(dots, filename='test/test_sa_surface_vec.xyz')
    assert dots.shape[0] == 73


def test_connolly_surface():
    coor = np.array([[0, 0, 0], [0, 0, 2.7]])
    elements = np.array(['C', 'O'])
    dots = connolly_surface(coor, elements, n=40, pr=1.4)
    logging.info(dots.shape)
    to_xyz(dots, filename='test/test_connolly_surface.xyz')
