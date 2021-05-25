from electrostatic import run_electrosatatic
from find_pocket import find_pocket
from pdb_io import read_pdb, to_pdb
import pytest

from hydrophobicity import run_hydro
import logging

logger = logging.getLogger(__name__)


def test_run_hydro():
    logging.info("test_run_hydro...")
    run_hydro('data/6FS6-mono_noe4z.pdb', dir='test')


def test_run_electro():
    logging.info("test_run_electro...")
    c, e, r = read_pdb('data/6FS6-mono_noe4z.pdb')
    grids = find_pocket(c, e)
    elecs = run_electrosatatic(grids, c, e, r)
    to_pdb(grids, elecs, filename="test/test.pdb")
