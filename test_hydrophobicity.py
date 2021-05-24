import pytest

from hydrophobicity import run_hydro
import logging

logger = logging.getLogger(__name__)


def test_run_hydro():
    logging.info("test_run_hydro...")
    run_hydro('data/6FS6-mono_noe4z.pdb', dir='test')
