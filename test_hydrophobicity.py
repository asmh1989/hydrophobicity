import pytest

from hydrophobicity import run_hydro_vec
import logging

logger = logging.getLogger(__name__)


def test_run_hydro_vec():
    logging.info("test_run_hydro_vec...")
    run_hydro_vec('data/6FS6-mono_noe4z.pdb', dir='test')
