import logging
import math

import numpy as np

from sitemap.hydrophobicity.electrostatic import run_electrosatatic
from sitemap.hydrophobicity.find_pocket import find_pocket
from sitemap.hydrophobicity.hydrophobicity import run_hydro
from sitemap.hydrophobicity.pdb_io import read_pdb, to_pdb

logger = logging.getLogger(__name__)

pdb_6fs6 = "data/6FS6-mono_noe4z.pdb"
pdb_4ey5 = "data/4ey5_clean.pdb"
pdb_6fyi = "data/6fyi_clean.pdb"
pdb = pdb_6fyi


def test_6fs6_run_hydro_rust():
    pdb = pdb_6fs6
    logging.info("test_6fs6_run_hydro_rust...")
    h = run_hydro(pdb, dir="test", n=100)
    hyo = h[:, 3]
    logger.info(
        "hyo max = %s, min = %s, m = %s, len = %s", np.max(hyo), np.min(hyo), np.mean(hyo), hyo.shape[0],
    )
    assert math.isclose(80.74128537450976, np.max(hyo), rel_tol=1e-6)
    assert math.isclose(-228.6843365052476, np.min(hyo), rel_tol=1e-6)
    assert math.isclose(-76.97541431222143, np.mean(hyo), rel_tol=1e-6)
    assert 6460 == hyo.shape[0]

    logging.info("test_6fs6_run_hydro_rust done ...")


def test_run_hydro_rust():
    logging.info("test_run_hydro_rust...")
    run_hydro(pdb, dir="test", n=100)


def test_4ey5_run_hydro_rust():
    pdb = pdb_4ey5
    logging.info("test_4ey5_run_hydro_rust...")
    h = run_hydro(pdb, dir="test", n=100)
    hyo = h[:, 3]
    logger.info(
        "hyo max = %s, min = %s, m = %s, len = %s", np.max(hyo), np.min(hyo), np.mean(hyo), hyo.shape[0],
    )
    assert math.isclose(76.50174143608243, np.max(hyo), rel_tol=1e-6)
    assert math.isclose(-197.84444459826696, np.min(hyo), rel_tol=1e-6)
    assert math.isclose(-75.44844935474565, np.mean(hyo), rel_tol=1e-6)
    assert 14389 == hyo.shape[0]

    logging.info("test_4ey5_run_hydro_rust done ...")


# 运行太慢注释
# def test_6fs6_run_hydro_python():
#     pdb = pdb_6fs6
#     logging.info("test_run_hydro_python...")
#     h = run_hydro(pdb, dir="test", n=100, enable_ext=False)
#     hyo = h[:, 3]

#     logger.info(
#         "hyo max = %s, min = %s, m = %s, len = %s", np.max(hyo), np.min(hyo), np.mean(hyo), hyo.shape[0],
#     )
#     logging.info("test_run_hydro_python done ...")


def test_run_electro():
    logging.info("test_run_electro...")
    c, e, r = read_pdb("data/6FS6-mono_noe4z.pdb")
    grids = find_pocket(c, e)
    elecs = run_electrosatatic(grids, c, e, r)
    to_pdb(grids, elecs, filename="test/test.pdb")
