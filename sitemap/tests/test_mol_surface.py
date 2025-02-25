import logging

import numpy as np

from sitemap.hydrophobicity.find_pocket import find_pocket, layer_grids
from sitemap.hydrophobicity.mol_surface import connolly_surface, sa_surface
from sitemap.hydrophobicity.pdb_io import read_pdb, to_xyz

logger = logging.getLogger(__name__)

pdb_6f6s = "data/6FS6-mono_noe4z.pdb"
pdb_4ey5 = "data/4ey5_clean.pdb"

pdb = pdb_4ey5

n = 100
c, e, r = read_pdb(pdb_4ey5)


def test_sa_surface_rust(p=pdb):
    print("parse : ", p)
    c, e, r = read_pdb(p)
    dots = sa_surface(c, e, n=n, pr=1.4)
    logger.info("dots = %s", dots.shape)
    to_xyz(dots, filename="test/{}-{}-cs_rust.xyz".format(p.replace("/", "_"), n))


# 运行太慢注释
# def test_sa_surface_python(p=pdb):
#     c, e, r = read_pdb(p)
#     dots = sa_surface(c, e, n=n, pr=1.4, enable_ext=False)
#     logger.info("dots = %s", dots.shape)

#     to_xyz(dots, filename="test/{}-{}-cs_rust.xyz".format(p.replace("/", "_"), n))


def test_connolly_surface_rust(p=pdb):
    c, e, r = read_pdb(p)
    dots = connolly_surface(c, e, n=n, pr=1.4)
    to_xyz(dots, filename="test/{}-{}-cs_rust.xyz".format(p.replace("/", "_"), n))


def test_connolly_surface_python(p=pdb):
    c, e, r = read_pdb(p)
    dots = connolly_surface(c, e, n=10, pr=1.4, enable_ext=False)
    to_xyz(
        dots, filename="test/{}-{}-cs_python.xyz".format(p.replace("/", "_"), n),
    )


def test_connolly_surface():
    coor = np.array([[0, 0, 0], [0, 0, 2.7]])
    elements = np.array(["C", "O"])
    dots = connolly_surface(coor, elements, n=40, pr=1.4)
    logging.info(dots.shape)
    to_xyz(dots, filename="test/test_connolly_surface.xyz")


def test_find_pockets_rust():
    n = 100
    p = pdb_4ey5
    logger.info("start ... test_find_pockets_rust")
    c, e, r = read_pdb(p)
    logger.info("start ... find_pocket")
    grids = find_pocket(c, e, n, 20, enable_ext=True)
    logger.info("grids = %s", grids.shape)
    assert grids.shape[0] == 14389


def test_find_pockets_rust2():
    n = 100
    p = pdb_6f6s
    logger.info("start ... test_find_pockets_rust")
    c, e, r = read_pdb(p)
    logger.info("start ... find_pocket")
    grids = find_pocket(c, e, n, 20, enable_ext=True)
    logger.info("grids = %s", grids.shape)
    assert grids.shape[0] == 6460


# 运行过慢注释
# def test_find_pockets_python():
#     n = 100
#     p = pdb_4ey5
#     c, e, r = read_pdb(p)
#     grids = find_pocket(c, e, n, 20, enable_ext=False)
#     print("grids = ", grids.shape)
#     assert grids.shape[0] == 14389


def test_find_pockets_python2():
    n = 100
    p = pdb_6f6s
    c, e, r = read_pdb(p)
    grids = find_pocket(c, e, n, 20, enable_ext=False)
    logger.info("grids = %s", grids.shape)
    assert grids.shape[0] == 6460


def test_find_layer_rust():
    logger.info("test_find_layer_rust...")
    n = 100
    p = pdb_4ey5
    c, e, _ = read_pdb(p)
    logger.info("start find layer...")
    grids = layer_grids(c, e, n, 20)
    logger.info("grids = %s", grids.shape)
    dm = grids[:, 3]
    assert grids[dm == -993].shape[0] == 9660
    assert grids[dm == -64].shape[0] == 593


#  运行过慢注释
# def test_find_layer_python():
#     logger.info("test_find_layer_python...")
#     n = 100
#     p = pdb_4ey5
#     c, e, r = read_pdb(p)
#     logger.info("start find layer...")
#     grids = layer_grids(c, e, n, 20, False)
#     logger.info("grids = %s", grids.shape)
#     dm = grids[:, 3]
#     assert grids[dm == -993].shape[0] == 9660
#     assert grids[dm == -64].shape[0] == 593


def test_custom_data():
    c = np.array(
        [
            [5.80400e00, 7.71280e01, 3.75770e01],
            [1.15920e01, 8.69370e01, 3.19960e01],
            [3.04400e00, 9.48200e01, 5.90210e01],
            [7.69800e00, 1.04841e02, 4.05320e01],
            [1.57460e01, 9.82030e01, 5.00190e01],
            [1.27360e01, 1.04393e02, 6.32570e01],
            [1.98680e01, 9.97520e01, 6.71680e01],
            [1.08790e01, 9.28970e01, 6.84030e01],
            [2.43770e01, 7.28520e01, 4.72010e01],
            [7.02600e00, 7.53090e01, 3.73980e01],
            [1.25710e01, 8.66980e01, 3.00440e01],
            [3.09400e00, 9.30790e01, 6.03660e01],
            [5.77400e00, 1.03860e02, 4.01630e01],
            [1.70960e01, 9.74770e01, 5.15840e01],
            [1.45720e01, 1.03200e02, 6.34270e01],
            [2.07020e01, 1.01729e02, 6.67280e01],
            [9.60500e00, 9.36190e01, 7.00330e01],
            [2.47620e01, 7.29910e01, 4.50380e01],
            [9.03600e00, 8.09360e01, 3.48480e01],
            [5.00000e-03, 9.38070e01, 3.97600e01],
            [4.39500e00, 8.90320e01, 3.82840e01],
            [1.14030e01, 1.00312e02, 3.92750e01],
            [1.86520e01, 9.02370e01, 3.51820e01],
            [1.99960e01, 8.57930e01, 3.65880e01],
            [8.24200e00, 8.60170e01, 5.20920e01],
            [5.43500e00, 1.05824e02, 5.37510e01],
            [1.44180e01, 9.72540e01, 4.74530e01],
            [2.50970e01, 1.13595e02, 5.18200e01],
            [1.79550e01, 1.06014e02, 6.69910e01],
            [1.95970e01, 9.96520e01, 5.07750e01],
            [3.03660e01, 8.34110e01, 5.82150e01],
            [2.87640e01, 9.12670e01, 4.82120e01],
            [2.63990e01, 8.39690e01, 5.99980e01],
            [1.90280e01, 9.09220e01, 6.35440e01],
            [1.72860e01, 8.86570e01, 7.04010e01],
            [7.12100e00, 8.89970e01, 7.06170e01],
            [8.21100e00, 7.78960e01, 5.15780e01],
            [2.46210e01, 8.73660e01, 4.91020e01],
            [3.20860e01, 8.06230e01, 3.66420e01],
            [7.41100e00, 8.22890e01, 3.54450e01],
            [-2.53000e-01, 9.23460e01, 4.13810e01],
            [5.37900e00, 8.75000e01, 3.70590e01],
            [1.06620e01, 1.01987e02, 3.80660e01],
            [1.97990e01, 9.19120e01, 3.60070e01],
            [2.03290e01, 8.62490e01, 3.44630e01],
            [6.66900e00, 8.61080e01, 5.36150e01],
            [6.10700e00, 1.07107e02, 5.54030e01],
            [1.26860e01, 9.73600e01, 4.61060e01],
            [2.51710e01, 1.13492e02, 5.40130e01],
            [1.70600e01, 1.04244e02, 6.79330e01],
            [1.99590e01, 9.75000e01, 5.05250e01],
            [3.07600e01, 8.14850e01, 5.72390e01],
            [3.07140e01, 9.03740e01, 4.86900e01],
            [2.58450e01, 8.19220e01, 6.05740e01],
            [1.94380e01, 8.96860e01, 6.53080e01],
            [1.83800e01, 9.02540e01, 7.14330e01],
            [6.30500e00, 8.74020e01, 6.93420e01],
            [7.13200e00, 7.70040e01, 5.32690e01],
            [2.36800e01, 8.53870e01, 4.89620e01],
            [3.09930e01, 7.93490e01, 3.80640e01],
            [1.58600e00, 9.70140e01, 3.88130e01],
            [6.20600e00, 9.69940e01, 3.08260e01],
            [6.55100e00, 8.39770e01, 3.19110e01],
            [2.05300e01, 8.99400e01, 3.80640e01],
            [1.74480e01, 1.15868e02, 4.62290e01],
            [2.92900e01, 1.09602e02, 4.80050e01],
            [1.24330e01, 1.03662e02, 7.07970e01],
            [2.21510e01, 9.61260e01, 4.91850e01],
            [2.44550e01, 9.91140e01, 4.41590e01],
            [3.44570e01, 1.04502e02, 4.74770e01],
            [2.48460e01, 1.14000e02, 5.72050e01],
            [2.77310e01, 1.05822e02, 6.51290e01],
            [1.90110e01, 8.68850e01, 6.52190e01],
            [4.49600e00, 8.48850e01, 3.82510e01],
            [1.20850e01, 1.01124e02, 6.68980e01],
            [4.97400e00, 1.04582e02, 4.85950e01],
            [1.71910e01, 1.02904e02, 3.82380e01],
            [2.06710e01, 1.04095e02, 6.44010e01],
            [2.03670e01, 8.55330e01, 4.68310e01],
            [3.00350e01, 8.73280e01, 4.86200e01],
            [1.88810e01, 9.12420e01, 6.74910e01],
            [6.34600e00, 8.81260e01, 6.24560e01],
            [7.51000e00, 8.14790e01, 5.39010e01],
            [2.08090e01, 7.75330e01, 5.66710e01],
            [8.67000e00, 7.34130e01, 4.91670e01],
            [2.74160e01, 7.66170e01, 4.81380e01],
            [3.40930e01, 8.78960e01, 4.53060e01],
            [6.15200e00, 8.47160e01, 3.66770e01],
            [1.38470e01, 1.02163e02, 6.58660e01],
            [5.81800e00, 1.04189e02, 5.06870e01],
            [1.59590e01, 1.01894e02, 3.65920e01],
            [1.90750e01, 1.02461e02, 6.45870e01],
            [2.09390e01, 8.39690e01, 4.52590e01],
            [2.80940e01, 8.85510e01, 4.86470e01],
            [1.75960e01, 9.23060e01, 6.90580e01],
            [4.23800e00, 8.75430e01, 6.31330e01],
            [7.23200e00, 8.31000e01, 5.23020e01],
            [2.12700e01, 7.53210e01, 5.62670e01],
            [6.48500e00, 7.35580e01, 4.85030e01],
            [2.61910e01, 7.47670e01, 4.75810e01],
            [3.43370e01, 8.56260e01, 4.51470e01],
            [1.81800e01, 9.60300e01, 5.04480e01],
            [1.63620e01, 9.84730e01, 4.80930e01],
        ]
    )
    e = np.array(
        [
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
            "CG",
        ]
    )
    # dot = sa_surface(c, e, n=100, pr=20)
    # print('shape = ', dot.shape)
    # pocket_grids = gen_grid(c, n=1)
    # print('shape = ', pocket_grids.shape)
    # pocket_grids = sas_search_del(c, e, pocket_grids, pr=1.4)
    # print('shape = ', pocket_grids.shape)
    # pocket_grids = pas_search_for_pocket(pocket_grids, dot, n=n, pr=20)
    # print('shape = ', pocket_grids.shape)
    grid = layer_grids(c, e, 100, 20)
    print("shape = ", grid)
