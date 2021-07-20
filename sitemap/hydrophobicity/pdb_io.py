import os

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb

from sitemap.core import mkdir_by_file


def to_pdb(coors, bfactor, filename="test.pdb"):
    """convert (xyz + b_facter + resnnum) to pdb file"""

    mkdir_by_file(filename)
    data = np.insert(coors, 3, bfactor, axis=1)

    lines = []
    for index, _t in enumerate(data):
        line = (
            "{:<6s}".format("ATOM")
            + "{:>5d}".format(index + 1)
            + " "
            + "{:<5s}".format("H")
            + "{:>3s}".format("GRD")
            + " "
            + "A"
            + "{:>4d}".format(1)
            + "    "
            + "{:>8.3f}".format(_t[0])
            + "{:>8.3f}".format(_t[1])
            + "{:>8.3f}".format(_t[2])
            + "{:>6.2f}".format(1)
            + "{:>6.2f}".format(_t[3])
            + "\n"
        )
        lines.append(line)
    with open(filename, "w") as f:
        f.writelines(lines)
    print("<<{}>> is created".format(filename))


def read_pdb(pdbfile):
    """read as pdb file and return its coors(xyz) and its elements
    """
    _t = PandasPdb().read_pdb(pdbfile)
    _a = _t.df["ATOM"]
    _b = _t.df["HETATM"]
    _a = pd.concat([_a, _b])
    x = _a["x_coord"]
    y = _a["y_coord"]
    z = _a["z_coord"]
    coors = pd.concat([x, y, z], axis=1).values
    eles = _a["atom_name"].values
    residue_name = _a["residue_name"].values
    return (coors, eles, residue_name)


def group_bfacter(data, sep):
    """伪代码
    data:arr:x y z bfacter
    sep: 间隔
    """
    _min = data[:, -1].min()
    _max = data[:, -1].max()
    ll = [i for i in range(int(_min) - 1, int(_max) + sep, sep)]
    df = pd.DataFrame(data)
    df["grouped"] = pd.cut(df.iloc[:, -1], ll)
    return df


def to_xyz(data, ele="H", filename="test.xyz"):

    mkdir_by_file(filename)

    t = pd.DataFrame(data)
    n_atom = int(t.shape[0])
    t.insert(0, "atom", ele)
    df1 = pd.DataFrame([[n_atom, "", "", ""], [filename, "", "", ""]], columns=["atom", 0, 1, 2],)
    res = pd.concat([df1, t])
    res.to_csv(filename, header=None, index=None, sep="\t")


def xyz2trj(datalist, ele="H", filename="test.xyz"):
    res = []
    mkdir_by_file(filename)
    for data in datalist:
        t = pd.DataFrame(data)
        n_atom = int(t.shape[0])
        t.insert(0, "atom", ele)
        df1 = pd.DataFrame([[n_atom, "", "", ""], [filename, "", "", ""]], columns=["atom", 0, 1, 2],)
        df2 = pd.concat([df1, t])
        res.append(df2)
    res2 = pd.concat(res)
    res2.to_csv(filename, header=None, index=None, sep="\t")


def read_xyz(filename, dir="."):
    """
    xyz format
    3 #numer of atoms
      # blank line
    H 0 0 0
    O 0 0 2
    H 0 0 4

    return coor,elements #in numpy array
    """
    file_path = os.path.join(dir, filename)
    df = pd.read_csv(file_path, header=None, sep="\\s+", skiprows=2)
    coors = df.iloc[:, 1:].values
    eles = df.iloc[:, 0].values.astype("str")
    eles = np.char.upper(eles)  # convert to uppercase
    return (coors, eles)
