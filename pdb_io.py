import pandas as pd
import numpy as np
from biopandas.pdb import PandasPdb

from core import mkdir_by_file


def to_pdb(coors, bfactor, filename='test.pdb'):
    '''convert (xyz + b_facter + resnnum) to pdb file'''

    mkdir_by_file(filename)
    data = np.insert(coors, 3, bfactor, axis=1)

    lines = []
    for index, _t in enumerate(data):
        line = "{:<6s}".format("ATOM") + "{:>5d}".format(index + 1) + " "\
            + "{:<5s}".format("H") + "{:>3s}".format("GRD") + " " + "A"\
            + "{:>4d}".format(1) + "    "\
            + "{:>8.3f}".format(_t[0]) + "{:>8.3f}".format(_t[1]) + "{:>8.3f}".format(_t[2])\
            + "{:>6.2f}".format(1) + "{:>6.2f}".format(_t[3]) + '\n'
        lines.append(line)
    with open(filename, 'w') as f:
        f.writelines(lines)
    print("<<{}>> is created".format(filename))


def read_pdb(pdbfile):
    '''read as pdb file and return its coors(xyz) and its elements
    '''
    _t = PandasPdb().read_pdb(pdbfile)
    _a = _t.df['ATOM']
    _b = _t.df['HETATM']
    _a = pd.concat([_a, _b])
    x = _a['x_coord']
    y = _a['y_coord']
    z = _a['z_coord']
    coors = pd.concat([x, y, z], axis=1).values
    eles = _a['atom_name'].values
    residue_name = _a['residue_name'].values
    return(coors, eles, residue_name)


def group_bfacter(data, sep):
    '''伪代码
    data:arr:x y z bfacter
    sep: 间隔
    '''
    _min = data[:, -1].min()
    _max = data[:, -1].max()
    ll = [i for i in range(int(_min)-1, int(_max)+sep, sep)]
    df = pd.DataFrame(data)
    df['grouped'] = pd.cut(df.iloc[:, -1], ll)
    return df


def to_xyz(data, ele='H', filename='test.xyz'):

    mkdir_by_file(filename)

    t = pd.DataFrame(data)
    t.insert(0, 'atom', ele)
    t.to_csv(filename, header=None, index=None, sep=' ')
    print('Done')
