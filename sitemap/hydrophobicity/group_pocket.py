""" 将pocket 分组
分组依据为两个点相邻（delta X + delta Y + delta Z )= 1
若A与B相邻，B和C相邻，则A与C也为同一组
实际为 并查集 问题
进一步优化： 现有程序为 广度优先搜索，时间复杂度为n^2.
可以用 并查集 进一步优化"""
import numpy as np

np.set_printoptions(precision=4)


def gen_isadjacent(grids):
    """
    判断格点与格点是否相邻，若相邻则为1.
    返回每个格点与其他格点的相邻关系，correlation matrix
    """
    import numpy as np

    n_grids = grids.shape[0]
    isadjacent = np.zeros((n_grids, n_grids))
    for i in range(n_grids):
        for j in range(n_grids):
            if np.sum(np.abs(grids[i] - grids[j])) == 1:
                isadjacent[i, j] = 1
    return isadjacent


def group_pocket(isadjacent):
    from collections import deque

    grids_number = isadjacent.shape[0]
    visited = set()
    res = []
    for i in range(grids_number):
        if i not in visited:
            q = deque([i])
            tmp = set()
            while q:
                j = q.popleft()
                visited.add(j)
                tmp.add(j)
                for k in range(grids_number):
                    if isadjacent[j, k] == 1 and k not in visited:
                        q.append(k)
                        tmp.add(k)
            res.append(tmp)
    return res


# def to_xyz(data, filename='test.xyz'):
#     import pandas as pd
#     t = pd.DataFrame(data)
#     t.insert(0, 'atom', 'H')
#     t.to_csv(filename, header=None, index=None, sep=' ')
#     print('Done')

# def test(grids):
#     res = group_pocket(gen_isadjacent(grids))
#     for i in range(len(res)):
#         to_xyz(grids[list(res[i]), :], 'pocket_number%s.xyz' % (i+1))

# if __name__ ==  '__main__':
#    test()
