use std::{
    cmp::min,
    collections::HashSet,
    sync::{Arc, Mutex},
};

use ndarray::{s, Array, ArrayView2, Axis};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    config::{get_all_vdw, get_vdw_radii},
    surface::sa_surface,
};

///
/// 网格生成
///
fn gen_grid(
    coors: &ArrayView2<'_, f64>,
    n: usize,
    buf: f64,
    xyz: &mut [f64],
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> {
    fn tuple3<T>(a: &[T]) -> (&T, &T, &T) {
        (&a[0], &a[1], &a[2])
    }
    // 取出每列的最大最小值, 生成一个单元格(n)队列
    let xyz = (0..3)
        .into_iter()
        .map(|i| {
            let mut x = coors.column(i).to_vec();

            x.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let min = (x[0] - buf).trunc();
            let max = (x[x.len() - 1] + buf).trunc() + 1.;

            let r = Array::<f64, _>::range(min, max, n as f64).to_vec();
            xyz[3 * i] = min;
            xyz[3 * i + 1] = max;
            xyz[3 * i + 2] = r.len() as f64;

            r
        })
        .collect::<Vec<_>>();

    let (x, y, z) = tuple3(&xyz);
    let mut v: Vec<f64> = vec![0.; x.len() * y.len() * z.len() * 3];

    // 构建立方体网格
    v.par_iter_mut().enumerate().for_each(|(index, value)| {
        if index % 3 == 0 {
            let i = (index / 3) / (y.len() * z.len());
            *value = x[i % x.len()];
        } else if index % 3 == 1 {
            let i = ((index - 1) / 3) / z.len();
            *value = y[i % y.len()];
        } else {
            *value = z[((index - 2) / 3) % z.len()];
        }
    });
    Array::from_shape_vec((v.len() / 3, 3), v).unwrap()
}

///
/// 从网格grid中去除原子集合内的点
/// * `coors`: 原子集合
/// * `elements`: 原子半径集合
/// * `grid`: 网格, 包含所有原子集合的立方体网格
/// * `pr`: 半径补充
///  
/// 返回过滤后的`grid`
fn select_point(
    coors: &ArrayView2<'_, f64>,
    elements: Option<&Vec<&str>>,
    grid: &ArrayView2<'_, f64>,
    pr: Option<f64>,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> {
    let y_ptr = if pr.is_none() { 1.4f64 } else { pr.unwrap() };

    let filer = Arc::new(Mutex::new(Vec::<usize>::new()));

    let mm = get_all_vdw();

    let get_vdw_radii = move |elements: Option<&Vec<&str>>, pr: f64, i: usize| {
        if let Some(e) = elements {
            mm.get(e[i]).unwrap() + pr
        } else {
            pr
        }
    };

    (0..grid.nrows()).into_par_iter().for_each(|i| {
        let mut flags = true;
        let b1 = grid.row(i);

        for j in 0..coors.nrows() {
            let r = get_vdw_radii(elements, y_ptr, j).powi(2);
            let a1 = coors.row(j);
            let r1 = (b1[0] - a1[0]).powi(2) + (b1[1] - a1[1]).powi(2) + (b1[2] - a1[2]).powi(2);
            if r1 < r {
                flags = false;
                break;
            }
        }

        if flags {
            &mut filer.lock().unwrap().push(i);
        }
    });

    let d = grid.select(Axis(0), &filer.lock().unwrap());
    d
}

fn select_point2(
    coors: &ArrayView2<'_, f64>,
    elements: Option<&Vec<&str>>,
    grid: &ArrayView2<'_, f64>,
    pr: Option<f64>,
    xyz: &[f64],
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> {
    let y_ptr = if pr.is_none() { 1.4f64 } else { pr.unwrap() };

    let tmp = grid.to_owned();

    let row = grid.nrows();

    let filer = Mutex::new(HashSet::<usize>::with_capacity(row / 4));

    (0..coors.nrows()).into_par_iter().for_each(|i| {
        let b1 = coors.row(i);
        let r = get_vdw_radii(elements, y_ptr, i);

        let (x1, y1, z1) = (b1[0] - r - xyz[0], b1[1] - r - xyz[3], b1[2] - r - xyz[6]);

        let (c1, c2) = (xyz[5] * xyz[8], xyz[8]);

        let start = (x1 * c1 + y1 * c2 + z1) as usize;
        let (x2, y2, z2) = (b1[0] + r - xyz[0], b1[1] + r - xyz[3], b1[2] + r - xyz[6]);

        let end = min(row, (x2 * c1 + y2 * c2 + z2) as usize);
        let tmp_s = tmp.slice(s![start..end, ..]);

        let t = tmp_s.to_owned() - b1;

        t.mapv(|f| f * f)
            .sum_axis(Axis(1))
            .into_iter()
            .enumerate()
            .for_each(|(i, f)| {
                if *f < r * r {
                    filer.lock().unwrap().insert(start + i);
                }
            });
    });

    let d = filer.lock().unwrap().to_owned();

    let d = (0..row)
        .into_par_iter()
        .filter_map(|f| if d.contains(&f) { None } else { Some(f) })
        .collect::<Vec<usize>>();

    tmp.select(Axis(0), &d)
}

pub fn find_pockets(
    coors: &ArrayView2<'_, f64>,
    elements: Option<&Vec<&str>>,
    n: usize,
    pr: Option<f64>,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> {
    let mut xyz = [0.; 9];
    // 辅助sa surface
    let dot = sa_surface(&coors.view(), elements, Some(n), pr, false);

    log::info!("shape: {:?}", dot.shape());

    // 生成网格
    let grid = gen_grid(&coors.view(), 1, 0., &mut xyz);

    log::info!("shape: {:?} xyz = {:?}", grid.shape(), xyz);

    //去除原子集合内的格点
    let grid1 = select_point2(&coors.view(), elements, &grid.view(), Some(1.4), &xyz);
    log::info!("1111 shape: {:?}", grid1.shape());

    // 获得最后的pokcets
    let grid1 = select_point(&dot.view(), None, &grid1.view(), pr);
    log::info!("2222 shape: {:?}", grid1.shape());
    grid1
}

#[cfg(test)]
mod tests {
    use log::info;
    use ndarray::array;

    use crate::pockets::find_pockets;

    #[test]
    fn test_grid() {
        crate::config::init_config();

        let a = array![
            [5.80400e+00, 7.71280e+01, 3.75770e+01],
            [1.15920e+01, 8.69370e+01, 3.19960e+01],
            [3.04400e+00, 9.48200e+01, 5.90210e+01],
            [7.69800e+00, 1.04841e+02, 4.05320e+01],
            [1.57460e+01, 9.82030e+01, 5.00190e+01],
            [1.27360e+01, 1.04393e+02, 6.32570e+01],
            [1.98680e+01, 9.97520e+01, 6.71680e+01],
            [1.08790e+01, 9.28970e+01, 6.84030e+01],
            [2.43770e+01, 7.28520e+01, 4.72010e+01],
            [7.02600e+00, 7.53090e+01, 3.73980e+01],
            [1.25710e+01, 8.66980e+01, 3.00440e+01],
            [3.09400e+00, 9.30790e+01, 6.03660e+01],
            [5.77400e+00, 1.03860e+02, 4.01630e+01],
            [1.70960e+01, 9.74770e+01, 5.15840e+01],
            [1.45720e+01, 1.03200e+02, 6.34270e+01],
            [2.07020e+01, 1.01729e+02, 6.67280e+01],
            [9.60500e+00, 9.36190e+01, 7.00330e+01],
            [2.47620e+01, 7.29910e+01, 4.50380e+01],
            [9.03600e+00, 8.09360e+01, 3.48480e+01],
            [5.00000e-03, 9.38070e+01, 3.97600e+01],
            [4.39500e+00, 8.90320e+01, 3.82840e+01],
            [1.14030e+01, 1.00312e+02, 3.92750e+01],
            [1.86520e+01, 9.02370e+01, 3.51820e+01],
            [1.99960e+01, 8.57930e+01, 3.65880e+01],
            [8.24200e+00, 8.60170e+01, 5.20920e+01],
            [5.43500e+00, 1.05824e+02, 5.37510e+01],
            [1.44180e+01, 9.72540e+01, 4.74530e+01],
            [2.50970e+01, 1.13595e+02, 5.18200e+01],
            [1.79550e+01, 1.06014e+02, 6.69910e+01],
            [1.95970e+01, 9.96520e+01, 5.07750e+01],
            [3.03660e+01, 8.34110e+01, 5.82150e+01],
            [2.87640e+01, 9.12670e+01, 4.82120e+01],
            [2.63990e+01, 8.39690e+01, 5.99980e+01],
            [1.90280e+01, 9.09220e+01, 6.35440e+01],
            [1.72860e+01, 8.86570e+01, 7.04010e+01],
            [7.12100e+00, 8.89970e+01, 7.06170e+01],
            [8.21100e+00, 7.78960e+01, 5.15780e+01],
            [2.46210e+01, 8.73660e+01, 4.91020e+01],
            [3.20860e+01, 8.06230e+01, 3.66420e+01],
            [7.41100e+00, 8.22890e+01, 3.54450e+01],
            [-2.53000e-01, 9.23460e+01, 4.13810e+01],
            [5.37900e+00, 8.75000e+01, 3.70590e+01],
            [1.06620e+01, 1.01987e+02, 3.80660e+01],
            [1.97990e+01, 9.19120e+01, 3.60070e+01],
            [2.03290e+01, 8.62490e+01, 3.44630e+01],
            [6.66900e+00, 8.61080e+01, 5.36150e+01],
            [6.10700e+00, 1.07107e+02, 5.54030e+01],
            [1.26860e+01, 9.73600e+01, 4.61060e+01],
            [2.51710e+01, 1.13492e+02, 5.40130e+01],
            [1.70600e+01, 1.04244e+02, 6.79330e+01],
            [1.99590e+01, 9.75000e+01, 5.05250e+01],
            [3.07600e+01, 8.14850e+01, 5.72390e+01],
            [3.07140e+01, 9.03740e+01, 4.86900e+01],
            [2.58450e+01, 8.19220e+01, 6.05740e+01],
            [1.94380e+01, 8.96860e+01, 6.53080e+01],
            [1.83800e+01, 9.02540e+01, 7.14330e+01],
            [6.30500e+00, 8.74020e+01, 6.93420e+01],
            [7.13200e+00, 7.70040e+01, 5.32690e+01],
            [2.36800e+01, 8.53870e+01, 4.89620e+01],
            [3.09930e+01, 7.93490e+01, 3.80640e+01],
            [1.58600e+00, 9.70140e+01, 3.88130e+01],
            [6.20600e+00, 9.69940e+01, 3.08260e+01],
            [6.55100e+00, 8.39770e+01, 3.19110e+01],
            [2.05300e+01, 8.99400e+01, 3.80640e+01],
            [1.74480e+01, 1.15868e+02, 4.62290e+01],
            [2.92900e+01, 1.09602e+02, 4.80050e+01],
            [1.24330e+01, 1.03662e+02, 7.07970e+01],
            [2.21510e+01, 9.61260e+01, 4.91850e+01],
            [2.44550e+01, 9.91140e+01, 4.41590e+01],
            [3.44570e+01, 1.04502e+02, 4.74770e+01],
            [2.48460e+01, 1.14000e+02, 5.72050e+01],
            [2.77310e+01, 1.05822e+02, 6.51290e+01],
            [1.90110e+01, 8.68850e+01, 6.52190e+01],
            [4.49600e+00, 8.48850e+01, 3.82510e+01],
            [1.20850e+01, 1.01124e+02, 6.68980e+01],
            [4.97400e+00, 1.04582e+02, 4.85950e+01],
            [1.71910e+01, 1.02904e+02, 3.82380e+01],
            [2.06710e+01, 1.04095e+02, 6.44010e+01],
            [2.03670e+01, 8.55330e+01, 4.68310e+01],
            [3.00350e+01, 8.73280e+01, 4.86200e+01],
            [1.88810e+01, 9.12420e+01, 6.74910e+01],
            [6.34600e+00, 8.81260e+01, 6.24560e+01],
            [7.51000e+00, 8.14790e+01, 5.39010e+01],
            [2.08090e+01, 7.75330e+01, 5.66710e+01],
            [8.67000e+00, 7.34130e+01, 4.91670e+01],
            [2.74160e+01, 7.66170e+01, 4.81380e+01],
            [3.40930e+01, 8.78960e+01, 4.53060e+01],
            [6.15200e+00, 8.47160e+01, 3.66770e+01],
            [1.38470e+01, 1.02163e+02, 6.58660e+01],
            [5.81800e+00, 1.04189e+02, 5.06870e+01],
            [1.59590e+01, 1.01894e+02, 3.65920e+01],
            [1.90750e+01, 1.02461e+02, 6.45870e+01],
            [2.09390e+01, 8.39690e+01, 4.52590e+01],
            [2.80940e+01, 8.85510e+01, 4.86470e+01],
            [1.75960e+01, 9.23060e+01, 6.90580e+01],
            [4.23800e+00, 8.75430e+01, 6.31330e+01],
            [7.23200e+00, 8.31000e+01, 5.23020e+01],
            [2.12700e+01, 7.53210e+01, 5.62670e+01],
            [6.48500e+00, 7.35580e+01, 4.85030e+01],
            [2.61910e+01, 7.47670e+01, 4.75810e+01],
            [3.43370e+01, 8.56260e+01, 4.51470e+01],
            [1.81800e+01, 9.60300e+01, 5.04480e+01],
            [1.63620e+01, 9.84730e+01, 4.80930e+01]
        ];

        let b = vec![
            "N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "N", "CA", "C", "O", "CB", "CG",
            "OD1", "OD2", "N", "CA", "C", "O", "CB", "N", "CA", "C", "O", "CB", "CG", "CD", "OE1",
            "OE2", "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "N", "CA", "C", "O", "CB", "CG",
            "CD1", "CD2", "N", "CA", "C", "O", "CB", "CG1", "CG2", "N", "CA", "C", "O", "CB",
            "CG2", "OG1", "N", "CA", "C", "O", "CB", "CG1", "CG2", "N", "CA", "C", "O", "CB", "CG",
            "CD", "NE", "CZ", "NH1", "NH2", "N", "CA", "C", "O", "N", "CA", "C", "O", "N", "CA",
            "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "N", "CA", "C", "O", "CB", "CG",
        ];

        let n = 100;

        info!("start find pockets");

        let grid = find_pockets(&a.view(), Some(&b), n, Some(20.));

        assert_eq!(23831, grid.shape()[0]);
    }

    #[test]
    fn feature() {
        println!(
            "{}, {}, {}",
            (-0.941 as f64).trunc(),
            (29.682 as f64).trunc(),
            (29.682 as f64).trunc()
        );
    }
}
