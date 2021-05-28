use std::sync::{Arc, Mutex};

use ndarray::{Array, ArrayView2, Axis};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::surface::{get_vdw_radii, sa_surface};

///
/// 网格生成
///
fn gen_grid(
    coors: &ArrayView2<'_, f64>,
    n: usize,
    buf: f64,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> {
    fn tuple3<T>(a: &[T]) -> (&T, &T, &T) {
        (&a[0], &a[1], &a[2])
    }
    // 取出每列的最大最小值, 生成一个单元格(n)队列
    let xyz = (0..3)
        .into_par_iter()
        .map(|i| {
            let mut x = coors.column(i).to_vec();

            x.sort_by(|a, b| a.partial_cmp(b).unwrap());

            Array::<f64, _>::range(
                (x[0] - buf).trunc(),
                (x[x.len() - 1] + buf).trunc() + 1.,
                n as f64,
            )
            .to_vec()
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

    (0..grid.nrows()).into_par_iter().for_each(|i| {
        let mut flags = true;
        (0..coors.nrows()).into_iter().for_each(|j| {
            let r = get_vdw_radii(elements, y_ptr, j).powi(2);
            let b1 = grid.row(i);
            let a1 = coors.row(j);
            let r1 = (b1[0] - a1[0]).powi(2) + (b1[1] - a1[1]).powi(2) + (b1[2] - a1[2]).powi(2);
            if r1 < r {
                flags = false;
            }
        });

        if flags {
            &mut filer.lock().unwrap().push(i);
        }
    });

    let d = grid.select(Axis(0), &filer.lock().unwrap());
    d
}

pub fn find_pockets(
    coors: &ArrayView2<'_, f64>,
    elements: Option<&Vec<&str>>,
    n: usize,
    pr: Option<f64>,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> {
    // 辅助sa surface
    let dot = sa_surface(&coors.view(), elements, Some(n), pr);

    println!("shape: {:?}", dot.shape());

    // 生成网格
    let grid = gen_grid(&coors.view(), 1, 0.);
    println!("shape: {:?}", grid.shape());

    //去除原子集合内的格点
    let grid1 = select_point(&coors.view(), elements, &grid.view(), Some(1.4));
    println!("shape: {:?}", grid1.shape());

    // 获得最后的pokcets
    select_point(&dot.view(), None, &grid1.view(), pr)
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
            [4.39500e+00, 8.90320e+01, 3.82840e+01]
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

        let grid = find_pockets(&a.view(), Some(&b), n, Some(20.));

        info!("gen_grid3: {:?}", grid.shape());

        assert_eq!(6013, grid.shape()[0]);
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
