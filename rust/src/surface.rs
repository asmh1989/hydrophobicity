use std::{
    f64::consts::PI,
    sync::{Arc, Mutex},
};

use ndarray::{concatenate, Array, ArrayBase, ArrayView1, ArrayView2, Axis, Dim, OwnedRepr};
use rayon::prelude::*;

use crate::config;

/// vec 并行计算 求球的均等分点 效率最高
#[inline]
pub fn dotsphere(
    n: usize,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> {
    let golden_ratio: f64 = (1f64 + 5f64.powf(0.5)) / 2f64;

    let mut data = vec![0.; n * 3];
    data.par_iter_mut().enumerate().for_each(|(i, x)| {
        let index = i % 3;
        let j = (i - index) / 3;
        let phi: f64 = (1f64 - 2f64 * ((j as f64) + 0.5f64) / (n as f64)).acos();

        if 2 == index {
            *x = phi.cos();
        } else {
            let theta = 2f64 * PI * (j as f64) / golden_ratio;
            if 0 == index {
                *x = theta.cos() * phi.sin();
            } else {
                *x = theta.sin() * phi.sin();
            }
        }
    });

    Array::from_shape_vec((n, 3), data).unwrap()
}

#[inline]
pub fn get_vdw_radii(elements: Option<&Vec<&str>>, pr: f64, i: usize) -> f64 {
    match elements {
        Some(e) => config::get_vdw_radii(e[i]) + pr,
        None => pr,
    }
}

// 比较两个球心是不是太远, 一样的点 也认为是太远
pub fn compare_two(a1: &ArrayView1<f64>, a2: &ArrayView1<f64>, r1: f64, r2: f64) -> bool {
    let a = a1 - a2;
    let r = a.mapv(|i| i * i).sum();

    r < 1e-6 || r > (r1 + r2) * (r1 + r2)
}

// 球体相交的原子关系列表
fn find_cross_ball(
    coors: &ArrayView2<'_, f64>,
    elements: Option<&Vec<&str>>,
    v: &mut Vec<Vec<usize>>,
    y_ptr: f64,
) {
    let n = coors.nrows();
    let vv = Arc::new(Mutex::new(v));
    (0..n).into_par_iter().for_each(|i| {
        (i..n)
            .into_iter()
            .filter(|x| {
                !compare_two(
                    &coors.row(i),
                    &coors.row(*x),
                    get_vdw_radii(elements, y_ptr, i),
                    get_vdw_radii(elements, y_ptr, *x),
                )
            })
            .for_each(|x| {
                let vvv = &mut vv.lock().unwrap();
                &mut vvv[i].push(x);
                &mut vvv[x].push(i);
            });
    });
}

/// 并行化 去除球体重叠部分, 效率优
pub fn sa_surface(
    coors: &ArrayView2<'_, f64>,
    elements: Option<&Vec<&str>>,
    n: Option<usize>,
    pr: Option<f64>,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    let count = if n.is_none() { 100 } else { n.unwrap() };
    let y_ptr = if pr.is_none() { 1.4f64 } else { pr.unwrap() };
    let y = dotsphere(count);

    if count == 1 {
        // 一个原子直接返回所有点
        return concatenate![
            Axis(1),
            y.mapv(|b| b * get_vdw_radii(elements, y_ptr, 0)) + coors.row(0),
            Array::<f64, _>::zeros((count, 1))
        ];
    }

    let mut v = vec![Vec::<usize>::new(); coors.nrows()];
    find_cross_ball(coors, elements, &mut v, y_ptr);

    let dd = Arc::new(Mutex::new(Vec::<f64>::new()));

    (0..coors.nrows()).into_par_iter().for_each(|i| {
        let r = get_vdw_radii(elements, y_ptr, i);
        let b2 = y.mapv(|b| b * r) + coors.row(i);

        let filer = Arc::new(Mutex::new(Vec::<usize>::new()));

        (0..b2.nrows()).into_par_iter().for_each(|i2| {
            let mut result = false;
            for &j in &v[i] {
                let r = get_vdw_radii(elements, y_ptr, j).powi(2);
                let b1 = coors.row(j);
                let a1 = b2.row(i2);
                let r1 =
                    (b1[0] - a1[0]).powi(2) + (b1[1] - a1[1]).powi(2) + (b1[2] - a1[2]).powi(2);
                if r1 < r && r - r1 > 1e-6 {
                    result = true;
                    break;
                }
            }

            if result == false {
                filer.lock().unwrap().push(i2);
            }
        });

        let u = b2.select(Axis(0), &filer.lock().unwrap());
        let four = Array::<f64, _>::zeros((u.nrows(), 1)).mapv(|_| i as f64);

        let mut v = vec![0.; u.nrows() * 4];
        let data = concatenate![Axis(1), u, four];

        v.par_iter_mut().enumerate().for_each(|(i, value)| {
            let rows = (i - i % 4) / 4;
            *value = data[[rows, i % 4]];
        });
        dd.lock().unwrap().extend(v.iter());
    });

    let ddd = dd.lock().unwrap().to_owned();

    Array::from_shape_vec((ddd.len() / 4, 4), ddd).unwrap()
}

#[cfg(test)]
mod tests {
    use log::info;
    use ndarray::array;

    use crate::surface::{dotsphere, sa_surface};

    #[test]
    fn test_surface() {
        crate::config::init_config();

        let a = array![[0., 0., 0.], [0., 0., 1.7], [0., 0., 10.7]];

        let b = vec!["C", "O", "CD1"];
        let n = 10000;

        info!("start ....");
        let d = sa_surface(&a.view(), Some(&b), Some(n), Some(1.4));
        info!("done ....{:?}", d.shape());

        assert_eq!(22795, d.shape()[0]);
    }

    #[test]
    fn test_ndarray() {
        crate::config::init_config();

        // let a = Array::<f64, _>::zeros((3, 2).f());
        // let b = Array::from_iter(0..10);

        // let a = array![[1., 2., 3.], [4., 5., 6.]];

        // let b = array!["C", "O"];

        let n = 1000000;
        info!("start dotsphere");
        let a = dotsphere(n);
        info!("dotsphere done");
        println!("{:?} ", a);

        assert!(true);
    }
}
