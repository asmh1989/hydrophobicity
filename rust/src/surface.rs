use std::{
    f64::consts::PI,
    sync::{Arc, Mutex},
};

use ndarray::{concatenate, Array, ArrayBase, ArrayView2, Axis, Dim, OwnedRepr};
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

/// 行遍历 求球的均等分点
#[inline]
pub fn dotsphere2(
    n: usize,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> {
    let golden_ratio: f64 = (1f64 + 5f64.powf(0.5)) / 2f64;

    let mut a = Array::<f64, _>::zeros((n, 3));

    let mut i = 0;
    a.map_axis_mut(Axis(1), |mut a| {
        let theta = 2f64 * PI * (i as f64) / golden_ratio;
        let phi: f64 = (1f64 - 2f64 * ((i as f64) + 0.5f64) / (n as f64)).acos();
        a[0] = theta.cos() * phi.sin();
        a[1] = theta.sin() * phi.sin();
        a[2] = phi.cos();
        i += 1;
    });

    a
}

fn get_vdw_radii(elements: Option<&Vec<&str>>, pr: f64, i: usize) -> f64 {
    match elements {
        Some(e) => config::get_vdw_radii(e[i]) + pr,
        None => pr,
    }
}

/// 并行化 去除球体重叠部分, 效率优
pub fn sa_surface(
    coors: &ArrayView2<'_, f64>,
    elements: Option<&Vec<&str>>,
    n: Option<usize>,
    pr: Option<f64>,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    let c = if n.is_none() { 100 } else { n.unwrap() };
    let y_ptr = if pr.is_none() { 1.4f64 } else { pr.unwrap() };
    let dd = Arc::new(Mutex::new(Vec::<f64>::new()));
    let y = dotsphere(c);

    (0..coors.nrows()).into_par_iter().for_each(|i| {
        let r = get_vdw_radii(elements, y_ptr, i);
        let b2 = y.mapv(|b| b * r) + coors.row(i);

        let filer = Arc::new(Mutex::new(Vec::<usize>::new()));

        (0..b2.nrows()).into_par_iter().for_each(|i2| {
            let mut result = false;
            for j in 0..coors.nrows() {
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
        dd.clone().lock().unwrap().extend(v.iter());
    });

    let ddd = dd.lock().unwrap();

    Array::from_shape_vec((ddd.len() / 4, 4), ddd.to_owned()).unwrap()
}

#[cfg(test)]
mod tests {
    use log::info;
    use ndarray::array;

    use crate::surface::{dotsphere, dotsphere2, sa_surface};

    #[test]
    fn test_surface() {
        crate::config::init_config();

        let a = array![[0., 0., 0.], [0., 0., 1.7]];

        let b = vec!["C", "O"];

        info!("start ....");
        let d = sa_surface(&a.view(), Some(&b), Some(100), Some(1.4));
        info!("done ....{:?}", d);
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

        info!("start dotsphere2");
        dotsphere2(n);
        info!("dotsphere2 done");

        println!("{:?} ", a);

        assert!(true);
    }
}
