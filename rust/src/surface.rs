use std::{
    collections::HashMap,
    f64::consts::PI,
    sync::{Mutex, RwLock},
};

use ndarray::{concatenate, Array, ArrayBase, ArrayView1, ArrayView2, Axis, Dim, OwnedRepr};
use once_cell::sync::OnceCell;
use rayon::prelude::*;

use crate::config::get_vdw_vec;

static DOTS: OnceCell<RwLock<HashMap<usize, Vec<f64>>>> = OnceCell::new();

/// vec 并行计算 求球的均等分点 效率最高
#[inline]
pub fn dotsphere(
    n: usize,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> {
    let dot = DOTS.get();
    if let Some(e) = dot {
        let e1 = e.read().unwrap();
        if e1.contains_key(&n) {
            let v = e1.get(&n).unwrap().to_owned();
            log::info!("read from cache dots");
            return Array::from_shape_vec((n, 3), v).unwrap();
        }
    }
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

    match dot {
        Some(e) => {
            let e = &mut e.write().unwrap();
            e.insert(n, data.clone());
        }
        None => {
            let mut v = HashMap::<usize, Vec<f64>>::new();
            v.insert(n, data.clone());
            DOTS.set(RwLock::new(v)).unwrap();
        }
    }

    log::info!("new dots...");

    Array::from_shape_vec((n, 3), data).unwrap()
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
    elements: &Vec<f64>,
    v: &mut Vec<Vec<usize>>,
    pr: f64,
) {
    let n = coors.nrows();
    let vv = Mutex::new(v);
    (0..n).into_par_iter().for_each(|i| {
        (i..n)
            .into_iter()
            .filter(|x| {
                !compare_two(
                    &coors.row(i),
                    &coors.row(*x),
                    elements[i] + pr,
                    elements[*x] + pr,
                )
            })
            .for_each(|x| {
                let vvv = &mut vv.lock().unwrap();
                &mut vvv[i].push(x);
                &mut vvv[x].push(i);
            });
    });
}

///
/// 去除球体重叠部分
/// * `coors` : 原子坐标集合
/// * `elements` : 原子名称列表
/// * `n` : 球均分点数
/// * `pr` : 补充半径
/// * `index`: 返回矩阵是否包含index
///
pub fn sa_surface(
    coors: &ArrayView2<'_, f64>,
    elements: Option<&Vec<&str>>,
    n: Option<usize>,
    pr: Option<f64>,
    index: bool,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    let count = if n.is_none() { 100 } else { n.unwrap() };

    let mut radis_v = vec![0.; coors.nrows()];
    get_vdw_vec(elements, &mut radis_v);

    sa_surface_core(coors, &radis_v, count, pr, index)
}

pub fn sa_surface_core(
    coors: &ArrayView2<'_, f64>,
    elements: &Vec<f64>,
    n: usize,
    pr: Option<f64>,
    index: bool,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    let ball = dotsphere(n);

    let y_ptr = if pr.is_none() { 1.4f64 } else { pr.unwrap() };

    let col = if index { 4 } else { 3 };

    let dd = Mutex::new(Vec::<f64>::new());

    let ele = elements
        .into_par_iter()
        .map(|f| (*f + y_ptr).powi(2) - 1e-6)
        .collect::<Vec<f64>>();

    (0..coors.nrows()).into_par_iter().for_each(|i| {
        let r = elements[i] + y_ptr;
        let ball2 = ball.mapv(|b| b * r) + coors.row(i);

        let filer = Mutex::new(Vec::<usize>::with_capacity(n / 4));

        (0..n).into_par_iter().for_each(|j| {
            let mut result = true;
            let b = ball2.row(j);
            for c in 0..coors.nrows() {
                let cc = coors.row(c);
                let r = ele[c];
                let r1 = (cc[0] - b[0]).powi(2) + (cc[1] - b[1]).powi(2) + (cc[2] - b[2]).powi(2);
                if r1 < r {
                    result = false;
                    break;
                }
            }

            if result {
                filer.lock().unwrap().push(j);
            }
        });

        let u = ball2.select(Axis(0), &filer.lock().unwrap());

        let data = if index {
            let four = Array::<f64, _>::zeros((u.nrows(), 1)).mapv(|_| i as f64);
            concatenate![Axis(1), u, four]
        } else {
            u
        };

        dd.lock().unwrap().extend(data.iter().map(|f| *f));
    });

    let ddd = dd.lock().unwrap().to_owned();

    Array::from_shape_vec((ddd.len() / col, col), ddd).unwrap()
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
        let d = sa_surface(&a.view(), Some(&b), Some(n), Some(1.4), true);
        info!("done ....{:?}", d.shape());

        let n = 200;

        info!("start ....");
        let d = sa_surface(&a.view(), Some(&b), Some(n), Some(1.4), true);
        info!("done ....{:?}", d.shape());

        assert_eq!(456, d.shape()[0]);
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
