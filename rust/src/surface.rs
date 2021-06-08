use std::{
    cmp::min,
    collections::HashMap,
    f64::consts::PI,
    sync::{Mutex, RwLock},
};

use ndarray::{concatenate, Array, ArrayBase, ArrayView2, Axis, Dim, OwnedRepr, Slice};
use once_cell::sync::OnceCell;
use rayon::prelude::*;

use crate::{config::get_vdw_vec, utils::distance};

/// 缓存单位球的均等分点
static DOTS: OnceCell<RwLock<HashMap<usize, Vec<f64>>>> = OnceCell::new();

/// vec 并行计算 求球的均等分点 效率最高
#[inline]
pub fn dotsphere(
    n: usize,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> {
    let dot = DOTS.get();

    // 判断是否有缓存
    if let Some(e) = dot {
        let e1 = e.read().unwrap();
        if e1.contains_key(&n) {
            let v = e1.get(&n).unwrap().to_owned();
            // log::info!("read from cache dots");
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

///
/// 求蛋白质sa平面点集合
///
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

    // 缓存半径计算
    let ele = elements
        .into_par_iter()
        .map(|f| (*f + y_ptr).powi(2) - 1e-6)
        .collect::<Vec<f64>>();

    // 遍历原子集合
    (0..coors.nrows()).into_par_iter().for_each(|i| {
        let r = elements[i] + y_ptr;

        // 生成该原子上的均等分点
        let ball2 = ball.mapv(|b| b * r) + coors.row(i);

        let filer = Mutex::new(Vec::<usize>::with_capacity(n / 4));

        // 遍历这些点, 开始刷选在其余原子半径内的点
        (0..n).into_par_iter().for_each(|j| {
            let mut result = true;
            let b = ball2.row(j);
            for c in 0..coors.nrows() {
                let cc = coors.row(c);
                let r = ele[c];
                let r1 = distance(&cc, &b);
                if r1 < r {
                    result = false;
                    break;
                }
            }
            // 不在半径内即为重叠部分, 选中
            if result {
                filer.lock().unwrap().push(j);
            }
        });

        let u = ball2.select(Axis(0), &filer.lock().unwrap());

        let data = if index {
            let four = Array::from_shape_fn((u.nrows(), 1), |(_, _)| i as f64);
            concatenate![Axis(1), u, four]
        } else {
            u
        };

        dd.lock().unwrap().extend(data.iter().map(|f| *f));
    });

    let ddd = dd.lock().unwrap().to_owned();

    Array::from_shape_vec((ddd.len() / col, col), ddd).unwrap()
}

pub fn sa_surface_from_prev(
    coors: &ArrayView2<'_, f64>,
    prev_dots: &ArrayView2<'_, f64>,
    elements: &Vec<f64>,
    n: usize,
    prev_ptr: f64,
    y_ptr: f64,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    let dd = Mutex::new(Vec::<f64>::new());

    // 缓存半径计算
    let ele = elements
        .into_par_iter()
        .map(|f| (*f + y_ptr).powi(2) - 1e-6)
        .collect::<Vec<f64>>();

    let mut cache_ball = vec![Vec::<usize>::with_capacity(n / 8); coors.nrows()];

    for row in prev_dots.axis_iter(Axis(0)).enumerate() {
        let index = row.1[3] as usize;
        let b = &mut cache_ball[index];
        b.push(row.0);
    }

    // 遍历原子集合
    (0..coors.nrows()).into_par_iter().for_each(|i| {
        let c_row = coors.row(i);

        // 还原出再单位球上的点坐标
        let ball = prev_dots.select(Axis(0), &cache_ball[i]);
        let ball = ball.slice_axis(Axis(1), Slice::from(0..3));
        let ball = &ball - &c_row;
        let prev_r = elements[i] + prev_ptr;
        let ball_ = ball.mapv(|f| f / prev_r);

        // 对应生成该原子上新的ptr对应的均等分点
        let r = elements[i] + y_ptr;
        let ball2 = ball_.mapv(|b| b * r) + c_row;

        let len = ball2.nrows();

        let filer = Mutex::new(Vec::<usize>::with_capacity(min(len / 2, 1)));

        // 遍历这些点, 开始刷选在其余原子半径内的点
        (0..len).into_par_iter().for_each(|j| {
            let mut result = true;
            let b = ball2.row(j);
            for c in 0..coors.nrows() {
                let cc = coors.row(c);
                let r = ele[c];
                let r1 = distance(&cc, &b);
                if r1 < r {
                    result = false;
                    break;
                }
            }
            // 不在半径内即为重叠部分, 选中
            if result {
                filer.lock().unwrap().push(j);
            }
        });

        let u = ball2.select(Axis(0), &filer.lock().unwrap());

        dd.lock().unwrap().extend(u.iter().map(|f| *f));
    });

    let ddd = dd.lock().unwrap().to_owned();

    Array::from_shape_vec((ddd.len() / 3, 3), ddd).unwrap()
}

///
/// 求蛋白质sa平面点集合, 返回字典数据, 对应每个原子上的返回点个数百分比
///
pub fn sa_surface_return_map(
    coors: &ArrayView2<'_, f64>,
    elements: &Vec<f64>,
    n: usize,
    pr: Option<f64>,
) -> HashMap<usize, f64> {
    let ball = dotsphere(n);

    let y_ptr = if pr.is_none() { 1.4f64 } else { pr.unwrap() };

    let dd = Mutex::new(HashMap::<usize, f64>::new());

    // 缓存半径计算
    let ele = elements
        .into_par_iter()
        .map(|f| (*f + y_ptr).powi(2) - 1e-6)
        .collect::<Vec<f64>>();

    // 遍历原子集合
    (0..coors.nrows()).into_par_iter().for_each(|i| {
        let r = elements[i] + y_ptr;

        // 生成该原子上的均等分点
        let ball2 = ball.mapv(|b| b * r) + coors.row(i);

        let filer = Mutex::new(0);

        // 遍历这些点, 开始刷选在其余原子半径内的点
        (0..n).into_par_iter().for_each(|j| {
            let mut result = true;
            let b = ball2.row(j);
            for c in 0..coors.nrows() {
                let cc = coors.row(c);
                let r = ele[c];
                let r1 = distance(&cc, &b);
                if r1 < r {
                    result = false;
                    break;
                }
            }
            // 不在半径内即为重叠部分, 选中
            if result {
                *filer.lock().unwrap() += 1;
            }
        });

        let pecent = filer.lock().unwrap().to_owned() as f64 / n as f64;

        dd.lock().unwrap().insert(i, pecent);
    });

    let m = dd.lock().unwrap().to_owned();
    m
}

#[cfg(test)]
mod tests {
    use log::info;
    use ndarray::array;

    use super::*;

    #[test]
    fn test_surface() {
        crate::config::init_config();

        let a = array![[0., 0., 0.], [0., 0., 1.7], [0., 0., 10.7]];

        let b = vec!["C", "O", "CD1"];
        let n = 10;

        info!("start ....");
        let d = sa_surface(&a.view(), Some(&b), Some(n), Some(1.4), true);
        info!("done ....{:?}", d.shape());

        let mut radis_v = vec![0.; a.nrows()];
        get_vdw_vec(Some(&b), &mut radis_v);

        info!("start form prev....");
        let d = sa_surface_from_prev(&a.view(), &d.view(), &radis_v, n, 1.4, 2.8);
        info!("done form prev ....{:?}", d);

        info!("start ....");
        let d = sa_surface(&a.view(), Some(&b), Some(n), Some(2.8), false);
        info!("done ....{:?}", d);

        // let n = 200;

        // info!("start ....");
        // let d = sa_surface(&a.view(), Some(&b), Some(n), Some(1.4), true);
        // info!("done ....{:?}", d);

        // assert_eq!(456, d.shape()[0]);
    }

    #[test]
    fn test_ndarray() {
        crate::config::init_config();
        let n = 1000000;
        info!("start dotsphere");
        let a = dotsphere(n);
        info!("dotsphere done");
        println!("{:?} ", a);

        assert!(true);
    }
}
