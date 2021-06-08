use std::{
    cmp::min,
    collections::HashMap,
    f64::consts::PI,
    sync::{Mutex, RwLock},
    usize,
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

pub struct Protein<'a> {
    pub coors: ArrayView2<'a, f64>,
    pub radis_v: Vec<f64>,
    pub n: usize,
    cache: Vec<(f64, ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>)>,
}

pub const DEFAULT_PTR: f64 = 1.4;

#[inline]
fn get_with_index(
    dots: &ArrayView2<'_, f64>,
    index: bool,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    if index {
        dots.to_owned()
    } else {
        dots.slice_axis(Axis(1), Slice::from(0..3)).to_owned()
    }
}

impl<'a> Protein<'a> {
    pub fn new(coors: ArrayView2<'a, f64>, elements: Option<&Vec<&str>>, n: usize) -> Self {
        // 求得原子半径集合缓存, 下面多个方法需要使用
        let mut radis_v = vec![0.; coors.nrows()];
        get_vdw_vec(elements, &mut radis_v);

        let data = sa_surface_core(&coors, &radis_v, n, Some(DEFAULT_PTR), true);
        let mut cache = vec![];
        cache.push((DEFAULT_PTR, data));
        Self {
            coors,
            radis_v,
            n,
            cache,
        }
    }

    ///
    /// 计算当前蛋白质的sa平面集合
    ///
    pub fn sa_surface(
        &mut self,
        pr: f64,
        index: bool,
    ) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
        let mut i = 0;
        for (k, v) in &self.cache {
            if *k == pr {
                return get_with_index(&v.view(), index);
            } else if *k < pr {
                break;
            } else {
                i += 1;
            }
        }

        // 未发现缓存
        if i == 0 {
            let data = sa_surface_core(&self.coors, &self.radis_v, self.n, Some(pr), true);
            self.cache.insert(i, (pr, data.clone()));
            return get_with_index(&data.view(), index);
        } else {
            let prev = self.cache[min(self.cache.len() - 1, i)].clone();
            let data = sa_surface_from_prev(
                &self.coors,
                &prev.1.view(),
                &self.radis_v,
                self.n,
                prev.0,
                pr,
                index,
            );
            self.cache
                .insert(min(self.cache.len(), i), (pr, data.clone()));
            return get_with_index(&data.view(), index);
        }
    }

    ///
    /// 计算`pr` 对应结果中, 均分点在每个原子下的百分比
    ///
    pub fn get_index_map_by_ptr(&mut self, pr: f64) -> HashMap<usize, f64> {
        let tmp = self.sa_surface(pr, true);
        let sa = Mutex::new(HashMap::<usize, f64>::new());

        (0..self.coors.nrows()).into_par_iter().for_each(|f| {
            let count = tmp
                .axis_iter(Axis(0))
                .filter(|r| (r[3] as usize) == f)
                .count();

            let n = &mut sa.lock().unwrap();
            n.insert(f, count as f64 / self.n as f64);
        });

        let m = sa.lock().unwrap().to_owned();
        m
    }
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
fn sa_surface_core(
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

///
/// 利用缓存数据,计算蛋白质sa平面点集合
///
/// `coors`: 蛋白质原子集合
/// `prev_dots`: `prev_ptr`计算的缓存结果
/// `elements`: 原子半径集合
/// `prev_ptr`: 缓存结果的对应的辅助半径
/// `y_ptr`: 要计算的新半径
/// `n`: 均分点数
/// `index`: 是否包含原子索引
///
fn sa_surface_from_prev(
    coors: &ArrayView2<'_, f64>,
    prev_dots: &ArrayView2<'_, f64>,
    elements: &Vec<f64>,
    n: usize,
    prev_ptr: f64,
    y_ptr: f64,
    index: bool,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    let dd = Mutex::new(Vec::<f64>::new());

    let col = if index { 4 } else { 3 };

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

        if u.nrows() == 0 {
            return;
        }

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
        let n = 1000;

        let mut p = Protein::new(a.view(), Some(&b), n);

        info!("start ....");
        let d = sa_surface(&a.view(), Some(&b), Some(n), Some(DEFAULT_PTR), true);
        info!("done ....{:?}", d.shape());

        info!("start form prev....");
        let d = p.sa_surface(2.8, false);
        info!("done form prev ....{:?}", d.shape());

        info!("start form prev....");
        let d = p.sa_surface(3.4, false);
        info!("done form prev ....{:?}", d.shape());

        info!("start form prev....");
        let d = p.sa_surface(6.8, false);
        info!("done form prev ....{:?}", d.shape());

        info!("start form prev....");
        let d = p.sa_surface(1.0, false);
        info!("done form prev ....{:?}", d.shape());

        info!("start form prev....");
        let d = p.sa_surface(10.8, false);
        info!("done form prev ....{:?}", d.shape());

        info!("start ....");
        let d1 = sa_surface(&a.view(), Some(&b), Some(n), Some(10.8), false);
        info!("done ....{:?}", d1.shape());

        assert_eq!(d1.shape()[0], d.shape()[0]);
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
