use std::f64::consts::PI;

use ndarray::{ArrayView1, ArrayView2, Axis};

pub fn cal_electro(grid: ArrayView1<'_, f64>, atoms: ArrayView2<'_, f64>, n: usize) -> f64 {
    let (s1, s2) = atoms.view().split_at(Axis(1), 3);
    let s = &s1 - &grid;
    let tmp = s.mapv(|i| i * i).sum_axis(Axis(1)).mapv(|i| i.sqrt());
    let a = tmp.dot(&s2)[0];
    (1. / (4. * PI * (n as f64))) * a
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let grid = ndarray::array![9., 68., 44.];
        let a = ndarray::array![[5.8, 7.7, 3.75, -0.5], [11.5, 86.9, 31.9, -0.5]];

        super::cal_electro(grid.view(), a.view(), 4);
        assert_eq!(2 + 2, 4);
    }
}
