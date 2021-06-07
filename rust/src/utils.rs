use ndarray::ArrayView1;

///
/// 两点在三维空间的距离
///
#[inline]
pub fn distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}
