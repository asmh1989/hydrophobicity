#![allow(dead_code)]

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{
    c64, npyffi::NPY_ARRAY_WRITEABLE, IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArrayDyn,
};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

use crate::{electrostatic::cal_electro, surface::sa_surface};

mod config;
mod electrostatic;
mod surface;

#[pymodule]
pub fn sz_py_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // immutable example
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // mutable example (no return)
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    // complex example
    fn conj(x: ArrayViewD<'_, c64>) -> ArrayD<c64> {
        x.map(|c| c.conj())
    }

    // wrapper of `axpy`
    #[pyfn(m, "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<'_, f64>,
        y: PyReadonlyArrayDyn<'_, f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        axpy(a, x, y).into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(a: f64, x: &PyArrayDyn<f64>) {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
    }

    // wrapper of `conj`
    #[pyfn(m, "conj")]
    fn conj_py<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'_, c64>) -> &'py PyArrayDyn<c64> {
        conj(x.as_array()).into_pyarray(py)
    }

    #[pyfn(m, "cal_electro")]
    fn cal_electro_py<'py>(
        _py: Python<'py>,
        grid: PyReadonlyArray1<'_, f64>,
        atoms: PyReadonlyArray2<'_, f64>,
        n: usize,
    ) -> f64 {
        cal_electro(grid.as_array(), atoms.as_array(), n)
    }

    #[pyfn(m, "sa_surface")]
    fn sa_surface_py<'py>(
        py: Python<'py>,
        coors: PyReadonlyArray2<'_, f64>,
        elements: Vec<&str>,
        n: usize,
        pr: f64,
    ) -> &'py PyArray2<f64> {
        // println!("elements = {:?}", elements);
        let dot =
            sa_surface(&coors.as_array(), Some(&elements), Some(n), Some(pr)).into_pyarray(py);
        unsafe {
            (*dot.as_array_ptr()).flags |= NPY_ARRAY_WRITEABLE;
        }
        dot
    }

    #[pyfn(m, "sa_surface_no_ele")]
    fn sa_surface_no_ele_py<'py>(
        py: Python<'py>,
        coors: PyReadonlyArray2<'_, f64>,
        n: usize,
        pr: f64,
    ) -> &'py PyArray2<f64> {
        let dot = sa_surface(&coors.as_array(), None, Some(n), Some(pr)).into_pyarray(py);
        unsafe {
            (*dot.as_array_ptr()).flags |= NPY_ARRAY_WRITEABLE;
        }
        dot
    }

    Ok(())
}
