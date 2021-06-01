#![allow(dead_code)]

use numpy::{
    npyffi::NPY_ARRAY_WRITEABLE, IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

use crate::{electrostatic::cal_electro, pockets::find_pockets, surface::sa_surface};

mod config;
mod electrostatic;
mod pockets;
mod surface;

#[macro_export]
macro_rules! nparray_return {
    ($s:expr) => {{
        let dot = $s;
        unsafe {
            (*dot.as_array_ptr()).flags |= NPY_ARRAY_WRITEABLE;
        }
        dot
    }};
}

#[pymodule]
pub fn sz_py_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
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
        index: bool,
    ) -> &'py PyArray2<f64> {
        // println!("elements = {:?}", elements);
        nparray_return!(
            sa_surface(&coors.as_array(), Some(&elements), Some(n), Some(pr), index)
                .into_pyarray(py)
        )
    }

    #[pyfn(m, "sa_surface_no_ele")]
    fn sa_surface_no_ele_py<'py>(
        py: Python<'py>,
        coors: PyReadonlyArray2<'_, f64>,
        n: usize,
        pr: f64,
        index: bool,
    ) -> &'py PyArray2<f64> {
        nparray_return!(
            sa_surface(&coors.as_array(), None, Some(n), Some(pr), index).into_pyarray(py)
        )
    }

    #[pyfn(m, "find_pocket")]
    fn find_pocket_py<'py>(
        py: Python<'py>,
        coors: PyReadonlyArray2<'_, f64>,
        elements: Vec<&str>,
        n: usize,
        pr: f64,
    ) -> &'py PyArray2<f64> {
        crate::config::init_config();
        nparray_return!(
            find_pockets(&coors.as_array(), Some(&elements), n, Some(pr)).into_pyarray(py)
        )
    }

    Ok(())
}
