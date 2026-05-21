use pyo3::prelude::*;
use pyo3::types::PyBytes;
use numpy::{PyArray1, PyArrayMethods};

/// F16 decode: auto-vectorized via target-cpu=native + LTO + opt-level=3.
/// Uses powi (integer exponent) for fast power-of-2 computation.
#[pyfunction]
fn f16_decode<'py>(py: Python<'py>, data: Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let bytes = data.as_bytes();
    let n = bytes.len() / 2;
    let mut result: Vec<f32> = Vec::with_capacity(n);

    for i in 0..n {
        let off = i * 2;
        let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        let sign = ((raw >> 15) & 1) as f32;
        let exp = ((raw >> 10) & 0x1F) as i32;
        let mant = (raw & 0x3FF) as f32;

        let val = if exp == 0 {
            mant * (1.0 / 1024.0) * 2.0_f32.powi(-14)
        } else {
            (1.0 + mant * (1.0 / 1024.0)) * 2.0_f32.powi(exp - 15)
        };

        result.push(if sign > 0.0 { -val } else { val });
    }

    Ok(PyArray1::from_slice_bound(py, &result))
}

/// Identity-block orthogonal projection for N Cores. Guaranteed 0 cross-talk.
#[pyfunction]
fn orthogonal_project<'py>(
    py: Python<'py>,
    n_cores: usize,
    d_dims: usize,
) -> PyResult<Vec<Bound<'py, PyArray1<f32>>>> {
    let dims_per_core = d_dims / n_cores;
    let mut projections: Vec<Bound<'py, PyArray1<f32>>> = Vec::with_capacity(n_cores);

    for i in 0..n_cores {
        let mut flat: Vec<f32> = vec![0.0; d_dims * d_dims];
        let start = i * dims_per_core;
        for j in start..(start + dims_per_core) {
            flat[j * d_dims + j] = 1.0;
        }
        projections.push(PyArray1::from_slice_bound(py, &flat));
    }

    Ok(projections)
}

/// Fast non-cryptographic tape hash.
#[pyfunction]
fn tape_hash(data: Bound<PyBytes>) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    data.as_bytes().hash(&mut h);
    format!("{:016x}", h.finish())
}

#[pymodule]
fn catalytic_ffi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(f16_decode, m)?)?;
    m.add_function(wrap_pyfunction!(orthogonal_project, m)?)?;
    m.add_function(wrap_pyfunction!(tape_hash, m)?)?;
    Ok(())
}
