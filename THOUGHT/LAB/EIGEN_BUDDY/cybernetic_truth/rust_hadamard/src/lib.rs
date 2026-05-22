use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2};
use rayon::prelude::*;

const INV_SQRT2: f32 = 0.7071067811865475;

/// Apply Walsh-Hadamard transform to each row. O(N log N) per row.
/// Input: (m, 2^d) float32. Output: (m, 2^d).
#[pyfunction]
fn hadamard_transform<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let arr = input.as_array();
    let m = arr.shape()[0];
    let n = arr.shape()[1];
    let d = n.ilog2() as usize;
    
    // Verify n is power of 2
    if 1usize << d != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "input width must be power of 2"
        ));
    }
    
    let mut result = arr.to_owned();
    
    // Apply H to each dimension. Parallel across rows.
    result.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let mut buf = vec![0.0f32; n];
            for t in 0..d {
                let step = 1 << t;
                for i in (0..n).step_by(step * 2) {
                    for j in 0..step {
                        let a = row[i + j];
                        let b = row[i + step + j];
                        buf[i + j] = (a + b) * INV_SQRT2;
                        buf[i + step + j] = (a - b) * INV_SQRT2;
                    }
                }
                if t < d - 1 {
                    row.as_slice_mut().unwrap().copy_from_slice(&buf);
                }
            }
            row.as_slice_mut().unwrap().copy_from_slice(&buf);
        });
    
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pymodule]
fn rust_hadamard(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hadamard_transform, m)?)?;
    Ok(())
}
