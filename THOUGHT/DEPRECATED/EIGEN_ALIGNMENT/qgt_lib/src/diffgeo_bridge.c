// Thin bridge: exposes diffgeo_ functions as a shared library
#include "quantum_geometric/distributed/differential_geometry.h"

diffgeo_engine_t* bridge_engine_create(void) {
    return diffgeo_engine_create();
}

bool bridge_compute_berry_curvature(
    diffgeo_engine_t* engine,
    const ComplexDouble* state, size_t dim,
    const ComplexDouble* param_derivs, size_t num_params,
    double* curvature_out)
{
    return diffgeo_compute_berry_curvature(engine, state, dim, param_derivs, num_params, curvature_out);
}

bool bridge_compute_fubini_study(
    diffgeo_engine_t* engine,
    const ComplexDouble* state, size_t dim,
    const ComplexDouble* param_derivs, size_t num_params,
    double* metric_out)
{
    return diffgeo_compute_fubini_study(engine, state, dim, param_derivs, num_params, metric_out);
}

ComplexDouble bridge_compute_berry_phase(
    diffgeo_engine_t* engine,
    const ComplexDouble** states,
    size_t num_states,
    size_t dim)
{
    return diffgeo_compute_berry_phase(engine, states, num_states, dim);
}
