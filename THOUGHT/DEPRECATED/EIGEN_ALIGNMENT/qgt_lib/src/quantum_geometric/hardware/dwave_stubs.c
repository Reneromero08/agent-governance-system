#include "quantum_geometric/hardware/quantum_dwave_backend.h"
#include <stddef.h>
DWaveConfig* init_dwave_backend(const DWaveBackendConfig* c) { return NULL; }
DWaveProblem* create_dwave_problem(size_t n, size_t m) { return NULL; }
char* submit_dwave_job(DWaveConfig* c, const DWaveJobConfig* j) { return NULL; }
DWaveJobStatus get_dwave_job_status(DWaveConfig* c, const char* id) { return (DWaveJobStatus){0}; }
DWaveJobResult* get_dwave_job_result(DWaveConfig* c, const char* id) { return NULL; }
bool cancel_dwave_job(DWaveConfig* c, const char* id) { return false; }
char* get_dwave_error_info(DWaveConfig* c, const char* id) { return NULL; }
void cleanup_dwave_config(DWaveConfig* c) {}
void cleanup_dwave_problem(DWaveProblem* p) {}
void cleanup_dwave_result(DWaveJobResult* r) {}
void cleanup_dwave_backend(DWaveConfig* c) {}