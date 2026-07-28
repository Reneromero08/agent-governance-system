#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: topology-compiled shared nonlinear phase graph.
 *
 * A public ordered interaction DAG is compiled without inspecting carrier
 * values. Each edge consumes an actual resident source phasor and applies a
 * nonlinear torus shear to an actual resident target through multiply_cell.
 * Sources may feed multiple later shears; no intermediate phase is copied,
 * decoded, serialized, or expanded into paths. The inverse walks the exact
 * compiled edge custody in reverse, then the restored carrier is reused.
 */

#define main algebraic_series_parallel_embedded_main
#include "algebraic_series_parallel_phase.c"
#undef main

#include <errno.h>

#define NPG_MAX_WIDTH 32U
#define NPG_MAX_EDGES 64U
#define NPG_MIN_ROUNDS 1U
#define NPG_MAX_ROUNDS 4096U
#define NPG_REUSE_CYCLES 8U
#define NPG_HASH_SCALE 1000000000.0

static const double NPG_PHASE_TOLERANCE = 2.0e-11;
static const double NPG_CONTROL_MINIMUM = 1.0e-6;

enum npg_program {
    NPG_PRIMARY = 0,
    NPG_REUSE = 1
};

enum npg_mode {
    NPG_CORRECT = 0,
    NPG_MISSING_INVERSE = 1,
    NPG_WRONG_INVERSE = 2,
    NPG_REORDERED_INVERSE = 3,
    NPG_SNAPSHOT = 4,
    NPG_DISABLED = 5
};

struct npg_edge {
    size_t public_id;
    size_t target;
    size_t source;
    int strength_milli;
};

struct npg_graph {
    size_t width;
    size_t output[2];
    size_t edge_count;
    struct npg_edge edge[NPG_MAX_EDGES];
    unsigned char dependency[NPG_MAX_EDGES][NPG_MAX_EDGES];
    size_t source_use[NPG_MAX_WIDTH];
    uint64_t topology_hash;
};

struct npg_stats {
    uint64_t forward_shears;
    uint64_t inverse_shears;
    uint64_t resident_source_reads;
    uint64_t native_phase_updates;
    uint64_t final_phase_decodes;
    double maximum_unit_modulus_error;
};

struct npg_projection {
    double complex phase[2];
    double interference_probability;
    uint64_t hash;
};

struct npg_execution {
    struct npg_projection projection;
    struct npg_stats stats;
    double restoration_max_abs;
    double integrity_max_abs;
    uint64_t restoration_generation;
    int actual_inverse;
    int snapshot_loaded;
};

static uint64_t npg_hash_word(uint64_t hash, uint64_t value) {
    for (size_t byte = 0U; byte < sizeof(value); ++byte) {
        hash ^= (value >> (8U * byte)) & UINT64_C(255);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void npg_parse_failure(const char *message) {
    fail(message);
}

static size_t npg_parse_size(
    const char *text,
    size_t minimum,
    size_t maximum,
    const char *message
) {
    if (text == NULL || text[0] < '0' || text[0] > '9') {
        npg_parse_failure(message);
    }
    for (size_t index = 1U; text[index] != '\0'; ++index) {
        if (text[index] < '0' || text[index] > '9') {
            npg_parse_failure(message);
        }
    }
    errno = 0;
    char *tail = NULL;
    const unsigned long parsed = strtoul(text, &tail, 10);
    if (
        errno != 0
        || tail == text
        || *tail != '\0'
        || parsed < minimum
        || parsed > maximum
    ) {
        npg_parse_failure(message);
    }
    return (size_t)parsed;
}

static struct npg_graph npg_load_graph(const char *path) {
    FILE *stream = fopen(path, "r");
    if (stream == NULL) {
        fail("cannot open nonlinear phase graph");
    }
    struct npg_graph graph = {
        .topology_hash = UINT64_C(14695981039346656037)
    };
    char line[160];
    int header = 0;
    int width = 0;
    int outputs = 0;
    int dependencies = 0;
    int end = 0;
    while (fgets(line, sizeof(line), stream) != NULL) {
        if (strchr(line, '\n') == NULL && !feof(stream)) {
            fail("nonlinear phase graph line too long");
        }
        char extra = '\0';
        if (strcmp(line, "NONLINEAR_PHASE_GRAPH 1\n") == 0) {
            if (header || width || outputs || graph.edge_count != 0U) {
                fail("nonlinear phase graph header order");
            }
            header = 1;
            continue;
        }
        size_t parsed_width = 0U;
        if (sscanf(line, "WIDTH %zu %c", &parsed_width, &extra) == 1) {
            if (
                !header
                || width
                || outputs
                || parsed_width < 3U
                || parsed_width > NPG_MAX_WIDTH
            ) {
                fail("nonlinear phase graph width invalid");
            }
            graph.width = parsed_width;
            width = 1;
            continue;
        }
        size_t first = 0U;
        size_t second = 0U;
        if (
            sscanf(line, "OUTPUTS %zu %zu %c", &first, &second, &extra)
                == 2
        ) {
            if (
                !width
                || outputs
                || first >= graph.width
                || second >= graph.width
                || first == second
            ) {
                fail("nonlinear phase graph outputs invalid");
            }
            graph.output[0] = first;
            graph.output[1] = second;
            outputs = 1;
            continue;
        }
        size_t public_id = 0U;
        int strength = 0;
        if (
            sscanf(
                line,
                "SHEAR %zu %zu %zu %d %c",
                &public_id,
                &first,
                &second,
                &strength,
                &extra
            ) == 4
        ) {
            if (
                !outputs
                || dependencies
                || graph.edge_count >= NPG_MAX_EDGES
                || public_id == 0U
                || first >= graph.width
                || second >= graph.width
                || first == second
                || strength == 0
                || strength < -250
                || strength > 250
            ) {
                fail("nonlinear phase graph shear invalid");
            }
            for (
                size_t prior = 0U;
                prior < graph.edge_count;
                ++prior
            ) {
                if (graph.edge[prior].public_id == public_id) {
                    fail("nonlinear phase graph duplicate shear id");
                }
            }
            graph.edge[graph.edge_count] = (struct npg_edge){
                .public_id = public_id,
                .target = first,
                .source = second,
                .strength_milli = strength
            };
            ++graph.edge_count;
            continue;
        }
        size_t dependent_id = 0U;
        size_t prerequisite_id = 0U;
        if (
            sscanf(
                line,
                "AFTER %zu %zu %c",
                &dependent_id,
                &prerequisite_id,
                &extra
            ) == 2
        ) {
            if (graph.edge_count < 2U) {
                fail("nonlinear phase graph dependency order");
            }
            dependencies = 1;
            size_t dependent = SIZE_MAX;
            size_t prerequisite = SIZE_MAX;
            for (
                size_t index = 0U;
                index < graph.edge_count;
                ++index
            ) {
                if (graph.edge[index].public_id == dependent_id) {
                    dependent = index;
                }
                if (graph.edge[index].public_id == prerequisite_id) {
                    prerequisite = index;
                }
            }
            if (
                dependent == SIZE_MAX
                || prerequisite == SIZE_MAX
                || dependent == prerequisite
                || graph.dependency[dependent][prerequisite] != 0U
            ) {
                fail("nonlinear phase graph dependency invalid");
            }
            graph.dependency[dependent][prerequisite] = 1U;
            continue;
        }
        if (strcmp(line, "END\n") == 0 || strcmp(line, "END") == 0) {
            if (!outputs || graph.edge_count < 2U || end) {
                fail("nonlinear phase graph end invalid");
            }
            end = 1;
            break;
        }
        fail("nonlinear phase graph syntax invalid");
    }
    if (fclose(stream) != 0 || !header || !width || !outputs || !end) {
        fail("nonlinear phase graph incomplete");
    }
    unsigned char reach[NPG_MAX_EDGES][NPG_MAX_EDGES] = {{0}};
    memcpy(reach, graph.dependency, sizeof(reach));
    for (size_t pivot = 0U; pivot < graph.edge_count; ++pivot) {
        for (size_t row = 0U; row < graph.edge_count; ++row) {
            for (size_t column = 0U; column < graph.edge_count; ++column) {
                if (reach[row][pivot] && reach[pivot][column]) {
                    reach[row][column] = 1U;
                }
            }
        }
    }
    for (size_t edge = 0U; edge < graph.edge_count; ++edge) {
        if (reach[edge][edge]) {
            fail("nonlinear phase graph contains cycle");
        }
    }
    for (size_t left = 0U; left < graph.edge_count; ++left) {
        for (size_t right = left + 1U; right < graph.edge_count; ++right) {
            const struct npg_edge *a = &graph.edge[left];
            const struct npg_edge *b = &graph.edge[right];
            const int hazard = (
                a->target == b->target
                || a->target == b->source
                || b->target == a->source
            );
            if (
                hazard
                && !reach[left][right]
                && !reach[right][left]
            ) {
                fail("nonlinear phase graph has unordered hazard");
            }
        }
    }

    struct npg_edge scheduled[NPG_MAX_EDGES] = {{0}};
    unsigned char emitted[NPG_MAX_EDGES] = {0};
    size_t raw_index[NPG_MAX_EDGES] = {0};
    for (size_t cursor = 0U; cursor < graph.edge_count; ++cursor) {
        size_t selected = SIZE_MAX;
        for (size_t candidate = 0U; candidate < graph.edge_count; ++candidate) {
            if (emitted[candidate]) {
                continue;
            }
            int ready = 1;
            for (
                size_t prerequisite = 0U;
                prerequisite < graph.edge_count;
                ++prerequisite
            ) {
                if (
                    graph.dependency[candidate][prerequisite]
                    && !emitted[prerequisite]
                ) {
                    ready = 0;
                    break;
                }
            }
            if (
                ready
                && (
                    selected == SIZE_MAX
                    || graph.edge[candidate].public_id
                        < graph.edge[selected].public_id
                )
            ) {
                selected = candidate;
            }
        }
        if (selected == SIZE_MAX) {
            fail("nonlinear phase graph scheduling failed");
        }
        emitted[selected] = 1U;
        scheduled[cursor] = graph.edge[selected];
        raw_index[cursor] = selected;
    }
    memcpy(graph.edge, scheduled, sizeof(scheduled));
    memset(graph.dependency, 0, sizeof(graph.dependency));
    for (size_t dependent = 0U; dependent < graph.edge_count; ++dependent) {
        for (
            size_t prerequisite = 0U;
            prerequisite < graph.edge_count;
            ++prerequisite
        ) {
            graph.dependency[dependent][prerequisite] =
                reach[raw_index[dependent]][raw_index[prerequisite]];
        }
    }

    size_t shared_sources = 0U;
    for (size_t edge = 0U; edge < graph.edge_count; ++edge) {
        ++graph.source_use[graph.edge[edge].source];
    }
    for (size_t cell = 0U; cell < graph.width; ++cell) {
        if (graph.source_use[cell] > 1U) {
            ++shared_sources;
        }
    }
    if (shared_sources < 2U) {
        fail("nonlinear phase graph must contain shared sources");
    }
    graph.topology_hash = npg_hash_word(
        graph.topology_hash, graph.width
    );
    graph.topology_hash = npg_hash_word(
        graph.topology_hash, graph.output[0]
    );
    graph.topology_hash = npg_hash_word(
        graph.topology_hash, graph.output[1]
    );
    for (size_t index = 0U; index < graph.edge_count; ++index) {
        graph.topology_hash = npg_hash_word(
            graph.topology_hash, graph.edge[index].public_id
        );
        graph.topology_hash = npg_hash_word(
            graph.topology_hash, graph.edge[index].target
        );
        graph.topology_hash = npg_hash_word(
            graph.topology_hash, graph.edge[index].source
        );
        graph.topology_hash = npg_hash_word(
            graph.topology_hash,
            (uint64_t)(int64_t)graph.edge[index].strength_milli
        );
        for (
            size_t prerequisite = 0U;
            prerequisite < graph.edge_count;
            ++prerequisite
        ) {
            if (graph.dependency[index][prerequisite]) {
                graph.topology_hash = npg_hash_word(
                    graph.topology_hash,
                    graph.edge[prerequisite].public_id
                );
            }
        }
    }
    return graph;
}

static void npg_observe(
    const struct carrier *carrier,
    struct npg_stats *stats
) {
    for (size_t cell = 0U; cell < carrier->cells; ++cell) {
        const double error = fabs(cabs(carrier->working[cell]) - 1.0);
        if (error > stats->maximum_unit_modulus_error) {
            stats->maximum_unit_modulus_error = error;
        }
        if (error > NPG_PHASE_TOLERANCE) {
            fail("nonlinear phase graph left unit torus");
        }
    }
}

static double npg_initial_angle(
    enum npg_program program,
    size_t cell
) {
    const size_t residue = (
        (program == NPG_PRIMARY ? 7U : 11U) * cell
        + (program == NPG_PRIMARY ? 3U : 5U)
    ) % 23U;
    return 0.071 * (double)(residue + 1U)
        - (program == NPG_PRIMARY ? 0.91 : 1.17);
}

static void npg_seal(
    struct carrier *carrier,
    const struct npg_graph *graph,
    enum npg_program program,
    int inverse,
    struct npg_stats *stats
) {
    for (size_t cell = 0U; cell < graph->width; ++cell) {
        const double complex phase = cexp(
            I * npg_initial_angle(program, cell)
        );
        multiply_cell(
            carrier, cell, inverse ? conj(phase) : phase
        );
        ++stats->native_phase_updates;
    }
    npg_observe(carrier, stats);
}

static void npg_shear(
    struct carrier *carrier,
    const struct npg_edge *edge,
    size_t round,
    int inverse,
    int wrong,
    enum npg_mode mode,
    struct npg_stats *stats
) {
    const double complex source = relative(carrier, edge->source);
    ++stats->resident_source_reads;
    const double round_scale =
        1.0 + 0.03125 * (double)(round % 5U);
    double strength =
        (double)edge->strength_milli * 0.001 * round_scale;
    if (inverse && !wrong) {
        strength = -strength;
    }
    if (mode == NPG_DISABLED) {
        strength = 0.0;
    }
    multiply_cell(
        carrier,
        edge->target,
        cexp(I * strength * cimag(source))
    );
    ++stats->native_phase_updates;
    if (inverse) {
        ++stats->inverse_shears;
    } else {
        ++stats->forward_shears;
    }
    npg_observe(carrier, stats);
}

static void npg_boundary(
    struct carrier *carrier,
    const struct npg_graph *graph,
    int inverse,
    struct npg_stats *stats
) {
    for (size_t output = 0U; output < 2U; ++output) {
        const double complex phase = relative(
            carrier, graph->output[output]
        );
        ++stats->resident_source_reads;
        multiply_cell(
            carrier,
            graph->width + output,
            inverse ? conj(phase) : phase
        );
        ++stats->native_phase_updates;
    }
    npg_observe(carrier, stats);
}

static int64_t npg_quantize(double value) {
    return (int64_t)llround(value * NPG_HASH_SCALE);
}

static struct npg_projection npg_project(
    const struct carrier *carrier,
    const struct npg_graph *graph,
    struct npg_stats *stats
) {
    struct npg_projection projection = {
        .hash = UINT64_C(14695981039346656037)
    };
    for (size_t output = 0U; output < 2U; ++output) {
        projection.phase[output] = relative(
            carrier, graph->width + output
        );
        const int64_t real = npg_quantize(
            creal(projection.phase[output])
        );
        const int64_t imaginary = npg_quantize(
            cimag(projection.phase[output])
        );
        projection.hash = npg_hash_word(
            projection.hash, (uint64_t)real
        );
        projection.hash = npg_hash_word(
            projection.hash, (uint64_t)imaginary
        );
        ++stats->final_phase_decodes;
    }
    projection.interference_probability = 0.5 * (
        1.0 + creal(
            projection.phase[0] * conj(projection.phase[1])
        )
    );
    return projection;
}

static struct npg_execution npg_execute(
    struct carrier *carrier,
    const struct npg_graph *graph,
    size_t rounds,
    enum npg_program program,
    enum npg_mode mode
) {
    if (
        carrier == NULL
        || graph == NULL
        || carrier->cells != graph->width + 2U
        || rounds < NPG_MIN_ROUNDS
        || rounds > NPG_MAX_ROUNDS
    ) {
        fail("invalid topology nonlinear phase transaction");
    }
    struct carrier borrowed = snapshot_carrier(carrier);
    struct npg_execution execution = {0};
    npg_seal(carrier, graph, program, 0, &execution.stats);
    for (size_t round = 0U; round < rounds; ++round) {
        for (size_t edge = 0U; edge < graph->edge_count; ++edge) {
            npg_shear(
                carrier,
                &graph->edge[edge],
                round,
                0,
                0,
                mode,
                &execution.stats
            );
        }
    }
    npg_boundary(carrier, graph, 0, &execution.stats);
    execution.projection = npg_project(
        carrier, graph, &execution.stats
    );

    if (mode == NPG_SNAPSHOT) {
        memcpy(
            carrier->working,
            borrowed.working,
            carrier->cells * sizeof(*carrier->working)
        );
        execution.snapshot_loaded = 1;
    } else {
        npg_boundary(carrier, graph, 1, &execution.stats);
        const size_t operations = rounds * graph->edge_count;
        for (size_t reverse = 0U; reverse < operations; ++reverse) {
            if (mode == NPG_MISSING_INVERSE && reverse == 0U) {
                continue;
            }
            size_t selected = reverse;
            if (mode == NPG_REORDERED_INVERSE) {
                /*
                 * Public operations 40 (3<-2) and 50 (0<-3) are an
                 * applicability-gated adjacent noncommuting pair.
                 */
                if (reverse == 1U) {
                    selected = 2U;
                } else if (reverse == 2U) {
                    selected = 1U;
                }
            }
            const size_t ordinal = operations - 1U - selected;
            const size_t round = ordinal / graph->edge_count;
            const size_t edge = ordinal % graph->edge_count;
            npg_shear(
                carrier,
                &graph->edge[edge],
                round,
                1,
                mode == NPG_WRONG_INVERSE && reverse == 0U,
                mode,
                &execution.stats
            );
        }
        npg_seal(carrier, graph, program, 1, &execution.stats);
        if (mode == NPG_CORRECT) {
            execution.actual_inverse = 1;
            execution.restoration_generation = 1U;
        }
    }
    execution.restoration_max_abs = restoration(carrier, &borrowed);
    execution.integrity_max_abs = integrity(carrier);
    free_carrier(&borrowed);
    return execution;
}

#ifndef NPG_NO_REQUIRE
static void npg_require_correct(
    const struct npg_graph *graph,
    size_t rounds,
    const struct npg_execution *execution
) {
    const uint64_t operations =
        (uint64_t)(rounds * graph->edge_count);
    if (
        !execution->actual_inverse
        || execution->snapshot_loaded
        || execution->restoration_generation != 1U
        || execution->stats.forward_shears != operations
        || execution->stats.inverse_shears != operations
        || execution->stats.final_phase_decodes != 2U
        || execution->restoration_max_abs > NPG_PHASE_TOLERANCE
        || execution->integrity_max_abs > NPG_PHASE_TOLERANCE
        || execution->stats.maximum_unit_modulus_error
            > NPG_PHASE_TOLERANCE
    ) {
        fail("topology nonlinear phase carrier law failed");
    }
}
#endif

#ifndef NPG_EMBEDDED
static struct npg_execution npg_control(
    const struct npg_graph *graph,
    enum npg_mode mode
) {
    const struct process process = {
        .carrier_cells = graph->width + 2U
    };
    struct carrier carrier = make_carrier(
        &process, 12100 + (int)mode
    );
    struct npg_execution execution = npg_execute(
        &carrier, graph, 3U, NPG_PRIMARY, mode
    );
    free_carrier(&carrier);
    return execution;
}

int main(int argc, char **argv) {
    if (
        argc == 2
        && strcmp(argv[1], "--project-intermediate") == 0
    ) {
        fail("nonlinear graph intermediate projection denied");
    }
    if (
        argc == 2
        && strcmp(argv[1], "--null-carrier") == 0
    ) {
        (void)npg_execute(
            NULL,
            NULL,
            1U,
            NPG_PRIMARY,
            NPG_CORRECT
        );
        fail("topology nonlinear null-carrier control failed open");
    }
    if (argc != 3) {
        fail(
            "usage: topology_nonlinear_phase_graph "
            "TOPOLOGY ROUNDS"
        );
    }
    const struct npg_graph graph = npg_load_graph(argv[1]);
    const size_t rounds = npg_parse_size(
        argv[2],
        NPG_MIN_ROUNDS,
        NPG_MAX_ROUNDS,
        "rounds must be canonical decimal from 1 through 4096"
    );
    const struct process process = {
        .carrier_cells = graph.width + 2U
    };
    struct carrier carrier = make_carrier(&process, 12001);
    const struct npg_execution primary = npg_execute(
        &carrier, &graph, rounds, NPG_PRIMARY, NPG_CORRECT
    );
    npg_require_correct(&graph, rounds, &primary);
    const struct npg_execution reuse = npg_execute(
        &carrier, &graph, rounds, NPG_REUSE, NPG_CORRECT
    );
    npg_require_correct(&graph, rounds, &reuse);
    if (primary.projection.hash == reuse.projection.hash) {
        fail("topology nonlinear phase reuse boundary unchanged");
    }
    double repeated_max = 0.0;
    for (size_t cycle = 0U; cycle < NPG_REUSE_CYCLES; ++cycle) {
        const struct npg_execution repeated = npg_execute(
            &carrier,
            &graph,
            rounds,
            cycle % 2U == 0U ? NPG_PRIMARY : NPG_REUSE,
            NPG_CORRECT
        );
        npg_require_correct(&graph, rounds, &repeated);
        if (repeated.restoration_max_abs > repeated_max) {
            repeated_max = repeated.restoration_max_abs;
        }
    }
    free_carrier(&carrier);

    const struct npg_execution missing = npg_control(
        &graph, NPG_MISSING_INVERSE
    );
    const struct npg_execution wrong = npg_control(
        &graph, NPG_WRONG_INVERSE
    );
    const struct npg_execution reordered = npg_control(
        &graph, NPG_REORDERED_INVERSE
    );
    const struct npg_execution snapshot = npg_control(
        &graph, NPG_SNAPSHOT
    );
    const struct npg_execution disabled = npg_control(
        &graph, NPG_DISABLED
    );
    if (
        missing.restoration_max_abs < NPG_CONTROL_MINIMUM
        || wrong.restoration_max_abs < NPG_CONTROL_MINIMUM
        || reordered.restoration_max_abs < NPG_CONTROL_MINIMUM
        || !snapshot.snapshot_loaded
        || snapshot.actual_inverse
        || snapshot.restoration_max_abs > NPG_PHASE_TOLERANCE
        || disabled.projection.hash == primary.projection.hash
    ) {
        fail("topology nonlinear phase control failed");
    }

    size_t shared_sources = 0U;
    size_t maximum_fanout = 0U;
    size_t precedence_pairs = 0U;
    for (size_t cell = 0U; cell < graph.width; ++cell) {
        if (graph.source_use[cell] > 1U) {
            ++shared_sources;
        }
        if (graph.source_use[cell] > maximum_fanout) {
            maximum_fanout = graph.source_use[cell];
        }
    }
    for (size_t edge = 0U; edge < graph.edge_count; ++edge) {
        for (
            size_t prerequisite = 0U;
            prerequisite < graph.edge_count;
            ++prerequisite
        ) {
            precedence_pairs +=
                (size_t)graph.dependency[edge][prerequisite];
        }
    }
    printf(
        "{\"mode\":\"topology-compiled-shared-nonlinear-phase-graph\","
        "\"claim\":\"BOUNDED_TOPOLOGY_COMPILED_SHARED_NONLINEAR_PHASE_GRAPH_WITH_ACTUAL_RESTORATION_AND_REUSE\","
        "\"width\":%zu,\"rounds\":%zu,\"edges\":%zu,"
        "\"shared_sources\":%zu,\"maximum_source_fanout\":%zu,"
        "\"compiled_precedence_pairs\":%zu,"
        "\"topology_fnv1a64\":\"%016llx\","
        "\"resident_phase_cells\":%zu,\"boundary_phase_cells\":2,"
        "\"carrier_cells\":%zu,\"live_carrier_bytes\":%zu,"
        "\"compiled_topology_bytes\":%zu,"
        "\"best_matched_classical_state_bytes\":%zu,"
        "\"primary_boundary_fnv1a64\":\"%016llx\","
        "\"reuse_boundary_fnv1a64\":\"%016llx\","
        "\"primary_interference_probability\":%.17g,"
        "\"forward_shears\":%llu,\"inverse_shears\":%llu,"
        "\"resident_source_reads\":%llu,"
        "\"native_phase_updates\":%llu,"
        "\"final_phase_decodes\":%llu,"
        "\"maximum_unit_modulus_error\":%.17g,"
        "\"maximum_repeated_restoration_error\":%.17g,"
        "\"missing_inverse_residual\":%.17g,"
        "\"wrong_inverse_residual\":%.17g,"
        "\"reordered_inverse_residual\":%.17g,"
        "\"snapshot_residual\":%.17g,"
        "\"projected_intermediate_phases\":0,"
        "\"serialized_intermediate_phases\":0,"
        "\"materialized_phase_paths\":0,"
        "\"topology_only_compilation\":true,"
        "\"deterministic_topological_scheduler\":true,"
        "\"unordered_hazards_rejected\":true,"
        "\"shared_resident_phase_sources\":true,"
        "\"nonlinear_phase_coupling\":true,"
        "\"native_phase_updates_only\":true,"
        "\"actual_inverse_restoration\":true,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"fixed_carrier_across_rounds\":true,"
        "\"snapshot_loaded_separately\":true,"
        "\"compact_classical_width_angle_recurrence_exists\":true,"
        "\"machine_enforced_hidden_intermediate\":false,"
        "\"genuinely_distinct_phase_resource\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_execution\":false,"
        "\"unlimited_catalytic_computation\":false,"
        "\"terminal\":false}\n",
        graph.width,
        rounds,
        graph.edge_count,
        shared_sources,
        maximum_fanout,
        precedence_pairs,
        (unsigned long long)graph.topology_hash,
        graph.width,
        graph.width + 2U,
        2U * (graph.width + 2U) * sizeof(double complex),
        sizeof(struct npg_graph),
        graph.width * sizeof(double),
        (unsigned long long)primary.projection.hash,
        (unsigned long long)reuse.projection.hash,
        primary.projection.interference_probability,
        (unsigned long long)primary.stats.forward_shears,
        (unsigned long long)primary.stats.inverse_shears,
        (unsigned long long)primary.stats.resident_source_reads,
        (unsigned long long)primary.stats.native_phase_updates,
        (unsigned long long)primary.stats.final_phase_decodes,
        primary.stats.maximum_unit_modulus_error,
        repeated_max,
        missing.restoration_max_abs,
        wrong.restoration_max_abs,
        reordered.restoration_max_abs,
        snapshot.restoration_max_abs
    );
    return 0;
}
#endif
