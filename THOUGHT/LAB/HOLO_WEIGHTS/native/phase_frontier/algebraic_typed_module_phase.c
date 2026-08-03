#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: typed phase-resident relation modules.
 *
 * Reuse the reviewed series/parallel parser and exact phase algebra in this
 * translation unit, but replace its CLI with a typed module compiler.  Leaf
 * modules compile independently.  Composite module descriptors contain only
 * nominal interface types and the resident phase address exported by each
 * child.  Parent composition consumes those actual addresses without decode,
 * serialization, hashing, or export copies.
 */

#define main algebraic_series_parallel_embedded_main
#include "algebraic_series_parallel_phase.c"
#undef main

#define MAX_DOMAINS 16U
#define MAX_MODULES 31U
#define PATH_CAPACITY 128U

enum module_kind {
    MODULE_LEAF = 0,
    MODULE_COMPOSITE = 1
};

struct domain {
    char name[MAX_IDENTIFIER + 1U];
};

struct typed_module {
    char name[MAX_IDENTIFIER + 1U];
    enum module_kind kind;
    size_t left_domain;
    size_t right_domain;
    char leaf_path[PATH_CAPACITY];
    size_t left_child;
    size_t right_child;
    size_t use_count;
    struct process *leaf;
    size_t input_offset;
    size_t operation_offset;
    size_t final_start;
    int final_transposed;
};

struct typed_source {
    struct domain domain[MAX_DOMAINS];
    struct typed_module module[MAX_MODULES];
    size_t domain_count;
    size_t module_count;
    size_t root;
    uint64_t source_hash;
};

static int valid_leaf_path(const char *text) {
    const size_t length = strlen(text);
    if (
        length == 0U
        || length >= PATH_CAPACITY
        || strstr(text, "..") != NULL
    ) {
        return 0;
    }
    for (size_t index = 0U; index < length; ++index) {
        const unsigned char byte = (unsigned char)text[index];
        if (!(
            isalnum(byte)
            || byte == '_'
            || byte == '-'
            || byte == '.'
        )) {
            return 0;
        }
    }
    return 1;
}

static size_t typed_find_domain(
    const struct typed_source *source,
    const char *name,
    size_t line
) {
    for (size_t index = 0U; index < source->domain_count; ++index) {
        if (strcmp(source->domain[index].name, name) == 0) {
            return index;
        }
    }
    fail_line("undeclared nominal domain", line);
    return 0U;
}

static size_t typed_find_module(
    const struct typed_source *source,
    const char *name,
    size_t line
) {
    for (size_t index = 0U; index < source->module_count; ++index) {
        if (strcmp(source->module[index].name, name) == 0) {
            return index;
        }
    }
    fail_line("undeclared or forward module reference", line);
    return 0U;
}

static int name_already_used(
    const struct typed_source *source,
    const char *name
) {
    for (size_t index = 0U; index < source->module_count; ++index) {
        if (strcmp(source->module[index].name, name) == 0) {
            return 1;
        }
    }
    return 0;
}

static struct typed_source read_typed_source(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }
    struct typed_source source = {
        .root = SIZE_MAX,
        .source_hash = UINT64_C(14695981039346656037)
    };
    char line[LINE_CAPACITY];
    size_t line_number = 0U;
    int header = 0;
    int composite_seen = 0;
    int project_seen = 0;
    int end = 0;
    while (fgets(line, sizeof(line), stream) != NULL) {
        ++line_number;
        const size_t length = strlen(line);
        source.source_hash = hash_bytes(
            source.source_hash,
            (const unsigned char *)line,
            length
        );
        if (length == 0U || line[length - 1U] != '\n') {
            fail_line("every typed-module record must end with LF", line_number);
        }
        if (memchr(line, '\r', length) != NULL) {
            fail_line("CR bytes are forbidden", line_number);
        }
        line[length - 1U] = '\0';
        if (end) {
            fail_line("record after END", line_number);
        }
        char *token[TOKEN_CAPACITY] = {0};
        const size_t count = tokenize(line, token);
        if (count == 0U) {
            fail_line("blank records are forbidden", line_number);
        }
        if (!header) {
            if (
                count != 2U
                || strcmp(token[0], "CATCAS_TYPED_RELATION_MODULE") != 0
                || strcmp(token[1], "1") != 0
            ) {
                fail_line("invalid typed-module header", line_number);
            }
            header = 1;
        } else if (strcmp(token[0], "DOMAIN") == 0) {
            if (
                composite_seen
                || source.module_count != 0U
                || count != 3U
                || source.domain_count == MAX_DOMAINS
                || !valid_identifier(token[1])
                || strcmp(token[2], "BOOLEAN_F3") != 0
            ) {
                fail_line("invalid nominal DOMAIN", line_number);
            }
            for (size_t index = 0U; index < source.domain_count; ++index) {
                if (strcmp(source.domain[index].name, token[1]) == 0) {
                    fail_line("duplicate nominal DOMAIN", line_number);
                }
            }
            memcpy(
                source.domain[source.domain_count++].name,
                token[1],
                strlen(token[1]) + 1U
            );
        } else if (strcmp(token[0], "LEAF") == 0) {
            if (
                source.domain_count == 0U
                || composite_seen
                || project_seen
                || count != 5U
                || source.module_count == MAX_MODULES
                || !valid_identifier(token[1])
                || name_already_used(&source, token[1])
                || !valid_leaf_path(token[4])
            ) {
                fail_line("invalid LEAF module", line_number);
            }
            struct typed_module *module =
                &source.module[source.module_count++];
            memcpy(module->name, token[1], strlen(token[1]) + 1U);
            module->kind = MODULE_LEAF;
            module->left_domain =
                typed_find_domain(&source, token[2], line_number);
            module->right_domain =
                typed_find_domain(&source, token[3], line_number);
            memcpy(
                module->leaf_path,
                token[4],
                strlen(token[4]) + 1U
            );
        } else if (strcmp(token[0], "COMPOSE") == 0) {
            if (
                source.module_count < 2U
                || project_seen
                || count != 4U
                || source.module_count == MAX_MODULES
                || !valid_identifier(token[1])
                || name_already_used(&source, token[1])
            ) {
                fail_line("invalid COMPOSE module", line_number);
            }
            composite_seen = 1;
            const size_t left =
                typed_find_module(&source, token[2], line_number);
            const size_t right =
                typed_find_module(&source, token[3], line_number);
            if (left == right) {
                fail_line("a module cannot compose with itself", line_number);
            }
            if (
                source.module[left].right_domain
                != source.module[right].left_domain
            ) {
                fail_line(
                    "nominal module interface type mismatch",
                    line_number
                );
            }
            struct typed_module *module =
                &source.module[source.module_count++];
            memcpy(module->name, token[1], strlen(token[1]) + 1U);
            module->kind = MODULE_COMPOSITE;
            module->left_domain = source.module[left].left_domain;
            module->right_domain = source.module[right].right_domain;
            module->left_child = left;
            module->right_child = right;
            ++source.module[left].use_count;
            ++source.module[right].use_count;
        } else if (strcmp(token[0], "PROJECT") == 0) {
            if (
                !composite_seen
                || project_seen
                || count != 2U
            ) {
                fail_line("invalid PROJECT", line_number);
            }
            project_seen = 1;
            source.root =
                typed_find_module(&source, token[1], line_number);
        } else if (strcmp(token[0], "END") == 0) {
            if (!project_seen || count != 1U) {
                fail_line("invalid END", line_number);
            }
            end = 1;
        } else {
            fail_line("unknown typed-module record", line_number);
        }
    }
    if (ferror(stream) || fclose(stream) != 0) {
        fail("failed to read typed-module source");
    }
    if (
        !header
        || source.domain_count == 0U
        || source.module_count < 3U
        || source.root == SIZE_MAX
        || !end
        || source.module[source.root].kind != MODULE_COMPOSITE
    ) {
        fail("typed-module source is incomplete");
    }
    for (size_t index = 0U; index < source.module_count; ++index) {
        const size_t expected = index == source.root ? 0U : 1U;
        if (source.module[index].use_count != expected) {
            fail("every non-root module must have one owning parent");
        }
    }
    return source;
}

#ifndef CATCAS_TYPED_MODULE_NO_MAIN
static size_t remap_leaf_start(
    const struct process *leaf,
    size_t start,
    size_t input_offset,
    size_t operation_offset,
    size_t total_inputs
) {
    if ((start % CCOUNT) != 0U) {
        fail("unaligned leaf message address");
    }
    const size_t relation = start / CCOUNT;
    if (relation < leaf->input_count) {
        return (input_offset + relation) * CCOUNT;
    }
    const size_t operation = relation - leaf->input_count;
    if (operation >= leaf->operation_count) {
        fail("leaf message address is outside its operation arena");
    }
    return (total_inputs + operation_offset + operation) * CCOUNT;
}

static struct process compile_typed_source(
    struct typed_source *typed
) {
    size_t total_inputs = 0U;
    size_t total_operations = 0U;
    size_t total_eliminations = 0U;
    size_t leaf_count = 0U;
    for (size_t index = 0U; index < typed->module_count; ++index) {
        struct typed_module *module = &typed->module[index];
        if (module->kind != MODULE_LEAF) {
            continue;
        }
        for (size_t earlier = 0U; earlier < index; ++earlier) {
            if (
                typed->module[earlier].kind == MODULE_LEAF
                && strcmp(
                    typed->module[earlier].leaf_path,
                    module->leaf_path
                ) == 0
            ) {
                module->leaf = typed->module[earlier].leaf;
                break;
            }
        }
        if (module->leaf == NULL) {
            module->leaf = checked_calloc(1U, sizeof(*module->leaf));
            *module->leaf = read_process(module->leaf_path);
        }
        module->input_offset = total_inputs;
        module->operation_offset = total_operations;
        if (
            module->leaf->input_count > MAX_INPUTS - total_inputs
            || module->leaf->operation_count
                > MAX_OPERATIONS - total_operations
        ) {
            fail("typed module program exceeds global capacity");
        }
        total_inputs += module->leaf->input_count;
        total_operations += module->leaf->operation_count;
        total_eliminations += module->leaf->elimination_count;
        ++leaf_count;
    }
    const size_t composite_count = typed->module_count - leaf_count;
    if (composite_count > MAX_OPERATIONS - total_operations) {
        fail("typed composite operation capacity exceeded");
    }
    total_operations += composite_count;

    struct process process = {
        .input_count = total_inputs,
        .source_hash = typed->source_hash
    };
    memcpy(process.node[0].name, "ROOT_LEFT", sizeof("ROOT_LEFT"));
    memcpy(process.node[1].name, "ROOT_RIGHT", sizeof("ROOT_RIGHT"));
    process.node[0].kind = NODE_EXTERNAL;
    process.node[1].kind = NODE_EXTERNAL;
    process.node_count = 2U;
    process.external[0] = 0U;
    process.external[1] = 1U;
    process.elimination_count = total_eliminations;

    for (size_t index = 0U; index < typed->module_count; ++index) {
        struct typed_module *module = &typed->module[index];
        if (module->kind != MODULE_LEAF) {
            continue;
        }
        const struct process *leaf = module->leaf;
        process.source_hash ^= leaf->source_hash;
        process.source_hash *= UINT64_C(1099511628211);
        for (size_t input = 0U; input < leaf->input_count; ++input) {
            process.input[module->input_offset + input] = leaf->input[input];
        }
        for (
            size_t operation = 0U;
            operation < leaf->operation_count;
            ++operation
        ) {
            const struct operation *source = &leaf->operation[operation];
            struct operation *target =
                &process.operation[process.operation_count++];
            *target = *source;
            target->left_start = remap_leaf_start(
                leaf,
                source->left_start,
                module->input_offset,
                module->operation_offset,
                total_inputs
            );
            target->right_start = remap_leaf_start(
                leaf,
                source->right_start,
                module->input_offset,
                module->operation_offset,
                total_inputs
            );
            target->output_start = (
                total_inputs
                + module->operation_offset
                + operation
            ) * CCOUNT;
            if (target->kind == OP_COMPOSE) {
                ++process.composition_count;
            } else {
                ++process.intersection_count;
            }
        }
        module->final_start = remap_leaf_start(
            leaf,
            leaf->final_start,
            module->input_offset,
            module->operation_offset,
            total_inputs
        );
        module->final_transposed = leaf->final_transposed;
    }

    for (size_t index = 0U; index < typed->module_count; ++index) {
        struct typed_module *module = &typed->module[index];
        if (module->kind != MODULE_COMPOSITE) {
            continue;
        }
        const struct typed_module *left =
            &typed->module[module->left_child];
        const struct typed_module *right =
            &typed->module[module->right_child];
        module->final_start = add_operation(
            &process,
            OP_COMPOSE,
            left->final_start,
            left->final_transposed,
            right->final_start,
            right->final_transposed
        );
        module->final_transposed = 0;
    }
    if (
        process.operation_count != total_operations
        || typed->module[typed->root].final_start
            != process.operation[
                process.operation_count - 1U
            ].output_start
    ) {
        fail("typed-module postorder compilation invariant failed");
    }
    process.final_start = typed->module[typed->root].final_start;
    process.final_transposed =
        typed->module[typed->root].final_transposed;
    process.carrier_cells = (
        process.input_count
        + process.operation_count
        + 1U
    ) * CCOUNT;
    return process;
}
#endif

static void free_typed_source(struct typed_source *typed) {
    for (size_t index = 0U; index < typed->module_count; ++index) {
        struct typed_module *module = &typed->module[index];
        if (module->kind != MODULE_LEAF || module->leaf == NULL) {
            continue;
        }
        int first_owner = 1;
        for (size_t earlier = 0U; earlier < index; ++earlier) {
            if (
                typed->module[earlier].kind == MODULE_LEAF
                && strcmp(
                    typed->module[earlier].leaf_path,
                    module->leaf_path
                ) == 0
            ) {
                first_owner = 0;
                break;
            }
        }
        if (first_owner) {
            free(module->leaf);
        }
        module->leaf = NULL;
    }
}

#ifndef CATCAS_TYPED_MODULE_NO_MAIN
static int same_typed_geometry(
    const struct typed_source *left_source,
    const struct process *left,
    const struct typed_source *right_source,
    const struct process *right
) {
    if (
        left_source->domain_count != right_source->domain_count
        || left_source->module_count != right_source->module_count
        || left_source->root != right_source->root
        || left->input_count != right->input_count
        || left->operation_count != right->operation_count
        || left->carrier_cells != right->carrier_cells
    ) {
        return 0;
    }
    for (size_t index = 0U; index < left_source->domain_count; ++index) {
        if (
            strcmp(
                left_source->domain[index].name,
                right_source->domain[index].name
            ) != 0
        ) {
            return 0;
        }
    }
    for (size_t index = 0U; index < left_source->module_count; ++index) {
        const struct typed_module *a = &left_source->module[index];
        const struct typed_module *b = &right_source->module[index];
        if (
            strcmp(a->name, b->name) != 0
            || a->kind != b->kind
            || a->left_domain != b->left_domain
            || a->right_domain != b->right_domain
            || a->left_child != b->left_child
            || a->right_child != b->right_child
        ) {
            return 0;
        }
    }
    return 1;
}

static int module_handoffs_are_direct(
    const struct typed_source *typed,
    const struct process *process
) {
    size_t composite_count = 0U;
    for (size_t index = 0U; index < typed->module_count; ++index) {
        composite_count +=
            typed->module[index].kind == MODULE_COMPOSITE;
    }
    size_t composite_operation =
        process->operation_count - composite_count;
    for (size_t index = 0U; index < typed->module_count; ++index) {
        const struct typed_module *module = &typed->module[index];
        if (module->kind != MODULE_COMPOSITE) {
            continue;
        }
        const struct operation *operation =
            &process->operation[composite_operation++];
        if (
            operation->kind != OP_COMPOSE
            || operation->left_start
                != typed->module[module->left_child].final_start
            || operation->right_start
                != typed->module[module->right_child].final_start
        ) {
            return 0;
        }
    }
    return composite_operation == process->operation_count;
}
#endif

static size_t module_depth(
    const struct typed_source *typed,
    size_t module_index
) {
    const struct typed_module *module = &typed->module[module_index];
    if (module->kind == MODULE_LEAF) {
        return 1U;
    }
    const size_t left = module_depth(typed, module->left_child);
    const size_t right = module_depth(typed, module->right_child);
    return 1U + (left > right ? left : right);
}

static size_t unique_leaf_source_count(
    const struct typed_source *typed
) {
    size_t unique = 0U;
    for (size_t index = 0U; index < typed->module_count; ++index) {
        const struct typed_module *module = &typed->module[index];
        if (module->kind != MODULE_LEAF) {
            continue;
        }
        int seen = 0;
        for (size_t earlier = 0U; earlier < index; ++earlier) {
            if (
                typed->module[earlier].kind == MODULE_LEAF
                && strcmp(
                    typed->module[earlier].leaf_path,
                    module->leaf_path
                ) == 0
            ) {
                seen = 1;
                break;
            }
        }
        unique += !seen;
    }
    return unique;
}

#ifndef CATCAS_TYPED_MODULE_NO_MAIN
static void print_typed_execution(
    const char *mode,
    const struct typed_source *typed,
    const struct process *process,
    const struct execution *execution,
    int direct_handoffs
) {
    size_t leaf_count = 0U;
    for (size_t index = 0U; index < typed->module_count; ++index) {
        leaf_count += typed->module[index].kind == MODULE_LEAF;
    }
    printf(
        "{\"mode\":\"%s\","
        "\"claim\":\"BOUNDED_RECURSIVE_TYPED_RELATIONAL_MODULE_COMPOSITION\","
        "\"source_fnv1a64\":\"%016llx\","
        "\"nominal_domain_count\":%zu,"
        "\"module_descriptor_count\":%zu,"
        "\"leaf_module_count\":%zu,"
        "\"composite_module_count\":%zu,"
        "\"module_tree_depth\":%zu,"
        "\"unique_leaf_source_count\":%zu,"
        "\"compiled_leaf_definition_reuse\":%s,"
        "\"module_handoffs_direct\":%s,"
        "\"module_export_copy_cells\":0,"
        "\"decoded_module_coefficients\":0,"
        "\"serialized_module_coefficients\":0,"
        "\"expanded_module_truth_tables\":0,"
        "\"compact_definition_reuse\":false,"
        "\"expanded_native_operation_descriptors\":%zu,"
        "\"compiled_typed_source_storage_bytes\":%zu,"
        "\"compiled_leaf_definition_storage_bytes\":%zu,"
        "\"compiled_native_process_storage_bytes\":%zu,"
        "\"native_operation_descriptor_bytes\":%zu,"
        "\"input_relation_count\":%zu,"
        "\"native_composition_operations\":%zu,"
        "\"native_intersection_operations\":%zu,"
        "\"phase_resident_relation_messages\":%zu,"
        "\"carrier_cells\":%zu,"
        "\"live_carrier_complex_values\":%zu,"
        "\"verification_peak_complex_values\":%zu,"
        "\"live_carrier_bytes\":%zu,"
        "\"retained_inverse_factors\":0,"
        "\"boundary_coefficients\":[%d,%d,%d,%d],"
        "\"maximum_root_error\":%.12g,"
        "\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"carrier_integrity_max_abs\":%.12g}\n",
        mode,
        (unsigned long long)process->source_hash,
        typed->domain_count,
        typed->module_count,
        leaf_count,
        typed->module_count - leaf_count,
        module_depth(typed, typed->root),
        unique_leaf_source_count(typed),
        unique_leaf_source_count(typed) < leaf_count ? "true" : "false",
        direct_handoffs ? "true" : "false",
        process->operation_count,
        sizeof(*typed),
        unique_leaf_source_count(typed) * sizeof(struct process),
        sizeof(*process),
        process->operation_count * sizeof(struct operation),
        process->input_count,
        process->composition_count,
        process->intersection_count,
        process->operation_count,
        process->carrier_cells,
        2U * process->carrier_cells,
        4U * process->carrier_cells,
        2U * process->carrier_cells * sizeof(double complex),
        execution->boundary.coefficient[0],
        execution->boundary.coefficient[1],
        execution->boundary.coefficient[2],
        execution->boundary.coefficient[3],
        execution->boundary.maximum_root_error,
        execution->displacement_l2,
        execution->restoration_max_abs,
        execution->integrity_max_abs
    );
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        fprintf(
            stderr,
            "usage: %s PROCESS.atrm [REUSE_PROCESS.atrm]\n",
            argv[0]
        );
        return 2;
    }
    struct typed_source typed = read_typed_source(argv[1]);
    struct process process = compile_typed_source(&typed);
    struct typed_source reuse_typed =
        argc == 3 ? read_typed_source(argv[2]) : read_typed_source(argv[1]);
    struct process reuse_process = compile_typed_source(&reuse_typed);
    if (!same_typed_geometry(
        &typed,
        &process,
        &reuse_typed,
        &reuse_process
    )) {
        fail("reuse typed process must have identical module geometry");
    }
    const int direct_handoffs =
        module_handoffs_are_direct(&typed, &process);
    if (!direct_handoffs) {
        fail("composite module does not consume actual child exports");
    }

    struct carrier carrier = make_carrier(&process, 5303);
    const struct execution nominal =
        execute(&carrier, &process, MODE_CORRECT);
    const struct execution reuse =
        execute(&carrier, &reuse_process, MODE_CORRECT);
    free_carrier(&carrier);

    carrier = make_carrier(&process, 5303);
    const struct execution wrong = execute(
        &carrier,
        &process,
        MODE_WRONG_BOUNDARY_INVERSE
    );
    free_carrier(&carrier);
    carrier = make_carrier(&process, 5303);
    const struct execution omitted = execute(
        &carrier,
        &process,
        MODE_OMITTED_MESSAGE_INVERSE
    );
    free_carrier(&carrier);
    carrier = make_carrier(&process, 5303);
    const struct execution bypass = execute(
        &carrier,
        &process,
        MODE_BYPASS_INTERSECTION
    );
    free_carrier(&carrier);
    carrier = make_carrier(&process, 5303);
    const struct execution ordinary = execute(
        &carrier,
        &process,
        MODE_ORDINARY_SUM_INTERSECTION
    );
    free_carrier(&carrier);

    print_typed_execution(
        "typed-module-relational-phase",
        &typed,
        &process,
        &nominal,
        direct_handoffs
    );
    print_typed_execution(
        argc == 3
            ? "actual-restored-cross-program-module-reuse"
            : "actual-restored-module-reuse",
        &reuse_typed,
        &reuse_process,
        &reuse,
        module_handoffs_are_direct(&reuse_typed, &reuse_process)
    );
    print_typed_execution(
        "wrong-boundary-inverse",
        &typed,
        &process,
        &wrong,
        direct_handoffs
    );
    print_typed_execution(
        "omitted-resident-module-inverse",
        &typed,
        &process,
        &omitted,
        direct_handoffs
    );
    print_typed_execution(
        "bypassed-leaf-intersection",
        &typed,
        &process,
        &bypass,
        direct_handoffs
    );
    print_typed_execution(
        "ordinary-sum-leaf-intersection",
        &typed,
        &process,
        &ordinary,
        direct_handoffs
    );
    const int bypass_applicable =
        boundary_differs(&nominal.boundary, &bypass.boundary);
    const int ordinary_applicable =
        boundary_differs(&nominal.boundary, &ordinary.boundary);
    printf(
        "{\"mode\":\"module-control-applicability\","
        "\"wrong_boundary\":%s,"
        "\"omitted_resident_module\":%s,"
        "\"bypassed_leaf_intersection\":%s,"
        "\"ordinary_sum_leaf_intersection\":%s}\n",
        wrong.wrong_applicable ? "true" : "false",
        omitted.omitted_applicable ? "true" : "false",
        bypass_applicable ? "true" : "false",
        ordinary_applicable ? "true" : "false"
    );
    const int valid = (
        nominal.boundary.maximum_root_error <= ROOT_TOLERANCE
        && reuse.boundary.maximum_root_error <= ROOT_TOLERANCE
        && bypass.boundary.maximum_root_error <= ROOT_TOLERANCE
        && ordinary.boundary.maximum_root_error <= ROOT_TOLERANCE
        && nominal.restoration_max_abs <= RESTORATION_TOLERANCE
        && reuse.restoration_max_abs <= RESTORATION_TOLERANCE
        && bypass.restoration_max_abs <= RESTORATION_TOLERANCE
        && ordinary.restoration_max_abs <= RESTORATION_TOLERANCE
        && (
            !wrong.wrong_applicable
            || wrong.restoration_max_abs >= CONTROL_MINIMUM
        )
        && (
            !omitted.omitted_applicable
            || omitted.restoration_max_abs >= CONTROL_MINIMUM
        )
    );
    free_typed_source(&typed);
    free_typed_source(&reuse_typed);
    return valid ? 0 : 1;
}
#endif
