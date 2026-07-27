#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: compact execution of typed relation modules.
 *
 * Leaf definitions are compiled once and shared.  Each module instance keeps
 * only a typed descriptor and phase-resident message addresses.  At runtime a
 * single transient operation descriptor relocates the shared body into the
 * instance's carrier region.  No per-instance native instruction records are
 * retained.  Carrier messages remain distinct so the actual forward history
 * can be reversed.
 */

#define CATCAS_TYPED_MODULE_NO_MAIN 1
#include "algebraic_typed_module_phase.c"

struct compact_layout {
    size_t input_relations;
    size_t operation_messages;
    size_t carrier_cells;
    size_t first_intersection;
    size_t leaf_instances;
    size_t unique_leaf_operations;
    size_t composite_descriptors;
};

struct compact_resource_ledger {
    size_t coexisting_typed_source_storage_bytes;
    size_t coexisting_leaf_definition_storage_bytes;
    size_t coexisting_layout_storage_bytes;
};

struct compact_execution {
    struct boundary boundary;
    double displacement_l2;
    double restoration_max_abs;
    double integrity_max_abs;
    int wrong_applicable;
    int omitted_parent_applicable;
};

static void load_shared_leaf_definitions(
    struct typed_source *typed
) {
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
    }
}

static size_t compact_remap_leaf_start(
    const struct typed_module *module,
    size_t start,
    size_t total_inputs
) {
    const struct process *leaf = module->leaf;
    if ((start % CCOUNT) != 0U) {
        fail("unaligned compact leaf message address");
    }
    const size_t relation = start / CCOUNT;
    if (relation < leaf->input_count) {
        return (module->input_offset + relation) * CCOUNT;
    }
    const size_t operation = relation - leaf->input_count;
    if (operation >= leaf->operation_count) {
        fail("compact leaf message address outside definition");
    }
    return (
        total_inputs
        + module->operation_offset
        + operation
    ) * CCOUNT;
}

static struct compact_layout prepare_compact_layout(
    struct typed_source *typed
) {
    load_shared_leaf_definitions(typed);
    struct compact_layout layout = {
        .first_intersection = SIZE_MAX
    };
    for (size_t index = 0U; index < typed->module_count; ++index) {
        struct typed_module *module = &typed->module[index];
        if (module->kind != MODULE_LEAF) {
            continue;
        }
        ++layout.leaf_instances;
        module->input_offset = layout.input_relations;
        if (
            module->leaf->input_count
            > MAX_INPUTS - layout.input_relations
        ) {
            fail("compact input relation capacity exceeded");
        }
        layout.input_relations += module->leaf->input_count;
    }

    size_t message = 0U;
    size_t unique_operations = 0U;
    for (size_t index = 0U; index < typed->module_count; ++index) {
        struct typed_module *module = &typed->module[index];
        if (module->kind != MODULE_LEAF) {
            continue;
        }
        module->operation_offset = message;
        for (
            size_t operation = 0U;
            operation < module->leaf->operation_count;
            ++operation
        ) {
            if (
                layout.first_intersection == SIZE_MAX
                && module->leaf->operation[operation].kind
                    == OP_INTERSECT
            ) {
                layout.first_intersection = message + operation;
            }
        }
        module->final_start = compact_remap_leaf_start(
            module,
            module->leaf->final_start,
            layout.input_relations
        );
        module->final_transposed = module->leaf->final_transposed;
        message += module->leaf->operation_count;

        int first_source = 1;
        for (size_t earlier = 0U; earlier < index; ++earlier) {
            if (
                typed->module[earlier].kind == MODULE_LEAF
                && typed->module[earlier].leaf == module->leaf
            ) {
                first_source = 0;
                break;
            }
        }
        if (first_source) {
            unique_operations += module->leaf->operation_count;
        }
    }
    for (size_t index = 0U; index < typed->module_count; ++index) {
        struct typed_module *module = &typed->module[index];
        if (module->kind != MODULE_COMPOSITE) {
            continue;
        }
        module->operation_offset = message;
        module->final_start = (
            layout.input_relations + message
        ) * CCOUNT;
        module->final_transposed = 0;
        ++message;
        ++layout.composite_descriptors;
    }
    if (
        layout.first_intersection == SIZE_MAX
        || typed->module[typed->root].final_start
            != (
                layout.input_relations + message - 1U
            ) * CCOUNT
    ) {
        fail("compact module layout invariant failed");
    }
    layout.operation_messages = message;
    layout.unique_leaf_operations = unique_operations;
    layout.carrier_cells = (
        layout.input_relations
        + layout.operation_messages
        + 1U
    ) * CCOUNT;
    return layout;
}

static size_t compact_boundary_start(
    const struct compact_layout *layout
) {
    return layout->carrier_cells - CCOUNT;
}

static struct operation relocated_leaf_operation(
    const struct typed_module *module,
    size_t operation,
    size_t total_inputs
) {
    const struct operation *definition =
        &module->leaf->operation[operation];
    return (struct operation){
        .kind = definition->kind,
        .left_start = compact_remap_leaf_start(
            module,
            definition->left_start,
            total_inputs
        ),
        .left_transposed = definition->left_transposed,
        .right_start = compact_remap_leaf_start(
            module,
            definition->right_start,
            total_inputs
        ),
        .right_transposed = definition->right_transposed,
        .output_start = (
            total_inputs
            + module->operation_offset
            + operation
        ) * CCOUNT
    };
}

static struct operation composite_operation(
    const struct typed_source *typed,
    const struct typed_module *module
) {
    const struct typed_module *left =
        &typed->module[module->left_child];
    const struct typed_module *right =
        &typed->module[module->right_child];
    return (struct operation){
        .kind = OP_COMPOSE,
        .left_start = left->final_start,
        .left_transposed = left->final_transposed,
        .right_start = right->final_start,
        .right_transposed = right->final_transposed,
        .output_start = module->final_start
    };
}

static void apply_compact_forward(
    struct carrier *carrier,
    const struct typed_source *typed,
    const struct compact_layout *layout,
    enum execution_mode mode
) {
    for (size_t index = 0U; index < typed->module_count; ++index) {
        const struct typed_module *module = &typed->module[index];
        if (module->kind == MODULE_LEAF) {
            for (
                size_t operation = 0U;
                operation < module->leaf->operation_count;
                ++operation
            ) {
                const struct operation relocated =
                    relocated_leaf_operation(
                        module,
                        operation,
                        layout->input_relations
                    );
                apply_operation(
                    carrier,
                    &relocated,
                    mode,
                    layout->first_intersection,
                    module->operation_offset + operation,
                    0
                );
            }
        } else {
            const struct operation operation =
                composite_operation(typed, module);
            apply_operation(
                carrier,
                &operation,
                mode,
                layout->first_intersection,
                module->operation_offset,
                0
            );
        }
    }
}

static void apply_compact_inverse(
    struct carrier *carrier,
    const struct typed_source *typed,
    const struct compact_layout *layout,
    enum execution_mode mode,
    int omit_root
) {
    for (size_t cursor = typed->module_count; cursor > 0U; --cursor) {
        const size_t index = cursor - 1U;
        const struct typed_module *module = &typed->module[index];
        if (module->kind == MODULE_COMPOSITE) {
            if (omit_root && index == typed->root) {
                continue;
            }
            const struct operation operation =
                composite_operation(typed, module);
            apply_operation(
                carrier,
                &operation,
                mode,
                layout->first_intersection,
                module->operation_offset,
                1
            );
            continue;
        }
        for (
            size_t operation = module->leaf->operation_count;
            operation > 0U;
            --operation
        ) {
            const struct operation relocated =
                relocated_leaf_operation(
                    module,
                    operation - 1U,
                    layout->input_relations
                );
            apply_operation(
                carrier,
                &relocated,
                mode,
                layout->first_intersection,
                module->operation_offset + operation - 1U,
                1
            );
        }
    }
}

static void encode_compact_inputs(
    struct carrier *carrier,
    const struct typed_source *typed,
    int inverse
) {
    if (!inverse) {
        for (size_t module_index = 0U;
             module_index < typed->module_count;
             ++module_index) {
            const struct typed_module *module =
                &typed->module[module_index];
            if (module->kind != MODULE_LEAF) {
                continue;
            }
            for (size_t input = 0U;
                 input < module->leaf->input_count;
                 ++input) {
                apply_encoding(
                    carrier,
                    (module->input_offset + input) * CCOUNT,
                    module->leaf->input[input].coefficient,
                    0
                );
            }
        }
        return;
    }
    for (size_t cursor = typed->module_count; cursor > 0U; --cursor) {
        const struct typed_module *module =
            &typed->module[cursor - 1U];
        if (module->kind != MODULE_LEAF) {
            continue;
        }
        for (size_t input = module->leaf->input_count;
             input > 0U;
             --input) {
            apply_encoding(
                carrier,
                (module->input_offset + input - 1U) * CCOUNT,
                module->leaf->input[input - 1U].coefficient,
                1
            );
        }
    }
}

static struct boundary compact_latch_boundary(
    const struct carrier *carrier,
    size_t start
) {
    struct boundary boundary = {
        .hash = UINT64_C(14695981039346656037)
    };
    static const unsigned char name[] = "COMPACT_ROOT";
    boundary.hash = hash_bytes(
        boundary.hash,
        name,
        sizeof(name) - 1U
    );
    for (size_t index = 0U; index < CCOUNT; ++index) {
        double distance = 0.0;
        boundary.coefficient[index] = decode_root(
            relative(carrier, start + index),
            &distance
        );
        if (distance > boundary.maximum_root_error) {
            boundary.maximum_root_error = distance;
        }
        const unsigned char coefficient =
            (unsigned char)boundary.coefficient[index];
        boundary.hash = hash_bytes(
            boundary.hash,
            &coefficient,
            1U
        );
    }
    return boundary;
}

static struct compact_execution compact_execute(
    struct carrier *carrier,
    const struct typed_source *typed,
    const struct compact_layout *layout,
    enum execution_mode mode
) {
    struct carrier borrowed = snapshot_carrier(carrier);
    encode_compact_inputs(carrier, typed, 0);
    apply_compact_forward(carrier, typed, layout, mode);

    double complex boundary_factor[CCOUNT];
    read_poly(
        carrier,
        typed->module[typed->root].final_start,
        typed->module[typed->root].final_transposed,
        boundary_factor
    );
    const size_t public_start = compact_boundary_start(layout);
    apply_factor(carrier, public_start, boundary_factor, 0);
    struct compact_execution execution = {
        .boundary = compact_latch_boundary(carrier, public_start),
        .displacement_l2 = displacement(carrier, &borrowed),
        .omitted_parent_applicable = factor_nontrivial(boundary_factor)
    };
    double complex rotated[CCOUNT];
    for (size_t index = 0U; index < CCOUNT; ++index) {
        rotated[index] = boundary_factor[(index + 1U) % CCOUNT];
        execution.wrong_applicable |=
            cabs(rotated[index] - boundary_factor[index])
                > ROOT_TOLERANCE;
    }
    apply_factor(
        carrier,
        public_start,
        mode == MODE_WRONG_BOUNDARY_INVERSE
            ? rotated
            : boundary_factor,
        1
    );
    apply_compact_inverse(
        carrier,
        typed,
        layout,
        mode,
        mode == MODE_OMITTED_MESSAGE_INVERSE
    );
    encode_compact_inputs(carrier, typed, 1);
    execution.restoration_max_abs = restoration(carrier, &borrowed);
    execution.integrity_max_abs = integrity(carrier);
    free_carrier(&borrowed);
    return execution;
}

static int same_compact_geometry(
    const struct typed_source *left,
    const struct compact_layout *left_layout,
    const struct typed_source *right,
    const struct compact_layout *right_layout
) {
    if (
        left->domain_count != right->domain_count
        || left->module_count != right->module_count
        || left->root != right->root
        || left_layout->input_relations != right_layout->input_relations
        || left_layout->operation_messages
            != right_layout->operation_messages
        || left_layout->carrier_cells != right_layout->carrier_cells
    ) {
        return 0;
    }
    for (size_t index = 0U; index < left->module_count; ++index) {
        if (
            strcmp(left->module[index].name, right->module[index].name) != 0
            || left->module[index].kind != right->module[index].kind
            || left->module[index].left_domain
                != right->module[index].left_domain
            || left->module[index].right_domain
                != right->module[index].right_domain
            || left->module[index].left_child
                != right->module[index].left_child
            || left->module[index].right_child
                != right->module[index].right_child
        ) {
            return 0;
        }
        if (
            left->module[index].kind == MODULE_LEAF
            && (
                left->module[index].leaf->input_count
                    != right->module[index].leaf->input_count
                || left->module[index].leaf->operation_count
                    != right->module[index].leaf->operation_count
            )
        ) {
            return 0;
        }
    }
    return 1;
}

static void print_compact_execution(
    const char *mode,
    const struct typed_source *typed,
    const struct compact_layout *layout,
    const struct compact_execution *execution,
    const struct compact_resource_ledger *resources
) {
    const size_t unique_leaf_sources =
        unique_leaf_source_count(typed);
    const size_t carrier_and_snapshot_complex_values =
        4U * layout->carrier_cells;
    const size_t maximum_native_operator_workspace_complex_values =
        13U * CCOUNT;
    const size_t root_transaction_temporary_complex_values =
        2U * CCOUNT;
    const size_t logical_phase_execution_peak_complex_values =
        carrier_and_snapshot_complex_values
        + maximum_native_operator_workspace_complex_values
        + root_transaction_temporary_complex_values;
    printf(
        "{\"mode\":\"%s\","
        "\"claim\":\"COMPACT_COMPILED_BODY_TYPED_RELATIONAL_MODULE_EXECUTION\","
        "\"module_tree_depth\":%zu,"
        "\"module_descriptors\":%zu,"
        "\"unique_leaf_sources\":%zu,"
        "\"leaf_instances\":%zu,"
        "\"compiled_leaf_definition_reuse\":%s,"
        "\"compact_definition_execution\":true,"
        "\"persistent_per_instance_native_operation_descriptors\":0,"
        "\"persistent_unique_leaf_operation_descriptors\":%zu,"
        "\"persistent_composite_module_descriptors\":%zu,"
        "\"transient_operation_descriptors\":1,"
        "\"transient_operation_descriptor_bytes\":%zu,"
        "\"executed_native_operations\":%zu,"
        "\"phase_resident_operation_messages\":%zu,"
        "\"module_export_copy_cells\":0,"
        "\"decoded_module_coefficients\":0,"
        "\"serialized_module_coefficients\":0,"
        "\"input_relation_count\":%zu,"
        "\"carrier_cells\":%zu,"
        "\"live_carrier_complex_values\":%zu,"
        "\"live_carrier_bytes\":%zu,"
        "\"carrier_and_snapshot_complex_values\":%zu,"
        "\"maximum_native_operator_workspace_complex_values\":%zu,"
        "\"retained_boundary_inverse_factor_sets\":1,"
        "\"retained_boundary_inverse_factor_complex_values\":%u,"
        "\"control_rotation_complex_values\":%u,"
        "\"logical_phase_execution_peak_complex_values\":%zu,"
        "\"logical_phase_execution_peak_bytes\":%zu,"
        "\"coexisting_typed_source_storage_bytes\":%zu,"
        "\"coexisting_leaf_definition_storage_bytes\":%zu,"
        "\"coexisting_layout_storage_bytes\":%zu,"
        "\"boundary_coefficients\":[%d,%d,%d,%d],"
        "\"maximum_root_error\":%.12g,"
        "\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"carrier_integrity_max_abs\":%.12g}\n",
        mode,
        module_depth(typed, typed->root),
        typed->module_count,
        unique_leaf_sources,
        layout->leaf_instances,
        unique_leaf_sources < layout->leaf_instances
            ? "true"
            : "false",
        layout->unique_leaf_operations,
        layout->composite_descriptors,
        sizeof(struct operation),
        layout->operation_messages,
        layout->operation_messages,
        layout->input_relations,
        layout->carrier_cells,
        2U * layout->carrier_cells,
        2U * layout->carrier_cells * sizeof(double complex),
        carrier_and_snapshot_complex_values,
        maximum_native_operator_workspace_complex_values,
        CCOUNT,
        CCOUNT,
        logical_phase_execution_peak_complex_values,
        logical_phase_execution_peak_complex_values
            * sizeof(double complex),
        resources->coexisting_typed_source_storage_bytes,
        resources->coexisting_leaf_definition_storage_bytes,
        resources->coexisting_layout_storage_bytes,
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
    const struct compact_layout layout =
        prepare_compact_layout(&typed);
    struct typed_source reuse_typed =
        argc == 3 ? read_typed_source(argv[2]) : read_typed_source(argv[1]);
    const struct compact_layout reuse_layout =
        prepare_compact_layout(&reuse_typed);
    if (!same_compact_geometry(
        &typed,
        &layout,
        &reuse_typed,
        &reuse_layout
    )) {
        fail("reuse compact process must have identical module geometry");
    }
    const struct compact_resource_ledger resources = {
        .coexisting_typed_source_storage_bytes =
            2U * sizeof(struct typed_source),
        .coexisting_leaf_definition_storage_bytes = (
            unique_leaf_source_count(&typed)
            + unique_leaf_source_count(&reuse_typed)
        ) * sizeof(struct process),
        .coexisting_layout_storage_bytes =
            2U * sizeof(struct compact_layout)
    };

    struct process carrier_shape = {
        .carrier_cells = layout.carrier_cells
    };
    struct carrier carrier = make_carrier(&carrier_shape, 5303);
    const struct compact_execution nominal = compact_execute(
        &carrier,
        &typed,
        &layout,
        MODE_CORRECT
    );
    const struct compact_execution reuse = compact_execute(
        &carrier,
        &reuse_typed,
        &reuse_layout,
        MODE_CORRECT
    );
    free_carrier(&carrier);

    carrier = make_carrier(&carrier_shape, 5303);
    const struct compact_execution wrong = compact_execute(
        &carrier,
        &typed,
        &layout,
        MODE_WRONG_BOUNDARY_INVERSE
    );
    free_carrier(&carrier);
    carrier = make_carrier(&carrier_shape, 5303);
    const struct compact_execution omitted = compact_execute(
        &carrier,
        &typed,
        &layout,
        MODE_OMITTED_MESSAGE_INVERSE
    );
    free_carrier(&carrier);
    carrier = make_carrier(&carrier_shape, 5303);
    const struct compact_execution bypass = compact_execute(
        &carrier,
        &typed,
        &layout,
        MODE_BYPASS_INTERSECTION
    );
    free_carrier(&carrier);
    carrier = make_carrier(&carrier_shape, 5303);
    const struct compact_execution ordinary = compact_execute(
        &carrier,
        &typed,
        &layout,
        MODE_ORDINARY_SUM_INTERSECTION
    );
    free_carrier(&carrier);

    print_compact_execution(
        "compact-typed-module-phase",
        &typed,
        &layout,
        &nominal,
        &resources
    );
    print_compact_execution(
        "actual-restored-cross-program-compact-module-reuse",
        &reuse_typed,
        &reuse_layout,
        &reuse,
        &resources
    );
    print_compact_execution(
        "wrong-boundary-inverse",
        &typed,
        &layout,
        &wrong,
        &resources
    );
    print_compact_execution(
        "omitted-parent-module-inverse",
        &typed,
        &layout,
        &omitted,
        &resources
    );
    print_compact_execution(
        "bypassed-leaf-intersection",
        &typed,
        &layout,
        &bypass,
        &resources
    );
    print_compact_execution(
        "ordinary-sum-leaf-intersection",
        &typed,
        &layout,
        &ordinary,
        &resources
    );
    const int bypass_applicable =
        boundary_differs(&nominal.boundary, &bypass.boundary);
    const int ordinary_applicable =
        boundary_differs(&nominal.boundary, &ordinary.boundary);
    printf(
        "{\"mode\":\"compact-module-control-applicability\","
        "\"wrong_boundary\":%s,"
        "\"omitted_parent\":%s,"
        "\"bypassed_leaf_intersection\":%s,"
        "\"ordinary_sum_leaf_intersection\":%s}\n",
        wrong.wrong_applicable ? "true" : "false",
        omitted.omitted_parent_applicable ? "true" : "false",
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
            !omitted.omitted_parent_applicable
            || omitted.restoration_max_abs >= CONTROL_MINIMUM
        )
    );
    free_typed_source(&typed);
    free_typed_source(&reuse_typed);
    return valid ? 0 : 1;
}
