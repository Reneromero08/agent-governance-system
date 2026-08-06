# Baseline Challenge Report

Status: branch-local obstruction evidence only. Not canonical. Small Wall not crossed.

## Method

The fixed-schema QANF obstruction was reconstructed from the public schema with a symbolic GF(2) polynomial model and a separate exhaustive public-program table. The source-selected baseline was challenged by searching for a compact direct formula over the declared public schema.

## Independent Result

- Public programs checked: 64
- Unique final boundaries: 16
- Compact direct formula count: 4 AND operations
- Raw five-bit table size: 320 bits
- Packed nonconstant table size: 256 bits
- Boundary table hash: `7850283fbf9568610123887cff9b5a8d9f0b106e827f7bcaec88d4f178c770b3`

## Mutation Cross-Check

- Mutation boundary map hash: `b0f14038b537b404b22a0ec18bd1b813f293661c0a7ac468b0f01f20f9886051`
- Equivalent public-program collisions: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 21, 21]
- Maximum public programs per final boundary: 21

## Decision

The obstruction survives the branch-local challenge as a negative law: the in-place carrier can satisfy transaction semantics, but a compact public-schema baseline still dominates for this fixed schema. This prevents false promotion of the source CATVM result and is transferable as an obstruction-testing method, not as a positive non-collapse claim.
