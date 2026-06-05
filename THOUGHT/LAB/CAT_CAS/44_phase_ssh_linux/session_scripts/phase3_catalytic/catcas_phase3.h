#ifndef CATCAS_PHASE3_H
#define CATCAS_PHASE3_H

#include <stdint.h>

#define CATCAS_TAPE_SIZE 256
#define CATCAS_MAX_SLOTS (CATCAS_TAPE_SIZE / 8)
#define CATCAS_SHA256_LEN 32

typedef struct catcas_tape_t catcas_tape_t;

catcas_tape_t* catcas_tape_init(void);
void           catcas_tape_destroy(catcas_tape_t *t);
void           catcas_tape_snapshot(catcas_tape_t *t, unsigned char *hash_out);
int            catcas_tape_verify(catcas_tape_t *t, const unsigned char *hash);
void           catcas_tape_fill_random(catcas_tape_t *t, uint64_t seed);

uint64_t catcas_slot_read(catcas_tape_t *t, int slot);
void     catcas_slot_write(catcas_tape_t *t, int slot, uint64_t value);

void catcas_xor_bind(catcas_tape_t *t, int slot, uint64_t value);
void catcas_rotate_left(catcas_tape_t *t, int slot, int bits);
void catcas_rotate_right(catcas_tape_t *t, int slot, int bits);
void catcas_phase_tag(catcas_tape_t *t, int slot, uint64_t phase);
void catcas_sign_bind(catcas_tape_t *t, int slot, uint64_t symbol, uint64_t context);
void catcas_permute_slots(catcas_tape_t *t, int slot_a, int slot_b);
void catcas_checksum_bind(catcas_tape_t *t, int ck_slot, const int *data_slots, int num_slots);

uint64_t catcas_compute_parity(catcas_tape_t *t, const int *slots, int num_slots, int *restored);
uint64_t catcas_compute_hash_fragment(catcas_tape_t *t, const int *slots, int num_slots, int *restored);
uint64_t catcas_compute_fsm_transition(catcas_tape_t *t, int state_slot, int trigger_slot, int *restored);

typedef uint64_t (*catcas_path_fn)(catcas_tape_t *t, void *userdata);
int      catcas_oracle_run(catcas_tape_t *t, catcas_path_fn *paths, void **userdatas, int num_paths, int *winner_out);
uint64_t catcas_oracle_get_winner(catcas_tape_t *t);
void     catcas_oracle_restore(catcas_tape_t *t);

void catcas_hash_print(const unsigned char *hash);

#endif
