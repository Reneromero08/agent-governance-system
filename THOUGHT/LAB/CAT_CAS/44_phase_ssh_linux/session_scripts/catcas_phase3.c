#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>
#include "catcas_phase3.h"

struct catcas_tape_t {
    unsigned char data[CATCAS_TAPE_SIZE];
    unsigned char snapshot[CATCAS_SHA256_LEN];
    uint64_t work_baseline[8];
    int has_snapshot;
};

static uint64_t lcg(uint64_t *s) { *s=(*s*0x41C64E6D+0x3039); *s=(*s>>13)^*s; *s=(*s<<17)+*s; return *s; }

catcas_tape_t* catcas_tape_init(void) {
    catcas_tape_t *t = calloc(1, sizeof(catcas_tape_t));
    if (t) { memset(t->data, 0xAA, CATCAS_TAPE_SIZE); t->has_snapshot = 0; }
    return t;
}

void catcas_tape_destroy(catcas_tape_t *t) {
    if (t) { memset(t->data, 0, CATCAS_TAPE_SIZE); free(t); }
}

void catcas_tape_snapshot(catcas_tape_t *t, unsigned char *hash_out) {
    SHA256(t->data, CATCAS_TAPE_SIZE, hash_out);
    memcpy(t->snapshot, hash_out, CATCAS_SHA256_LEN);
    t->has_snapshot = 1;
}

int catcas_tape_verify(catcas_tape_t *t, const unsigned char *hash) {
    unsigned char cur[CATCAS_SHA256_LEN];
    SHA256(t->data, CATCAS_TAPE_SIZE, cur);
    return memcmp(cur, hash, CATCAS_SHA256_LEN) == 0;
}

void catcas_tape_fill_random(catcas_tape_t *t, uint64_t seed) {
    uint64_t s = seed;
    for (int i = 0; i < CATCAS_MAX_SLOTS; i++) ((uint64_t*)t->data)[i] = lcg(&s);
    t->has_snapshot = 0;
}

uint64_t catcas_slot_read(catcas_tape_t *t, int slot) { return ((uint64_t*)t->data)[slot]; }
void catcas_slot_write(catcas_tape_t *t, int slot, uint64_t value) { ((uint64_t*)t->data)[slot] = value; }

void catcas_xor_bind(catcas_tape_t *t, int slot, uint64_t value) { ((uint64_t*)t->data)[slot] ^= value; }

void catcas_rotate_left(catcas_tape_t *t, int slot, int bits) {
    uint64_t *p = &((uint64_t*)t->data)[slot]; bits %= 64;
    *p = (*p << bits) | (*p >> (64 - bits));
}

void catcas_rotate_right(catcas_tape_t *t, int slot, int bits) {
    uint64_t *p = &((uint64_t*)t->data)[slot]; bits %= 64;
    *p = (*p >> bits) | (*p << (64 - bits));
}

void catcas_phase_tag(catcas_tape_t *t, int slot, uint64_t phase) { catcas_xor_bind(t, slot, phase); }
void catcas_sign_bind(catcas_tape_t *t, int slot, uint64_t symbol, uint64_t context) { catcas_xor_bind(t, slot, symbol ^ context); }

void catcas_permute_slots(catcas_tape_t *t, int a, int b) {
    uint64_t *p = &((uint64_t*)t->data)[a], *q = &((uint64_t*)t->data)[b], x = *p; *p = *q; *q = x;
}

void catcas_checksum_bind(catcas_tape_t *t, int cs, const int *ds, int n) {
    uint64_t c = 0;
    for (int i = 0; i < n; i++) c ^= catcas_slot_read(t, ds[i]);
    catcas_xor_bind(t, cs, c);
}

uint64_t catcas_compute_parity(catcas_tape_t *t, const int *slots, int n, int *restored) {
    unsigned char h0[CATCAS_SHA256_LEN];
    catcas_tape_snapshot(t, h0);
    uint64_t p = 0;
    for (int i = 0; i < n; i++) p ^= catcas_slot_read(t, slots[i]);
    p &= 1;
    int out = slots[n - 1];
    uint64_t orig = catcas_slot_read(t, out);
    catcas_slot_write(t, out, p);
    catcas_slot_write(t, out, orig);
    *restored = catcas_tape_verify(t, h0);
    return p;
}

uint64_t catcas_compute_hash_fragment(catcas_tape_t *t, const int *slots, int n, int *restored) {
    unsigned char h0[CATCAS_SHA256_LEN];
    catcas_tape_snapshot(t, h0);
    uint64_t m = 0;
    for (int i = 0; i < n; i++) {
        m ^= catcas_slot_read(t, slots[i]);
        m = (m << 17) | (m >> 47);
    }
    int out = slots[n - 1];
    uint64_t orig = catcas_slot_read(t, out);
    catcas_slot_write(t, out, m);
    catcas_slot_write(t, out, orig);
    *restored = catcas_tape_verify(t, h0);
    return m;
}

uint64_t catcas_compute_fsm_transition(catcas_tape_t *t, int state_slot, int trig_slot, int *restored) {
    unsigned char h0[CATCAS_SHA256_LEN];
    catcas_tape_snapshot(t, h0);
    uint64_t st = catcas_slot_read(t, state_slot);
    uint64_t tr = catcas_slot_read(t, trig_slot);
    uint64_t ns = st ^ tr;
    catcas_slot_write(t, state_slot, ns);
    catcas_slot_write(t, state_slot, st);
    *restored = catcas_tape_verify(t, h0);
    return ns;
}

int catcas_oracle_run(catcas_tape_t *t, catcas_path_fn *paths, void **userdatas, int n, int *winner) {
    for (int i = 0; i < 8; i++) t->work_baseline[i] = catcas_slot_read(t, i);
    catcas_slot_write(t, 5, UINT64_MAX);
    catcas_slot_write(t, 4, 0);
    int best = -1; uint64_t bs = UINT64_MAX;
    for (int p = 0; p < n; p++) {
        uint64_t sc = paths[p](t, userdatas ? userdatas[p] : NULL);
        if (sc < bs) { bs = sc; best = p; catcas_slot_write(t, 4, sc); catcas_slot_write(t, 5, sc); }
        for (int i = 0; i < 4; i++) if (catcas_slot_read(t, i) != t->work_baseline[i]) return -1;
    }
    if (winner) *winner = best;
    return 0;
}

uint64_t catcas_oracle_get_winner(catcas_tape_t *t) { return catcas_slot_read(t, 4); }
void catcas_oracle_restore(catcas_tape_t *t) {
    for (int i = 4; i < 8; i++) catcas_slot_write(t, i, t->work_baseline[i]);
}

void catcas_hash_print(const unsigned char *h) { for (int i = 0; i < 8; i++) printf("%02x", h[i]); printf("..."); }
