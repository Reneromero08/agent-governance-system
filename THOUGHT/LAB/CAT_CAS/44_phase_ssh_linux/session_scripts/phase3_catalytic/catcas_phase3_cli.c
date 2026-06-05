#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "catcas_phase3.h"

uint64_t path_xor(catcas_tape_t *t, void *ud) { int s=*(int*)ud; catcas_xor_bind(t,s,0xFFFFFFFFFFFFFFFFULL); uint64_t sc=catcas_slot_read(t,s); catcas_xor_bind(t,s,0xFFFFFFFFFFFFFFFFULL); return sc; }

int main(int argc, char **argv) {
    printf("=== CATCAS PHASE 3 CLI ===\n\n");
    if (argc < 2) { printf("Usage: %s <test|xor|oracle>\n",argv[0]); return 1; }

    if (strcmp(argv[1],"test")==0) {
        printf("Running operator tests...\n\n");
        catcas_tape_t *t=catcas_tape_init(); unsigned char h[CATCAS_SHA256_LEN];
        catcas_tape_fill_random(t,0xDEAD0001); catcas_tape_snapshot(t,h);
        printf("Initial SHA256: "); catcas_hash_print(h); printf("\n\nOperator tests:\n");

        catcas_xor_bind(t,0,0xCAFE); catcas_xor_bind(t,0,0xCAFE);
        printf("  XOR_BIND: %s\n",catcas_tape_verify(t,h)?"PASS":"FAIL");

        uint64_t o1=catcas_slot_read(t,1); catcas_rotate_left(t,1,13); catcas_rotate_right(t,1,13);
        printf("  ROTATE: %s\n",catcas_slot_read(t,1)==o1?"PASS":"FAIL");

        catcas_phase_tag(t,2,0xABCD); catcas_phase_tag(t,2,0xABCD);
        printf("  PHASE_TAG: %s\n",catcas_tape_verify(t,h)?"PASS":"FAIL");

        catcas_sign_bind(t,3,0xAAAA,0xBBBB); catcas_sign_bind(t,3,0xAAAA,0xBBBB);
        printf("  SIGN_BIND: %s\n",catcas_tape_verify(t,h)?"PASS":"FAIL");

        uint64_t a=catcas_slot_read(t,4),b=catcas_slot_read(t,5);
        catcas_permute_slots(t,4,5); catcas_permute_slots(t,4,5);
        printf("  PERMUTE: %s\n",(catcas_slot_read(t,4)==a&&catcas_slot_read(t,5)==b)?"PASS":"FAIL");

        int ds[]={0,1,2,3}; catcas_checksum_bind(t,7,ds,4); catcas_checksum_bind(t,7,ds,4);
        printf("  CHECKSUM: %s\n\n",catcas_tape_verify(t,h)?"PASS":"FAIL");
        printf("Full tape: %s\n",catcas_tape_verify(t,h)?"PASS":"FAIL");
        catcas_tape_destroy(t);

    } else if (strcmp(argv[1],"xor")==0) {
        catcas_tape_t *t=catcas_tape_init(); unsigned char h[CATCAS_SHA256_LEN]; catcas_tape_snapshot(t,h);
        int s=argc>2?atoi(argv[2]):0; uint64_t v=argc>3?strtoull(argv[3],NULL,16):0xCAFEBABEULL;
        printf("XOR slot %d with 0x%016lx\n",s,v);
        catcas_xor_bind(t,s,v); printf("After forward: 0x%016lx\n",catcas_slot_read(t,s));
        catcas_xor_bind(t,s,v); printf("After reverse: 0x%016lx\n",catcas_slot_read(t,s));
        printf("Restored: %s\n",catcas_tape_verify(t,h)?"YES":"NO");
        catcas_tape_destroy(t);

    } else if (strcmp(argv[1],"oracle")==0) {
        catcas_tape_t *t=catcas_tape_init(); catcas_tape_fill_random(t,0xBEEF0002);
        catcas_slot_write(t,0,100); catcas_slot_write(t,1,50); catcas_slot_write(t,2,200);
        unsigned char h[CATCAS_SHA256_LEN]; catcas_tape_snapshot(t,h);
        printf("Oracle 3 paths: [%lu,%lu,%lu]\n",catcas_slot_read(t,0),catcas_slot_read(t,1),catcas_slot_read(t,2));
        int ss0=0,ss1=1,ss2=2; void *ud[3]={&ss0,&ss1,&ss2};
        catcas_path_fn ps[3]={path_xor,path_xor,path_xor}; int w;
        catcas_oracle_run(t,ps,ud,3,&w);
        printf("Winner: path %d value %lu\n",w,catcas_oracle_get_winner(t));
        catcas_oracle_restore(t);
        printf("Restored: %s\n",catcas_tape_verify(t,h)?"YES":"NO");
        catcas_tape_destroy(t);
    }
    return 0;
}
