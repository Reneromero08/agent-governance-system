/*
 * catalytic_tape_lifecycle.c -- Exp 50 L2 Gate: Catalytic Tape Lifecycle.
 *
 * Hardened: per-trial CSV logging, explicit identity no-op control,
 * compute checksum, deterministic replay with checksum comparison.
 *
 * Build: gcc -O2 -Wall -Wextra catalytic_tape_lifecycle.c -o tape_lifecycle -lssl -lcrypto
 * Run:   ./tape_lifecycle --seeds 10 --tape-kb 256 --trials 5 --csv results.csv
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <openssl/sha.h>

static uint64_t rng_s;
static uint64_t rng64(void) { uint64_t x=rng_s; x^=x>>12; x^=x<<25; x^=x>>27; rng_s=x; return x*0x2545F4914F6CDD1DULL; }
static void fill_tape(uint8_t *t, size_t n, uint64_t s) { rng_s=s|1; for(size_t i=0;i<n;i++)t[i]=(uint8_t)(rng64()&0xFF); }
static void tape_sha256(const uint8_t *t, size_t n, uint8_t h[32]) { SHA256(t,n,h); }
static void xor_mask(uint8_t *t, size_t n, uint64_t s) { rng_s=s|1; for(size_t i=0;i<n;i++)t[i]^=(uint8_t)(rng64()&0xFF); }
static int hash_eq(const uint8_t *a, const uint8_t *b) { return memcmp(a,b,32)==0; }
static void hash_hex(const uint8_t h[32], char o[65]) { for(int i=0;i<32;i++)sprintf(o+i*2,"%02x",h[i]); o[64]=0; }

/* Reversible compute: byte rotation + XOR. Checksum = sum of tape bytes after compute. */
static uint64_t compute_phase(uint8_t *t, size_t n) {
    uint64_t cs = 0;
    for (size_t i=0;i<n;i++) { uint8_t b=t[i]; t[i]=(b<<((i&3)+1))|(b>>(8-((i&3)+1))); cs+=t[i]; }
    return cs;
}
static uint64_t compute_inverse(uint8_t *t, size_t n) {
    uint64_t cs = 0;
    for (size_t i=0;i<n;i++) { uint8_t b=t[i]; int sh=(int)((i&3)+1); t[i]=(b>>sh)|(b<<(8-sh)); cs+=t[i]; }
    return cs;
}

int main(int argc, char **argv) {
    int ns=10, tkb=256, nt=5; uint64_t bs=42; const char *csv="results.csv";
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--seeds")&&i+1<argc)ns=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--tape-kb")&&i+1<argc)tkb=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--trials")&&i+1<argc)nt=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--seed")&&i+1<argc)bs=(uint64_t)atol(argv[++i]);
        else if(!strcmp(argv[i],"--csv")&&i+1<argc)csv=argv[++i];
    }
    size_t n=(size_t)tkb*1024;
    uint8_t *tape=malloc(n), *tape2=malloc(n);
    if(!tape||!tape2){fprintf(stderr,"malloc\n");return 1;}

    FILE *f=fopen(csv,"w");
    fprintf(f,"trial_id,mode,seed,tape_kb,before_hash,after_hash,restore_pass,compute_cs,replay_cs_match\n");

    printf("EXP50 L2 TAPE LIFECYCLE seeds=%d tape_kb=%d trials=%d\n",ns,tkb,nt);

    int total=0, restore_ok=0, replay_ok=0, wrong_ok=0, identity_ok=0;

    for(int s=0;s<ns;s++) {
        uint64_t sd=bs+(uint64_t)s*1000;
        for(int t=0;t<nt;t++) {
            uint64_t tid=sd+(uint64_t)t; total++;

            /* --- NORMAL RESTORE MODE --- */
            uint8_t hb[32], ha[32];
            fill_tape(tape,n,tid);
            tape_sha256(tape,n,hb);
            uint64_t ms=tid^0xA5A5A5A5A5A5A5A5ULL;
            xor_mask(tape,n,ms);
            uint64_t cs=compute_phase(tape,n);
            uint64_t cs_inv=compute_inverse(tape,n);
            xor_mask(tape,n,ms);
            tape_sha256(tape,n,ha);
            int ok=hash_eq(hb,ha);
            char hbs[65],has[65]; hash_hex(hb,hbs); hash_hex(ha,has);
            if(ok)restore_ok++;

            /* Deterministic replay */
            fill_tape(tape2,n,tid);
            uint8_t h2b[32],h2a[32];
            tape_sha256(tape2,n,h2b);
            xor_mask(tape2,n,ms);
            uint64_t cs2=compute_phase(tape2,n);
            uint64_t cs2_inv=compute_inverse(tape2,n);
            xor_mask(tape2,n,ms);
            tape_sha256(tape2,n,h2a);
            int rp_ok=hash_eq(h2b,h2a)&&hash_eq(hb,h2b)&&(cs==cs2)&&(cs_inv==cs2_inv);
            if(rp_ok)replay_ok++;

            fprintf(f,"%d,normal,%llu,%d,%s,%s,%s,%llu,%s\n",
                    total,(unsigned long long)tid,tkb,hbs,has,ok?"PASS":"FAIL",
                    (unsigned long long)cs,rp_ok?"PASS":"FAIL");

            /* Wrong-mask negative */
            fill_tape(tape,n,tid);
            tape_sha256(tape,n,hb);
            xor_mask(tape,n,ms);
            compute_phase(tape,n);
            compute_inverse(tape,n);
            xor_mask(tape,n,ms^0xDEADBEEFULL);
            tape_sha256(tape,n,ha);
            int wm=!hash_eq(hb,ha);
            if(wm)wrong_ok++;
            fprintf(f,"%d,wrong_mask,%llu,%d,%s,%s,%s,0,NA\n",
                    total,(unsigned long long)tid,tkb,hbs,has,wm?"FAIL_OK":"UNEXPECTED_PASS");

            /* Identity no-op */
            fill_tape(tape,n,tid);
            tape_sha256(tape,n,hb);
            tape_sha256(tape,n,ha);
            int id=hash_eq(hb,ha);
            if(id)identity_ok++;
            fprintf(f,"%d,identity,%llu,%d,%s,%s,%s,0,NA\n",
                    total,(unsigned long long)tid,tkb,hbs,has,id?"PASS":"FAIL");
        }
    }

    fclose(f);
    printf("\n=== RESULTS ===\n");
    printf("total_trials: %d\n",total);
    printf("restore_pass: %d/%d\n",restore_ok,total);
    printf("replay_pass: %d/%d\n",replay_ok,total);
    printf("wrong_mask_pass: %d/%d\n",wrong_ok,total);
    printf("identity_pass: %d/%d\n",identity_ok,total);
    int all=restore_ok==total&&replay_ok==total&&wrong_ok==total&&identity_ok==total;
    printf("VERDICT: %s\n",all?"L2_TAPE_LIFECYCLE_PASS_HARDENED":"L2_TAPE_LIFECYCLE_FAIL");
    printf("wrote %s\n",csv);
    free(tape);free(tape2);
    return all?0:1;
}
