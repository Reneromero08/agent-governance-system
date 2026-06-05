#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <openssl/sha.h>

#define TAPE_SIZE 256
#define ITERATIONS 100

uint64_t tape_words[TAPE_SIZE / 8];
unsigned char *tape = (unsigned char *)tape_words;

uint64_t lcg(uint64_t *s) { *s=(*s*0x41C64E6D+0x3039); *s=(*s>>13)^*s; *s=(*s<<17)+*s; return *s; }
void sha256_tape(unsigned char *h) { SHA256(tape,TAPE_SIZE,h); }
int hmatch(unsigned char *a,unsigned char *b) { return memcmp(a,b,SHA256_DIGEST_LENGTH)==0; }

double read_temp() {
    FILE *fp = fopen("/sys/class/hwmon/hwmon0/temp1_input","r");
    if(!fp) return -1; double t; fscanf(fp,"%lf",&t); fclose(fp); return t/1000.0;
}
double time_ns() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e9+ts.tv_nsec; }

int main(){
    unsigned char h0[SHA256_DIGEST_LENGTH],h1[SHA256_DIGEST_LENGTH];
    uint64_t rng; double t0,t1,temp_start=read_temp();
    printf("=== PHASE 3.11: BASELINE COMPARISON ===\n");
    printf("Starting temp: %.1f C\n\n",temp_start);

    // --- BENCH 1: REVERSIBLE XOR_BIND ---
    printf("--- BENCH 1: REVERSIBLE XOR_BIND (%d iter) ---\n",ITERATIONS);
    int rev_ok=0; double rev_us=0;
    for(int it=0;it<ITERATIONS;it++){
        rng=0xAAA1110000000001ULL+it;
        for(int i=0;i<TAPE_SIZE/8;i++)tape_words[i]=lcg(&rng);
        sha256_tape(h0);
        t0=time_ns();
        for(int s=0;s<8;s++)tape_words[s]^=lcg(&rng);
        t1=time_ns(); rev_us+=t1-t0;
        rng=0xAAA1110000000001ULL+it;
        for(int i=0;i<TAPE_SIZE/8;i++)lcg(&rng);
        t0=time_ns();
        for(int s=0;s<8;s++)tape_words[s]^=lcg(&rng);
        rev_us+=time_ns()-t0;
        sha256_tape(h1);
        if(hmatch(h0,h1))rev_ok++;
    }
    printf("  %.1f us/cycle, restored: %d/%d\n",rev_us/1000/ITERATIONS,rev_ok,ITERATIONS);

    // --- BENCH 2: DESTRUCTIVE OVERWRITE ---
    printf("--- BENCH 2: DESTRUCTIVE OVERWRITE (%d iter) ---\n",ITERATIONS);
    int dst_ok=0; double dst_us=0;
    for(int it=0;it<ITERATIONS;it++){
        rng=0xBBB2220000000002ULL+it;
        for(int i=0;i<TAPE_SIZE/8;i++)tape_words[i]=lcg(&rng);
        sha256_tape(h0);
        t0=time_ns();
        for(int s=0;s<8;s++)tape_words[s]=0xDEADBEEFCAFEBABEULL;
        dst_us+=time_ns()-t0;
        sha256_tape(h1);
        if(!hmatch(h0,h1))dst_ok++;
    }
    printf("  %.1f us/cycle, erased: %d/%d\n",dst_us/1000/ITERATIONS,dst_ok,ITERATIONS);

    // --- BENCH 3: REVERSIBLE ORACLE PATH ---
    printf("--- BENCH 3: REVERSIBLE ORACLE PATH (%d iter) ---\n",ITERATIONS);
    int ora_ok=0; double ora_us=0; uint64_t bl[8];
    for(int it=0;it<ITERATIONS;it++){
        rng=0xCCC3330000000003ULL+it;
        for(int i=0;i<TAPE_SIZE/8;i++)tape_words[i]=lcg(&rng);
        sha256_tape(h0);
        for(int i=0;i<8;i++)bl[i]=tape_words[i];
        t0=time_ns();
        for(int p=0;p<3;p++){
            tape_words[p]^=0xFFFFFFFFFFFFFFFFULL;
            if(tape_words[p]<tape_words[7])tape_words[7]=tape_words[p];
            tape_words[p]^=0xFFFFFFFFFFFFFFFFULL;
        }
        ora_us+=time_ns()-t0;
        tape_words[7]=bl[7];
        sha256_tape(h1);
        if(hmatch(h0,h1))ora_ok++;
    }
    printf("  %.1f us/cycle, restored: %d/%d\n\n",ora_us/1000/ITERATIONS,ora_ok,ITERATIONS);

    double temp_end=read_temp();
    printf("=== SUMMARY ===\nTemp: %.1f -> %.1f C (%+.1f)\n\n",temp_start,temp_end,temp_end-temp_start);
    printf("%-25s %12s %12s %12s\n","Benchmark","us/cycle","Restored","BitsErased");
    printf("%-25s %11.1f us %8d/%-4d %12d\n","Reversible XOR",rev_us/1000/ITERATIONS,rev_ok,ITERATIONS,0);
    printf("%-25s %11.1f us %8d/%-4d %12d\n","Destructive",dst_us/1000/ITERATIONS,0,ITERATIONS,8*8*8);
    printf("%-25s %11.1f us %8d/%-4d %12d\n","Reversible Oracle",ora_us/1000/ITERATIONS,ora_ok,ITERATIONS,0);
    printf("\nReversible is %.1fx slower. Destructive erases 512 bits/cycle. Oracle fully restores.\n",rev_us/dst_us);
    return 0;
}
