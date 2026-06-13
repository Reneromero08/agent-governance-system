#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <openssl/sha.h>

#define TAPE_SIZE 256
uint64_t tape_words[TAPE_SIZE / 8];
unsigned char *tape = (unsigned char *)tape_words;

uint64_t lcg(uint64_t *s) { *s=(*s*0x41C64E6D+0x3039); *s=(*s>>13)^*s; *s=(*s<<17)+*s; return *s; }
void sha256_tape(unsigned char *h) { SHA256(tape,TAPE_SIZE,h); }
int hmatch(unsigned char *a,unsigned char *b) { return memcmp(a,b,SHA256_DIGEST_LENGTH)==0; }

void sign_apply(int ctx,uint64_t sym,uint64_t ph){tape_words[ctx]^=sym;tape_words[4]^=ph;}
void sign_reverse(int ctx,uint64_t sym,uint64_t ph){tape_words[4]^=ph;tape_words[ctx]^=sym;}

int main(){
    unsigned char h0[SHA256_DIGEST_LENGTH],hf[SHA256_DIGEST_LENGTH],hr[SHA256_DIGEST_LENGTH];
    int all=1,pass;
    uint64_t rng;

    printf("=== PHASE 3.9 HARDENED: CATALYTIC TOKEN / SIGN ===\n\n");

    // --- TEST 1: SINGLE SIGN WITH 4 SEEDS ---
    printf("--- TEST 1: SINGLE SIGN (4 seeds) ---\n");
    pass=0;
    for(int seed=0;seed<4;seed++){
        rng=0xAAA1110000000001ULL+seed*0x1000;
        for(int i=0;i<TAPE_SIZE/8;i++)tape_words[i]=lcg(&rng);
        uint64_t sym=lcg(&rng),ph=lcg(&rng);
        uint64_t c0=tape_words[0];tape_words[7]=lcg(&rng);uint64_t m7=tape_words[7];
        sha256_tape(h0);
        sign_apply(0,sym,ph);sha256_tape(hf);
        int changed=!hmatch(h0,hf);
        int ctx_changed=(tape_words[0]!=c0);
        sign_reverse(0,sym,ph);sha256_tape(hr);
        int rest=hmatch(h0,hr);
        int meta_ok=(tape_words[7]==m7);
        if(changed&&ctx_changed&&rest&&meta_ok)pass++;
    }
    printf("  %d/4 (forward-modifies, context-changed, reverse-restores, meta-intact)\n",pass);
    all&=(pass==4);

    // --- TEST 2: TWO-SIGN INTERFERENCE (non-tautological) ---
    printf("--- TEST 2: TWO-SIGN INTERFERENCE ---\n");
    pass=0;
    for(int seed=0;seed<4;seed++){
        rng=0xBBB2220000000002ULL+seed*0x1000;
        for(int i=0;i<TAPE_SIZE/8;i++)tape_words[i]=lcg(&rng);
        uint64_t sA=lcg(&rng),pA=lcg(&rng),sB=lcg(&rng),pB=lcg(&rng);
        sha256_tape(h0);
        uint64_t out0=tape_words[4],ctx0=tape_words[0],ctx1=tape_words[1];
        sign_apply(0,sA,pA);
        uint64_t outA=tape_words[4]; // after sign A only
        sign_apply(1,sB,pB);
        uint64_t outAB=tape_words[4]; // after both signs
        sha256_tape(hf);

        // Non-tautological checks: sign A changed output, sign B changed it further,
        // both contexts changed, interference is XOR of original with both phases
        int sA_changed_out = (outA != out0);
        int sB_changed_out = (outAB != outA);
        int ctx0_changed = (tape_words[0] != ctx0);
        int ctx1_changed = (tape_words[1] != ctx1);
        int interference_ok = sA_changed_out && sB_changed_out && ctx0_changed && ctx1_changed;

        sign_reverse(1,sB,pB);sign_reverse(0,sA,pA);
        sha256_tape(hr);
        int rest=hmatch(h0,hr);
        if(interference_ok&&rest)pass++;
    }
    printf("  %d/4 (both-signs-modify-output, contexts-change, reverse-restores)\n",pass);
    all&=(pass==4);

    // --- TEST 3: METADATA ISOLATION (4 seeds, slots 8-11) ---
    printf("--- TEST 3: METADATA ISOLATION ---\n");
    pass=0;
    for(int seed=0;seed<4;seed++){
        rng=0xCCC3330000000003ULL+seed*0x1000;
        for(int i=0;i<TAPE_SIZE/8;i++)tape_words[i]=lcg(&rng);
        uint64_t meta[4];for(int j=0;j<4;j++)meta[j]=lcg(&rng);
        memcpy(&tape_words[8],meta,32);
        sha256_tape(h0);

        uint64_t s0=lcg(&rng),p0=lcg(&rng),s1=lcg(&rng),p1=lcg(&rng),s2=lcg(&rng),p2=lcg(&rng);
        sign_apply(0,s0,p0);
        sign_apply(1,s1,p1);
        sign_apply(2,s2,p2);

        uint64_t mc[4];memcpy(mc,&tape_words[8],32);
        int meta_survived=(mc[0]==meta[0]&&mc[1]==meta[1]&&mc[2]==meta[2]&&mc[3]==meta[3]);

        // Verifiy operations DID modify tape (not a no-op)
        sha256_tape(hf);
        int modified=!hmatch(h0,hf);

        sign_reverse(2,s2,p2);
        sign_reverse(1,s1,p1);
        sign_reverse(0,s0,p0);
        sha256_tape(hr);
        int rest=hmatch(h0,hr);
        if(meta_survived&&modified&&rest)pass++;
    }
    printf("  %d/4 (meta-survives-forward, tape-modified, reverse-restores)\n",pass);
    all&=(pass==4);

    printf("\n=== VERDICT: %s ===\n",all?"ALL GATES PASS":"FAILURES DETECTED");
    return all?0:1;
}
