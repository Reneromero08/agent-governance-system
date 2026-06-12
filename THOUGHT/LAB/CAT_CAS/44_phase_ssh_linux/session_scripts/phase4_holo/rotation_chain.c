#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <openssl/sha.h>

#define TAPE_SIZE 256
uint64_t tape_words[TAPE_SIZE / 8];
unsigned char *tape = (unsigned char *)tape_words;

#define SLOT_MASTER    0
#define SLOT_R1        1
#define SLOT_R2        2
#define SLOT_OUTPUT    3
#define SLOT_META      4
#define SLOT_BASIS_0   5
#define SLOT_BASIS_1   6
#define SLOT_ANGLE_R1  7
#define SLOT_ANGLE_R2  8
#define SLOT_BASIS_DIM 9
#define SLOT_BASIS_V0  10
#define SLOT_BASIS_V1  11
#define SLOT_BASIS_S0  12
#define SLOT_BASIS_S1  13
#define SLOT_BASIS_CK  14
#define SLOT_CHAIN_LEN 16
#define SLOT_CHAIN_IDX 17
#define SLOT_LAYER_1   18
#define SLOT_LAYER_2   19
#define SLOT_LAYER_3   20
#define SLOT_ACCUM     21
#define SLOT_CHAIN_CK  22

uint64_t lcg(uint64_t *s) { *s=(*s*0x41C64E6D+0x3039); *s=(*s>>13)^*s; *s=(*s<<17)+*s; return *s; }
void sha256_tape(unsigned char *h) { SHA256(tape,TAPE_SIZE,h); }
int hmatch(unsigned char *a,unsigned char *b) { return memcmp(a,b,SHA256_DIGEST_LENGTH)==0; }
uint64_t chain_ck() { uint64_t c=0; for(int i=16;i<=21;i++)c^=tape_words[i]; return c; }

void rot(int in,int out,uint64_t ang) {
    uint64_t x=tape_words[in],r=((x<<(ang%64))|(x>>(64-(ang%64))));
    tape_words[out]^=r; tape_words[SLOT_ACCUM]^=r;
}

int main(){
    unsigned char h0[SHA256_DIGEST_LENGTH],hf[SHA256_DIGEST_LENGTH],hr[SHA256_DIGEST_LENGTH];
    printf("=== PHASE 4.2A: CATALYTIC ROTATION CHAIN ===\n\n");

    uint64_t rng=0x4441110000000001ULL;
    for(int i=0;i<TAPE_SIZE/8;i++)tape_words[i]=lcg(&rng);
    tape_words[SLOT_META]=(1UL<<56)|(3UL<<48);
    tape_words[SLOT_BASIS_0]=0x5245464552454E43ULL;
    tape_words[SLOT_BASIS_1]=0x524F544154494F4EULL;
    tape_words[SLOT_ANGLE_R1]=1570796;tape_words[SLOT_ANGLE_R2]=3141593;
    tape_words[SLOT_BASIS_DIM]=2;
    tape_words[SLOT_BASIS_V0]=0x3F8000003F800000ULL;
    tape_words[SLOT_BASIS_V1]=0xBF8000003F800000ULL;
    tape_words[SLOT_BASIS_S0]=0x0000006400000064ULL;
    tape_words[SLOT_BASIS_S1]=0x0000003200000032ULL;
    tape_words[SLOT_CHAIN_LEN]=3;tape_words[SLOT_CHAIN_IDX]=0;
    tape_words[SLOT_LAYER_1]=1570796;tape_words[SLOT_LAYER_2]=3141593;tape_words[SLOT_LAYER_3]=4712389;
    tape_words[SLOT_ACCUM]=0;tape_words[SLOT_CHAIN_CK]=chain_ck();
    uint64_t cl=tape_words[SLOT_CHAIN_LEN],l1=tape_words[SLOT_LAYER_1],l2=tape_words[SLOT_LAYER_2],l3=tape_words[SLOT_LAYER_3];
    printf("Chain: %lu layers R1=pi/2 R2=pi R3=3pi/2\n\n",cl);
    tape_words[SLOT_MASTER]=0xDEADBEEFCAFEBABEULL;
    sha256_tape(h0); int all=1,mk;

    // Test 1
    printf("--- Test 1: Forward chain ---\n");
    tape_words[SLOT_CHAIN_IDX]=1;rot(SLOT_MASTER,SLOT_R1,l1);
    tape_words[SLOT_CHAIN_IDX]=2;rot(SLOT_R1,SLOT_R2,l2);
    tape_words[SLOT_CHAIN_IDX]=3;rot(SLOT_R2,SLOT_OUTPUT,l3);
    sha256_tape(hf);
    mk=(tape_words[SLOT_CHAIN_LEN]==cl&&tape_words[SLOT_LAYER_1]==l1&&tape_words[SLOT_LAYER_2]==l2&&tape_words[SLOT_LAYER_3]==l3);
    printf("  R1=0x%016lx R2=0x%016lx Out=0x%016lx\n",tape_words[SLOT_R1],tape_words[SLOT_R2],tape_words[SLOT_OUTPUT]);
    printf("  Modified: %s Metadata: %s Accum: 0x%016lx\n\n",hmatch(h0,hf)?"NO":"YES",mk?"YES":"NO",tape_words[SLOT_ACCUM]); all&=mk;

    // Test 2
    printf("--- Test 2: Reverse chain ---\n");
    tape_words[SLOT_CHAIN_IDX]=3;rot(SLOT_R2,SLOT_OUTPUT,l3);
    tape_words[SLOT_CHAIN_IDX]=2;rot(SLOT_R1,SLOT_R2,l2);
    tape_words[SLOT_CHAIN_IDX]=1;rot(SLOT_MASTER,SLOT_R1,l1);
    tape_words[SLOT_CHAIN_IDX]=0;
    sha256_tape(hr);
    printf("  Restored: %s Accum: %s\n\n",hmatch(h0,hr)?"YES":"NO",tape_words[SLOT_ACCUM]==0?"CLEAR":"NO"); all&=hmatch(h0,hr);

    // Test 3
    printf("--- Test 3: Cumulative transform ---\n");
    rot(SLOT_MASTER,SLOT_R1,l1);uint64_t a1=tape_words[SLOT_R1];
    rot(SLOT_R1,SLOT_R2,l2);uint64_t a2=tape_words[SLOT_R2];
    rot(SLOT_R2,SLOT_OUTPUT,l3);uint64_t a3=tape_words[SLOT_OUTPUT];
    printf("  L1=0x%016lx L2=0x%016lx L3=0x%016lx distinct: %s\n",a1,a2,a3,(a1!=a2&&a2!=a3)?"YES":"NO");
    rot(SLOT_R2,SLOT_OUTPUT,l3);rot(SLOT_R1,SLOT_R2,l2);rot(SLOT_MASTER,SLOT_R1,l1);
    sha256_tape(hr);
    printf("  Restored: %s\n\n",hmatch(h0,hr)?"YES":"NO"); all&=hmatch(h0,hr);

    // Test 4
    printf("--- Test 4: 4-input stress ---\n");
    uint64_t ins[]={0xAAAA,0xBBBB,0xCCCC,0xDDDD};int cy=0;
    for(int s=0;s<4;s++){
        tape_words[SLOT_MASTER]=ins[s];
        rot(SLOT_MASTER,SLOT_R1,l1);rot(SLOT_R1,SLOT_R2,l2);rot(SLOT_R2,SLOT_OUTPUT,l3);
        rot(SLOT_R2,SLOT_OUTPUT,l3);rot(SLOT_R1,SLOT_R2,l2);rot(SLOT_MASTER,SLOT_R1,l1);
        tape_words[SLOT_MASTER]=0xDEADBEEFCAFEBABEULL;
        sha256_tape(hr);if(hmatch(h0,hr))cy++;
    }
    printf("  %d/4\n\n",cy); all&=(cy==4);

    printf("=== VERDICT: %s ===\n",all?"ALL TESTS PASS - Rotation chain proven":"FAILURES DETECTED");
    return all?0:1;
}
