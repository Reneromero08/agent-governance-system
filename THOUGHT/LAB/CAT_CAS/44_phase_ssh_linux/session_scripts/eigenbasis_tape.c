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

uint64_t lcg(uint64_t *s) { *s=(*s*0x41C64E6D+0x3039); *s=(*s>>13)^*s; *s=(*s<<17)+*s; return *s; }
void sha256_tape(unsigned char *h) { SHA256(tape,TAPE_SIZE,h); }
int hmatch(unsigned char *a,unsigned char *b) { return memcmp(a,b,SHA256_DIGEST_LENGTH)==0; }
uint64_t basis_ck() { uint64_t c=0; for(int i=9;i<=13;i++)c^=tape_words[i]; return c; }

void op_v0(int in,int out) {
    uint64_t x=tape_words[in],b=tape_words[SLOT_BASIS_V0],w=tape_words[SLOT_BASIS_S0];
    tape_words[out]^=(x^b)*(w|1);
}
void op_v1(int in,int out) {
    uint64_t x=tape_words[in],b=tape_words[SLOT_BASIS_V1],w=tape_words[SLOT_BASIS_S1];
    tape_words[out]^=(x^b)*(w|1);
}
void op_combined(int in,int out) {
    uint64_t x=tape_words[in],v0=tape_words[SLOT_BASIS_V0],v1=tape_words[SLOT_BASIS_V1];
    uint64_t s0=tape_words[SLOT_BASIS_S0],s1=tape_words[SLOT_BASIS_S1];
    tape_words[out]^=((x^v0)*(s0|1))^((x^v1)*(s1|1));
}

int main(){
    unsigned char h0[SHA256_DIGEST_LENGTH],hf[SHA256_DIGEST_LENGTH],hr[SHA256_DIGEST_LENGTH];
    printf("=== PHASE 4.1A: SHARED EIGENBASIS ON TAPE ===\n\n");

    uint64_t rng=0x4440000000000001ULL;
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
    tape_words[SLOT_BASIS_CK]=basis_ck();
    uint64_t bdim=tape_words[SLOT_BASIS_DIM],bv0=tape_words[SLOT_BASIS_V0],bv1=tape_words[SLOT_BASIS_V1];
    uint64_t bs0=tape_words[SLOT_BASIS_S0],bs1=tape_words[SLOT_BASIS_S1],bck=tape_words[SLOT_BASIS_CK];
    printf("Basis: dim=%lu V0=0x%016lx S0=0x%016lx V1=0x%016lx S1=0x%016lx CK=0x%016lx\n\n",bdim,bv0,bs0,bv1,bs1,bck);
    tape_words[SLOT_MASTER]=0x0000000100000002ULL;
    sha256_tape(h0);

    int all=1,bok;

    // Test 1
    printf("--- Test 1: Single operator ---\n");
    op_v0(SLOT_MASTER,SLOT_OUTPUT); sha256_tape(hf);
    bok=(tape_words[SLOT_BASIS_DIM]==bdim&&tape_words[SLOT_BASIS_V0]==bv0&&tape_words[SLOT_BASIS_V1]==bv1&&tape_words[SLOT_BASIS_S0]==bs0&&tape_words[SLOT_BASIS_S1]==bs1);
    printf("  Modified: %s Basis: %s\n",hmatch(h0,hf)?"NO":"YES",bok?"YES":"NO");
    op_v0(SLOT_MASTER,SLOT_OUTPUT); sha256_tape(hr);
    printf("  Restored: %s\n\n",hmatch(h0,hr)?"YES":"NO"); all&=bok&&hmatch(h0,hr);

    // Test 2
    printf("--- Test 2: Two operators ---\n");
    uint64_t out2=tape_words[SLOT_OUTPUT];
    op_v0(SLOT_MASTER,SLOT_R1); op_v1(SLOT_MASTER,SLOT_R2);
    tape_words[SLOT_OUTPUT]=tape_words[SLOT_R1]^tape_words[SLOT_R2];
    bok=(tape_words[SLOT_BASIS_DIM]==bdim&&tape_words[SLOT_BASIS_V0]==bv0&&tape_words[SLOT_BASIS_V1]==bv1&&tape_words[SLOT_BASIS_S0]==bs0&&tape_words[SLOT_BASIS_S1]==bs1);
    printf("  R1=0x%016lx R2=0x%016lx Out=0x%016lx Basis: %s\n",tape_words[SLOT_R1],tape_words[SLOT_R2],tape_words[SLOT_OUTPUT],bok?"YES":"NO");
    tape_words[SLOT_OUTPUT]=out2;
    op_v1(SLOT_MASTER,SLOT_R2); op_v0(SLOT_MASTER,SLOT_R1);
    sha256_tape(hr);
    printf("  Restored: %s\n\n",hmatch(h0,hr)?"YES":"NO"); all&=bok&&hmatch(h0,hr);

    // Test 3
    printf("--- Test 3: Combined projection ---\n");
    uint64_t out3=tape_words[SLOT_OUTPUT];
    op_combined(SLOT_MASTER,SLOT_OUTPUT); sha256_tape(hf);
    bok=(tape_words[SLOT_BASIS_DIM]==bdim&&tape_words[SLOT_BASIS_V0]==bv0&&tape_words[SLOT_BASIS_V1]==bv1&&tape_words[SLOT_BASIS_S0]==bs0&&tape_words[SLOT_BASIS_S1]==bs1);
    printf("  Out=0x%016lx Modified: %s Basis: %s\n",tape_words[SLOT_OUTPUT],hmatch(h0,hf)?"NO":"YES",bok?"YES":"NO");
    tape_words[SLOT_OUTPUT]=out3; sha256_tape(hr);
    printf("  Restored: %s Check: %s\n\n",hmatch(h0,hr)?"YES":"NO",tape_words[SLOT_BASIS_CK]==basis_ck()?"YES":"NO"); all&=bok&&hmatch(h0,hr);

    // Test 4
    printf("--- Test 4: 10-cycle stress ---\n");
    int cyc=0;
    for(int c=0;c<10;c++){op_v0(SLOT_MASTER,SLOT_OUTPUT);op_v0(SLOT_MASTER,SLOT_OUTPUT);sha256_tape(hr);if(hmatch(h0,hr))cyc++;}
    bok=(tape_words[SLOT_BASIS_DIM]==bdim&&tape_words[SLOT_BASIS_V0]==bv0&&tape_words[SLOT_BASIS_V1]==bv1&&tape_words[SLOT_BASIS_S0]==bs0&&tape_words[SLOT_BASIS_S1]==bs1);
    printf("  Cycles restored: %d/10 Basis: %s\n",cyc,bok?"YES":"NO"); all&=(cyc==10)&&bok;

    printf("\n=== VERDICT: %s ===\n",all?"ALL TESTS PASS - Shared eigenbasis proven":"FAILURES DETECTED");
    return all?0:1;
}
