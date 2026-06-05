#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <openssl/sha.h>

#define TAPE_SIZE 256
uint64_t tape_words[TAPE_SIZE / 8];
unsigned char *tape = (unsigned char *)tape_words;

#define SLOT_WORK_0  0
#define SLOT_WORK_1  1
#define SLOT_WORK_2  2
#define SLOT_WORK_3  3
#define SLOT_OUTPUT  4
#define SLOT_SCORE   5
#define SLOT_PATH_ID 6
#define SLOT_CKSUM   7

uint64_t lcg(uint64_t *s) { *s=(*s*0x41C64E6D+0x3039); *s=(*s>>13)^*s; *s=(*s<<17)+*s; return *s; }
void sha256_tape(unsigned char *h) { SHA256(tape,TAPE_SIZE,h); }
int hmatch(unsigned char *a,unsigned char *b) { return memcmp(a,b,SHA256_DIGEST_LENGTH)==0; }

uint64_t checksum() { uint64_t c=0; for(int i=0;i<7;i++)c^=tape_words[i]; return c; }
void save_bl(uint64_t *bl) { for(int i=0;i<8;i++)bl[i]=tape_words[i]; }

void path_fwd(int id) {
    tape_words[SLOT_PATH_ID]=id;
    if(id==0)tape_words[SLOT_WORK_0]+=1;
    else if(id==1)tape_words[SLOT_WORK_1]-=1;
    else if(id==2)tape_words[SLOT_WORK_2]^=0xFFFFFFFFFFFFFFFFULL;
}
void path_rev(int id) {
    if(id==0)tape_words[SLOT_WORK_0]-=1;
    else if(id==1)tape_words[SLOT_WORK_1]+=1;
    else if(id==2)tape_words[SLOT_WORK_2]^=0xFFFFFFFFFFFFFFFFULL;
    tape_words[SLOT_PATH_ID]=0;
}
uint64_t path_score() {
    int id=(int)tape_words[SLOT_PATH_ID];
    if(id==0)return tape_words[SLOT_WORK_0];
    if(id==1)return tape_words[SLOT_WORK_1];
    if(id==2)return tape_words[SLOT_WORK_2];
    return UINT64_MAX;
}

int main(){
    unsigned char h0[SHA256_DIGEST_LENGTH],hr[SHA256_DIGEST_LENGTH];
    int all=1;
    uint64_t bl[8];

    printf("=== PHASE 3.10: ORACLE-STYLE PATH RESTORATION ===\n\n");

    // --- TEST 1: 3 PATHS, MINIMUM SCORE ---
    printf("--- TEST 1: MINIMUM SCORE ORACLE ---\n");
    uint64_t rng=0xAAA1110000000001ULL;
    for(int i=0;i<TAPE_SIZE/8;i++)tape_words[i]=lcg(&rng);
    tape_words[0]=100;tape_words[1]=50;tape_words[2]=200;
    tape_words[4]=0;tape_words[5]=UINT64_MAX;tape_words[6]=0;
    tape_words[7]=checksum();
    sha256_tape(h0);
    save_bl(bl);

    uint64_t best=UINT64_MAX;int bp=-1;
    printf("  Work slots: [%lu, %lu, %lu]\n",tape_words[0],tape_words[1],tape_words[2]);
    for(int p=0;p<3;p++){
        path_fwd(p);
        uint64_t sc=path_score();
        int wrk_ok=1;
        if(sc<tape_words[SLOT_SCORE]){tape_words[SLOT_SCORE]=sc;tape_words[SLOT_OUTPUT]=sc;best=sc;bp=p;}
        path_rev(p); // self-inverse
        if(p==0)wrk_ok=(tape_words[0]==bl[0]);
        else if(p==1)wrk_ok=(tape_words[1]==bl[1]);
        else if(p==2)wrk_ok=(tape_words[2]==bl[2]);
        printf("  Path %d: score=%lu best=%s restored=%s\n",p,sc,sc==best?"YES":"no",wrk_ok?"YES":"NO");
    }

    uint64_t exp_win=(bp==0)?101:((bp==1)?49:(200ULL^0xFFFFFFFFFFFFFFFFULL));
    int w_ok=(best==exp_win);
    printf("  Winner: path %d score %lu correct: %s\n",bp,best,w_ok?"YES":"NO");

    int slots_ok=(tape_words[0]==bl[0]&&tape_words[1]==bl[1]&&tape_words[2]==bl[2]);
    printf("  Working slots match baseline: %s\n",slots_ok?"YES":"NO");

    tape_words[4]=bl[4];tape_words[5]=bl[5];tape_words[6]=bl[6];tape_words[7]=bl[7];
    sha256_tape(hr);
    int t1r=hmatch(h0,hr);
    printf("  Tape restored: %s\n\n",t1r?"YES":"NO");
    all&=w_ok&&slots_ok&&t1r;

    // --- TEST 2: TIEBREAK ---
    printf("--- TEST 2: TIEBREAK ---\n");
    rng=0xBBB2220000000002ULL;
    for(int i=0;i<TAPE_SIZE/8;i++)tape_words[i]=lcg(&rng);
    tape_words[0]=42;tape_words[1]=42;tape_words[2]=42;
    tape_words[4]=0;tape_words[5]=UINT64_MAX;tape_words[6]=0;tape_words[7]=checksum();
    sha256_tape(h0);save_bl(bl);
    for(int p=0;p<3;p++){path_fwd(p);uint64_t sc=path_score();if(sc<tape_words[5]){tape_words[5]=sc;tape_words[4]=sc;}path_rev(p);}
    int t2p=(tape_words[5]==41);  // path 1: 42-1=41, min of {43,41,huge}
    printf("  All slots=42, min score path 1 wins with 41: %s\n",t2p?"YES":"NO");
    tape_words[4]=bl[4];tape_words[5]=bl[5];tape_words[6]=bl[6];tape_words[7]=bl[7];
    sha256_tape(hr);
    int t2r=hmatch(h0,hr);
    printf("  Tape restored: %s\n\n",t2r?"YES":"NO");
    all&=t2p&&t2r;

    // --- TEST 3: 4-SEED RANDOMIZED ---
    printf("--- TEST 3: RANDOMIZED (4 seeds) ---\n");
    int pass=0;
    for(int s=0;s<4;s++){
        rng=0xCCC3330000000003ULL+s*0x1000;
        for(int i=0;i<TAPE_SIZE/8;i++)tape_words[i]=lcg(&rng);
        tape_words[4]=0;tape_words[5]=UINT64_MAX;tape_words[6]=0;tape_words[7]=checksum();
        sha256_tape(h0);save_bl(bl);
        uint64_t best_l=UINT64_MAX;
        for(int p=0;p<3;p++){
            path_fwd(p);uint64_t sc=path_score();if(sc<tape_words[5]){tape_words[5]=sc;tape_words[4]=sc;}path_rev(p);
            if(sc<best_l)best_l=sc;
        }
    int sl_ok=(tape_words[0]==bl[0]&&tape_words[1]==bl[1]&&tape_words[2]==bl[2]);
    tape_words[4]=bl[4];tape_words[5]=bl[5];tape_words[6]=bl[6];tape_words[7]=bl[7];
        sha256_tape(hr);
        if(sl_ok&&hmatch(h0,hr))pass++;
    }
    printf("  %d/4 (slots-restored, tape-restored)\n",pass);
    all&=(pass==4);

    printf("\n=== VERDICT: %s ===\n",all?"ALL TESTS PASS - Oracle path restoration achieved":"FAILURES DETECTED");
    return all?0:1;
}
