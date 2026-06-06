#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>

#define N 8
#define SAMPLES 300
#define TAPE_SIZE 256
#define ITERS 30000

volatile uint64_t *tape __attribute__((aligned(64)));
uint64_t lcg(uint64_t *s){*s=(*s*0x41C64E6D+0x3039);*s=(*s>>13)^*s;*s=(*s<<17)+*s;return *s;}

// ===== CHANNELS =====

// C1: QR orthogonal subspace partition (Exp 13 pattern)
// Core 3 writes to even-indexed tape slots, Core 4 to odd-indexed
// No cross-contamination by construction
void wrk_qr_even(int core,int iters,int *edges,int ne){
    cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(core,&cs);sched_setaffinity(0,sizeof(cs),&cs);
    uint64_t rng=0x1111000000000001ULL+core;
    for(int i=0;i<iters;i++){
        int e=lcg(&rng)%ne,a=edges[e*2],b=edges[e*2+1];
        if(a%2==0){ // even-indexed slot is "mine"
            uint64_t va=__atomic_load_n(&tape[a],__ATOMIC_RELAXED)&1;
            uint64_t vb=__atomic_load_n(&tape[b],__ATOMIC_RELAXED)&1;
            if(va!=vb)__atomic_fetch_xor(&tape[a],1,__ATOMIC_SEQ_CST);
        }
    }
}
void wrk_qr_odd(int core,int iters,int *edges,int ne){
    cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(core,&cs);sched_setaffinity(0,sizeof(cs),&cs);
    uint64_t rng=0x2222000000000001ULL+core;
    for(int i=0;i<iters;i++){
        int e=lcg(&rng)%ne,a=edges[e*2],b=edges[e*2+1];
        if(a%2==1){ // odd-indexed slot is "mine"
            uint64_t va=__atomic_load_n(&tape[a],__ATOMIC_RELAXED)&1;
            uint64_t vb=__atomic_load_n(&tape[b],__ATOMIC_RELAXED)&1;
            if(va!=vb)__atomic_fetch_xor(&tape[a],1,__ATOMIC_SEQ_CST);
        }
    }
}

// C2: Retrocausal 2-pass self-consistency (Exp 23 pattern)
// Pass 1: worker writes "prediction" to future-slot
// Pass 2: worker reads "future" state and self-corrects
void wrk_retro(int core,int iters,int *edges,int ne){
    cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(core,&cs);sched_setaffinity(0,sizeof(cs),&cs);
    uint64_t rng=0x3333000000000001ULL+core;
    for(int p=0;p<2;p++){ // 2-pass convergence (Exp 23: always converges in 2)
        for(int i=0;i<iters/2;i++){
            int e=lcg(&rng)%ne,a=edges[e*2],b=edges[e*2+1];
            uint64_t va=__atomic_load_n(&tape[a],__ATOMIC_RELAXED)&1;
            uint64_t vb=__atomic_load_n(&tape[b],__ATOMIC_RELAXED)&1;
            if(p==0){ // predict: write desired future state
                __atomic_fetch_xor(&tape[20+e],va^vb,__ATOMIC_RELAXED);
            }else{ // correct: read future, align to it
                uint64_t fut=__atomic_load_n(&tape[20+e],__ATOMIC_RELAXED)&1;
                if(va!=fut)__atomic_fetch_xor(&tape[a],1,__ATOMIC_SEQ_CST);
            }
        }
    }
}

// C3: Warm-tape fingerprint contention (Exp 12 pattern)
// Workers touch cache lines based on problem fingerprint
// Different fingerprints = different contention patterns
void wrk_fingerprint(int core,int iters,int *edges,int ne,uint64_t fp){
    cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(core,&cs);sched_setaffinity(0,sizeof(cs),&cs);
    uint64_t rng=0x4444000000000001ULL+core;
    for(int i=0;i<iters;i++){
        int e=lcg(&rng)%ne,a=edges[e*2],b=edges[e*2+1];
        // Access pattern shaped by fingerprint (not J_ij!)
        int slot=((a*b*fp)>>(core*8))&0x1F; // 0-31
        __atomic_fetch_add(&tape[slot],1,__ATOMIC_RELAXED);
        uint64_t va=__atomic_load_n(&tape[a],__ATOMIC_RELAXED)&1;
        uint64_t vb=__atomic_load_n(&tape[b],__ATOMIC_RELAXED)&1;
        if(va!=vb)__atomic_fetch_xor(&tape[a],1,__ATOMIC_SEQ_CST);
    }
}

// C4: Detuned DID frequency coupling
// Core 3 at DID=3 (200MHz), Core 4 at DID=4 (100MHz)
// Tests whether frequency mismatch creates asymmetric update bias
void wrk_did3(int core,int iters,int *edges,int ne){
    cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(core,&cs);sched_setaffinity(0,sizeof(cs),&cs);
    for(int i=0;i<iters;i++){
        int e=i%ne,a=edges[e*2],b=edges[e*2+1];
        uint64_t va=__atomic_load_n(&tape[a],__ATOMIC_RELAXED)&1;
        uint64_t vb=__atomic_load_n(&tape[b],__ATOMIC_RELAXED)&1;
        if(va!=vb)__atomic_fetch_xor(&tape[a],1,__ATOMIC_SEQ_CST);
    }
}

int64_t score(uint64_t *s,int64_t (*J)[N]){
    int64_t E=0;
    for(int i=0;i<N;i++){int si=(s[i]&1)?1:-1;for(int j=i+1;j<N;j++){int sj=(s[j]&1)?1:-1;E-=J[i][j]*si*sj;}}
    return E;
}

void run(const char *l,int mode,int paired,int *rt,int ne,int64_t (*J)[N],int64_t g,int *h,double *a){
    *h=0;*a=0;
    for(int s=0;s<SAMPLES;s++){
        uint64_t rng=0x5555000000000001ULL+s;
        for(int i=0;i<N;i++)tape[i]=lcg(&rng)&1;
        for(int i=8;i<32;i++)tape[i]=0;
        int cores[2]={3,4};uint64_t fp=lcg(&rng);
        if(paired){
            pid_t p1=fork();if(p1==0){
                switch(mode){case 0:wrk_qr_even(3,ITERS,rt,ne);break;case 1:wrk_retro(3,ITERS,rt,ne);break;case 2:wrk_fingerprint(3,ITERS,rt,ne,fp);break;case 3:wrk_did3(3,ITERS,rt,ne);break;}
                _exit(0);}
            pid_t p2=fork();if(p2==0){
                switch(mode){case 0:wrk_qr_odd(4,ITERS,rt,ne);break;case 1:wrk_retro(4,ITERS,rt,ne);break;case 2:wrk_fingerprint(4,ITERS,rt,ne,fp);break;case 3:wrk_did3(4,ITERS,rt,ne);break;}
                _exit(0);}
            waitpid(p1,NULL,0);waitpid(p2,NULL,0);
        }else{
            pid_t p=fork();if(p==0){
                switch(mode){case 0:wrk_qr_even(3,ITERS*2,rt,ne);break;case 1:wrk_retro(3,ITERS*2,rt,ne);break;case 2:wrk_fingerprint(3,ITERS*2,rt,ne,fp);break;case 3:wrk_did3(3,ITERS*2,rt,ne);break;}
                _exit(0);}
            waitpid(p,NULL,0);
        }
        uint64_t fs[N];for(int i=0;i<N;i++)fs[i]=tape[i]&1;
        int64_t E=score(fs,J);*a+=E;if(E==g)(*h)++;
    }
    *a/=SAMPLES;
    printf("  %-35s %3d/%d %8.2f\n",l,*h,SAMPLES,*a);
}

int main(){
    printf("=== PHASE 2B.4: CHANNEL MATRIX ===\n\n");
    printf("4 channels from CAT_CAS lab techniques:\n");
    printf("  C1: QR orthogonal subspace partition (Exp 13)\n");
    printf("  C2: Retrocausal 2-pass self-consistency (Exp 23)\n");
    printf("  C3: Warm-tape fingerprint contention (Exp 12)\n");
    printf("  C4: Detuned DID frequency coupling\n\n");
    printf("CONTAMINATION: Workers NEVER access J_ij or compute energy.\n\n");

    tape=(volatile uint64_t*)mmap(NULL,TAPE_SIZE,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
    int h;double a;

    // Test on 3 problem types
    const char *names[]={"Ferro (J=+1)","Anti-Ferro (J=-1)","Mixed (+1/-1)"};
    for(int pt=0;pt<3;pt++){
        int64_t J[N][N]={{0}};
        int sgn=(pt==1)?-1:1;
        for(int i=0;i<N-1;i++){J[i][i+1]=J[i+1][i]=(pt==2&&i%2)?1*sgn:-1*sgn;}
        int64_t g=-(N-1);
        int edges[]={0,1,1,2,2,3,3,4,4,5,5,6,6,7};int ne=7;

        printf("=== %s (ground=%ld) ===\n",names[pt],g);
        printf("%-35s %8s %8s\n","CHANNEL","HITS","MEAN_E");
        for(int m=0;m<4;m++){
            char b[64];sprintf(b,"C%d:2w shared",m+1);
            run(b,m,1,edges,ne,J,g,&h,&a);
            sprintf(b,"C%d:1w null",m+1);
            run(b,m,0,edges,ne,J,g,&h,&a);
        }
        printf("\n");
    }
    printf("CONTAMINATION: Workers NEVER access J_ij.\n");
    printf("All channels: if 2w shared beats 1w null = candidate.\n");
    return 0;
}
