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

#define NUM_SPINS 8
#define SAMPLES 200
#define TAPE_SIZE 256
#define ITERS 50000

volatile uint64_t *tape __attribute__((aligned(64)));
uint64_t lcg(uint64_t *s) { *s=(*s*0x41C64E6D+0x3039); *s=(*s>>13)^*s; *s=(*s<<17)+*s; return *s; }

static int *route_signs;
void worker_p1(int core,int iters,int *route,int rlen){
    cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(core,&cs);sched_setaffinity(0,sizeof(cs),&cs);
    uint64_t rng=0x1111000000000001ULL+core;
    for(int i=0;i<iters;i++){
        int ri=lcg(&rng)%rlen,si=route[ri*2],sj=route[ri*2+1];
        uint64_t vi=__atomic_load_n(&tape[si],__ATOMIC_RELAXED)&1;
        uint64_t vj=__atomic_load_n(&tape[sj],__ATOMIC_RELAXED)&1;
        // P1: if different, flip to align (ferromagnetic bias ONLY)
        if(vi!=vj)__atomic_fetch_xor(&tape[si],1,__ATOMIC_SEQ_CST);
    }
}
void worker_p2(int core,int iters,int *route,int rlen){
    cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(core,&cs);sched_setaffinity(0,sizeof(cs),&cs);
    uint64_t rng=0x2222000000000001ULL+core;
    for(int i=0;i<iters;i++){
        int ri=lcg(&rng)%rlen,si=route[ri*2],sj=route[ri*2+1],sgn=route_signs[ri];
        uint64_t vi=__atomic_load_n(&tape[si],__ATOMIC_RELAXED)&1;
        uint64_t vj=__atomic_load_n(&tape[sj],__ATOMIC_RELAXED)&1;
        int sat=(vi==vj)?(sgn>0):(sgn<0);
        if(!sat)__atomic_fetch_xor(&tape[si],1,__ATOMIC_SEQ_CST);
    }
}
int64_t score(uint64_t *s,int64_t (*J)[NUM_SPINS]){
    int64_t E=0;
    for(int i=0;i<NUM_SPINS;i++){int si=(s[i]&1)?1:-1;for(int j=i+1;j<NUM_SPINS;j++){int sj=(s[j]&1)?1:-1;E-=J[i][j]*si*sj;}}
    return E;
}
void run(const char *l,void (*w)(int,int,int*,int),int *rt,int rl,int64_t (*J)[NUM_SPINS],int64_t g,int *h,double *a,int p){
    *h=0;*a=0;
    for(int s=0;s<SAMPLES;s++){
        uint64_t rng=0x3333000000000001ULL+s;
        for(int i=0;i<NUM_SPINS;i++)tape[i]=lcg(&rng)&1;
        for(int i=8;i<32;i++)tape[i]=0;
        if(p){pid_t p1=fork();if(p1==0){w(3,ITERS,rt,rl);_exit(0);}pid_t p2=fork();if(p2==0){w(4,ITERS,rt,rl);_exit(0);}waitpid(p1,NULL,0);waitpid(p2,NULL,0);}
        else{pid_t p=fork();if(p==0){w(3,ITERS*2,rt,rl);_exit(0);}waitpid(p,NULL,0);}
        uint64_t fs[NUM_SPINS];for(int i=0;i<NUM_SPINS;i++)fs[i]=tape[i]&1;
        int64_t E=score(fs,J);*a+=E;if(E==g)(*h)++;
    }
    *a/=SAMPLES;
    printf("  %-30s %3d/%d %8.2f\n",l,*h,SAMPLES,*a);
}

int main(){
    printf("=== PHASE 2B.3B: ANTI-FERROMAGNETIC ACID TEST ===\n\n");
    printf("If P1 works only because problem is ferromagnetic,\n");
    printf("it should FAIL on antiferromagnetic where ground != all-aligned.\n\n");

    // Antiferromagnetic chain: J[i][i+1] = -1 (Néel state ground)
    int64_t Jaf[NUM_SPINS][NUM_SPINS]={{0}};
    for(int i=0;i<NUM_SPINS-1;i++)Jaf[i][i+1]=Jaf[i+1][i]=-1;
    int64_t ground_af=-(NUM_SPINS-1); // all anti-aligned = 7 satisfied edges

    int rt[]={0,1,1,2,2,3,3,4,4,5,5,6,6,7,0,7};
    int sg[]={-1,-1,-1,-1,-1,-1,-1,-1};
    route_signs=sg;
    int rl=8;

    tape=(volatile uint64_t*)mmap(NULL,TAPE_SIZE,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
    int h;double a;

    printf("Anti-ferromagnetic chain J=-1, ground=%ld\n",ground_af);
    printf("MODE       WORKERS       HITS     MEAN_E\n");
    run("P1:2w shared",worker_p1,rt,rl,Jaf,ground_af,&h,&a,1);
    run("P1:1w null",worker_p1,rt,rl,Jaf,ground_af,&h,&a,0);
    run("P2:2w shared",worker_p2,rt,rl,Jaf,ground_af,&h,&a,1);
    run("P2:1w null",worker_p2,rt,rl,Jaf,ground_af,&h,&a,0);

    printf("\nP1=ferro-bias-only P2=sign-aware\n");
    printf("If P1 fails on anti-ferro: P1 was just ferro bias, not attractor.\n");
    printf("If P2 passes on anti-ferro: sign-aware edge rule works.\n");
    printf("CONTAMINATION: Workers NEVER access J_ij.\n");
    return 0;
}
