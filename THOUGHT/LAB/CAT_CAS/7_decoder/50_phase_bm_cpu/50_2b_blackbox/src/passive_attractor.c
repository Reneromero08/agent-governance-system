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
#define NUM_SAMPLES 200
#define TAPE_SIZE 256
#define WORKER_ITERS 10000

volatile uint64_t *tape __attribute__((aligned(64)));

uint64_t lcg(uint64_t *s) { *s=(*s*0x41C64E6D+0x3039); *s=(*s>>13)^*s; *s=(*s<<17)+*s; return *s; }

void passive_worker(int core,int iters){
    cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(core,&cs);sched_setaffinity(0,sizeof(cs),&cs);
    uint64_t rng=0x1111000000000001ULL+core;
    for(int i=0;i<iters;i++){
        int a=lcg(&rng)%NUM_SPINS,b=lcg(&rng)%NUM_SPINS;
        if(a==b)continue;
        uint64_t sa=__atomic_load_n(&tape[a],__ATOMIC_RELAXED);
        uint64_t sb=__atomic_load_n(&tape[b],__ATOMIC_RELAXED);
        int ba=sa&1,bb=sb&1;
        if(ba!=bb){
            if(a>b)__atomic_fetch_xor(&tape[a],1,__ATOMIC_RELAXED);
            else __atomic_fetch_xor(&tape[b],1,__ATOMIC_RELAXED);
        }else{
            uint64_t ct=__atomic_fetch_add(&tape[31],1,__ATOMIC_RELAXED);
            if(ct&1)__atomic_fetch_xor(&tape[a],1,__ATOMIC_RELAXED);
        }
    }
}

int64_t score_energy(uint64_t *spins,int64_t J[NUM_SPINS][NUM_SPINS]){
    int64_t E=0;
    for(int i=0;i<NUM_SPINS;i++){
        int si=(spins[i]&1)?1:-1;
        for(int j=i+1;j<NUM_SPINS;j++){
            int sj=(spins[j]&1)?1:-1;
            E-=J[i][j]*si*sj;
        }
    }
    return E;
}

int main(){
    printf("=== PHASE 2B.1: PASSIVE HIDDEN-ATTRACTOR HARNESS ===\n\n");
    printf("CONTAMINATION: Worker NEVER accesses J, local field, or energy.\n");
    printf("Worker logic: pick 2 spins, if different flip the higher-slot one.\n");
    printf("If same, flip based on shared contention counter parity.\n\n");

    int64_t J[NUM_SPINS][NUM_SPINS]={{0}};
    // 8-spin linear chain: J[i][i+1] = +1 (ferromagnetic)
    for(int i=0;i<NUM_SPINS-1;i++)J[i][i+1]=J[i+1][i]=1;
    int64_t ground_energy=-(NUM_SPINS-1); // all aligned = 7 pairs

    printf("Problem: 8-spin linear chain, J[i][i+1]=+1\n");
    printf("Ground energy: %ld\n\n",ground_energy);

    tape=(volatile uint64_t*)mmap(NULL,TAPE_SIZE,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);

    // Test 1: Shared tape, 2 passive workers
    printf("--- Test 1: Shared tape, 2 workers ---\n");
    int64_t energies[NUM_SAMPLES];int hits=0;double avg=0;
    for(int s=0;s<NUM_SAMPLES;s++){
        uint64_t rng=0x2222000000000001ULL+s;
        for(int i=0;i<NUM_SPINS;i++)tape[i]=lcg(&rng)&1;
        tape[31]=0;
        pid_t p3=fork();if(p3==0){passive_worker(3,WORKER_ITERS);_exit(0);}
        pid_t p4=fork();if(p4==0){passive_worker(4,WORKER_ITERS);_exit(0);}
        waitpid(p3,NULL,0);waitpid(p4,NULL,0);
        uint64_t fs[NUM_SPINS];for(int i=0;i<NUM_SPINS;i++)fs[i]=tape[i]&1;
        int64_t E=score_energy(fs,J);
        energies[s]=E;avg+=E;if(E==ground_energy)hits++;
    }
    avg/=NUM_SAMPLES;double var=0;
    for(int s=0;s<NUM_SAMPLES;s++)var+=(energies[s]-avg)*(energies[s]-avg);
    var/=NUM_SAMPLES;
    printf("  Hits: %d/%d (%.1f%%) Mean: %.2f Std: %.2f\n",hits,NUM_SAMPLES,100.0*hits/NUM_SAMPLES,avg,sqrt(var));

    // Test 2: Single worker null
    printf("--- Test 2: Single worker null ---\n");
    int nh=0;double na=0;
    for(int s=0;s<NUM_SAMPLES;s++){
        uint64_t rng=0x3333000000000001ULL+s;
        for(int i=0;i<NUM_SPINS;i++)tape[i]=lcg(&rng)&1;tape[31]=0;
        pid_t p=fork();if(p==0){passive_worker(3,WORKER_ITERS*2);_exit(0);}
        waitpid(p,NULL,0);
        uint64_t fs[NUM_SPINS];for(int i=0;i<NUM_SPINS;i++)fs[i]=tape[i]&1;
        int64_t E=score_energy(fs,J);
        na+=E;if(E==ground_energy)nh++;
    }
    na/=NUM_SAMPLES;
    printf("  Hits: %d/%d (%.1f%%) Mean: %.2f\n",nh,NUM_SAMPLES,100.0*nh/NUM_SAMPLES,na);

    // Test 3: Independent tapes null
    printf("--- Test 3: Independent tapes null ---\n");
    int ih=0;double ia=0;
    for(int s=0;s<NUM_SAMPLES;s++){
        volatile uint64_t *ta=(volatile uint64_t*)mmap(NULL,TAPE_SIZE,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
        volatile uint64_t *tb=(volatile uint64_t*)mmap(NULL,TAPE_SIZE,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
        uint64_t rng=0x4444000000000001ULL+s;
        for(int i=0;i<NUM_SPINS;i++){uint64_t sp=lcg(&rng)&1;ta[i]=sp;tb[i]=sp;}
        pid_t pa=fork();if(pa==0){tape=ta;passive_worker(3,WORKER_ITERS);_exit(0);}
        pid_t pb=fork();if(pb==0){tape=tb;passive_worker(4,WORKER_ITERS);_exit(0);}
        waitpid(pa,NULL,0);waitpid(pb,NULL,0);
        uint64_t fs[NUM_SPINS];for(int i=0;i<NUM_SPINS;i++)fs[i]=ta[i]&1;
        int64_t E=score_energy(fs,J);ia+=E;if(E==ground_energy)ih++;
        munmap((void*)ta,TAPE_SIZE);munmap((void*)tb,TAPE_SIZE);
    }
    ia/=NUM_SAMPLES;
    printf("  Hits: %d/%d (%.1f%%) Mean: %.2f\n\n",ih,NUM_SAMPLES,100.0*ih/NUM_SAMPLES,ia);

    printf("=== RESULTS ===\n");
    printf("%-30s %10s %15s\n","Condition","Hits","Mean Energy");
    printf("%-30s %9d/%d %14.2f\n","Shared tape (2 workers)",hits,NUM_SAMPLES,avg);
    printf("%-30s %9d/%d %14.2f\n","Single worker (null)",nh,NUM_SAMPLES,na);
    printf("%-30s %9d/%d %14.2f\n","Independent tapes (null)",ih,NUM_SAMPLES,ia);

    if(hits>nh&&hits>ih)printf("\nShared tape beats both nulls. Candidate signal.\n");
    else if(hits>nh)printf("\nBeats single-worker only. Weak signal.\n");
    else printf("\nNo evidence shared tape outperforms nulls.\n");
    printf("CONTAMINATION: Workers NEVER accessed J, local field, or energy.\n");
    return 0;
}
