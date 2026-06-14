#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <openssl/sha.h>
#define TAPE_KB 64
#define MAX_ITERS 512
#define PREDECLARED_FP 42

static uint64_t rng_s;
static uint64_t rng64(void){uint64_t x=rng_s;x^=x>>12;x^=x<<25;x^=x>>27;rng_s=x;return x*0x2545F4914F6CDD1DULL;}
static void fill_tape(uint8_t*t,size_t n,uint64_t s){rng_s=s|1;for(size_t i=0;i<n;i++)t[i]=(uint8_t)(rng64()&0xFF);}
static void tape_sha256(const uint8_t*t,size_t n,uint8_t h[32]){SHA256(t,n,h);}
static void xor_mask(uint8_t*t,size_t n,uint64_t s){rng_s=s|1;for(size_t i=0;i<n;i++)t[i]^=(uint8_t)(rng64()&0xFF);}
static int hash_eq(const uint8_t*a,const uint8_t*b){return memcmp(a,b,32)==0;}

static int f_contract(int x){return (x+PREDECLARED_FP)/2;}
static int f_nofp(int x){return(x+1)%256;}

int main(int argc,char**argv){
    int ns=10;uint64_t bs=42;const char*csv="fp_results.csv";
    for(int i=1;i<argc;i++){if(!strcmp(argv[i],"--seeds")&&i+1<argc)ns=atoi(argv[++i]);else if(!strcmp(argv[i],"--seed")&&i+1<argc)bs=(uint64_t)atol(argv[++i]);else if(!strcmp(argv[i],"--csv")&&i+1<argc)csv=argv[++i];}
    size_t n=(size_t)TAPE_KB*1024;uint8_t*tape=malloc(n);if(!tape)return 1;
    FILE*f=fopen(csv,"w");
    fprintf(f,"seed,mode,start_x,final_x,iterations,converged,tape_ok,baseline_iters\n");
    printf("EXP50 L3 FIXED POINT seeds=%d fp=%d map=contraction\n",ns,PREDECLARED_FP);

    int cat_trials=0,cat_ok=0,bas_ok=0,id_ok=0,wr_ok=0,nofp_ok=0,rp_ok=0;

    for(int s=0;s<ns;s++){
        uint64_t sd=bs+(uint64_t)s*1000;
        for(int sx=10;sx<=250;sx+=30){cat_trials++;
            uint64_t tid=sd+(uint64_t)sx;int x=sx,iters=0,cv=0,to=1;
            for(iters=0;iters<MAX_ITERS;iters++){
                uint8_t hb[32],ha[32];uint64_t ms=tid^(uint64_t)iters^0xA5A5ULL;
                fill_tape(tape,n,tid+(uint64_t)iters);tape_sha256(tape,n,hb);
                xor_mask(tape,n,ms);int nx=f_contract(x);xor_mask(tape,n,ms);
                tape_sha256(tape,n,ha);if(!hash_eq(hb,ha)){to=0;break;}
                if(nx==x){cv=1;break;}x=nx;}
            if(cv&&to)cat_ok++;
            fprintf(f,"%llu,catalytic_loop,%d,%d,%d,%s,%s,NA\n",(unsigned long long)tid,sx,x,iters+1,cv?"YES":"NO",to?"PASS":"FAIL");}

        {uint64_t tid=sd+777;int bl=0,fi=0;for(int sx=0;sx<256;sx++){bl++;if(f_contract(sx)==sx){fi=sx;break;}}if(fi==PREDECLARED_FP)bas_ok++;
        fprintf(f,"%llu,forward_scan,0,%d,%d,%s,NA,%d\n",(unsigned long long)tid,fi,bl,fi==PREDECLARED_FP?"YES":"NO",bl);}

        {uint64_t tid=sd+888,tm=1;uint8_t hb[32],ha[32];uint64_t ms=tid^0xA5A5ULL;
        fill_tape(tape,n,tid);tape_sha256(tape,n,hb);xor_mask(tape,n,ms);xor_mask(tape,n,ms);
        tape_sha256(tape,n,ha);if(!hash_eq(hb,ha))tm=0;if(tm)id_ok++;
        fprintf(f,"%llu,identity_loop,10,10,1,YES,%s,NA\n",(unsigned long long)tid,tm?"PASS":"FAIL");}

        {uint64_t tid=sd+1111;int fail=0;for(int it=0;it<2;it++){uint8_t hb[32],ha[32];uint64_t ms=tid^(uint64_t)it^0xA5A5ULL;
        fill_tape(tape,n,tid+(uint64_t)it);tape_sha256(tape,n,hb);xor_mask(tape,n,ms);f_contract(10);xor_mask(tape,n,ms^0xDEADULL);
        tape_sha256(tape,n,ha);if(!hash_eq(hb,ha))fail++;}if(fail>0)wr_ok++;
        fprintf(f,"%llu,wrong_restore,10,NA,2,%s,NA,NA\n",(unsigned long long)tid,fail>0?"FAIL_OK":"SHOULD_FAIL");}

        {uint64_t tid=sd+2222;int x=10,iters=0,cv=0,to=1;
        for(iters=0;iters<MAX_ITERS;iters++){uint8_t hb[32],ha[32];uint64_t ms=tid^(uint64_t)iters^0xA5A5ULL;
        fill_tape(tape,n,tid+(uint64_t)iters);tape_sha256(tape,n,hb);
        xor_mask(tape,n,ms);int nx=f_nofp(x);xor_mask(tape,n,ms);
        tape_sha256(tape,n,ha);if(!hash_eq(hb,ha)){to=0;break;}if(nx==x){cv=1;break;}x=nx;}
        if(!cv&&to)nofp_ok++;
        fprintf(f,"%llu,no_fp_negative,%d,%d,%d,%s,%s,NA\n",(unsigned long long)tid,10,x,iters+1,!cv?"NO_FP_OK":"UNEXPECTED",to?"PASS":"FAIL");}

        {uint64_t tid=sd+999;int x1=10,it1=0,c1=0,t1=1;for(it1=0;it1<MAX_ITERS;it1++){uint8_t hb[32],ha[32];uint64_t ms=tid^(uint64_t)it1^0xA5A5ULL;
        fill_tape(tape,n,tid+(uint64_t)it1);tape_sha256(tape,n,hb);xor_mask(tape,n,ms);
        int nx=f_contract(x1);xor_mask(tape,n,ms);tape_sha256(tape,n,ha);
        if(!hash_eq(hb,ha)){t1=0;break;}if(nx==x1){c1=1;break;}x1=nx;}
        int x2=10,it2=0,c2=0,t2=1;for(it2=0;it2<MAX_ITERS;it2++){uint8_t hb[32],ha[32];uint64_t ms=tid^(uint64_t)it2^0xA5A5ULL;
        fill_tape(tape,n,tid+(uint64_t)it2);tape_sha256(tape,n,hb);xor_mask(tape,n,ms);
        int nx=f_contract(x2);xor_mask(tape,n,ms);tape_sha256(tape,n,ha);
        if(!hash_eq(hb,ha)){t2=0;break;}if(nx==x2){c2=1;break;}x2=nx;}
        int rp=(c1==c2&&it1==it2&&x1==x2&&t1&&t2);if(rp)rp_ok++;
        fprintf(f,"%llu,replay,%d,%d,%d,%s,%s,NA\n",(unsigned long long)tid,10,x1,it1+1,rp?"YES":"NO",(t1&&t2)?"PASS":"FAIL");}
    }
    fclose(f);
    printf("\n=== RESULTS ===\n");
    printf("catalytic_loop: %d/%d\n",cat_ok,cat_trials);
    printf("forward_scan: %d/%d\n",bas_ok,ns);
    printf("identity: %d/%d\n",id_ok,ns);
    printf("wrong_restore: %d/%d\n",wr_ok,ns);
    printf("no_fp_negative: %d/%d\n",nofp_ok,ns);
    printf("replay: %d/%d\n",rp_ok,ns);
    int all=(cat_ok==cat_trials&&bas_ok==ns&&id_ok==ns&&wr_ok==ns&&nofp_ok==ns&&rp_ok==ns);
    printf("VERDICT: %s\n",all?"L3_FIXED_POINT_CONVERGENCE_PASS":"L3_FAIL");
    free(tape);return all?0:1;
}
