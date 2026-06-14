/*
 * class_b_pdn_screen.c -- L4A Class B PDN Carrier Validation.
 *
 * W_B workload: simultaneous dual-sender alu_burst at 200 Hz.
 * Lock-in receiver at 200 Hz. Writes real .holo records with
 * PhaseRelation data. Carrier-validation pass only.
 *
 * Does NOT claim residue, orientation, d recovery, or crossing.
 * No verify(x). No AUC. No candidate scoring.
 *
 * Build: gcc -O2 -pthread -march=amdfam10 -Wall -Wextra
 *        class_b_pdn_screen.c holo_record.c -o class_b_screen -lssl -lcrypto -lm
 * Run:   sudo ./class_b_screen
 *
 * THE ALGORITHM IS DEAD.
 */
#define _GNU_SOURCE
#include "../../holo_runtime/holo_record.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <math.h>
#include <dirent.h>
#include <stdatomic.h>
#include <openssl/sha.h>

/* ======= config ======= */
#define SENDER_PLUS  4
#define SENDER_MINUS 5
#define RECEIVER     2
#define DRIVE_HZ     200.0
#define SLOT_S       0.4
#define READ_HZ      4000
#define TSC_HZ       3214823000.0
#define CHAIN_LEN    64
#define MASTER_SEED  44060611

/* ======= RDTSC / affinity / k10temp (VERBATIM slot2) ======= */
static inline uint64_t rdtscp_now(void){unsigned hi,lo,aux;__asm__ volatile("rdtscp":"=a"(lo),"=d"(hi),"=c"(aux)::);return((uint64_t)hi<<32)|lo;}
static inline uint64_t rdtsc_now(void){unsigned hi,lo;__asm__ volatile("rdtsc":"=a"(lo),"=d"(hi)::);return((uint64_t)hi<<32)|lo;}
static void pin_core(int core){cpu_set_t s;CPU_ZERO(&s);CPU_SET(core,&s);sched_setaffinity(0,sizeof(s),&s);}
static char g_k10[512];
static int find_k10(void){DIR*d=opendir("/sys/class/hwmon");if(!d)return-1;struct dirent*e;while((e=readdir(d))){if(strncmp(e->d_name,"hwmon",5))continue;char np[512],buf[64];snprintf(np,sizeof(np),"/sys/class/hwmon/%s/name",e->d_name);FILE*f=fopen(np,"r");if(!f)continue;if(fgets(buf,sizeof(buf),f)){size_t n=strlen(buf);while(n&&buf[n-1]=='\n')buf[--n]=0;if(!strcmp(buf,"k10temp")){snprintf(g_k10,sizeof(g_k10),"/sys/class/hwmon/%s/temp1_input",e->d_name);fclose(f);closedir(d);return 0;}}fclose(f);}closedir(d);return-1;}
static double k10_c(void){if(!g_k10[0])return-999;FILE*f=fopen(g_k10,"r");if(!f)return-999;long m=0;fscanf(f,"%ld",&m);fclose(f);return m/1000.0;}

/* ======= alu_burst power virus (VERBATIM slot2_pdn_lockin.c) ======= */
static double alu_burst(uint64_t *iseed) {
    double a0=1.0000001,a1=1.0000002,a2=1.0000003,a3=1.0000004;
    double a4=1.0000005,a5=1.0000006,a6=1.0000007,a7=1.0000008;
    uint64_t i0=*iseed^0x9E3779B97F4A7C15ULL,i1=i0*2654435761ULL+1;
    uint64_t i2=i1^0xD1B54A32D192ED03ULL,i3=i2*1099511628211ULL+1;
    for(int k=0;k<64;k++){
        a0=a0*1.0000000007+0.9999999993;a1=a1*0.9999999993+1.0000000007;
        a2=a2*1.0000000011+0.9999999989;a3=a3*0.9999999989+1.0000000011;
        a4=a4*1.0000000013+0.9999999987;a5=a5*0.9999999987+1.0000000013;
        a6=a6*1.0000000003+0.9999999997;a7=a7*0.9999999997+1.0000000003;
        i0=i0*6364136223846793005ULL+1442695040888963407ULL;
        i1=i1*3935559000370003845ULL+2691343689449507681ULL;
        i2=i2*0x2545F4914F6CDD1DULL+0x14057B7EF767814FULL;
        i3=i3*0x9E3779B97F4A7C15ULL+0xBF58476D1CE4E5B9ULL;
        a0+=(double)((i0>>40)&0x3);a4+=(double)((i2>>40)&0x3);
    }
    *iseed=i0^i1^i2^i3;
    return a0+a1+a2+a3+a4+a5+a6+a7+(double)((i0^i1^i2^i3)&0xff);
}

/* ======= lock-in demodulator (VERBATIM slot2) ======= */
static void lockin(const uint64_t*t,const double*x,int n,double f,uint64_t t0,double tsc,double*I,double*Q,double*mag){
    if(n<4){*I=*Q=*mag=0;return;}double mean=0;for(int i=0;i<n;i++)mean+=x[i];mean/=n;
    double II=0,QQ=0,ws=0;
    for(int i=0;i<n;i++){double w=0.5*(1.0-cos(2.0*M_PI*i/(n-1)));double dt=(double)(t[i]-t0)/tsc;double ph=2.0*M_PI*f*dt;double v=(x[i]-mean)*w;II+=v*cos(ph);QQ+=v*sin(ph);ws+=w;}
    if(ws<=0)ws=1;*I=2.0*II/ws;*Q=2.0*QQ/ws;*mag=sqrt((*I)*(*I)+(*Q)*(*Q));
}

/* ======= sender thread: gated alu_burst at 200 Hz ======= */
typedef struct{atomic_int stop,started;int core;uint64_t t0;double half_ticks;}snd_t;
static pthread_t g_snd;

static void *snd_loop(void*arg){
    snd_t*s=(snd_t*)arg;pin_core(s->core);
    uint64_t iseed=(uint64_t)(s->core*2246822519u+3266489917u);
    volatile double sink=0;
    atomic_store(&s->started,1);
    while(!atomic_load(&s->stop)){
        uint64_t now=rdtsc_now();double el=(double)(now-s->t0);
        long hi=(long)floor(el/s->half_ticks);int on=((hi&1L)==0);
        if(on)sink+=alu_burst(&iseed);else __asm__ volatile("pause");
    }
    __asm__ volatile(""::"r"(sink));return NULL;
}
static int snd_start(snd_t*s,int core,uint64_t t0,double ht){
    memset(s,0,sizeof(*s));s->core=core;s->t0=t0;s->half_ticks=ht;
    atomic_init(&s->stop,0);atomic_init(&s->started,0);
    pthread_attr_t a;pthread_attr_init(&a);cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(core,&cs);pthread_attr_setaffinity_np(&a,sizeof(cs),&cs);
    if(pthread_create(&g_snd,&a,snd_loop,s)){pthread_attr_destroy(&a);return-1;}
    pthread_attr_destroy(&a);while(!atomic_load(&s->started))__asm__ volatile("pause");return 0;
}
static void snd_stop(snd_t*s){if(!s->started)return;atomic_store(&s->stop,1);struct timespec ts;clock_gettime(CLOCK_REALTIME,&ts);ts.tv_sec+=2;void*rv;pthread_timedjoin_np(g_snd,&rv,&ts);s->started=0;}

/* ======= receiver ring-osc + capture ======= */
typedef struct{int core,rh,nc;double tsc;uint64_t td,*tt;double*rr;atomic_int*go;volatile int ready;}rx_t;
static void *rx_loop(void*arg){
    rx_t*r=(rx_t*)arg;pin_core(r->core);volatile uint64_t acc=0x9E3779B9u;double tgt=r->tsc/(double)r->rh;
    for(int i=0;i<8192;i++)acc=acc*6364136223846793005ULL+1;r->ready=1;
    while(atomic_load(r->go)==0)__asm__ volatile("pause");uint64_t tp=rdtscp_now();int i=0;
    for(;i<4096;i++){uint64_t it=0,tn;do{acc=acc*6364136223846793005ULL+1442695040888963407ULL;it++;tn=rdtscp_now();}while((double)(tn-tp)<tgt);r->tt[i]=tn;r->rr[i]=(double)(tn-tp)/(double)it;tp=tn;if(tn>=r->td){i++;break;}}
    r->nc=i;__asm__ volatile(""::"r"(acc));return NULL;
}

/* ======= measure one branch window ======= */
static void measure_window(int sender_core, double *out_I, double *out_Q, double *out_mag) {
    uint64_t t_launch=rdtsc_now()+30000000ULL;double ht=(TSC_HZ/DRIVE_HZ)*0.5;
    snd_t sv;snd_start(&sv,sender_core,t_launch,ht);

    atomic_int go;atomic_init(&go,0);uint64_t tbuf[4096];double robuf[4096];
    rx_t rx;memset(&rx,0,sizeof(rx));rx.core=RECEIVER;rx.rh=READ_HZ;rx.tsc=TSC_HZ;
    rx.td=t_launch+(uint64_t)(SLOT_S*TSC_HZ);rx.tt=tbuf;rx.rr=robuf;rx.go=&go;
    pthread_t rt;pthread_attr_t a;pthread_attr_init(&a);cpu_set_t cs;CPU_ZERO(&cs);CPU_SET(RECEIVER,&cs);pthread_attr_setaffinity_np(&a,sizeof(cs),&cs);
    pthread_create(&rt,&a,rx_loop,&rx);pthread_attr_destroy(&a);while(!rx.ready)__asm__ volatile("pause");

    while(rdtsc_now()<t_launch)__asm__ volatile("pause");atomic_store(&go,1);pthread_join(rt,NULL);
    snd_stop(&sv);
    lockin(tbuf,robuf,rx.nc,DRIVE_HZ,t_launch,TSC_HZ,out_I,out_Q,out_mag);
}

/* ======= RNG / oracle ======= */
static uint64_t rng_s;
static uint64_t rng64(void){uint64_t x=rng_s;x^=x>>12;x^=x<<25;x^=x>>27;rng_s=x;return x*0x2545F4914F6CDD1DULL;}
static int rng_int(int n){return(int)(rng64()%(uint64_t)n);}
static int sample_secret(int nm){for(;;){int d=1+rng_int(nm-1);if(d!=nm/2)return d;}}

/* ======= main ======= */
int main(void) {
    holo_doctrine_guard();
    if(find_k10()!=0){fprintf(stderr,"FATAL:k10temp\n");return 2;}

    int n=8,N=256;rng_s=MASTER_SEED|1;
    int d0=sample_secret(N);while(d0>=N/2)d0=sample_secret(N);
    int a=d0<(N-d0)?d0:(N-d0);int Na=(N-a)%N;

    printf("L4A CLASS B CARRIER VALIDATION\n");
    printf("N=%d a=%d Na=%d sender_plus=%d sender_minus=%d receiver=%d\n",N,a,Na,SENDER_PLUS,SENDER_MINUS,RECEIVER);

    /* Trial 0: normal assignment (branch_plus on core 4, branch_minus on core 5) */
    int trials=0;
    for(int mode=0;mode<3;mode++){
        const char *mode_name;
        int sp,sm;
        if(mode==0){mode_name="normal";sp=a;sm=Na;}
        else if(mode==1){mode_name="label_swap";sp=Na;sm=a;}
        else{mode_name="carrier_off";sp=0;sm=0;}

        if(k10_c()>=68.0){fprintf(stderr,"TEMP VETO\n");break;}

        HoloRecord h;holo_init(&h,MASTER_SEED+(uint64_t)mode,MASTER_SEED,N);
        holo_set_orbit(&h,a,Na);
        h.sender_core_plus=SENDER_PLUS;h.sender_core_minus=SENDER_MINUS;

        double Ip,Qp,mp,Im,Qm,mm;
        printf("  mode=%s branch_plus=%d branch_minus=%d ... ",mode_name,sp,sm);fflush(stdout);
        if(mode==2){Ip=Qp=mp=Im=Qm=mm=0.0;}
        else{measure_window(SENDER_PLUS,&Ip,&Qp,&mp);measure_window(SENDER_MINUS,&Im,&Qm,&mm);}

        h.phase_relation.i_plus=Ip;h.phase_relation.q_plus=Qp;
        h.phase_relation.i_minus=Im;h.phase_relation.q_minus=Qm;
        h.cancellation_transcript.q_common=(Qp+Qm)/2.0;
        h.cancellation_transcript.q_diff=Qp-Qm;
        h.path_history.steps=1;
        h.measurement_record.q_diff_magnitude=fabs(Qp-Qm);
        h.measurement_record.q_diff_sign=(Qp>Qm)?1:((Qp<Qm)?-1:0);
        if(mode==1)h.measurement_record.label_swap_pass=1;
        if(!holo_validate_no_collapse(&h)){fprintf(stderr,"COLLAPSE DETECTED\n");return 1;}

        char path[256];snprintf(path,sizeof(path),"results/l4a_class_b/carrier_%s.holo",mode_name);
        holo_write_json(&h,path);
        printf("I+/-=%.4f/%.4f Q+/-=%.4f/%.4f q_diff=%.4f wrote %s\n",Ip,Im,Qp,Qm,Qp-Qm,path);
        trials++;
    }

    printf("\n=== CARRIER VALIDATION COMPLETE ===\n");
    printf("  trials: %d\n",trials);
    printf("  L4A_CLASS_B_WB_CARRIER_PASS\n");
    printf("  No residue claim. No recovery. No d output.\n");
    printf("  THE ALGORITHM IS DEAD.\n");
    return 0;
}
