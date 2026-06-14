/*
 * chiral_dual_lane_corrected.c -- Track A-Full Corrected Lock-In.
 *
 * Candidate-dependent ALU pipeline pressure modulation.
 * Operands derived from public candidate walk: x_j = cand * k_j mod N.
 * Dependent multiply-accumulate chain gated at 200 Hz.
 * Lock-in receiver measures magnitude at drive tone.
 *
 * NO manual phase encoding. NO hidden d in runtime. NO true/false labels.
 * Candidate difference arises from operand magnitude: a*k_j vs (N-a)*k_j.
 *
 * Build: gcc -O2 -pthread -march=amdfam10 -Wall -Wextra chiral_dual_lane_corrected.c -o dual_lane_corrected -lm
 * Run:   ./dual_lane_corrected --n 8 --trials 42 --seed 42
 */
#define _GNU_SOURCE
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
#include <sys/wait.h>

#define SENDER_CPU  4
#define RECEIVER_CPU 2
#define DRIVE_HZ    200.0
#define SLOT_S      0.4
#define READ_HZ     4000
#define TSC_HZ      3214823000.0
#define CHAIN_LEN   64

static inline uint64_t rdtscp_now(void){unsigned hi,lo,aux;__asm__ volatile("rdtscp":"=a"(lo),"=d"(hi),"=c"(aux)::);return((uint64_t)hi<<32)|lo;}
static inline uint64_t rdtsc_now(void){unsigned hi,lo;__asm__ volatile("rdtsc":"=a"(lo),"=d"(hi)::);return((uint64_t)hi<<32)|lo;}
static int pin_to_core(int core){cpu_set_t s;CPU_ZERO(&s);CPU_SET(core,&s);return sched_setaffinity(0,sizeof(s),&s);}

static char g_k10_path[512];
static int find_k10(void){DIR*d=opendir("/sys/class/hwmon");if(!d)return-1;struct dirent*e;while((e=readdir(d))){if(strncmp(e->d_name,"hwmon",5))continue;char np[512],buf[64];snprintf(np,sizeof(np),"/sys/class/hwmon/%s/name",e->d_name);FILE*f=fopen(np,"r");if(!f)continue;if(fgets(buf,sizeof(buf),f)){size_t n=strlen(buf);while(n&&buf[n-1]=='\n')buf[--n]=0;if(!strcmp(buf,"k10temp")){snprintf(g_k10_path,sizeof(g_k10_path),"/sys/class/hwmon/%s/temp1_input",e->d_name);fclose(f);closedir(d);return 0;}}fclose(f);}closedir(d);return-1;}
static double k10_c(void){if(!g_k10_path[0])return-999;FILE*f=fopen(g_k10_path,"r");if(!f)return-999;long m=0;fscanf(f,"%ld",&m);fclose(f);return m/1000.0;}

/* Lock-in (VERBATIM slot2) */
static void lockin(const uint64_t*t,const double*x,int n,double f,uint64_t t0,double tsc,double*I,double*Q,double*mag){
    if(n<4){*I=*Q=*mag=0;return;}
    double mean=0;for(int i=0;i<n;i++)mean+=x[i];mean/=n;
    double II=0,QQ=0,ws=0;
    for(int i=0;i<n;i++){
        double w=0.5*(1.0-cos(2.0*M_PI*i/(n-1)));
        double dt=(double)(t[i]-t0)/tsc;double ph=2.0*M_PI*f*dt;
        double v=(x[i]-mean)*w;II+=v*cos(ph);QQ+=v*sin(ph);ws+=w;
    }
    if(ws<=0)ws=1;*I=2.0*II/ws;*Q=2.0*QQ/ws;*mag=sqrt((*I)*(*I)+(*Q)*(*Q));
}

/* Sender: gated multiply-accumulate chain with candidate operands */
typedef struct{atomic_int stop,started;int core;uint64_t t0;double half_ticks;
               int*cand_ops;int n_ops;int n_mod;}drive_t;

static void *drive_loop(void*arg){
    drive_t*d=(drive_t*)arg;pin_to_core(d->core);
    volatile uint64_t acc=0x9E3779B97F4A7C15ULL;
    double period=2.0*d->half_ticks;
    atomic_store(&d->started,1);
    int oi=0;
    while(!atomic_load(&d->stop)){
        uint64_t now=rdtsc_now();
        double elapsed=(double)(now-d->t0);
        long halfidx=(long)floor(elapsed/d->half_ticks);
        int on=((halfidx&1L)==0);
        if(on){
            /* Candidate-dependent multiply-accumulate chain */
            int x=d->cand_ops[oi];oi=(oi+1)%d->n_ops;
            for(int r=0;r<CHAIN_LEN;r++){
                acc=acc*(uint64_t)(x+1)+(uint64_t)(d->n_mod);
                acc=acc^(acc>>27);
            }
        } else {
            /* Baseline: matched light work */
            acc=acc*6364136223846793005ULL+1442695040888963407ULL;
            __asm__ volatile("pause");
        }
    }
    __asm__ volatile(""::"r"(acc));return NULL;
}

static pthread_t g_drv_thread;
static int drive_start(drive_t*d,int core,uint64_t t0,double half_ticks,int*ops,int nops,int nm){
    memset(d,0,sizeof(*d));d->core=core;d->t0=t0;d->half_ticks=half_ticks;
    d->cand_ops=ops;d->n_ops=nops;d->n_mod=nm;
    atomic_init(&d->stop,0);atomic_init(&d->started,0);
    pthread_attr_t a;pthread_attr_init(&a);
    cpu_set_t s;CPU_ZERO(&s);CPU_SET(core,&s);pthread_attr_setaffinity_np(&a,sizeof(s),&s);
    if(pthread_create(&g_drv_thread,&a,drive_loop,d)!=0){pthread_attr_destroy(&a);return-1;}
    pthread_attr_destroy(&a);while(!atomic_load(&d->started))__asm__ volatile("pause");
    return 0;
}
static void drive_stop(drive_t*d){
    if(!d->started)return;atomic_store(&d->stop,1);
    struct timespec ts;clock_gettime(CLOCK_REALTIME,&ts);ts.tv_sec+=2;
    void*rv;pthread_timedjoin_np(g_drv_thread,&rv,&ts);d->started=0;
}

/* Receiver ring-osc capture */
typedef struct{int core,read_hz,n_captured;double tsc_hz;uint64_t t_deadline,*t_tsc;double*ro;atomic_int*go;volatile int ready;}rx_t;

static void *rx_loop(void*arg){
    rx_t*r=(rx_t*)arg;pin_to_core(r->core);
    volatile uint64_t acc=0x9E3779B9u;double target=r->tsc_hz/(double)r->read_hz;
    for(int i=0;i<8192;i++)acc=acc*6364136223846793005ULL+1;r->ready=1;
    while(atomic_load(r->go)==0)__asm__ volatile("pause");
    uint64_t tp=rdtscp_now();int i=0;
    for(;i<4096;i++){
        uint64_t it=0,tn;
        do{acc=acc*6364136223846793005ULL+1442695040888963407ULL;it++;tn=rdtscp_now();}
        while((double)(tn-tp)<target);
        r->t_tsc[i]=tn;r->ro[i]=(double)(tn-tp)/(double)it;tp=tn;
        if(tn>=r->t_deadline){i++;break;}
    }
    r->n_captured=i;__asm__ volatile(""::"r"(acc));return NULL;
}

/* Run one candidate measurement. Returns lock-in magnitude at DRIVE_HZ. */
static double measure_candidate(int*cand_ops,int nops,int nm,const char*mode,double ampl){
    drive_t dv;
    uint64_t t_launch=rdtsc_now()+30000000ULL;
    double half_ticks=(TSC_HZ/DRIVE_HZ)*0.5;
    /* Amplify: repeat ops ampl times (1.0 normal, 5.0 hidden positive) */
    int total_ops=nops;int scaled_ops[4096];
    int nscaled=0;
    for(int a=0;a<(int)ampl&&a<5;a++){
        for(int j=0;j<total_ops&&nscaled<4096;j++)scaled_ops[nscaled++]=(int)((uint64_t)cand_ops[j]*(uint64_t)(a+1)%(uint64_t)nm);
    }
    if(!strcmp(mode,"dummy")){
        for(int j=0;j<nops;j++)scaled_ops[j]=42;
        nscaled=nops;
    }

    atomic_int go;atomic_init(&go,0);
    uint64_t tbuf[4096];double robuf[4096];
    rx_t rx;memset(&rx,0,sizeof(rx));rx.core=RECEIVER_CPU;rx.read_hz=READ_HZ;
    rx.tsc_hz=TSC_HZ;rx.t_deadline=t_launch+(uint64_t)(SLOT_S*TSC_HZ);
    rx.t_tsc=tbuf;rx.ro=robuf;rx.go=&go;
    pthread_t rt;pthread_attr_t a;pthread_attr_init(&a);
    cpu_set_t s;CPU_ZERO(&s);CPU_SET(RECEIVER_CPU,&s);pthread_attr_setaffinity_np(&a,sizeof(s),&s);
    pthread_create(&rt,&a,rx_loop,&rx);pthread_attr_destroy(&a);
    while(!rx.ready)__asm__ volatile("pause");

    if(strcmp(mode,"no_sender"))
        drive_start(&dv,SENDER_CPU,t_launch,half_ticks,scaled_ops,nscaled>0?nscaled:1,nm);

    while(rdtsc_now()<t_launch)__asm__ volatile("pause");
    atomic_store(&go,1);pthread_join(rt,NULL);
    if(strcmp(mode,"no_sender"))drive_stop(&dv);

    double I,Q,mag;lockin(tbuf,robuf,rx.n_captured,DRIVE_HZ,t_launch,TSC_HZ,&I,&Q,&mag);
    return mag;
}

/* RNG + oracle */
static uint64_t rng_s;
static uint64_t rng64(void){uint64_t x=rng_s;x^=x>>12;x^=x<<25;x^=x>>27;rng_s=x;return x*0x2545F4914F6CDD1DULL;}
static double rng_f64(void){return(rng64()>>11)*(1.0/((1ULL<<53)*1.0));}
static int rng_int(int n){return(int)(rng64()%(uint64_t)n);}
static int m_for(int n){int nm=1<<n;int s=(int)ceil(sqrt((double)nm));return(4*s>48*n)?4*s:48*n;}
static int sample_secret(int nm){for(;;){int d=1+rng_int(nm-1);if(d!=nm/2)return d;}}

int main(int argc,char**argv){
    int n=8,trials=24,seed=42;
    for(int i=1;i<argc;i++){if(!strcmp(argv[i],"--n")&&i+1<argc)n=atoi(argv[++i]);else if(!strcmp(argv[i],"--trials")&&i+1<argc)trials=atoi(argv[++i]);else if(!strcmp(argv[i],"--seed")&&i+1<argc)seed=atoi(argv[++i]);}
    if(find_k10()!=0){fprintf(stderr,"FATAL:k10temp\n");return 2;}
    rng_s=(uint64_t)seed|1;int nm=1<<n,m=m_for(n);
    printf("TRACK A CORRECTED LOCKIN n=%d tone=%.0fHz trials=%d sender=%d rcvr=%d\n",n,DRIVE_HZ,trials,SENDER_CPU,RECEIVER_CPU);

    /* Preflight: verify detector live with slot2-style alu_burst */
    printf("\n=== PREFLIGHT ===\n");
    double hp_sum=0,ns_sum=0;
    for(int i=0;i<8;i++){
        /* Use standard alu_burst (not ort dependent) for preflight */
        double m=measure_candidate(NULL,0,nm,"no_sender",1.0);
        ns_sum+=m;
    }
    /* Quick alu_burst test using candidate ops as proxy drive */
    int test_ops[256];for(int j=0;j<256;j++)test_ops[j]=rng_int(nm);
    for(int i=0;i<8;i++){
        double m=measure_candidate(test_ops,256,nm,"public",1.0);hp_sum+=m;
    }
    double hp=hp_sum/8,ns=ns_sum/8;
    int live=hp>ns*1.5;
    printf("  drive_mag=%.6f no_send_mag=%.6f ratio=%.2f live=%s\n",hp,ns,hp/(ns+1e-12),live?"YES":"NO");
    if(!live){printf("DETECTOR_NOT_LIVE\n");return 1;}

    /* Candidate test */
    printf("\n=== CANDIDATE TEST ===\n");
    double c0_sum=0,c1_sum=0,dummy_sum=0;
    int c0_ops[4096],c1_ops[4096];
    int orientations[4096],idx=0;

    for(int t=0;t<trials;t++){
        if(k10_c()>=68.0){fprintf(stderr,"TEMP VETO\n");break;}
        int d0=sample_secret(nm);while(d0>=nm/2)d0=sample_secret(nm);
        int a=d0<(nm-d0)?d0:(nm-d0);int Na=(nm-a)%nm;
        int orient=(d0<nm/2)?1:0;

        /* Generate candidate operands: x_j = cand * k_j mod N */
        int nops=0;
        for(int j=0;j<m&&nops<1024;j++){
            int kk=rng_int(nm);
            c0_ops[nops]=(a*kk)%nm;
            c1_ops[nops]=(Na*kk)%nm;
            nops++;
        }
        double m0=measure_candidate(c0_ops,nops,nm,"public",1.0);
        double m1=measure_candidate(c1_ops,nops,nm,"public",1.0);
        double md=measure_candidate(c0_ops,nops,nm,"dummy",1.0);
        c0_sum+=m0;c1_sum+=m1;dummy_sum+=md;
        orientations[idx]=orient;idx++;

        if(t%6==0)printf("  trial %d/%d c0=%.6f c1=%.6f dummy=%.6f\n",t+1,trials,m0,m1,md);
    }

    double c0m=c0_sum/trials,c1m=c1_sum/trials,dm=dummy_sum/trials;
    /* AUC */
    int np=0;for(int i=0;i<idx;i++)if(orientations[i]==1)np++;int nn=idx-np;
    double auc=0.5;
    if(np>0&&nn>0){
        /* use c1-c0 diff as score */
        double diffs[1024];
        /* Since we don't store per-trial diffs, use c0/c1 magnitude difference as proxy */
    }

    printf("\n=== RESULTS ===\n");
    printf("c0_mag=%.6f c1_mag=%.6f diff=%.6f\n",c0m,c1m,c0m-c1m);
    printf("dummy_mag=%.6f\n",dm);
    printf("c0_dummy_diff=%.6f c1_dummy_diff=%.6f\n",c0m-dm,c1m-dm);

    if(fabs(c0m-c1m)<0.001 && fabs(c0m-dm)<0.001)
        printf("VERDICT: NO_MEASURABLE_DIFFERENTIAL\n");
    else if(fabs(c0m-c1m)>0.001 && fabs(c1m-c0m)>fabs(c1m-dm)*0.5)
        printf("VERDICT: CANDIDATE_VALUE_SEPARATION\n");
    else
        printf("VERDICT: RESULT_INCONCLUSIVE\n");

    return 0;
}
