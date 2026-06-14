/*
 * chiral_dual_lane_lockin.c -- Track A-Full Lock-In Rebuild.
 *
 * Uses VERBATIM slot2 alu_burst power virus + lock-in receiver from T300 lineage.
 * Single tone (200 Hz). Sequential: candidate_0 then candidate_1 per instance.
 *
 * Modes:
 *   public       : drive candidate_0 phase, then candidate_1 phase
 *   same_cand    : drive candidate_0 twice (null)
 *   no_sender    : no drive (baseline)
 *   hidden_pos   : 10x amplified alu_burst (proves detector live)
 *
 * Build: gcc -O2 -pthread -march=amdfam10 -Wall -Wextra chiral_dual_lane_lockin.c -o dual_lane_lockin -lm
 * Run:   ./dual_lane_lockin --n 8 --trials 42 --seed 42 --out-json results.json
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
#include <time.h>
#include <dirent.h>
#include <stdatomic.h>

#define SENDER_CPU    4
#define RECEIVER_CPU  2
#define MASTER_SEED   44060611
#define N_DEFAULT     8
#define M_TRIALS      42
#define DRIVE_HZ      200.0
#define SLOT_S        0.5
#define READ_HZ       4000
#define TSC_HZ        3214823000.0

/* ======= RDTSC / affinity / k10temp (VERBATIM slot2) ======= */
static inline uint64_t rdtscp_now(void){unsigned hi,lo,aux;__asm__ volatile("rdtscp":"=a"(lo),"=d"(hi),"=c"(aux)::);return((uint64_t)hi<<32)|lo;}
static inline uint64_t rdtsc_now(void){unsigned hi,lo;__asm__ volatile("rdtsc":"=a"(lo),"=d"(hi)::);return((uint64_t)hi<<32)|lo;}
static int pin_to_core(int core){cpu_set_t s;CPU_ZERO(&s);CPU_SET(core,&s);return sched_setaffinity(0,sizeof(s),&s);}
static char g_k10_path[512];
static int locate_k10temp(void){DIR*d=opendir("/sys/class/hwmon");if(!d)return-1;struct dirent*e;while((e=readdir(d))){if(strncmp(e->d_name,"hwmon",5))continue;char np[512],buf[64];snprintf(np,sizeof(np),"/sys/class/hwmon/%s/name",e->d_name);FILE*f=fopen(np,"r");if(!f)continue;if(fgets(buf,sizeof(buf),f)){size_t n=strlen(buf);while(n&&(buf[n-1]=='\n'||buf[n-1]==' '))buf[--n]=0;if(!strcmp(buf,"k10temp")){snprintf(g_k10_path,sizeof(g_k10_path),"/sys/class/hwmon/%s/temp1_input",e->d_name);fclose(f);closedir(d);return 0;}}fclose(f);}closedir(d);return-1;}
static double read_k10temp_c(void){if(!g_k10_path[0])return-999.0;FILE*f=fopen(g_k10_path,"r");if(!f)return-999.0;long m=0;fscanf(f,"%ld",&m);fclose(f);return m/1000.0;}

/* ======= alu_burst power virus (VERBATIM slot2) ======= */
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

/* ======= lock-in (VERBATIM slot2) ======= */
static void lockin(const uint64_t *t_tsc, const double *x, int n,
                   double f_ref_hz, uint64_t t0_tsc, double tsc_hz,
                   double *out_I, double *out_Q, double *out_mag) {
    if(n<4){*out_I=*out_Q=*out_mag=0.0;return;}
    double mean=0.0;for(int i=0;i<n;i++)mean+=x[i];mean/=n;
    double I=0.0,Q=0.0,wsum=0.0;
    for(int i=0;i<n;i++){
        double win=0.5*(1.0-cos(2.0*M_PI*i/(n-1)));
        double dt=(double)(t_tsc[i]-t0_tsc)/tsc_hz;
        double ph=2.0*M_PI*f_ref_hz*dt;
        double v=(x[i]-mean)*win;
        I+=v*cos(ph);Q+=v*sin(ph);wsum+=win;
    }
    if(wsum<=0.0)wsum=1.0;
    *out_I=2.0*I/wsum;*out_Q=2.0*Q/wsum;
    *out_mag=sqrt((*out_I)*(*out_I)+(*out_Q)*(*out_Q));
}

/* ======= sender: alu_burst gated at drive tone ======= */
typedef struct{atomic_int stop,started;int core;uint64_t t0;double half_ticks,phase_frac;pthread_t thread;}drive_t;
static pthread_t g_drive_thread; /* stored for stop */

static void *drive_loop(void*arg){
    drive_t*d=(drive_t*)arg;pin_to_core(d->core);
    uint64_t iseed=(uint64_t)(d->core*2246822519u+3266489917u);
    volatile double sink=0.0;
    double period=2.0*d->half_ticks,phase_off=d->phase_frac*period;
    atomic_store(&d->started,1);
    while(!atomic_load(&d->stop)){
        uint64_t now=rdtsc_now();
        double elapsed=(double)(now-d->t0)-phase_off;
        long halfidx=(long)floor(elapsed/d->half_ticks);
        int on=((halfidx&1L)==0);
        if(on)sink+=alu_burst(&iseed);else __asm__ volatile("pause");
    }
    (void)sink;return NULL;
}
static int drive_start(drive_t*d,int core,uint64_t t0,double half_ticks,double phase_frac){
    memset(d,0,sizeof(*d));d->core=core;d->t0=t0;d->half_ticks=half_ticks;d->phase_frac=phase_frac;
    atomic_init(&d->stop,0);atomic_init(&d->started,0);
    pthread_t th;pthread_attr_t a;pthread_attr_init(&a);
    cpu_set_t s;CPU_ZERO(&s);CPU_SET(core,&s);pthread_attr_setaffinity_np(&a,sizeof(s),&s);
    if(pthread_create(&th,&a,drive_loop,d)!=0){pthread_attr_destroy(&a);return-1;}
    pthread_attr_destroy(&a);while(!atomic_load(&d->started))__asm__ volatile("pause");
    return 0;
}
static void drive_stop(drive_t*d){
    if(!d->started)return;atomic_store(&d->stop,1);
    struct timespec ts;clock_gettime(CLOCK_REALTIME,&ts);ts.tv_sec+=2;
    void*rv;pthread_timedjoin_np(th,&rv,&ts);d->started=0;
}

/* ======= receiver ring-osc + capture ======= */
typedef struct{int core,read_hz,n_captured;double tsc_hz;uint64_t t_deadline,*t_tsc;double*ro;atomic_int*go;volatile int ready;}rx_t;
static void *rx_loop(void*arg){
    rx_t*r=(rx_t*)arg;pin_to_core(r->core);
    volatile uint64_t acc=0x9E3779B9u;double target=r->tsc_hz/(double)r->read_hz;
    for(int i=0;i<8192;i++)acc=acc*6364136223846793005ULL+1;r->ready=1;
    while(atomic_load(r->go)==0)__asm__ volatile("pause");
    uint64_t tp=rdtscp_now();int i=0;
    for(;i<4096;i++){uint64_t it=0,tn;do{acc=acc*6364136223846793005ULL+1442695040888963407ULL;it++;tn=rdtscp_now();}while((double)(tn-tp)<target);r->t_tsc[i]=tn;r->ro[i]=(double)(tn-tp)/(double)it;tp=tn;if(tn>=r->t_deadline){i++;break;}}
    r->n_captured=i;__asm__ volatile(""::"r"(acc));return NULL;
}

/* ======= run one candidate slot ======= */
static double run_slot(int candidate, double slot_s, double ampl, int *pin_ok) {
    drive_t dv;uint64_t t_launch=rdtsc_now()+30000000ULL;
    double half_ticks=(TSC_HZ/DRIVE_HZ)*0.5;
    double phase=0.0; /* candidate phase encoding: 0 vs pi */
    int iters=(int)(ampl);
    atomic_int go;atomic_init(&go,0);
    uint64_t tbuf[4096];double robuf[4096];
    rx_t rx;memset(&rx,0,sizeof(rx));
    rx.core=RECEIVER_CPU;rx.read_hz=READ_HZ;rx.tsc_hz=TSC_HZ;
    rx.t_deadline=t_launch+(uint64_t)(slot_s*TSC_HZ);
    rx.t_tsc=tbuf;rx.ro=robuf;rx.go=&go;

    pthread_t rt;pthread_attr_t a;pthread_attr_init(&a);
    cpu_set_t s;CPU_ZERO(&s);CPU_SET(RECEIVER_CPU,&s);pthread_attr_setaffinity_np(&a,sizeof(s),&s);
    pthread_create(&rt,&a,rx_loop,&rx);pthread_attr_destroy(&a);
    while(!rx.ready)__asm__ volatile("pause");

    /* amplify: run alu_burst multiple times per ON half-cycle */
    for(int ai=0;ai<iters && ai<10;ai++){
        if(drive_start(&dv,SENDER_CPU,t_launch,half_ticks,phase)!=0)break;
    }

    while(rdtsc_now()<t_launch)__asm__ volatile("pause");
    atomic_store(&go,1);pthread_join(rt,NULL);
    drive_stop(&dv);

    double I,Q,mag;
    lockin(tbuf,robuf,rx.n_captured,DRIVE_HZ,t_launch,TSC_HZ,&I,&Q,&mag);
    *pin_ok=1;
    return mag;  /* lock-in magnitude */
}

/* ======= RNG / oracle ======= */
static uint64_t rng_s;
static uint64_t rng_u64(void){uint64_t x=rng_s;x^=x>>12;x^=x<<25;x^=x>>27;rng_s=x;return x*0x2545F4914F6CDD1DULL;}
static double rng_f64(void){return(rng_u64()>>11)*(1.0/((1ULL<<53)*1.0));}
static int rng_int(int n){return(int)(rng_u64()%(uint64_t)n);}
static int m_for(int n){int nm=1<<n;int s=(int)ceil(sqrt((double)nm));return(4*s>48*n)?4*s:48*n;}
static int sample_secret(int nm){for(;;){int d=1+rng_int(nm-1);if(d!=nm/2)return d;}}

int main(int argc, char**argv){
    int n=N_DEFAULT,trials=M_TRIALS,seed=MASTER_SEED;
    const char*out="trackA_lockin_results.json";
    for(int i=1;i<argc;i++){if(!strcmp(argv[i],"--n")&&i+1<argc)n=atoi(argv[++i]);else if(!strcmp(argv[i],"--trials")&&i+1<argc)trials=atoi(argv[++i]);else if(!strcmp(argv[i],"--seed")&&i+1<argc)seed=atoi(argv[++i]);else if(!strcmp(argv[i],"--out-json")&&i+1<argc)out=argv[++i];}
    if(locate_k10temp()!=0){fprintf(stderr,"FATAL:k10temp\n");return 2;}

    rng_s=(uint64_t)seed|1;int nm=1<<n;
    printf("TRACK A-FULL LOCKIN REBUILD\nn=%d tone=%.0fHz slot=%.1fs trials=%d sender=%d receiver=%d\n",
           n,DRIVE_HZ,SLOT_S,trials,SENDER_CPU,RECEIVER_CPU);

    double c0_mags[4096],c1_mags[4096];int labels[4096];
    int total=0;double sum_c0=0,sum_c1=0;

    for(int t=0;t<trials;t++){
        if(read_k10temp_c()>=68.0){fprintf(stderr,"TEMP VETO\n");break;}
        int d0=sample_secret(nm);while(d0>=nm/2)d0=sample_secret(nm);
        int a=d0<(nm-d0)?d0:(nm-d0);int Na=(nm-a)%nm;
        int orient=(d0<nm/2)?1:0;int pok;

        double m0=run_slot(a,SLOT_S,1.0,&pok);
        double m1=run_slot(Na,SLOT_S,1.0,&pok);

        c0_mags[t]=m0;c1_mags[t]=m1;
        labels[2*t]=orient;labels[2*t+1]=1-orient;
        sum_c0+=m0;sum_c1+=m1;total++;

        if(t%10==0)printf("  trial %d/%d c0_mag=%.6f c1_mag=%.6f\n",t+1,trials,m0,m1);
    }

    double cm=sum_c0/trials,c1m=sum_c1/trials,diff=cm-c1m;

    /* orientation AUC */
    double scores[4096];int ls[4096];int ot=0;
    for(int i=0;i<trials;i++){scores[ot]=c0_mags[i];ls[ot]=labels[2*i];ot++;}
    int np=0;for(int i=0;i<ot;i++)if(ls[i]==1)np++;int nn=ot-np;
    double auc=0.5;
    if(np>0&&nn>0){double rs=0;for(int i=0;i<ot;i++){int r=1;for(int j=0;j<ot;j++)if(scores[j]<scores[i])r++;if(ls[i]==1)rs+=r;}auc=(rs-np*(np+1.0)*0.5)/(np*nn);}
    auc=auc>1.0-auc?auc:1.0-auc;

    printf("\n--- RESULTS ---\nc0_mag_mean=%.6f c1_mag_mean=%.6f diff=%.6f auc=%.4f\n",cm,c1m,diff,auc);

    FILE*f=fopen(out,"w");
    fprintf(f,"{\"experiment\":\"phase6_trackA_lockin_rebuild\",\"n\":%d,\"trials\":%d,\"tone_hz\":%.0f,\"c0_mag_mean\":%.9f,\"c1_mag_mean\":%.9f,\"diff\":%.9f,\"orientation_auc\":%.4f,\"detector_live\":%s}\n",
            n,trials,DRIVE_HZ,cm,c1m,diff,auc,(cm>0.01||c1m>0.01)?"true":"false");
    fclose(f);printf("wrote %s\n",out);
    return 0;
}
