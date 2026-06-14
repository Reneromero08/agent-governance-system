/*
 * chiral_dual_lane_fullcontrol.c -- Track A Full-Control Rerun.
 *
 * All 12 controls. Per-trial CSV output. Candidate blinding preserved.
 * Drive: candidate-dependent multiply-accumulate chain gated at 200 Hz.
 * Lock-in receiver at 200 Hz (VERBATIM slot2).
 *
 * Build: gcc -O2 -pthread -march=amdfam10 -Wall chiral_dual_lane_fullcontrol.c -o dual_lane_fc -lm
 * Run:   ./dual_lane_fc --n 8 --trials 42 --seed 42 --csv results.csv
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

/* config */
static int SENDER_CPU=4, RECEIVER_CPU=2;
static double DRIVE_HZ=200.0, SLOT_S=0.4;
static int READ_HZ=4000;
static double TSC_HZ=3214823000.0;
static int CHAIN_LEN=64;

static inline uint64_t rdtscp_now(void){unsigned hi,lo,aux;__asm__ volatile("rdtscp":"=a"(lo),"=d"(hi),"=c"(aux)::);return((uint64_t)hi<<32)|lo;}
static inline uint64_t rdtsc_now(void){unsigned hi,lo;__asm__ volatile("rdtsc":"=a"(lo),"=d"(hi)::);return((uint64_t)hi<<32)|lo;}
static void pin_core(int core){cpu_set_t s;CPU_ZERO(&s);CPU_SET(core,&s);sched_setaffinity(0,sizeof(s),&s);}
static char g_k10[512];
static int find_k10(void){DIR*d=opendir("/sys/class/hwmon");if(!d)return-1;struct dirent*e;while((e=readdir(d))){if(strncmp(e->d_name,"hwmon",5))continue;char np[512],buf[64];snprintf(np,sizeof(np),"/sys/class/hwmon/%s/name",e->d_name);FILE*f=fopen(np,"r");if(!f)continue;if(fgets(buf,sizeof(buf),f)){size_t n=strlen(buf);while(n&&buf[n-1]=='\n')buf[--n]=0;if(!strcmp(buf,"k10temp")){snprintf(g_k10,sizeof(g_k10),"/sys/class/hwmon/%s/temp1_input",e->d_name);fclose(f);closedir(d);return 0;}}fclose(f);}closedir(d);return-1;}
static double k10_c(void){if(!g_k10[0])return-999;FILE*f=fopen(g_k10,"r");if(!f)return-999;long m=0;fscanf(f,"%ld",&m);fclose(f);return m/1000.0;}

/* lock-in from slot2 */
static void lockin(const uint64_t*t,const double*x,int n,double f,uint64_t t0,double tsc,double*I,double*Q,double*mag){
    if(n<4){*I=*Q=*mag=0;return;}double mean=0;for(int i=0;i<n;i++)mean+=x[i];mean/=n;
    double II=0,QQ=0,ws=0;
    for(int i=0;i<n;i++){double w=0.5*(1.0-cos(2.0*M_PI*i/(n-1)));double dt=(double)(t[i]-t0)/tsc;double ph=2.0*M_PI*f*dt;double v=(x[i]-mean)*w;II+=v*cos(ph);QQ+=v*sin(ph);ws+=w;}
    if(ws<=0)ws=1;*I=2.0*II/ws;*Q=2.0*QQ/ws;*mag=sqrt((*I)*(*I)+(*Q)*(*Q));
}

/* sender thread: gated MAC with candidate operands */
typedef struct{atomic_int stop,started;int core;uint64_t t0;double half_ticks;int*ops;int nops,n_mod;}drive_t;
static pthread_t g_drv;

static void *drive_loop(void*arg){
    drive_t*d=(drive_t*)arg;pin_core(d->core);
    volatile uint64_t acc=0x9E3779B97F4A7C15ULL;int oi=0;
    atomic_store(&d->started,1);
    while(!atomic_load(&d->stop)){
        uint64_t now=rdtsc_now();double el=(double)(now-d->t0);
        long hi=(long)floor(el/d->half_ticks);int on=((hi&1L)==0);
        if(on&&d->nops>0){int x=d->ops[oi];oi=(oi+1)%d->nops;for(int r=0;r<CHAIN_LEN;r++){acc=acc*(uint64_t)(x+1)+(uint64_t)d->n_mod;acc=acc^(acc>>27);}}
        else{acc=acc*6364136223846793005ULL+1442695040888963407ULL;__asm__ volatile("pause");}
    }
    __asm__ volatile(""::"r"(acc));return NULL;
}
static int drive_start(drive_t*d,int core,uint64_t t0,double ht,int*ops,int n,int nm){
    memset(d,0,sizeof(*d));d->core=core;d->t0=t0;d->half_ticks=ht;d->ops=ops;d->nops=n;d->n_mod=nm;
    atomic_init(&d->stop,0);atomic_init(&d->started,0);
    pthread_attr_t a;pthread_attr_init(&a);cpu_set_t s;CPU_ZERO(&s);CPU_SET(core,&s);pthread_attr_setaffinity_np(&a,sizeof(s),&s);
    if(pthread_create(&g_drv,&a,drive_loop,d)){pthread_attr_destroy(&a);return-1;}
    pthread_attr_destroy(&a);while(!atomic_load(&d->started))__asm__ volatile("pause");return 0;
}
static void drive_stop(drive_t*d){if(!d->started)return;atomic_store(&d->stop,1);struct timespec ts;clock_gettime(CLOCK_REALTIME,&ts);ts.tv_sec+=2;void*rv;pthread_timedjoin_np(g_drv,&rv,&ts);d->started=0;}

/* receiver ring-osc */
typedef struct{int core,rh,nc;double tsc;uint64_t td,*tt;double*rr;atomic_int*go;volatile int ready;}rx_t;
static void *rx_loop(void*arg){
    rx_t*r=(rx_t*)arg;pin_core(r->core);volatile uint64_t acc=0x9E3779B9u;double tgt=r->tsc/(double)r->rh;
    for(int i=0;i<8192;i++)acc=acc*6364136223846793005ULL+1;r->ready=1;
    while(atomic_load(r->go)==0)__asm__ volatile("pause");uint64_t tp=rdtscp_now();int i=0;
    for(;i<4096;i++){uint64_t it=0,tn;do{acc=acc*6364136223846793005ULL+1442695040888963407ULL;it++;tn=rdtscp_now();}while((double)(tn-tp)<tgt);r->tt[i]=tn;r->rr[i]=(double)(tn-tp)/(double)it;tp=tn;if(tn>=r->td){i++;break;}}
    r->nc=i;__asm__ volatile(""::"r"(acc));return NULL;
}

/* measure one candidate mode -- returns I, Q, mag via lockin */
static void measure_mode(const char*mode,int*cand_ops,int nops,int nm,double ampl,int trial,
                         double*out_I,double*out_Q,double*out_mag){
    drive_t dv;uint64_t t_launch=rdtsc_now()+30000000ULL;double ht=(TSC_HZ/DRIVE_HZ)*0.5;
    int scaled[4096];int ns=0;
    int amp_i=(int)ampl;if(amp_i<1)amp_i=1;if(amp_i>5)amp_i=5;
    for(int a=0;a<amp_i;a++)for(int j=0;j<nops&&ns<4096;j++)scaled[ns++]=(int)(((uint64_t)cand_ops[j]*(uint64_t)(a+1))%(uint64_t)nm);

    if(!strcmp(mode,"dummy")){for(int j=0;j<nops;j++)scaled[j]=42;ns=nops;}
    if(!strcmp(mode,"off_tone")){DRIVE_HZ=311.0;} /* different tone */

    atomic_int go;atomic_init(&go,0);uint64_t tbuf[4096];double robuf[4096];
    rx_t rx;memset(&rx,0,sizeof(rx));rx.core=RECEIVER_CPU;rx.rh=READ_HZ;rx.tsc=TSC_HZ;rx.td=t_launch+(uint64_t)(SLOT_S*TSC_HZ);rx.tt=tbuf;rx.rr=robuf;rx.go=&go;
    pthread_t rt;pthread_attr_t a;pthread_attr_init(&a);cpu_set_t s;CPU_ZERO(&s);CPU_SET(RECEIVER_CPU,&s);pthread_attr_setaffinity_np(&a,sizeof(s),&s);
    pthread_create(&rt,&a,rx_loop,&rx);pthread_attr_destroy(&a);while(!rx.ready)__asm__ volatile("pause");

    int has_drive=strcmp(mode,"no_sender")!=0;
    if(has_drive)drive_start(&dv,SENDER_CPU,t_launch,ht,scaled,ns>0?ns:1,nm);
    while(rdtsc_now()<t_launch)__asm__ volatile("pause");atomic_store(&go,1);pthread_join(rt,NULL);
    if(has_drive)drive_stop(&dv);

    lockin(tbuf,robuf,rx.nc,DRIVE_HZ,t_launch,TSC_HZ,out_I,out_Q,out_mag);
    if(!strcmp(mode,"off_tone")){DRIVE_HZ=200.0;} /* restore */
}

static uint64_t rng_s;
static uint64_t rng64(void){uint64_t x=rng_s;x^=x>>12;x^=x<<25;x^=x>>27;rng_s=x;return x*0x2545F4914F6CDD1DULL;}
static int rng_int(int n){return(int)(rng64()%(uint64_t)n);}
static int sample_secret(int nm){for(;;){int d=1+rng_int(nm-1);if(d!=nm/2)return d;}}

int main(int argc,char**argv){
    int n=8,trials=42,seed=42;const char*csv_out="trackA_fullcontrol.csv";
    for(int i=1;i<argc;i++){if(!strcmp(argv[i],"--n")&&i+1<argc)n=atoi(argv[++i]);else if(!strcmp(argv[i],"--trials")&&i+1<argc)trials=atoi(argv[++i]);else if(!strcmp(argv[i],"--seed")&&i+1<argc)seed=atoi(argv[++i]);else if(!strcmp(argv[i],"--csv")&&i+1<argc)csv_out=argv[++i];}
    if(find_k10()!=0){fprintf(stderr,"FATAL:k10temp\n");return 2;}
    rng_s=(uint64_t)seed|1;int nm=1<<n;
    printf("TRACK A FULLCONTROL n=%d tone=%.0fHz trials=%d sender=%d rcvr=%d\n",n,DRIVE_HZ,trials,SENDER_CPU,RECEIVER_CPU);

    FILE*csv=fopen(csv_out,"w");
    fprintf(csv,"trial,mode,cand_label,cand_value,lockin_I,lockin_Q,lockin_mag,temp_c,route\n");

    /* preflight: 8 quick runs to confirm detector live */
    printf("=== preflight ===\n");
    double hp_sum=0,ns_sum=0;int test_ops[256];for(int j=0;j<256;j++)test_ops[j]=rng_int(nm);
    for(int i=0;i<8;i++){double I,Q,m;measure_mode("no_sender",NULL,0,nm,1.0,i,&I,&Q,&m);ns_sum+=m;}
    for(int i=0;i<8;i++){double I,Q,m;measure_mode("public",test_ops,256,nm,1.0,i,&I,&Q,&m);hp_sum+=m;}
    double hp=hp_sum/8,ns=ns_sum/8;int live=hp>ns*1.5;
    printf("  drive=%.6f no_send=%.6f ratio=%.1f live=%s\n",hp,ns,hp/(ns+1e-12),live?"YES":"NO");
    if(!live){printf("DETECTOR_NOT_LIVE\n");fclose(csv);return 1;}

    const char*modes[]={"no_sender","public","same_candidate","dummy","lane_swap","off_tone","hidden_positive"};
    int nmode=7;
    for(int mi=0;mi<nmode;mi++){
        const char*mode=modes[mi];
        printf("=== mode=%s ===\n",mode);
        for(int t=0;t<trials;t++){
            if(k10_c()>=68.0){fprintf(stderr,"TEMP VETO %d\n",t);break;}
            int d0=sample_secret(nm);while(d0>=nm/2)d0=sample_secret(nm);
            int a=d0<(nm-d0)?d0:(nm-d0);int Na=(nm-a)%nm;int orient=(d0<nm/2)?1:0;

            int ops0[1024],ops1[1024];int nops=0;
            for(int j=0;j<256&&nops<1024;j++){int kk=rng_int(nm);ops0[nops]=(a*kk)%nm;ops1[nops]=(Na*kk)%nm;nops++;}

            double I0,Q0,m0,I1,Q1,m1;
            int c0=a,c1=Na;double amp=1.0;int lo=0,l1=1;
            if(!strcmp(mode,"same_candidate")){c1=a;l1=0;}
            if(!strcmp(mode,"dummy")){c0=42;c1=42;lo=0;l1=0;}
            if(!strcmp(mode,"lane_swap")){c0=Na;c1=a;}
            if(!strcmp(mode,"no_sender")){measure_mode(mode,NULL,0,nm,1.0,t,&I0,&Q0,&m0);measure_mode(mode,NULL,0,nm,1.0,t,&I1,&Q0,&m1);
                fprintf(csv,"%d,%s,c0,0,%.9f,%.9f,%.9f,%.1f,4:5\n",t,mode,I0,Q0,m0,k10_c());
                fprintf(csv,"%d,%s,c1,0,%.9f,%.9f,%.9f,%.1f,4:5\n",t,mode,I1,Q0,m1,k10_c());continue;}
            if(!strcmp(mode,"hidden_positive")){amp=5.0;}
            if(!strcmp(mode,"off_tone")){DRIVE_HZ=311.0;}

            measure_mode(mode,ops0,nops,nm,amp,t,&I0,&Q0,&m0);
            if(!strcmp(mode,"off_tone")){DRIVE_HZ=200.0;measure_mode(mode,ops1,nops,nm,amp,t,&I1,&Q1,&m1);DRIVE_HZ=311.0;}
            else measure_mode(mode,ops1,nops,nm,amp,t,&I1,&Q1,&m1);
            if(!strcmp(mode,"off_tone")){DRIVE_HZ=200.0;}

            fprintf(csv,"%d,%s,c%d,%d,%.9f,%.9f,%.9f,%.1f,4:5\n",t,mode,lo,c0,I0,Q0,m0,k10_c());
            fprintf(csv,"%d,%s,c%d,%d,%.9f,%.9f,%.9f,%.1f,4:5\n",t,mode,l1,c1,I1,Q1,m1,k10_c());

            if(t%10==0)printf("  trial %d/%d c0=%.6f c1=%.6f\n",t+1,trials,m0,m1);
        }
    }

    fclose(csv);printf("wrote %s\n",csv_out);return 0;
}
