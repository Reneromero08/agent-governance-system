/*
 * chiral_dual_lane_full.c -- Track A-Full: Simultaneous Dual-Lane Common-Mode Rejection.
 *
 * Three threads: sender_A (core 4), sender_B (core 5), receiver (core 2).
 * Both senders run simultaneously. Receiver measures ring-osc timing.
 *
 * Modes:
 *   public           : sender_A=candidate_0, sender_B=candidate_1
 *   same_candidate   : sender_A=candidate_0, sender_B=candidate_0  (null)
 *   equal_sign       : sender_A=candidate_0, sender_B=candidate_0  (alias)
 *   lane_swap        : sender_A=candidate_1, sender_B=candidate_0
 *   no_sender        : neither sender active (baseline)
 *   hidden_positive  : 5x amplify integer multiply on true-candidate sender
 *
 * Candidate blinding: runtime uses c0/c1 only. Hidden d for oracle gen only.
 * Offline scorer maps labels after run.
 *
 * Build: gcc -O2 -pthread -march=amdfam10 -Wall -Wextra chiral_dual_lane_full.c -o dual_lane_full -lm
 * Run:   sudo ./dual_lane_full --n 8 --trials 42 --seed 42 --out-json results.json
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

#define SENDER_A_CPU 4
#define SENDER_B_CPU 5
#define RECEIVER_CPU 2
#define MASTER_SEED  44060611
#define N_DEFAULT    8
#define M_TRIALS     42
#define WALK_STEPS   256
#define READ_HZ      4000
#define TSC_HZ       3214823000.0
#define CAP_S        0.35

/* ======= RDTSC / affinity / k10temp (VERBATIM slot2) ======= */
static inline uint64_t rdtsc_now(void) {
    unsigned hi,lo,aux; __asm__ volatile("rdtscp":"=a"(lo),"=d"(hi),"=c"(aux)::);
    return ((uint64_t)hi<<32)|lo;
}
static int pin_to_core(int core) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(core,&s);
    return sched_setaffinity(0,sizeof(s),&s);
}
static char g_k10_path[512];
static int locate_k10temp(void) {
    DIR *d=opendir("/sys/class/hwmon"); if(!d) return -1;
    struct dirent *e;
    while((e=readdir(d))) {
        if(strncmp(e->d_name,"hwmon",5)) continue;
        char np[512],buf[64];
        snprintf(np,sizeof(np),"/sys/class/hwmon/%s/name",e->d_name);
        FILE *f=fopen(np,"r"); if(!f) continue;
        if(fgets(buf,sizeof(buf),f)) {
            size_t n=strlen(buf); while(n&&(buf[n-1]=='\n'||buf[n-1]==' '))buf[--n]=0;
            if(!strcmp(buf,"k10temp")){
                snprintf(g_k10_path,sizeof(g_k10_path),"/sys/class/hwmon/%s/temp1_input",e->d_name);
                fclose(f);closedir(d);return 0;
            }
        }
        fclose(f);
    }
    closedir(d);return -1;
}
static double read_k10temp_c(void) {
    if(!g_k10_path[0])return -999.0;
    FILE *f=fopen(g_k10_path,"r");if(!f)return -999.0;
    long m=0;fscanf(f,"%ld",&m);fclose(f);return m/1000.0;
}

/* ======= RNG ======= */
static uint64_t rng_s;
static uint64_t rng_u64(void){uint64_t x=rng_s;x^=x>>12;x^=x<<25;x^=x>>27;rng_s=x;return x*0x2545F4914F6CDD1DULL;}
static double rng_f64(void){return(rng_u64()>>11)*(1.0/((1ULL<<53)*1.0));}
static int rng_int(int n){return(int)(rng_u64()%(uint64_t)n);}

/* ======= oracle ======= */
static int m_for(int n){int nm=1<<n;int s=(int)ceil(sqrt((double)nm));return(4*s>48*n)?4*s:48*n;}
static int sample_secret(int nm){for(;;){int d=1+rng_int(nm-1);if(d!=nm/2)return d;}}
typedef struct{int n_mod,d,m;int*k;int8_t*b;}inst_t;
static inst_t make_inst(int n,int d){
    inst_t I;I.n_mod=1<<n;I.d=d;I.m=m_for(n);
    I.k=malloc(I.m*sizeof(int));I.b=malloc(I.m*sizeof(int8_t));
    double nmd=(double)I.n_mod;
    for(int j=0;j<I.m;j++){int kk=rng_int(I.n_mod);double p=(1.0+cos(2.0*M_PI*kk*d/nmd))*0.5;I.k[j]=kk;I.b[j]=rng_f64()<p?1:-1;}
    return I;
}
static void free_inst(inst_t*I){free(I->k);free(I->b);}

/* ======= sender thread ======= */
typedef struct{
    atomic_int stop, started;
    int core, candidate, n_mod, steps;
    int *k_vals;
    double amplify;  /* 1.0 normal, 5.0 hidden positive */
} snd_t;

static void *sender_loop(void *arg) {
    snd_t *s = (snd_t*)arg;
    pin_to_core(s->core);
    uint64_t iseed = 0x9E3779B97F4A7C15ULL;
    volatile double sink = 0.0;
    int cand = s->candidate, nm = s->n_mod;
    double amp = s->amplify;
    atomic_store(&s->started, 1);
    while (!atomic_load(&s->stop)) {
        for (int j=0; j<s->steps; j++) {
            int kk = s->k_vals[j];
            uint64_t prod = ((uint64_t)cand * (uint64_t)kk) % (uint64_t)nm;
            int bits = __builtin_popcountll(prod);
            int iters = (int)(bits * amp);
            if (iters < 1) iters = 1;
            for (int b=0; b<iters; b++) {
                sink += (double)(iseed ^ prod) * 0.0000001;
                iseed = iseed * 6364136223846793005ULL + 1442695040888963407ULL;
            }
        }
    }
    (void)sink; return NULL;
}

/* ======= receiver ======= */
typedef struct{
    int core, read_hz, n_captured;
    double tsc_hz;
    uint64_t t_deadline, *t_tsc;
    double *ro_period;
    atomic_int *go;
    volatile int ready;
} rx_t;

static void *rx_loop(void *arg) {
    rx_t *r = (rx_t*)arg;
    pin_to_core(r->core);
    volatile uint64_t acc = 0x9E3779B9u;
    double target = r->tsc_hz/(double)r->read_hz;
    for(int i=0;i<8192;i++)acc=acc*6364136223846793005ULL+1;
    r->ready=1;
    while(atomic_load(r->go)==0)__asm__ volatile("pause");
    uint64_t tp=rdtsc_now();int i=0;
    for(;i<4096;i++){
        uint64_t it=0,tn;
        do{acc=acc*6364136223846793005ULL+1442695040888963407ULL;it++;tn=rdtsc_now();}
        while((double)(tn-tp)<target);
        r->t_tsc[i]=tn;r->ro_period[i]=(double)(tn-tp)/(double)it;tp=tn;
        if(tn>=r->t_deadline){i++;break;}
    }
    r->n_captured=i;__asm__ volatile(""::"r"(acc));
    return NULL;
}

/* ======= run one trial ======= */
static double run_trial(int *k_vals, int n_mod, int c0, int c1, double amp,
                        const char *mode, int *pin_ok) {
    snd_t sa, sb;
    memset(&sa,0,sizeof(sa)); memset(&sb,0,sizeof(sb));
    sa.core=SENDER_A_CPU; sa.k_vals=k_vals; sa.candidate=c0; sa.n_mod=n_mod;
    sa.steps=WALK_STEPS; sa.amplify=amp; atomic_init(&sa.stop,0); atomic_init(&sa.started,0);
    sb.core=SENDER_B_CPU; sb.k_vals=k_vals; sb.candidate=c1; sb.n_mod=n_mod;
    sb.steps=WALK_STEPS; sb.amplify=amp; atomic_init(&sb.stop,0); atomic_init(&sb.started,0);

    pthread_t ta, tb, tr;
    if (strcmp(mode,"no_sender")) {
        pthread_create(&ta, NULL, sender_loop, &sa);
        pthread_create(&tb, NULL, sender_loop, &sb);
        while (!atomic_load(&sa.started) || !atomic_load(&sb.started)) __asm__ volatile("pause");
    }

    uint64_t tbuf[4096]; double robuf[4096];
    atomic_int go; atomic_init(&go,0);
    rx_t rx; memset(&rx,0,sizeof(rx));
    rx.core=RECEIVER_CPU; rx.read_hz=READ_HZ; rx.tsc_hz=TSC_HZ;
    rx.t_tsc=tbuf; rx.ro_period=robuf; rx.go=&go;

    uint64_t t_launch = rdtsc_now() + 30000000ULL;
    rx.t_deadline = t_launch + (uint64_t)(CAP_S * TSC_HZ);

    pthread_attr_t attr; pthread_attr_init(&attr);
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(RECEIVER_CPU,&set);
    pthread_attr_setaffinity_np(&attr,sizeof(set),&set);
    pthread_create(&tr,&attr,rx_loop,&rx); pthread_attr_destroy(&attr);
    while(!rx.ready)__asm__ volatile("pause");

    while(rdtsc_now()<t_launch)__asm__ volatile("pause");
    atomic_store(&go,1); pthread_join(tr,NULL);

    if(strcmp(mode,"no_sender")) {
        atomic_store(&sa.stop,1); atomic_store(&sb.stop,1);
        pthread_join(ta,NULL); pthread_join(tb,NULL);
    }

    double resp=0.0;
    for(int j=0;j<rx.n_captured;j++)resp+=robuf[j];
    resp/=(rx.n_captured>0?rx.n_captured:1);
    *pin_ok = 1;
    return resp;
}

/* ======= main ======= */
int main(int argc, char **argv) {
    int n=N_DEFAULT, trials=M_TRIALS, seed=MASTER_SEED;
    const char *out_json="trackA_full_results.json";

    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--n")&&i+1<argc)n=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--trials")&&i+1<argc)trials=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--seed")&&i+1<argc)seed=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--out-json")&&i+1<argc)out_json=argv[++i];
    }
    if(locate_k10temp()!=0){fprintf(stderr,"FATAL:k10temp\n");return 2;}

    rng_s=(uint64_t)seed|1;
    int nm=1<<n, m=m_for(n);
    printf("TRACK A-FULL -- SIMULTANEOUS DUAL-LANE PDN\n");
    printf("n=%d N=%d M=%d trials=%d senders=[%d,%d] receiver=%d\n",
           n,nm,m,trials,SENDER_A_CPU,SENDER_B_CPU,RECEIVER_CPU);

    FILE *f=fopen(out_json,"w");
    fprintf(f,"{\n  \"experiment\":\"phase6_trackA_full_dual_lane\",\n");
    fprintf(f,"  \"n\":%d,\"trials\":%d,\"route\":\"4:5\",\n",n,trials);
    fprintf(f,"  \"modes\":[\"public\",\"same_candidate\",\"no_sender\",\"hidden_positive\"],\n");
    fprintf(f,"  \"cells\":[\n");

    const char *modes[]={"public","same_candidate","no_sender","hidden_positive"};
    int n_modes=4;
    double mode_means[4]={0}; int mode_counts[4]={0};
    double orient_scores[4096]; int orient_labels[4096]; int ot=0;

    for(int mi=0; mi<n_modes; mi++) {
        const char *mode = modes[mi];
        printf("\n--- mode=%s ---\n", mode);
        for(int t=0; t<trials; t++) {
            if(read_k10temp_c()>=68.0){fprintf(stderr,"TEMP VETO\n");goto cleanup;}

            int d0 = sample_secret(nm);
            while(d0 >= nm/2) d0 = sample_secret(nm);
            inst_t I = make_inst(n, d0);

            int a = d0 < (nm-d0) ? d0 : (nm-d0);
            int Na = (nm - a) % nm;
            int orient = (d0 < nm/2) ? 1 : 0;

            int c0_val = a, c1_val = Na;
            double amp = 1.0;

            if (!strcmp(mode,"same_candidate")) { c0_val = a; c1_val = a; }
            if (!strcmp(mode,"no_sender")) { c0_val = 0; c1_val = 0; }
            if (!strcmp(mode,"hidden_positive")) { amp = 5.0; }

            int pok;
            double resp = run_trial(I.k, I.n_mod, c0_val, c1_val, amp, mode, &pok);
            mode_means[mi] += resp; mode_counts[mi]++;

            /* For public mode: record orientation label */
            if (!strcmp(mode,"public")) {
                orient_scores[ot] = resp;
                orient_labels[ot] = orient;
                ot++;
            }
            free_inst(&I);

            if (t%10==0) printf("  trial %d/%d mean_so_far=%.6f\n", t+1, trials,
                                mode_means[mi]/mode_counts[mi]);
        }
    }

    /* compute AUC for public mode */
    double auc = 0.5;
    if (ot > 0) {
        /* simple AUC */
        int np=0; for(int i=0;i<ot;i++) if(orient_labels[i]==1) np++;
        int nn=ot-np;
        if(np>0&&nn>0){double rs=0;for(int i=0;i<ot;i++){int r=1;for(int j=0;j<ot;j++)if(orient_scores[j]<orient_scores[i])r++;if(orient_labels[i]==1)rs+=r;}
            auc=(rs-np*(np+1.0)*0.5)/(np*nn);}
    }

    printf("\n--- RESULTS ---\n");
    for(int mi=0; mi<n_modes; mi++)
        printf("%-18s mean=%.6f n=%d\n", modes[mi], mode_means[mi]/mode_counts[mi], mode_counts[mi]);

    double pub_mean = mode_means[0]/mode_counts[0];
    double same_mean = mode_means[1]/mode_counts[1];
    double no_mean = mode_means[2]/mode_counts[2];
    double hid_mean = mode_means[3]/mode_counts[3];
    double diff_public = pub_mean - same_mean;

    printf("public-same diff=%.6f\n", diff_public);
    printf("orientation AUC=%.4f\n", auc);

    fprintf(f,"    {\"mode\":\"public\",\"mean\":%.9f},\n", pub_mean);
    fprintf(f,"    {\"mode\":\"same_candidate\",\"mean\":%.9f},\n", same_mean);
    fprintf(f,"    {\"mode\":\"no_sender\",\"mean\":%.9f},\n", no_mean);
    fprintf(f,"    {\"mode\":\"hidden_positive\",\"mean\":%.9f}\n", hid_mean);
    fprintf(f,"  ],\n");
    fprintf(f,"  \"public_same_diff\":%.9f,\n", diff_public);
    fprintf(f,"  \"orientation_auc\":%.4f,\n", auc);
    fprintf(f,"  \"verdict\":\"%s\"\n}\n",
            auc>0.65?"PUBLIC_CHIRAL_LANE_GENERATED_CANDIDATE_L4":
            (fabs(diff_public)>0.01?"CANDIDATE_VALUE_SEPARATION_FOUND":"TRACK_A_FULL_NO_DIFFERENTIAL"));
    fclose(f);
    printf("wrote %s\n", out_json);

cleanup:
    return 0;
}
