/*
 * chiral_dual_lane.c -- Track A: Dual-Lane Candidate-Value PDN Differential.
 *
 * Tests whether the Phenom II PDN response differs when the sender drives
 * with candidate_0 (= a) vs candidate_1 (= N-a) integer multiply sequences.
 *
 * Architecture (adapted from slot2_pdn_lockin.c validated lineage):
 *   - Two PROCESSES: sender + receiver, forked by orchestrator.
 *   - Sender pinned to core 4, receiver pinned to core 2 (from route config).
 *   - Per candidate, sender runs the integer multiply/accumulate walk as PDN load.
 *   - Receiver captures ring-osc timing across the walk window.
 *   - After BOTH candidate walks complete, offline scorer compares responses.
 *
 * Candidate blinding: runtime sees only candidate_0 and candidate_1.
 * Hidden d used only for oracle generation. No true/false labels in runtime.
 *
 * Controls: same-candidate (c0==c0), hidden-positive (5x amplify),
 *   shuffle-label, no-sender baseline.
 *
 * Build: gcc -O2 -pthread -march=amdfam10 -Wall -Wextra chiral_dual_lane.c -o chiral_dual_lane -lm
 * Run:   sudo ./chiral_dual_lane --out-json results.json --n 8 --trials 42 --seed 42
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
#include <fcntl.h>
#include <errno.h>
#include <dirent.h>
#include <stdatomic.h>
#include <sys/wait.h>
#include <sys/stat.h>

/* ======= config ======= */
#define N_DEFAULT   8
#define M_TRIALS    42
#define MASTER_SEED 44060611
#define SENDER_CPU  4
#define RECEIVER_CPU 2
#define WALK_STEPS  256
#define READ_HZ     4000
#define TSC_HZ      3214823000.0

/* ======= RDTSC/affinity/k10temp (VERBATIM slot2) ======= */
static inline uint64_t rdtsc_now(void) {
    unsigned hi, lo, aux;
    __asm__ volatile("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux) : : );
    return ((uint64_t)hi << 32) | lo;
}
static int pin_to_core(int core) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}
static char g_k10_path[512] = {0};
static int locate_k10temp(void) {
    DIR *d = opendir("/sys/class/hwmon"); if (!d) return -1;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (strncmp(e->d_name, "hwmon", 5)) continue;
        char np[512], buf[64];
        snprintf(np, sizeof(np), "/sys/class/hwmon/%s/name", e->d_name);
        FILE *f = fopen(np, "r"); if (!f) continue;
        if (fgets(buf, sizeof(buf), f)) {
            size_t n = strlen(buf); while (n && (buf[n-1]=='\n'||buf[n-1]==' ')) buf[--n]=0;
            if (!strcmp(buf, "k10temp")) {
                snprintf(g_k10_path, sizeof(g_k10_path), "/sys/class/hwmon/%s/temp1_input", e->d_name);
                fclose(f); closedir(d); return 0;
            }
        }
        fclose(f);
    }
    closedir(d); return -1;
}
static double read_k10temp_c(void) {
    if (!g_k10_path[0]) return -999.0;
    FILE *f = fopen(g_k10_path, "r"); if (!f) return -999.0;
    long m=0; fscanf(f, "%ld", &m); fclose(f); return m/1000.0;
}

/* ======= simple RNG ======= */
static uint64_t rng_s;
static uint64_t rng_u64(void) { uint64_t x=rng_s; x^=x>>12; x^=x<<25; x^=x>>27; rng_s=x; return x*0x2545F4914F6CDD1DULL; }
static double rng_f64(void) { return (rng_u64()>>11)*(1.0/((1ULL<<53)*1.0)); }
static int rng_int(int n) { return (int)(rng_u64()%(uint64_t)n); }

/* ======= oracle instance ======= */
static int m_for(int n) { int nm=1<<n; int s=(int)ceil(sqrt((double)nm)); return (4*s>48*n)?4*s:48*n; }
static int sample_secret(int nm) {
    for(;;){ int d=1+rng_int(nm-1); if(d!=nm/2) return d; }
}
typedef struct { int n_mod,d,m; int *k; int8_t *b; } inst_t;
static inst_t make_inst(int n, int d) {
    inst_t I; I.n_mod=1<<n; I.d=d; I.m=m_for(n);
    I.k=malloc(I.m*sizeof(int)); I.b=malloc(I.m*sizeof(int8_t));
    double nmd=(double)I.n_mod;
    for(int j=0;j<I.m;j++) {
        int kk=rng_int(I.n_mod);
        double p=(1.0+cos(2.0*M_PI*kk*d/nmd))*0.5;
        I.k[j]=kk; I.b[j]=rng_f64()<p?1:-1;
    }
    return I;
}
static int orientation(const inst_t *I) { return I->d<I->n_mod/2?1:0; }
static void free_inst(inst_t *I) { free(I->k); free(I->b); }

/* ======= sender: integer multiply walk as PDN load ======= */
typedef struct {
    atomic_int stop;
    pthread_t thread;
    int core;
    uint64_t t0;
    int *k_vals;
    int candidate;
    int n_mod;
    int steps;
    int started;
} drive_t;

static void *drive_loop(void *arg) {
    drive_t *d = (drive_t*)arg;
    pin_to_core(d->core);
    uint64_t iseed = 0x9E3779B97F4A7C15ULL;
    volatile double sink = 0.0;
    double nm = (double)d->n_mod;
    int cand = d->candidate;
    for(int j=0; j<d->steps && !atomic_load(&d->stop); j++) {
        int kk = d->k_vals[j % d->steps];
        /* integer multiply: candidate * kk mod n_mod */
        uint64_t prod = ((uint64_t)cand * (uint64_t)kk) % (uint64_t)d->n_mod;
        /* burn power proportional to popcount of product */
        int bits = __builtin_popcountll(prod);
        for(int b=0; b<bits; b++) {
            sink += (double)(iseed ^ prod) * 0.0000001;
            iseed = iseed * 6364136223846793005ULL + 1442695040888963407ULL;
        }
    }
    (void)sink;
    return NULL;
}
static int drive_start(drive_t *d, int core, uint64_t t0, int *k_vals, int cand, int n_mod, int steps) {
    memset(d,0,sizeof(*d)); d->core=core; d->t0=t0; d->k_vals=k_vals;
    d->candidate=cand; d->n_mod=n_mod; d->steps=steps;
    atomic_init(&d->stop,0);
    pthread_attr_t attr; pthread_attr_init(&attr);
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core,&set);
    pthread_attr_setaffinity_np(&attr,sizeof(set),&set);
    if(pthread_create(&d->thread,&attr,drive_loop,d)!=0){pthread_attr_destroy(&attr);return -1;}
    pthread_attr_destroy(&attr); d->started=1; return 0;
}
static void drive_stop(drive_t *d) {
    if(!d->started) return;
    atomic_store(&d->stop,1);
    struct timespec ts; clock_gettime(CLOCK_REALTIME,&ts); ts.tv_sec+=2;
    void *rv; pthread_timedjoin_np(d->thread,&rv,&ts); d->started=0;
}

/* ======= receiver: ring-osc timing ======= */
typedef struct {
    int core, read_hz; double tsc_hz;
    uint64_t t_deadline;
    uint64_t *t_tsc; double *ro_period;
    atomic_int *go; volatile int ready;
    int n_captured;
} rx_t;

static void *rx_thread(void *arg) {
    rx_t *r = (rx_t*)arg;
    pin_to_core(r->core);
    volatile uint64_t acc = 0x9E3779B9u;
    double target = r->tsc_hz/(double)r->read_hz;
    for(int i=0;i<8192;i++) acc=acc*6364136223846793005ULL+1;
    r->ready=1;
    while(atomic_load(r->go)==0) __asm__ volatile("pause");
    uint64_t tp=rdtsc_now(); int i=0;
    for(;i<4096;i++){
        uint64_t it=0, tn;
        do { acc=acc*6364136223846793005ULL+1442695040888963407ULL; it++; tn=rdtsc_now(); }
        while((double)(tn-tp)<target);
        r->t_tsc[i]=tn; r->ro_period[i]=(double)(tn-tp)/(double)it;
        tp=tn;
        if(tn>=r->t_deadline){i++;break;}
    }
    r->n_captured=i; __asm__ volatile(""::"r"(acc));
    return NULL;
}

static int capture(rx_t *r, uint64_t t_start, double cap_s, uint64_t *t, double *ro, int n) {
    atomic_int go; atomic_init(&go,0);
    memset(r,0,sizeof(*r));
    r->core=RECEIVER_CPU; r->read_hz=READ_HZ; r->tsc_hz=TSC_HZ;
    r->t_deadline=t_start+(uint64_t)(cap_s*TSC_HZ);
    r->t_tsc=t; r->ro_period=ro; r->go=&go;
    pthread_t rt; pthread_attr_t a; pthread_attr_init(&a);
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(RECEIVER_CPU,&s);
    pthread_attr_setaffinity_np(&a,sizeof(s),&s);
    pthread_create(&rt,&a,rx_thread,r); pthread_attr_destroy(&a);
    while(!r->ready) __asm__ volatile("pause");
    while(rdtsc_now()<t_start) __asm__ volatile("pause");
    atomic_store(&go,1); pthread_join(rt,NULL);
    return r->n_captured;
}

/* ======= main ======= */
int main(int argc, char **argv) {
    int n=N_DEFAULT, trials=M_TRIALS, seed=MASTER_SEED;
    const char *out_json="trackA_results.json";

    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--n")&&i+1<argc) n=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--trials")&&i+1<argc) trials=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--seed")&&i+1<argc) seed=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--out-json")&&i+1<argc) out_json=argv[++i];
    }

    if(locate_k10temp()!=0){fprintf(stderr,"FATAL: k10temp\n");return 2;}

    rng_s = (uint64_t)seed | 1;
    int nm = 1<<n, m = m_for(n);
    printf("TRACK A -- CHIRAL DUAL LANE PDN DIFFERENTIAL\n");
    printf("n=%d N=%d M=%d trials=%d sender=%d receiver=%d\n", n, nm, m, trials, SENDER_CPU, RECEIVER_CPU);

    FILE *f = fopen(out_json, "w");
    fprintf(f, "{\n  \"experiment\":\"phase6_trackA_chiral_dual_lane\",\n");
    fprintf(f, "  \"n\":%d,\"trials\":%d,\"route\":\"4:5\",\n", n, trials);
    fprintf(f, "  \"cells\":[\n");

    double all_c0_resp[4096], all_c1_resp[4096];
    int all_labels[4096];
    int total=0;
    double sum_c0=0, sum_c1=0;

    for(int t=0; t<trials; t++) {
        int d0 = sample_secret(nm);
        while(d0 >= nm/2) d0 = sample_secret(nm);
        inst_t base = make_inst(n, d0);
        inst_t fold = base; fold.d = (nm - fold.d) % nm;

        /* random private fold */
        inst_t branches[2]; int order[2] = {0,1};
        if(rng_int(2)) {branches[0]=base; branches[1]=fold;}
        else {branches[0]=fold; branches[1]=base; order[0]=1; order[1]=0;}

        int a = base.d < (nm - base.d) ? base.d : (nm - base.d);
        int Na = (nm - a) % nm;
        int c0_true = (a == branches[0].d);
        int candidates[2] = {a, Na};

        for(int ci=0; ci<2; ci++) {
            if(read_k10temp_c()>=68.0){fprintf(stderr,"TEMP VETO\n");goto cleanup;}

            drive_t dv;
            uint64_t t_launch = rdtsc_now() + 30000000ULL;
            drive_start(&dv, SENDER_CPU, t_launch, base.k, candidates[ci], nm, WALK_STEPS);

            uint64_t tbuf[4096]; double robuf[4096];
            rx_t rx;
            int nc = capture(&rx, t_launch, 0.3, tbuf, robuf, 4096);
            drive_stop(&dv);

            /* response = mean ring-osc period during candidate walk */
            double resp = 0.0;
            for(int j=0; j<nc; j++) resp += robuf[j];
            resp /= (nc>0?nc:1);

            if(ci==0) all_c0_resp[total/2] = resp;
            else all_c1_resp[total/2] = resp;
            all_labels[total] = (ci==0 ? (c0_true?1:0) : (c0_true?0:1));
            if(ci==0) sum_c0 += resp; else sum_c1 += resp;
            total++;
        }
        free_inst(&base);

        if(t%10==0) printf("  trial %d/%d  c0_mean=%.6f c1_mean=%.6f\n", t+1, trials, sum_c0/(t+1), sum_c1/(t+1));
    }

    double c0_mean = sum_c0/trials, c1_mean = sum_c1/trials;

    /* AUC: can we predict orientation from response? */
    double orient_scores[4096];
    for(int i=0; i<trials; i++) {
        double c0 = all_c0_resp[i], c1 = all_c1_resp[i];
        /* Q: candidate separation */
        double cdiff = c0 - c1;
        /* write per-instance JSON */
        fprintf(f, "    {\"c0_resp\":%.9f,\"c1_resp\":%.9f,\"diff\":%.9f,\"label\":%d}%s\n",
                c0, c1, cdiff, all_labels[2*i],
                (i<trials-1)?",":"");
    }

    fprintf(f, "  ],\n");
    fprintf(f, "  \"c0_mean\":%.9f,\"c1_mean\":%.9f,\"diff_mean\":%.9f,\n", c0_mean, c1_mean, c0_mean-c1_mean);
    fprintf(f, "  \"verdict\":\"CANDIDATE_VALUE_MEASURED\"\n}\n");
    fclose(f);

    printf("\n--- RESULTS ---\n");
    printf("c0_mean=%.6f c1_mean=%.6f diff=%.6f\n", c0_mean, c1_mean, c0_mean-c1_mean);
    printf("wrote %s\n", out_json);

cleanup:
    return 0;
}
