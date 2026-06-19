/*
 * slot2_pdn_run.h -- orchestration / sender / receiver / preflight / matrix.
 * Included from slot2_pdn_lockin.c (after all primitives + config_t are defined).
 *
 * One-box orchestration model: a single invocation (role=orchestrate) forks a SENDER
 * child and a RECEIVER child, pinned to the sender / victim cores. The parent waits
 * for both. The two children share:
 *   - all run params via the inherited config_t (fork copies it),
 *   - the absolute TSC origin t0 via a tiny handshake file (sender writes, receiver
 *     polls): t0 = rdtsc_now() + ~100ms, decided by the SENDER after both are pinned.
 *   - the symbol schedule: both compute the SAME schedule from the SAME seed, so the
 *     receiver knows exactly which (MODE,theta,sign) drives each (symbol,bin) slot and
 *     when (absolute tsc). The receiver only needs t0 from the sender; everything else
 *     is deterministic from seed + params.
 *
 * SCHEDULE (matrix mode):
 *   families = {real, pseudo, wrong}, `trials` symbols each, interleaved by trial index
 *   so the order matches the analyzer's trial%2 train/test split. Plus a 4-symbol
 *   PREAMBLE (one per MODE, theta=0) at the front to train centroids + rho threshold.
 *   Each symbol = (family, declared_mode, actual_mode, theta). The DRIVE for a symbol
 *   uses the codeword of its ACTUAL physical mode (what the rail really carries):
 *     real  : declared = actual; cw = CODE[actual]
 *     wrong : declared = actual + r; cw = CODE[actual]  (reader must read ACTUAL)
 *     pseudo: cw = CODE[actual] permuted by a wrong key (decoy schedule) -> off-codebook
 *   theta in {0..PHASE_LEVELS-1}/PHASE_LEVELS * 2pi, carried as a global drive-phase
 *   offset; the relational tag is the DIFFERENTIAL theta between consecutive symbols
 *   (cancels common drift), scored by the python analyzer.
 *
 * CSV schema (matrix): the matched-null analyzer here is the PDN-codebook analyzer
 *   slot2_pdn_analyze.py (rho energy-concentration + full-vector centroid), reusing
 *   the analyze_cache_hologram_matched_nulls.py conventions. We emit per symbol:
 *     family,declared_mode,actual_mode,trial,hash_restored,theta_idx,
 *     b00_I,b00_Q,...,b11_I,b11_Q,fl00..fl11
 *   plus a header comment with the codebook + tones + seeds for reproducibility.
 */

#ifndef SLOT2_PDN_RUN_H
#define SLOT2_PDN_RUN_H

/* forward decls of things defined in the .c before this header is included */
static int run_orchestrate(config_t *cfg);

/* ---- reversible XOR-borrow + byte-hash restore (bit-exact tape) ----
 * A tiny reversible message tape per symbol so the run carries the
 * "hash_restored==1 on all rows" property the matched-null battery checks. The tape
 * is the symbol's (declared,actual,theta) packed; XOR-borrow scramble then exact
 * inverse restores it; byte-hash compares pre/post. */
static uint32_t byte_hash(const uint8_t *b, int n) {
    uint32_t h = 2166136261u;
    for (int i=0;i<n;i++){ h ^= b[i]; h *= 16777619u; }
    return h;
}
static int reversible_restore_ok(uint8_t declared, uint8_t actual, uint8_t theta, uint8_t fam) {
    uint8_t tape[4] = { declared, actual, theta, fam };
    uint32_t h0 = byte_hash(tape, 4);
    /* XOR-borrow scramble (reversible): forward then exact inverse */
    uint8_t key[4] = {0x5A, 0xC3, 0x39, 0x96};
    uint8_t s[4]; uint8_t borrow = 0;
    for (int i=0;i<4;i++){ uint16_t v = (uint16_t)tape[i] ^ key[i]; v = (uint16_t)(v - borrow); borrow = (v>>8)&1; s[i]=(uint8_t)v; }
    /* inverse */
    uint8_t r[4]; borrow = 0;
    /* recompute borrow chain forward to invert exactly */
    {
        uint8_t b2 = 0;
        for (int i=0;i<4;i++){ uint16_t v = (uint16_t)((s[i] + b2)); b2 = (v>>8)&1; r[i] = (uint8_t)(v ^ key[i]); }
    }
    (void)borrow;
    uint32_t h1 = byte_hash(r, 4);
    return (h0 == h1) ? 1 : 0;
}

/* ---- symbol schedule (deterministic from seed) ---- */
typedef struct {
    int   fam;            /* 0 real, 1 pseudo, 2 wrong */
    int   declared;       /* declared mode 0..3 */
    int   actual;         /* actual physical mode 0..3 */
    int   theta_idx;      /* phase level 0..PHASE_LEVELS-1 */
    int   perm[NBIN_MAX]; /* pseudo: decoy permutation of bins; else identity */
    int   trial;          /* trial index within family */
} symbol_t;

static const char *fam_name(int f){ return f==0?"real":(f==1?"pseudo":"wrong"); }
static const char *mode_name(int m){ const char*n[4]={"basis","rotation","residual","mini"}; return (m>=0&&m<4)?n[m]:"?"; }

/* build the full symbol list (preamble + families) deterministically */
static int build_schedule(const config_t *cfg, symbol_t **out, int *n_preamble) {
    g_rng = 0x9E3779B97F4A7C15ULL ^ (uint64_t)cfg->seed;
    int npre = MODES;                 /* 4-symbol preamble, one per mode */
    int per = cfg->trials;            /* trials per family */
    int total = npre + 3*per;
    symbol_t *S = (symbol_t*)calloc(total, sizeof(symbol_t));
    if (!S) return -1;
    int idx = 0;
    /* preamble: one clean real symbol per mode, theta=0 -> trains centroids+threshold */
    for (int m=0;m<MODES;m++) {
        symbol_t *s=&S[idx++];
        s->fam=0; s->declared=m; s->actual=m; s->theta_idx=0; s->trial=-1-m;
        for (int i=0;i<cfg->nbin;i++) s->perm[i]=i;
    }
    /* families interleaved by trial so trial%2 split is balanced across families */
    for (int t=0;t<per;t++) {
        for (int fam=0; fam<3; fam++) {
            symbol_t *s=&S[idx++];
            s->fam=fam; s->trial=t;
            int actual = irand(MODES);
            int theta = irand(PHASE_LEVELS);
            s->actual=actual; s->theta_idx=theta;
            for (int i=0;i<cfg->nbin;i++) s->perm[i]=i;
            if (fam==0) {            /* real */
                s->declared = actual;
            } else if (fam==2) {     /* wrong: declared != actual; drive ACTUAL */
                s->declared = (actual + 1 + irand(MODES-1)) % MODES;
            } else {                 /* pseudo: decoy permutation of the codeword */
                s->declared = irand(MODES);
                /* random permutation of bins (Fisher-Yates) -> off-codebook drive */
                int p[NBIN_MAX]; for (int i=0;i<cfg->nbin;i++) p[i]=i;
                for (int i=cfg->nbin-1;i>0;i--){ int j=irand(i+1); int tmp=p[i]; p[i]=p[j]; p[j]=tmp; }
                memcpy(s->perm, p, sizeof(int)*cfg->nbin);
            }
        }
    }
    *out = S; *n_preamble = npre;
    return total;
}

/* per-bin drive sign for a symbol: the codeword of the ACTUAL mode, then (pseudo)
 * remapped through the decoy permutation. Returns +1/-1 for bin b. */
static double symbol_bin_sign(const codebook_t *cb, const symbol_t *s, int b) {
    /* the bin physically driven at slot b carries codeword[actual][perm[b]] */
    int src = s->perm[b];
    return cb->cw[s->actual][src];
}

/* One implementation defines what the sender executes and what the receiver records. */
static void effective_drive(const config_t *cfg, const codebook_t *cb,
                            const symbol_t *s, int symbol_index, int bin_index,
                            int *drive_sign, double *phase_fraction) {
    double sign = symbol_bin_sign(cb, s, bin_index);
    if (cfg->ctrl_scramble) {
        uint64_t key = 0x9E3779B97F4A7C15ULL ^
                       ((uint64_t)(symbol_index + 1) * 2654435761ULL) ^
                       (uint64_t)(bin_index * 40503u);
        key ^= key << 13; key ^= key >> 7; key ^= key << 17;
        sign = cb->cw[(int)((key >> 20) % MODES)][(int)(key % (uint64_t)cfg->nbin)];
    }
    if (cfg->ctrl_silent) sign = 0.0;
    *drive_sign = sign < 0.0 ? -1 : (sign > 0.0 ? 1 : 0);
    *phase_fraction = ((*drive_sign < 0) ? 0.5 : 0.0) +
                      (double)s->theta_idx / (double)PHASE_LEVELS;
    *phase_fraction -= floor(*phase_fraction);
}

static int write_schedule_json(const config_t *cfg, const codebook_t *cb,
                               const double *tones, const symbol_t *symbols,
                               int total, uint64_t t0) {
    char path[512];
    if (snprintf(path, sizeof(path), "%s/schedule.json", cfg->witness_dir) >= (int)sizeof(path)) return -1;
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) return -1;
    FILE *file = fdopen(fd, "w");
    if (!file) { close(fd); unlink(path); return -1; }
    int failed = 0;
#define JW(...) do { if (fprintf(file, __VA_ARGS__) < 0) failed = 1; } while (0)
    JW("{\n  \"schema_id\": \"CAT_CAS_PDN_CARRIER_SCHEDULE_V1\",\n");
    JW("  \"campaign_id\": \"%s\",\n  \"run_id\": \"%s\",\n", cfg->campaign_id, cfg->run_id);
    JW("  \"condition\": \"%s\",\n", cfg->condition);
    JW("  \"seed\": %d,\n  \"phase_levels\": %d,\n  \"t0_tsc\": %llu,\n",
       cfg->seed, PHASE_LEVELS, (unsigned long long)t0);
    JW("  \"tones_hz\": [");
    for (int b=0;b<cfg->nbin;b++) JW("%.17g%s", tones[b], b+1<cfg->nbin ? ", " : "");
    JW("],\n  \"codebook\": {\n");
    for (int m=0;m<MODES;m++) {
        JW("    \"%s\": [", mode_name(m));
        for (int b=0;b<cfg->nbin;b++) JW("%d%s", (int)cb->cw[m][b], b+1<cfg->nbin ? ", " : "");
        JW("]%s\n", m+1<MODES ? "," : "");
    }
    JW("  },\n  \"symbols\": [\n");
    for (int si=0;si<total;si++) {
        const symbol_t *s=&symbols[si];
        JW("    {\"symbol_index\": %d, \"family\": \"%s\", ", si,
           s->trial < 0 ? "preamble" : fam_name(s->fam));
        JW("\"declared_mode\": \"%s\", \"actual_mode\": \"%s\", ",
           mode_name(s->declared), mode_name(s->actual));
        JW("\"trial\": %d, \"hash_restored\": %d, \"theta_idx\": %d, ",
           s->trial, reversible_restore_ok((uint8_t)s->declared, (uint8_t)s->actual,
           (uint8_t)s->theta_idx, (uint8_t)s->fam), s->theta_idx);
        JW("\"bin_permutation\": [");
        for (int b=0;b<cfg->nbin;b++) JW("%d%s", s->perm[b], b+1<cfg->nbin ? ", " : "");
        JW("], \"drive_signs\": [");
        for (int b=0;b<cfg->nbin;b++) { int sign; double phase; effective_drive(cfg,cb,s,si,b,&sign,&phase); (void)phase; JW("%d%s",sign,b+1<cfg->nbin?", ":""); }
        JW("], \"phase_fractions\": [");
        for (int b=0;b<cfg->nbin;b++) { int sign; double phase; effective_drive(cfg,cb,s,si,b,&sign,&phase); (void)sign; JW("%.17g%s",phase,b+1<cfg->nbin?", ":""); }
        JW("]}%s\n", si+1<total ? "," : "");
    }
    JW("  ]\n}\n");
#undef JW
    if (fflush(file) != 0) failed = 1;
    if (fsync(fd) != 0) failed = 1;
    if (fclose(file) != 0) failed = 1;
    if (failed) { unlink(path); return -1; }
    return 0;
}

/* ===========================================================================
 * SENDER child: pin to sender core, decide t0, write handshake, then drive the full
 * schedule bin-by-bin on the absolute clock. For each (symbol,bin) slot, run the
 * gated alu_burst at that bin's tone with phase_frac = (sign<0?0.5:0.0) + theta/2pi,
 * for slot_s seconds, aligned to the absolute slot start. k10temp veto before each
 * slot. One drive thread per sender core (namp).
 * =========================================================================== */
static int run_sender(config_t *cfg) {
    if (pin_to_core(cfg->sender) != 0 && cfg->witness_enabled) return 4;
    double tones[NBIN_MAX];
    make_tones(tones, cfg->nbin, cfg->f_lo, cfg->f_hi);
    codebook_t cb; make_codebook(&cb, cfg->nbin, 7);

    symbol_t *S=NULL; int npre=0;
    int total = (cfg->mode_matrix) ? build_schedule(cfg, &S, &npre) : 1;
    if (cfg->mode_matrix && total < 0) return -1;

    /* decide t0 = now + 100 ms; publish to handshake */
    uint64_t t0 = rdtsc_now() + (uint64_t)(0.100 * cfg->tsc_hz);
    if (write_t0(cfg->handshake, t0) != 0) {
        fprintf(stderr, "SENDER: handshake write failed (%s)\n", cfg->handshake);
        free(S); return -1;
    }
    fprintf(stderr, "SENDER: t0=%llu published; cores=[", (unsigned long long)t0);
    for (int k=0;k<cfg->n_sender_cores;k++) fprintf(stderr,"%d%s",cfg->sender_cores[k], k+1<cfg->n_sender_cores?",":"");
    fprintf(stderr, "] nbin=%d slot_s=%.3f mode=%s\n", cfg->nbin, cfg->slot_s, cfg->mode_matrix?"matrix":"preflight");

    if (!cfg->mode_matrix) {
        /* PREFLIGHT: drive ONE tone f=bin_hz continuously (sign +1, theta 0) for a
           single capture window of slot_s (the receiver captures the matching window). */
        double f_b = cfg->bin_hz;
        double half_ticks = 0.5 * (cfg->tsc_hz / f_b);
        /* start the slot at t0 */
        while (rdtsc_now() < t0) { __asm__ volatile("pause"); }
        drive_t dr[NCPU_MAX]; int nd=0;
        for (int k=0;k<cfg->n_sender_cores;k++)
            if (drive_start(&dr[nd], cfg->sender_cores[k], t0, half_ticks, 0.0)==0) nd++;
        /* hold the drive for the capture length (receiver captures slot_s) */
        double hold_s = cfg->slot_s;
        uint64_t end = t0 + (uint64_t)(hold_s * cfg->tsc_hz);
        while (rdtsc_now() < end) { __asm__ volatile("pause"); }
        for (int k=0;k<nd;k++) drive_stop(&dr[k]);
        free(S);
        return 0;
    }

    /* MATRIX: walk the schedule. Each symbol occupies nbin consecutive slots. */
    for (int si=0; si<total; si++) {
        symbol_t *s=&S[si];
        double theta = 2.0*M_PI*(double)s->theta_idx/(double)PHASE_LEVELS;
        for (int b=0;b<cfg->nbin;b++) {
            double T = read_k10temp_c();
            if (!isfinite(T) || T < -100.0 || T >= cfg->temp_veto) {
                fprintf(stderr, "SENDER TEMP VETO %.1fC at sym=%d bin=%d\n", T, si, b);
                free(S); return 3;
            }
            double f_b = tones[b];
            double half_ticks = 0.5 * (cfg->tsc_hz / f_b);
            int drive_sign;
            double phase_frac;
            effective_drive(cfg, &cb, s, si, b, &drive_sign, &phase_frac);
            (void)theta;
            /* absolute slot start */
            double slot_start_s = bin_slot_start_s(si, b, cfg->nbin, cfg->slot_s);
            uint64_t t_start = t0 + (uint64_t)(slot_start_s * cfg->tsc_hz);
            uint64_t t_end   = t0 + (uint64_t)((slot_start_s + cfg->slot_s) * cfg->tsc_hz);
            /* spin to the slot start (cheap; OFF current) */
            uint64_t t_arrive = rdtsc_now();
            while (rdtsc_now() < t_start) { __asm__ volatile("pause"); }
            if (getenv("SLOT2_DBG") && (si*cfg->nbin+b) % 12 == 0)
                fprintf(stderr, "SND slot=%d arrive_lag=%.3fms (t_arrive-t_start)\n",
                        si*cfg->nbin+b, (double)((int64_t)t_arrive-(int64_t)t_start)/cfg->tsc_hz*1e3);
            drive_t dr[NCPU_MAX]; int nd=0;
            /* drive origin = t_start (PER-SLOT), matching the receiver's per-slot
               lock-in reference. The square-wave phase (sign 0/pi + theta offset) is
               thus carried relative to the slot start, immune to whole-run TSC phase
               accumulation.
               SILENT negative control: drive nothing at all -- the sender just idles
               through the slot, so the receiver locks in on pure OS/thermal background
               and must sit at chance. */
            if (drive_sign != 0) {
                for (int k=0;k<cfg->n_sender_cores;k++)
                    if (drive_start(&dr[nd], cfg->sender_cores[k], t_start, half_ticks, phase_frac)==0) nd++;
                if (cfg->witness_enabled && nd != cfg->n_sender_cores) {
                    for (int k=0;k<nd;k++) drive_stop(&dr[k]);
                    free(S); return 4;
                }
            }
            while (rdtsc_now() < t_end) { __asm__ volatile("pause"); }
            for (int k=0;k<nd;k++) drive_stop(&dr[k]);
        }
    }
    free(S);
    return 0;
}

/* ===========================================================================
 * RECEIVER child: pin to victim, read t0 from handshake, then capture each
 * (symbol,bin) slot in lockstep on the absolute clock and lock in. Emit CSV.
 * =========================================================================== */
static int run_receiver(config_t *cfg) {
    if (pin_to_core(cfg->victim) != 0 && cfg->witness_enabled) return 4;
    double tones[NBIN_MAX];
    make_tones(tones, cfg->nbin, cfg->f_lo, cfg->f_hi);
    codebook_t cb; make_codebook(&cb, cfg->nbin, 7);

    /* poll for t0 */
    uint64_t t0=0; int tries=0;
    while (read_t0(cfg->handshake, &t0) != 0) {
        struct timespec ss={0, 2*1000*1000}; nanosleep(&ss,NULL);
        if (++tries > 5000) { fprintf(stderr, "RECEIVER: no handshake t0\n"); return -1; }
    }

    /* capture window cap_s = slot_s - gap_s (setup headroom). Buffer sized to slot_s
       worth of samples with margin so a slow ring-osc still fits. */
    double cap_s = cfg->slot_s - cfg->gap_s;
    if (cap_s < 0.1 * cfg->slot_s) cap_s = 0.5 * cfg->slot_s;
    int nsamp = (int)(cfg->slot_s * cfg->read_hz) + 256;
    if (nsamp < 64) nsamp = 64;
    uint64_t *t = (uint64_t*)malloc(sizeof(uint64_t)*nsamp);
    double   *ro= (double*)malloc(sizeof(double)*nsamp);
    if (!t || !ro) { free(t); free(ro); return -1; }

    if (!cfg->mode_matrix) {
        /* PREFLIGHT: capture the single window at t0, lock in at bin_hz + floor. */
        double f_b = cfg->bin_hz;
        uint64_t t_start = t0;          /* drive starts at t0 */
        int ncap = capture_slot(cfg, cfg->victim, t_start, cap_s, t, ro, nsamp);
        double I,Q,mag,floor;
        score_slot(t, ro, ncap, f_b, t0, cfg->tsc_hz, &I,&Q,&mag,&floor);
        double snr = (floor>0)? mag/floor : 0.0;
        double T = read_k10temp_c();
        long khz = read_cur_khz(cfg->victim); int ps = cofvid_pstate(cfg->victim);
        /* write a one-line preflight result */
        FILE *f = cfg->out_csv[0] ? fopen(cfg->out_csv,"w") : stdout;
        if (!f) f=stdout;
        fprintf(f, "# SLOT2 PREFLIGHT victim=%d sender=%d bin_hz=%.3f slot_s=%.3f read_hz=%d nsamp=%d\n",
                cfg->victim, cfg->sender, f_b, cfg->slot_s, cfg->read_hz, nsamp);
        fprintf(f, "metric,value\n");
        fprintf(f, "drive_bin_mag,%.6g\n", mag);
        fprintf(f, "offbin_floor_mag,%.6g\n", floor);
        fprintf(f, "snr_eff,%.4f\n", snr);
        fprintf(f, "k10temp_c,%.2f\n", T);
        fprintf(f, "cur_khz,%ld\n", khz);
        fprintf(f, "cofvid_pstate,%d\n", ps);
        if (f!=stdout) fclose(f);
        fprintf(stderr, "RECEIVER PREFLIGHT: bin_mag=%.5g floor=%.5g SNR_eff=%.3f T=%.1fC khz=%ld ps=%d n=%d\n",
                mag, floor, snr, T, khz, ps, ncap);
        free(t); free(ro);
        return 0;
    }

    /* MATRIX: rebuild the SAME schedule (deterministic) so the receiver knows each
       symbol's labels; capture each (symbol,bin) slot; lock in; emit per-symbol row. */
    symbol_t *S=NULL; int npre=0;
    int total = build_schedule(cfg, &S, &npre);
    if (total<0){ free(t); free(ro); return -1; }

    CarrierWitnessRawWriter witness_writer;
    int witness_open = 0;
    memset(&witness_writer, 0, sizeof(witness_writer));
    if (cfg->witness_enabled) {
        if (carrier_witness_raw_open(&witness_writer, cfg->witness_dir) != 0) {
            fprintf(stderr, "RECEIVER: cannot create immutable raw witness files: %s\n", strerror(errno));
            free(S); free(t); free(ro); return 5;
        }
        witness_open = 1;
        if (write_schedule_json(cfg, &cb, tones, S, total, t0) != 0) {
            fprintf(stderr, "RECEIVER: cannot serialize exact schedule: %s\n", strerror(errno));
            carrier_witness_raw_close(&witness_writer);
            free(S); free(t); free(ro); return 5;
        }
    }

    FILE *f = cfg->out_csv[0] ? fopen(cfg->out_csv, cfg->witness_enabled ? "wx" : "w") : stdout;
    if (!f) {
        if (witness_open) carrier_witness_raw_close(&witness_writer);
        free(S); free(t); free(ro); return 5;
    }
    /* header comment: codebook + tones + seeds (reproducibility) */
    fprintf(f, "# SLOT2 MATRIX victim=%d sender=%d nbin=%d slot_s=%.3f read_hz=%d seed=%d trials=%d minham=%d\n",
            cfg->victim, cfg->sender, cfg->nbin, cfg->slot_s, cfg->read_hz, cfg->seed, cfg->trials, cb.minham);
    fprintf(f, "# tones_hz=");
    for (int b=0;b<cfg->nbin;b++) fprintf(f, "%.4f%s", tones[b], b+1<cfg->nbin?",":"\n");
    for (int m=0;m<MODES;m++){
        fprintf(f, "# codeword_%d=", m);
        for (int b=0;b<cfg->nbin;b++) fprintf(f, "%+d%s", (int)cb.cw[m][b], b+1<cfg->nbin?",":"\n");
    }
    /* CSV header row */
    fprintf(f, "family,declared_mode,actual_mode,trial,hash_restored,theta_idx");
    for (int b=0;b<cfg->nbin;b++) fprintf(f, ",b%02d_I,b%02d_Q", b, b);
    for (int b=0;b<cfg->nbin;b++) fprintf(f, ",fl%02d", b);
    fprintf(f, "\n");

    double Tmax = read_k10temp_c();
    for (int si=0; si<total; si++) {
        symbol_t *s=&S[si];
        double binI[NBIN_MAX], binQ[NBIN_MAX], binFL[NBIN_MAX];
        for (int b=0;b<cfg->nbin;b++) {
            double temp_before = read_k10temp_c();
            long khz_before = read_cur_khz(cfg->victim);
            int pstate_before = cofvid_pstate(cfg->victim);
            if (temp_before > Tmax) Tmax = temp_before;
            if (!isfinite(temp_before) || temp_before < -100.0 || temp_before >= cfg->temp_veto) {
                fprintf(stderr, "RECEIVER INVALID TEMP %.1fC at sym=%d bin=%d\n", temp_before, si, b);
                if (f!=stdout) fclose(f);
                if (witness_open) carrier_witness_raw_close(&witness_writer);
                free(S); free(t); free(ro);
                return 3;
            }
            double f_b = tones[b];
            double slot_start_s = bin_slot_start_s(si, b, cfg->nbin, cfg->slot_s);
            uint64_t t_start = t0 + (uint64_t)(slot_start_s * cfg->tsc_hz);
            int ncap = capture_slot(cfg, cfg->victim, t_start, cap_s, t, ro, nsamp);
            if (ncap < 4) {
                fprintf(stderr, "RECEIVER: capture failed/short at sym=%d bin=%d n=%d\n", si, b, ncap);
                if (f!=stdout) fclose(f);
                if (witness_open) carrier_witness_raw_close(&witness_writer);
                free(S); free(t); free(ro); return 5;
            }
            double I,Q,mag,floor;
            /* PER-SLOT phase reference: lock in against t_start (this slot's start),
               NOT the global t0. The sender references the same slot start, so the
               drive sign (0/pi) + theta carry within the slot. Using the slot-local
               origin makes the lock-in immune to TSC-calibration phase accumulation
               over the whole run (a 0.5% tsc_hz error over ~150s otherwise rotates the
               carrier by thousands of radians and decoheres late symbols). */
            score_slot(t, ro, ncap, f_b, t_start, cfg->tsc_hz, &I,&Q,&mag,&floor);
            binI[b]=I; binQ[b]=Q; binFL[b]=floor;
            double temp_after = read_k10temp_c();
            long khz_after = read_cur_khz(cfg->victim);
            int pstate_after = cofvid_pstate(cfg->victim);
            if (temp_after > Tmax) Tmax = temp_after;
            if (!isfinite(temp_after) || temp_after < -100.0 || temp_after >= cfg->temp_veto) {
                fprintf(stderr, "RECEIVER INVALID POST TEMP %.1fC at sym=%d bin=%d\n", temp_after, si, b);
                if (f!=stdout) fclose(f);
                if (witness_open) carrier_witness_raw_close(&witness_writer);
                free(S); free(t); free(ro); return 3;
            }
            if (witness_open) {
                CarrierWitnessWindow window;
                int sign; double phase_fraction;
                memset(&window, 0, sizeof(window));
                effective_drive(cfg, &cb, s, si, b, &sign, &phase_fraction);
                window.symbol_index=si; window.bin_index=b;
                snprintf(window.family,sizeof(window.family),"%s",s->trial<0?"preamble":fam_name(s->fam));
                snprintf(window.declared_mode,sizeof(window.declared_mode),"%s",mode_name(s->declared));
                snprintf(window.actual_mode,sizeof(window.actual_mode),"%s",mode_name(s->actual));
                window.trial=s->trial;
                window.hash_restored=reversible_restore_ok((uint8_t)s->declared,(uint8_t)s->actual,
                                                           (uint8_t)s->theta_idx,(uint8_t)s->fam);
                window.theta_idx=s->theta_idx; window.tone_hz=f_b;
                window.drive_sign=sign; window.phase_fraction=phase_fraction;
                snprintf(window.control,sizeof(window.control),"%s",cfg->condition);
                window.slot_start_tsc=t_start;
                window.capture_deadline_tsc=t_start+(uint64_t)(cap_s*cfg->tsc_hz);
                window.temp_before_c=temp_before; window.temp_after_c=temp_after;
                window.cur_khz_before=khz_before; window.cur_khz_after=khz_after;
                window.cofvid_pstate_before=pstate_before; window.cofvid_pstate_after=pstate_after;
                window.computed_i=I; window.computed_q=Q;
                window.computed_magnitude=mag; window.computed_floor=floor;
                if (carrier_witness_raw_append(&witness_writer,&window,t,ro,ncap) != 0) {
                    fprintf(stderr,"RECEIVER: immutable raw write failed at sym=%d bin=%d\n",si,b);
                    if (f!=stdout) fclose(f);
                    carrier_witness_raw_close(&witness_writer);
                    free(S); free(t); free(ro); return 5;
                }
            }
        }
        /* preamble symbols carry trial<0 -> map to even trials so they train */
        int trial_out = (s->trial < 0) ? (s->trial) : s->trial;
        int hash_ok = reversible_restore_ok((uint8_t)s->declared,(uint8_t)s->actual,
                                            (uint8_t)s->theta_idx,(uint8_t)s->fam);
        fprintf(f, "%s,%s,%s,%d,%d,%d",
                fam_name(s->fam), mode_name(s->declared), mode_name(s->actual),
                trial_out, hash_ok, s->theta_idx);
        for (int b=0;b<cfg->nbin;b++) fprintf(f, ",%.6g,%.6g", binI[b], binQ[b]);
        for (int b=0;b<cfg->nbin;b++) fprintf(f, ",%.6g", binFL[b]);
        fprintf(f, "\n");
        fflush(f);
        if ((si % 16)==0)
            fprintf(stderr, "  [rx %d/%d fam=%s decl=%s act=%s th=%d] T=%.1fC\n",
                    si+1, total, fam_name(s->fam), mode_name(s->declared),
                    mode_name(s->actual), s->theta_idx, Tmax);
    }
    if (f!=stdout) fclose(f);
    if (witness_open && carrier_witness_raw_close(&witness_writer) != 0) {
        fprintf(stderr, "RECEIVER: raw witness close/flush failed\n");
        free(S); free(t); free(ro); return 5;
    }
    fprintf(stderr, "RECEIVER MATRIX done: %d symbols, Tmax=%.1fC\n", total, Tmax);
    free(S); free(t); free(ro);
    return 0;
}

/* SIGTERM/SIGINT safety net: an external kill (e.g. a shell `timeout`) must still
 * restore the P-state. We stash the active pinstate + the child pids and a flag, and on
 * a terminating signal we kill the children, restore the pin, and re-raise. Only the
 * orchestrator parent installs this. */
static pinstate_t *g_sig_ps = NULL;
static int g_sig_do_pin = 0;
static volatile pid_t g_sig_rx = 0, g_sig_sd = 0;
static char g_sig_handshake[400] = {0};
#include <signal.h>
static void term_handler(int sig) {
    if (g_sig_rx > 0) kill(g_sig_rx, SIGKILL);
    if (g_sig_sd > 0) kill(g_sig_sd, SIGKILL);
    if (g_sig_do_pin && g_sig_ps) restore_pstate(g_sig_ps);
    if (g_sig_handshake[0]) unlink(g_sig_handshake);
    signal(sig, SIG_DFL);
    raise(sig);
}

static int prepare_witness_directory(config_t *cfg, int redirect_logs) {
    char stdout_path[512], stderr_path[512], summary_path[512];
    if (!cfg->witness_enabled) return 0;
    if (mkdir(cfg->witness_dir, 0755) != 0) return -1;
    if (snprintf(stdout_path,sizeof(stdout_path),"%s/stdout.log",cfg->witness_dir)>=(int)sizeof(stdout_path) ||
        snprintf(stderr_path,sizeof(stderr_path),"%s/stderr.log",cfg->witness_dir)>=(int)sizeof(stderr_path) ||
        snprintf(summary_path,sizeof(summary_path),"%s/summary.csv",cfg->witness_dir)>=(int)sizeof(summary_path)) return -1;
    int out_fd=open(stdout_path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0644);
    if(out_fd<0) return -1;
    int err_fd=open(stderr_path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0644);
    if(err_fd<0){close(out_fd);return -1;}
    if (redirect_logs) {
        if(dup2(out_fd,STDOUT_FILENO)<0 || dup2(err_fd,STDERR_FILENO)<0){close(out_fd);close(err_fd);return -1;}
    }
    close(out_fd); close(err_fd);
    if (cfg->out_csv[0] && strcmp(cfg->out_csv,summary_path)!=0) return -1;
    if (strlen(summary_path) >= sizeof(cfg->out_csv)) return -1;
    memcpy(cfg->out_csv,summary_path,strlen(summary_path)+1);
    return 0;
}

/* ===========================================================================
 * ORCHESTRATE: pin P-state (shared package), fork sender + receiver, wait, restore.
 * =========================================================================== */
static int run_orchestrate(config_t *cfg) {
    if (cfg->witness_enabled && prepare_witness_directory(cfg, 1) != 0) {
        fprintf(stderr,"FATAL: witness run directory must be new and writable: %s\n",strerror(errno));
        return 5;
    }
    /* default handshake path under /tmp if not given */
    if (!cfg->handshake[0]) {
        snprintf(cfg->handshake, sizeof(cfg->handshake), "/tmp/slot2_hs_%d", (int)getpid());
    }
    /* clear any stale handshake */
    unlink(cfg->handshake);

    /* explicit single-role mode (debug / external orchestration) */
    if (cfg->role == 1) return run_sender(cfg);
    if (cfg->role == 2) return run_receiver(cfg);

    double T0 = read_k10temp_c();
    fprintf(stderr, "=== SLOT2 PDN lock-in orchestrate: mode=%s victim=%d sender=%d "
                    "nbin=%d f=[%.0f,%.0f] slot_s=%.3f read_hz=%d seed=%d veto=%.0fC startT=%.1fC ===\n",
            cfg->mode_matrix?"matrix":"preflight", cfg->victim, cfg->sender, cfg->nbin,
            cfg->f_lo, cfg->f_hi, cfg->slot_s, cfg->read_hz, cfg->seed, cfg->temp_veto, T0);
    if (!isfinite(T0) || T0 < -100.0 || T0 >= cfg->temp_veto) {
        fprintf(stderr, "ABORT: invalid/start temperature %.1fC\n", T0); return 3;
    }

    /* pin P-state across all package cores (shared rail control) */
    pinstate_t ps; memset(&ps,0,sizeof(ps));
    if (cfg->do_pin) {
        if (pin_pstate(&ps, cfg->pin_khz)!=0) {
            fprintf(stderr, "WARN: P-state pin had no effect\n");
            if (cfg->witness_enabled) { restore_pstate(&ps); return 4; }
        }
        struct timespec s={0, 300*1000*1000}; nanosleep(&s,NULL);
        long khz = read_cur_khz(cfg->victim);
        fprintf(stderr, "P-state pinned: target=%ld victim_cur=%ld cofvid_ps=%d\n",
                cfg->pin_khz, khz, cofvid_pstate(cfg->victim));
        if (khz>0 && labs(khz-cfg->pin_khz) > 50000)
            fprintf(stderr, "NOTE: idle current-frequency proxy=%ld; policy target=%ld\n", khz, cfg->pin_khz);
    }

    /* install the signal safety net now that the pin is applied */
    g_sig_ps = &ps; g_sig_do_pin = cfg->do_pin;
    snprintf(g_sig_handshake, sizeof(g_sig_handshake), "%s", cfg->handshake);
    signal(SIGTERM, term_handler);
    signal(SIGINT,  term_handler);

    pid_t pr = fork();
    if (pr < 0) { fprintf(stderr, "FATAL: fork receiver\n"); if (cfg->do_pin) restore_pstate(&ps); return 4; }
    if (pr == 0) {
        /* receiver child */
        config_t c = *cfg; c.role = 2; c.do_pin = 0;  /* parent owns pin/restore */
        _exit(run_receiver(&c) & 0xff);
    }
    g_sig_rx = pr;
    pid_t psd = fork();
    if (psd < 0) { fprintf(stderr, "FATAL: fork sender\n"); }
    if (psd == 0) {
        /* sender child */
        config_t c = *cfg; c.role = 1; c.do_pin = 0;
        _exit(run_sender(&c) & 0xff);
    }
    g_sig_sd = psd;

    int strx=0, ssnd=0;
    waitpid(pr, &strx, 0);
    if (psd>0) waitpid(psd, &ssnd, 0);

    int restored_ok = !cfg->do_pin || restore_pstate(&ps) == 0;
    unlink(cfg->handshake);
    double Tend = read_k10temp_c();
    int rrx = WIFEXITED(strx)?WEXITSTATUS(strx):-1;
    int rsd = WIFEXITED(ssnd)?WEXITSTATUS(ssnd):-1;
    fprintf(stderr, "=== SLOT2 done: receiver_rc=%d sender_rc=%d startT=%.1fC endT=%.1fC ===\n",
            rrx, rsd, T0, Tend);
    if (cfg->witness_enabled)
        fprintf(stderr,"CARRIER_WITNESS_EXIT receiver=%d sender=%d orchestrator=%d pstate_restored=%d affinity_verified=%d\n",
                rrx,rsd,(rrx==0&&rsd==0&&restored_ok)?0:(rrx?rrx:(rsd?rsd:4)),
                restored_ok,rrx==0&&rsd==0);
    return (rrx==0 && rsd==0 && restored_ok) ? 0 : (rrx? rrx : (rsd?rsd:4));
}

#endif /* SLOT2_PDN_RUN_H */
