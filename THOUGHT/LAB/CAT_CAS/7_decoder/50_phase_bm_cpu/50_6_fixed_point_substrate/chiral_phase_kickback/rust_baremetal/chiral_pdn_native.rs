// chiral_pdn_native.rs
//
// Native Rust chiral phase-kickback probe for the local machine.
//
// This is the bare-metal-adjacent version of the Phase 6 chiral-prep test:
// two native threads are pinned to distinct logical CPUs. A sender drives a
// balanced high/low compute-load pattern; a receiver measures per-slot timing
// throughput. Public modes derive the pattern only from public (k,b,N). The
// hidden control flips the same balanced pattern by the secret orientation before
// the cosine projection, proving the local physical channel/gate lights up when
// the missing lane exists.
//
// No voltage, firmware, MSR writes, or privileged operations. ASCII only.

use std::cmp::Ordering as CmpOrdering;
use std::env;
use std::fs::{create_dir_all, File};
use std::hint::black_box;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const MASTER_SEED: u64 = 0x5060_6017_2026_0613;
const NSLOTS: usize = 64;
const REPEATS: usize = 8;
const SLOT_CYCLES: u64 = 820_000;
const START_DELAY_CYCLES: u64 = 30_000_000;
const PAIRS_PER_MODE: usize = 42;
const N_SHUFFLES: usize = 80;
const SENDER_CPU: usize = 4;
const RECEIVER_CPU: usize = 5;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

#[cfg(target_arch = "x86")]
#[inline(always)]
fn rdtsc() -> u64 {
    unsafe { core::arch::x86::_rdtsc() as u64 }
}

#[cfg(windows)]
mod affinity {
    use std::ffi::c_void;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> *mut c_void;
        fn SetThreadAffinityMask(hThread: *mut c_void, dwThreadAffinityMask: usize) -> usize;
        fn SetThreadPriority(hThread: *mut c_void, nPriority: i32) -> i32;
    }

    const THREAD_PRIORITY_ABOVE_NORMAL: i32 = 1;

    pub fn pin(cpu: usize) -> bool {
        unsafe {
            let h = GetCurrentThread();
            let mask = 1usize.checked_shl(cpu as u32).unwrap_or(0);
            let ok = mask != 0 && SetThreadAffinityMask(h, mask) != 0;
            let _ = SetThreadPriority(h, THREAD_PRIORITY_ABOVE_NORMAL);
            ok
        }
    }
}

#[cfg(not(windows))]
mod affinity {
    pub fn pin(_cpu: usize) -> bool {
        false
    }
}

#[derive(Clone)]
struct Rng {
    s: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { s: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn f64(&mut self) -> f64 {
        let x = self.next_u64() >> 11;
        (x as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }

    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.usize(i + 1);
            xs.swap(i, j);
        }
    }
}

#[derive(Clone)]
struct Instance {
    n_mod: usize,
    d: usize,
    k: Vec<usize>,
    b: Vec<i8>,
}

fn sample_secret(n_mod: usize, rng: &mut Rng) -> usize {
    loop {
        let d = 1 + rng.usize(n_mod - 1);
        if d != n_mod / 2 {
            return d;
        }
    }
}

fn m_for(n: usize) -> usize {
    let n_mod = 1usize << n;
    let sqrt_n = (n_mod as f64).sqrt().ceil() as usize;
    usize::max(4 * sqrt_n, 48 * n)
}

fn coset_instance(n: usize, d: usize, rng: &mut Rng) -> Instance {
    let n_mod = 1usize << n;
    let m = m_for(n);
    let mut k = Vec::with_capacity(m);
    let mut b = Vec::with_capacity(m);
    for _ in 0..m {
        let kk = rng.usize(n_mod);
        let p = (1.0 + (2.0 * std::f64::consts::PI * (kk as f64) * (d as f64) / (n_mod as f64)).cos()) * 0.5;
        let bit = if rng.f64() < p { 1 } else { -1 };
        k.push(kk);
        b.push(bit);
    }
    Instance { n_mod, d, k, b }
}

fn folded(inst: &Instance) -> Instance {
    let mut out = inst.clone();
    out.d = (inst.n_mod - inst.d) % inst.n_mod;
    out
}

fn orientation(inst: &Instance) -> i32 {
    if inst.d < inst.n_mod / 2 { 1 } else { 0 }
}

fn public_hash(inst: &Instance) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64 ^ (inst.n_mod as u64);
    for (&kk, &bb) in inst.k.iter().zip(inst.b.iter()) {
        h ^= (kk as u64).wrapping_mul(0x9E37_79B1_85EB_CA87);
        h = h.rotate_left(17).wrapping_mul(0x1000_0000_01B3);
        if bb < 0 {
            h ^= 0xD1B5_4A32_D192_ED03;
        }
    }
    h
}

fn public_chiral_pattern(inst: &Instance, shuffled: bool) -> Vec<i8> {
    let mut order: Vec<usize> = (0..inst.k.len()).collect();
    if shuffled {
        let mut rng = Rng::new(public_hash(inst) ^ 0xA5A5_5A5A_1337_4242);
        rng.shuffle(&mut order);
    } else {
        order.sort_by_key(|&i| inst.k[i]);
    }

    let mut scores = vec![0.0f64; NSLOTS];
    let mut z_re = 1.0f64;
    let mut z_im = 0.0f64;
    let mut tag = 0xC0DE_C0DE_C0DE_C0DEu64;
    for (step, &idx) in order.iter().enumerate() {
        let kk = inst.k[idx] as f64;
        let bb = inst.b[idx] as f64;
        let theta = 2.0 * std::f64::consts::PI * kk / (inst.n_mod as f64);
        let c = theta.cos();
        let s = theta.sin();
        let nr = 0.997 * z_re + bb * c + 0.003 * (z_re * c + z_im * s);
        let ni = 0.997 * z_im + bb * s + 0.003 * (z_re * s - z_im * c);
        z_re = nr;
        z_im = ni;
        let word = ((inst.k[idx] as u64 + 1).wrapping_mul(0x9E37_79B1_85EB_CA87))
            ^ ((step as u64 + 1).wrapping_mul(0x94D0_49BB_1331_11EB))
            ^ if inst.b[idx] < 0 { 0xD1B5_4A32_D192_ED03 } else { 0 };
        tag ^= word.rotate_left(((inst.k[idx] + step) & 63) as u32);
        let slot = (step * NSLOTS / order.len()).min(NSLOTS - 1);
        scores[slot] += bb * (s * z_re - c * z_im) + ((tag & 0xff) as f64 - 128.0) / 512.0;
    }

    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(CmpOrdering::Equal));
    let med = sorted[sorted.len() / 2];
    let mut pattern: Vec<i8> = scores.iter().map(|&x| if x >= med { 1 } else { -1 }).collect();

    // Force exact balance. The physical test should read phase, not total duty.
    let mut pos = pattern.iter().filter(|&&x| x > 0).count();
    let target = NSLOTS / 2;
    if pos > target {
        let mut idx: Vec<usize> = (0..NSLOTS).filter(|&i| pattern[i] > 0).collect();
        idx.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(CmpOrdering::Equal));
        for &i in idx.iter().take(pos - target) {
            pattern[i] = -1;
        }
    } else if pos < target {
        let mut idx: Vec<usize> = (0..NSLOTS).filter(|&i| pattern[i] < 0).collect();
        idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(CmpOrdering::Equal));
        for &i in idx.iter().take(target - pos) {
            pattern[i] = 1;
        }
    }
    pos = pattern.iter().filter(|&&x| x > 0).count();
    debug_assert_eq!(pos, target);
    pattern
}

fn drive_pattern(base: &[i8], inst: &Instance, mode: &str) -> Vec<i8> {
    let mut out = base.to_vec();
    if mode == "hidden_chiral_control" && orientation(inst) == 0 {
        for x in &mut out {
            *x = -*x;
        }
    }
    out
}

fn sender_heavy(seed: &mut u64, mem: &mut [u64]) -> u64 {
    let mut x = *seed;
    let mask = mem.len() - 1;
    for r in 0..448usize {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = ((x as usize) ^ (r * 131)) & mask;
        let v = mem[idx].wrapping_add(x.rotate_left((r & 31) as u32));
        mem[idx] = v ^ 0x9E37_79B1_85EB_CA87;
        black_box(mem[idx]);
    }
    *seed = x;
    x
}

fn sender_light(seed: &mut u64) -> u64 {
    let mut x = *seed;
    for _ in 0..8 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        std::hint::spin_loop();
    }
    *seed = x;
    x
}

fn receiver_tick(state: &mut u64, mem: &mut [u64]) {
    let mask = mem.len() - 1;
    let mut x = *state;
    for r in 0..12usize {
        x = x.wrapping_mul(0xD134_2543_DE82_EF95).wrapping_add(0x9E37_79B9);
        let idx = ((x >> 7) as usize ^ (r * 17)) & mask;
        x ^= mem[idx].rotate_left((x & 31) as u32);
        black_box(x);
    }
    *state = x;
}

fn run_physical_trial(pattern: &[i8], trial_seed: u64) -> (Vec<f64>, bool, bool) {
    let start_tsc = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicU64::new(0));
    let pat = Arc::new(pattern.to_vec());

    let sender_start = Arc::clone(&start_tsc);
    let sender_done = Arc::clone(&done);
    let sender_pat = Arc::clone(&pat);
    let sender = thread::spawn(move || {
        let pin_ok = affinity::pin(SENDER_CPU);
        let mut mem = vec![0u64; 1 << 18];
        let mut seed = trial_seed ^ 0xA5A5_D00D_1111_2222;
        let start = sender_start.load(Ordering::Acquire);
        while rdtsc() < start {
            std::hint::spin_loop();
        }
        let total_slots = NSLOTS * REPEATS;
        for s in 0..total_slots {
            let slot_end = start + ((s as u64 + 1) * SLOT_CYCLES);
            let sign = sender_pat[s % NSLOTS];
            while rdtsc() < slot_end {
                if sign > 0 {
                    black_box(sender_heavy(&mut seed, &mut mem));
                } else {
                    black_box(sender_light(&mut seed));
                }
            }
        }
        sender_done.store(1, Ordering::Release);
        pin_ok
    });

    let recv_start = Arc::clone(&start_tsc);
    let recv_done = Arc::clone(&done);
    let receiver = thread::spawn(move || {
        let pin_ok = affinity::pin(RECEIVER_CPU);
        let mut mem = vec![0u64; 1 << 15];
        let mut counts = vec![0u64; NSLOTS];
        let mut state = trial_seed ^ 0x1234_5678_9ABC_DEF0;
        let start = recv_start.load(Ordering::Acquire);
        while rdtsc() < start {
            std::hint::spin_loop();
        }
        let end = start + ((NSLOTS * REPEATS) as u64 * SLOT_CYCLES);
        loop {
            let now = rdtsc();
            if now >= end || recv_done.load(Ordering::Acquire) != 0 {
                break;
            }
            let elapsed = now.saturating_sub(start);
            let slot = ((elapsed / SLOT_CYCLES) as usize) % NSLOTS;
            receiver_tick(&mut state, &mut mem);
            counts[slot] += 1;
        }
        (counts, pin_ok)
    });

    let start = rdtsc() + START_DELAY_CYCLES;
    start_tsc.store(start, Ordering::Release);
    let sender_pin = sender.join().unwrap_or(false);
    let (counts_u64, receiver_pin) = receiver.join().unwrap_or((vec![0u64; NSLOTS], false));
    let counts: Vec<f64> = counts_u64.iter().map(|&x| x as f64).collect();
    (counts, sender_pin, receiver_pin)
}

fn features(counts: &[f64], pattern: &[i8]) -> Vec<f64> {
    let n = counts.len() as f64;
    let mean = counts.iter().sum::<f64>() / n;
    let var = counts.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let sd = var.sqrt() + 1e-12;
    let mut z: Vec<f64> = counts.iter().map(|x| (x - mean) / sd).collect();
    for x in &mut z {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
    let p: Vec<f64> = pattern.iter().map(|&x| x as f64).collect();
    let corr = z.iter().zip(p.iter()).map(|(a, b)| a * b).sum::<f64>() / n;
    let rev_corr = z.iter().zip(p.iter().rev()).map(|(a, b)| a * b).sum::<f64>() / n;
    let half = counts.len() / 2;
    let h0 = z[..half].iter().sum::<f64>() / (half as f64);
    let h1 = z[half..].iter().sum::<f64>() / (half as f64);
    let mut out = vec![mean, sd, corr, rev_corr, h0 - h1];
    for harm in 1..=5 {
        let mut ci = 0.0;
        let mut cq = 0.0;
        for (i, &v) in z.iter().enumerate() {
            let th = 2.0 * std::f64::consts::PI * (harm as f64) * (i as f64) / n;
            ci += v * th.cos();
            cq += v * th.sin();
        }
        out.push(ci / n);
        out.push(cq / n);
    }
    out
}

fn auc_one(scores: &[f64], labels: &[i32]) -> f64 {
    let mut pairs: Vec<(f64, i32)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(CmpOrdering::Equal));
    let n_pos = labels.iter().filter(|&&y| y == 1).count() as f64;
    let n_neg = labels.iter().filter(|&&y| y == 0).count() as f64;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    let mut rank_sum = 0.0;
    let mut i = 0usize;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && (pairs[j].0 - pairs[i].0).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = ((i + 1 + j) as f64) * 0.5;
        for item in pairs.iter().take(j).skip(i) {
            if item.1 == 1 {
                rank_sum += avg_rank;
            }
        }
        i = j;
    }
    (rank_sum - n_pos * (n_pos + 1.0) * 0.5) / (n_pos * n_neg)
}

fn best_signed_auc(x: &[Vec<f64>], labels: &[i32]) -> (f64, usize, f64) {
    let nf = x.first().map(|r| r.len()).unwrap_or(0);
    let mut best = 0.5;
    let mut best_idx = 0;
    let mut raw = 0.5;
    for j in 0..nf {
        let col: Vec<f64> = x.iter().map(|r| r[j]).collect();
        let a = auc_one(&col, labels);
        let signed = a.max(1.0 - a);
        if signed > best {
            best = signed;
            best_idx = j;
            raw = a;
        }
    }
    (best, best_idx, raw)
}

fn shuffle_null(x: &[Vec<f64>], labels: &[i32], seed: u64) -> (f64, f64) {
    let mut rng = Rng::new(seed);
    let mut nulls = Vec::with_capacity(N_SHUFFLES);
    for _ in 0..N_SHUFFLES {
        let mut y = labels.to_vec();
        rng.shuffle(&mut y);
        let (a, _, _) = best_signed_auc(x, &y);
        nulls.push(a);
    }
    nulls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(CmpOrdering::Equal));
    let p95 = nulls[((N_SHUFFLES as f64 * 0.95).floor() as usize).min(N_SHUFFLES - 1)];
    let mean = nulls.iter().sum::<f64>() / (nulls.len() as f64);
    (p95, mean)
}

#[derive(Clone)]
struct Cell {
    mode: String,
    n: usize,
    trials: usize,
    auc: f64,
    null95: f64,
    null_mean: f64,
    best_feature: usize,
    raw_auc: f64,
    verdict: String,
    sender_pin_ok: bool,
    receiver_pin_ok: bool,
    elapsed_s: f64,
}

fn run_cell(n: usize, mode: &str, seed: u64) -> (Cell, Vec<(i32, Vec<f64>)>) {
    let t0 = Instant::now();
    let mut rng = Rng::new(seed);
    let mut rows = Vec::with_capacity(PAIRS_PER_MODE * 2);
    let mut sender_pin_ok = true;
    let mut receiver_pin_ok = true;

    for pair in 0..PAIRS_PER_MODE {
        let n_mod = 1usize << n;
        let d0 = loop {
            let d = sample_secret(n_mod, &mut rng);
            if d < n_mod / 2 {
                break d;
            }
        };
        let base = coset_instance(n, d0, &mut rng);
        let fold = folded(&base);
        let mut branches = vec![base, fold];
        if rng.usize(2) == 1 {
            branches.swap(0, 1);
        }
        for inst in branches {
            let base_pat = match mode {
                "public_shuffle_null" => public_chiral_pattern(&inst, true),
                _ => public_chiral_pattern(&inst, false),
            };
            let drive = drive_pattern(&base_pat, &inst, mode);
            let (counts, sp, rp) = run_physical_trial(&drive, seed ^ ((pair as u64) << 9) ^ (inst.d as u64));
            sender_pin_ok &= sp;
            receiver_pin_ok &= rp;
            rows.push((orientation(&inst), features(&counts, &base_pat)));
        }
    }

    let labels: Vec<i32> = rows.iter().map(|r| r.0).collect();
    let x: Vec<Vec<f64>> = rows.iter().map(|r| r.1.clone()).collect();
    let (auc, best_feature, raw_auc) = best_signed_auc(&x, &labels);
    let (null95, null_mean) = shuffle_null(&x, &labels, seed ^ 0x7777_1111);
    let verdict = if auc > null95 + 0.03 {
        if mode == "hidden_chiral_control" {
            "PHYSICAL_HIDDEN_PREP_CHANNEL_LIVE"
        } else {
            "PHYSICAL_PUBLIC_CHIRAL_CANDIDATE"
        }
    } else {
        "PHYSICAL_PUBLIC_CHIRAL_NO_CROSSING"
    };

    let cell = Cell {
        mode: mode.to_string(),
        n,
        trials: rows.len(),
        auc,
        null95,
        null_mean,
        best_feature,
        raw_auc,
        verdict: verdict.to_string(),
        sender_pin_ok,
        receiver_pin_ok,
        elapsed_s: t0.elapsed().as_secs_f64(),
    };
    (cell, rows)
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn write_json(path: &str, cells: &[Cell], total_elapsed: f64) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "{{")?;
    writeln!(f, "  \"experiment\": \"phase6_chiral_pdn_native_rust\",")?;
    writeln!(f, "  \"master_seed\": {},", MASTER_SEED)?;
    writeln!(f, "  \"sender_cpu\": {},", SENDER_CPU)?;
    writeln!(f, "  \"receiver_cpu\": {},", RECEIVER_CPU)?;
    writeln!(f, "  \"nslots\": {},", NSLOTS)?;
    writeln!(f, "  \"repeats\": {},", REPEATS)?;
    writeln!(f, "  \"slot_cycles\": {},", SLOT_CYCLES)?;
    writeln!(f, "  \"pairs_per_mode\": {},", PAIRS_PER_MODE)?;
    writeln!(f, "  \"elapsed_s\": {:.6},", total_elapsed)?;
    writeln!(f, "  \"cells\": [")?;
    for (i, c) in cells.iter().enumerate() {
        writeln!(f, "    {{")?;
        writeln!(f, "      \"mode\": \"{}\",", escape(&c.mode))?;
        writeln!(f, "      \"n\": {},", c.n)?;
        writeln!(f, "      \"trials\": {},", c.trials)?;
        writeln!(f, "      \"auc\": {:.9},", c.auc)?;
        writeln!(f, "      \"null95\": {:.9},", c.null95)?;
        writeln!(f, "      \"null_mean\": {:.9},", c.null_mean)?;
        writeln!(f, "      \"best_feature\": {},", c.best_feature)?;
        writeln!(f, "      \"raw_auc\": {:.9},", c.raw_auc)?;
        writeln!(f, "      \"sender_pin_ok\": {},", c.sender_pin_ok)?;
        writeln!(f, "      \"receiver_pin_ok\": {},", c.receiver_pin_ok)?;
        writeln!(f, "      \"elapsed_s\": {:.6},", c.elapsed_s)?;
        writeln!(f, "      \"verdict\": \"{}\"", escape(&c.verdict))?;
        write!(f, "    }}")?;
        if i + 1 != cells.len() {
            writeln!(f, ",")?;
        } else {
            writeln!(f)?;
        }
    }
    writeln!(f, "  ]")?;
    writeln!(f, "}}")?;
    Ok(())
}

fn write_csv(path: &str, all_rows: &[(String, usize, i32, Vec<f64>)]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "mode,n,label,mean,sd,corr,rev_corr,half_delta,h1_i,h1_q,h2_i,h2_q,h3_i,h3_q,h4_i,h4_q,h5_i,h5_q")?;
    for (mode, n, label, feat) in all_rows {
        write!(f, "{},{},{}", mode, n, label)?;
        for v in feat {
            write!(f, ",{:.9}", v)?;
        }
        writeln!(f)?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let out_dir = env::args().nth(1).unwrap_or_else(|| {
        "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/chiral_phase_kickback/rust_baremetal/results".to_string()
    });
    create_dir_all(&out_dir)?;
    let t0 = Instant::now();
    let mut cells = Vec::new();
    let mut all_rows: Vec<(String, usize, i32, Vec<f64>)> = Vec::new();
    let modes = ["public_chiral_native", "public_shuffle_null", "hidden_chiral_control"];
    for &n in &[8usize, 10usize] {
        for (mi, mode) in modes.iter().enumerate() {
            let seed = MASTER_SEED ^ ((n as u64) << 32) ^ ((mi as u64) * 0x9E37_79B9);
            let (cell, rows) = run_cell(n, mode, seed);
            println!(
                "{:<24} n={} verdict={} auc={:.3}/{:.3} best_f={} pins={}/{} elapsed={:.1}s",
                cell.mode,
                cell.n,
                cell.verdict,
                cell.auc,
                cell.null95,
                cell.best_feature,
                cell.sender_pin_ok,
                cell.receiver_pin_ok,
                cell.elapsed_s
            );
            for (label, feat) in rows {
                all_rows.push((cell.mode.clone(), n, label, feat));
            }
            cells.push(cell);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    write_json(&format!("{}/chiral_pdn_native_result.json", out_dir), &cells, elapsed)?;
    write_csv(&format!("{}/chiral_pdn_native_features.csv", out_dir), &all_rows)?;
    println!("wrote {}/chiral_pdn_native_result.json", out_dir);
    println!("wrote {}/chiral_pdn_native_features.csv", out_dir);
    Ok(())
}
