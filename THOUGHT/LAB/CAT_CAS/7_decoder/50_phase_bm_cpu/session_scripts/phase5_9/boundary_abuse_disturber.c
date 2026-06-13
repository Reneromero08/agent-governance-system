#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

static int pin_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}

static void cache_abuse(size_t mb) {
    size_t bytes = mb * 1024ULL * 1024ULL;
    uint8_t *buf = NULL;
    if (posix_memalign((void **)&buf, 64, bytes) != 0 || !buf) {
        perror("posix_memalign");
        return;
    }
    memset(buf, 0xa5, bytes);
    uint64_t x = 0x123456789abcdefULL;
    while (!g_stop) {
        for (size_t i = 0; i < bytes; i += 64) {
            x = x * 6364136223846793005ULL + 1442695040888963407ULL;
            buf[i] ^= (uint8_t)(x >> 32);
        }
    }
    free(buf);
}

static void pagefault_abuse(size_t mb) {
    size_t bytes = mb * 1024ULL * 1024ULL;
    uint8_t *buf = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) {
        perror("mmap");
        return;
    }
    while (!g_stop) {
        for (size_t i = 0; i < bytes; i += 4096) {
            buf[i]++;
        }
        madvise(buf, bytes, MADV_DONTNEED);
    }
    munmap(buf, bytes);
}

static void syscall_abuse(void) {
    struct timespec ts;
    uint64_t x = 0;
    while (!g_stop) {
        syscall(SYS_gettid);
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        x ^= (uint64_t)ts.tv_nsec;
        if ((x & 0xfff) == 0) sched_yield();
    }
}

static void branch_abuse(void) {
    uint64_t x = 0xfeedfacecafebeefULL;
    volatile double sink = 0.0;
    while (!g_stop) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        if (x & 1) {
            sink += sqrt((double)(x & 0xffff) + 1.0);
        } else {
            sink -= sin((double)(x & 0xff) * 0.001);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODE CORE [MB]\n", argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    int core = atoi(argv[2]);
    size_t mb = (argc >= 4) ? (size_t)strtoull(argv[3], NULL, 10) : 128;

    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);

    if (pin_core(core) != 0) {
        fprintf(stderr, "WARN: failed to pin to core %d: %s\n", core, strerror(errno));
    }

    if (strcmp(mode, "cache") == 0) cache_abuse(mb);
    else if (strcmp(mode, "pagefault") == 0) pagefault_abuse(mb);
    else if (strcmp(mode, "syscall") == 0) syscall_abuse();
    else if (strcmp(mode, "branch") == 0) branch_abuse();
    else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        return 2;
    }

    return 0;
}
