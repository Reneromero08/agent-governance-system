#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <openssl/sha.h>

volatile uint64_t *tape __attribute__((aligned(64)));
#define TAPE_SIZE 256

#define SLOT_MASTER    0
#define SLOT_R1        1
#define SLOT_R2        2
#define SLOT_OUTPUT    3
#define SLOT_META      4
#define SLOT_BASIS_0   5
#define SLOT_BASIS_1   6
#define SLOT_ANGLE_R1  7
#define SLOT_ANGLE_R2  8

void holo_xor(int core, int slot, uint64_t seed, uint64_t iters) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    uint64_t x = seed;
    for (uint64_t i = 0; i < iters; i++) {
        x = (x * 0x41C64E6D + 0x3039);
        x = (x >> 13) ^ x;
        x = (x << 17) + x;
    }
    __atomic_fetch_xor(&tape[slot], x, __ATOMIC_SEQ_CST);
}

void sha256_tape(unsigned char *h) { SHA256((unsigned char *)tape, TAPE_SIZE, h); }

int main() {
    tape = (volatile uint64_t *)mmap(NULL, TAPE_SIZE, PROT_READ|PROT_WRITE,
                                      MAP_SHARED|MAP_ANONYMOUS, -1, 0);
    memset((void *)tape, 0xAA, TAPE_SIZE);

    // Write basis metadata before snapshot
    tape[SLOT_META]     = (1UL << 56) | (3UL << 48);
    tape[SLOT_BASIS_0]  = 0x5245464552454E43;
    tape[SLOT_BASIS_1]  = 0x524F544154494F4E;
    tape[SLOT_ANGLE_R1] = 1570796;
    tape[SLOT_ANGLE_R2] = 3141593;

    unsigned char h0[SHA256_DIGEST_LENGTH], h1[SHA256_DIGEST_LENGTH];
    sha256_tape(h0);

    uint64_t m0=tape[SLOT_META], b0=tape[SLOT_BASIS_0], b1=tape[SLOT_BASIS_1];
    uint64_t a1=tape[SLOT_ANGLE_R1], a2=tape[SLOT_ANGLE_R2];

    printf("=== PHASE 3.6: .HOLO EIGENBASIS WITH METADATA ===\n");
    printf("Tape SHA256: ");
    for (int i=0;i<8;i++) printf("%02x", h0[i]);
    printf("...\n");
    printf("Metadata: magic=HOLOBASI v=1 dims=3\n");
    printf("Basis 0=REFERENCE Basis 1=ROTATION\n");

    // FORWARD
    printf("\n=== FORWARD PASS ===\n");
    pid_t p;
    p=fork();if(p==0){holo_xor(5,SLOT_MASTER,0xCAFE0001,10000000);_exit(0);}waitpid(p,NULL,0);
    p=fork();if(p==0){holo_xor(3,SLOT_R1,0xBABE0002,10000000);_exit(0);}waitpid(p,NULL,0);
    p=fork();if(p==0){holo_xor(4,SLOT_R2,0xDEAD0003,10000000);_exit(0);}waitpid(p,NULL,0);
    p=fork();if(p==0){holo_xor(3,SLOT_OUTPUT,0xF00D0004,10000000);_exit(0);}waitpid(p,NULL,0);

    uint64_t out = tape[SLOT_OUTPUT];
    printf("Slot 0(Master): 0x%016lx\nSlot 1(R1): 0x%016lx\nSlot 2(R2): 0x%016lx\nSlot 3(Output): 0x%016lx\n",
           tape[SLOT_MASTER], tape[SLOT_R1], tape[SLOT_R2], out);

    int meta_fwd = (tape[SLOT_META]==m0&&tape[SLOT_BASIS_0]==b0&&tape[SLOT_BASIS_1]==b1
                    &&tape[SLOT_ANGLE_R1]==a1&&tape[SLOT_ANGLE_R2]==a2);
    printf("Metadata survived forward: %s\n", meta_fwd ? "YES" : "NO");

    // REVERSE
    printf("\n=== REVERSE PASS ===\n");
    p=fork();if(p==0){holo_xor(5,SLOT_MASTER,0xCAFE0001,10000000);_exit(0);}waitpid(p,NULL,0);
    p=fork();if(p==0){holo_xor(3,SLOT_R1,0xBABE0002,10000000);_exit(0);}waitpid(p,NULL,0);
    p=fork();if(p==0){holo_xor(4,SLOT_R2,0xDEAD0003,10000000);_exit(0);}waitpid(p,NULL,0);
    p=fork();if(p==0){holo_xor(3,SLOT_OUTPUT,0xF00D0004,10000000);_exit(0);}waitpid(p,NULL,0);

    sha256_tape(h1);
    int sha_ok = memcmp(h0, h1, SHA256_DIGEST_LENGTH) == 0;
    printf("Tape SHA256: ");
    for (int i=0;i<8;i++) printf("%02x", h1[i]);
    printf("... match=%s\n", sha_ok ? "YES" : "NO");

    int meta_rev = (tape[SLOT_META]==m0&&tape[SLOT_BASIS_0]==b0&&tape[SLOT_BASIS_1]==b1
                    &&tape[SLOT_ANGLE_R1]==a1&&tape[SLOT_ANGLE_R2]==a2);
    printf("Metadata restored: %s\n", meta_rev ? "YES" : "NO");

    if (sha_ok && meta_fwd && meta_rev) {
        printf("\n=== PHASE 3.6 COMPLETE ===\n");
        printf("Basis survived forward: YES\nTape restored: YES\nMetadata matches pre-encode: YES\n");
    } else {
        printf("\n=== PHASE 3.6 FAILED ===\n");
    }
    return sha_ok ? 0 : 1;
}
