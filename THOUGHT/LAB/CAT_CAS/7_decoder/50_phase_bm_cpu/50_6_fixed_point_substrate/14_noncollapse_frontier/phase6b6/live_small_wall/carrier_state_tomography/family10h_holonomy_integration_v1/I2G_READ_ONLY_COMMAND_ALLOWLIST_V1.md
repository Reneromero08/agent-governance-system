# Family 10h I2G Read-Only Target Inventory Allowlist V1

Authority scope: read-only inventory only.
Target: `root@192.168.137.100` (hostname expected: `catcas`).

No command may contain redirection, pipelines to writers, package installation, compilation,
process signaling, affinity mutation, PMU opening, file creation, file deletion, or sysfs writes.

SHA-256 of this complete file content is recorded in the accompanying authority grant.

## Commands

Execute each command individually over SSH. Do not copy a remote shell script.

```text
1. hostname
2. uname -a
3. cat /etc/os-release
4. lscpu --json
5. cat /sys/devices/system/cpu/online
6. sh -c 'for d in /sys/devices/system/cpu/cpu[0-9]*/cache/index*; do test -d "$d" || continue; printf "CACHE %s\n" "$d"; for f in level type size shared_cpu_list coherency_line_size number_of_sets ways_of_associativity; do test -r "$d/$f" && { printf "%s=" "$f"; cat "$d/$f"; }; done; done'
7. sh -c 'command -v numactl >/dev/null 2>&1 && numactl --hardware || { test -r /sys/devices/system/node/online && cat /sys/devices/system/node/online; }'
8. taskset -pc $$
9. cat /proc/sys/kernel/perf_event_paranoid
10. sh -c 'for d in /sys/bus/event_source/devices/*; do test -d "$d" || continue; printf "PMU %s\n" "$d"; test -r "$d/type" && { printf "type="; cat "$d/type"; }; test -d "$d/events" && find "$d/events" -maxdepth 1 -type f -printf "%f\n" | LC_ALL=C sort; done'
11. getconf PAGESIZE
12. grep -E '^(HugePages|Hugepagesize|AnonHugePages|ShmemHugePages|FileHugePages)' /proc/meminfo
13. sh -c 'for d in /sys/class/thermal/thermal_zone* /sys/class/hwmon/hwmon*; do test -d "$d" || continue; printf "SENSOR %s\n" "$d"; for f in type name; do test -r "$d/$f" && { printf "%s=" "$f"; cat "$d/$f"; }; done; find "$d" -maxdepth 1 -type f \( -name "temp*_label" -o -name "temp*_input" \) -printf "%m %u:%g %p\n" | LC_ALL=C sort; done'
14. sh -c 'for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_driver /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_min_freq /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_max_freq; do test -r "$f" && { printf "%s=" "$f"; cat "$f"; }; done'
15. sh -c 'cc --version 2>/dev/null | head -n 1; ldd --version 2>/dev/null | head -n 1; python3 --version 2>&1; getconf GNU_LIBC_VERSION 2>/dev/null || true'
16. sh -c 'root=/root/catcas_live_small_wall; if test -d "$root"; then find "$root" -maxdepth 3 -type f -printf "%p\n" | LC_ALL=C sort; fi'
17. sh -c 'root=/root/catcas_live_small_wall; if test -d "$root"; then find "$root" -maxdepth 3 -type f -size -16M -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum; fi'
18. ps -eo pid,ppid,user,stat,comm,args
19. ss -lntp
20. sh -c 'printf "write_attempt_count=0\nscientific_measurement_count=0\n"'
```

## Forbidden even during inventory

```text
perf, perf stat, perf record, perf_event_open helpers, or any PMU acquisition
taskset with a command or taskset -p -c with a new mask
tee, sed -i, printf redirected to a file, shell redirection, or file creation
cp, scp to target, rsync to target, mv, rm, chmod, chown, touch, mkdir
gcc, cc compilation, make, cmake, package managers, or executing experiment binaries
sysctl -w, writes under /sys or /proc, governor/frequency changes, MSR access
kill, pkill, systemctl mutations, reboot, shutdown
```
