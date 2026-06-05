#!/bin/bash
echo "=== ATTACK PATH: CLOCK GENERATOR AT SMBUS 0x69 ==="
modprobe i2c-dev 2>/dev/null

echo "Dumping 0x69 on bus 0 (first 64 registers):"
i2cdump -y 0 0x69 2>/dev/null || echo "i2cdump failed on 0x69"

echo ""
echo "=== ATTACK PATH: NB PCI CONFIG SPACE ==="
echo "NB PCI devices:"
lspci -s 00:18.0 -v 2>/dev/null | head -10
echo "---"
lspci -s 00:18.1 -v 2>/dev/null | head -10
echo "---"
lspci -s 00:18.2 -v 2>/dev/null | head -10
echo "---"
lspci -s 00:18.3 -v 2>/dev/null | head -10
echo "---"
lspci -s 00:18.4 -v 2>/dev/null | head -10

echo ""
echo "=== NB DRAM CONTROLLER CONFIG SPACE (0:24.2) ==="
xxd /sys/devices/pci0000:00/0000:00:18.2/config 2>/dev/null | head -20 || echo "Cannot read config space"

echo ""
echo "=== ALL SMBUS DEVICES ON BUS 0 ==="
i2cdetect -y 0 2>/dev/null

echo ""
echo "=== DONE ==="
