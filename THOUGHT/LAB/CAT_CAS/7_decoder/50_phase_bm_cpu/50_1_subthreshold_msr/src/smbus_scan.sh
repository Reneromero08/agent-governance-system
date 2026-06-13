#!/bin/bash
echo "=== LOADING I2C DEV MODULE ==="
modprobe i2c-dev
echo "done"

echo ""
echo "=== CHECKING FOR SMBus CONTROLLER ==="
lspci -v | grep -A5 SMBus 2>/dev/null || echo "No SMBus in lspci"

echo ""
echo "=== AVAILABLE I2C/SMBus ADAPTERS ==="
ls /dev/i2c-* 2>/dev/null || echo "No i2c devices found"

echo ""
echo "=== INSTALLING i2c-tools ==="
apt-get install -y i2c-tools 2>&1 | tail -3

echo ""
echo "=== SCANNING I2C BUS 0 ==="
i2cdetect -y 0 2>/dev/null || echo "Bus 0 not available"

echo ""
echo "=== SCANNING I2C BUS 1 ==="
i2cdetect -y 1 2>/dev/null || echo "Bus 1 not available"

echo ""
echo "=== SCANNING I2C BUS 2 ==="
i2cdetect -y 2 2>/dev/null || echo "Bus 2 not available"

echo ""
echo "=== SCANNING I2C BUS 3 ==="
i2cdetect -y 3 2>/dev/null || echo "Bus 3 not available"

echo ""
echo "=== LOADED I2C MODULES ==="
lsmod | grep i2c

echo ""
echo "=== DONE ==="
