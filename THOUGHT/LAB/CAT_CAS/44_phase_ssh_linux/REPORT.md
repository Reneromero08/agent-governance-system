# 44_phase_ssh_linux - Phenom II Bare Metal LAN Setup

## Target Hardware

| Component | Detail |
|-----------|--------|
| CPU | AMD Phenom II X6 (Thuban, Family 10h) |
| Motherboard | Gigabyte GA-970A-DS3P (AMD 970 chipset) |
| RAM | DDR3 1600 |
| GPU | NVIDIA GeForce GTX 1050 |
| NIC (onboard) | Realtek RTL8111F GbE (r8169 driver) |
| WiFi (USB) | Netgear WNA1100 (Atheros AR9271, ath9k_htc driver) |
| Boot Drive | Sabrent 128GB SATA SSD (hot-swap) |

## OS

| Component | Detail |
|-----------|--------|
| Distro | Debian 13 "Trixie" (netinst) |
| Kernel | 6.12.86+deb13-amd64 |
| Hostname | catcas |
| Root Password | linktome |

## Network Architecture

```
[Phenom II catcas] ---Ethernet--- [Windows ASSFACE3000] ---WiFi--- [Internet]
  192.168.137.100                  192.168.137.1 (Ethernet 2)
                                   10.5.0.2 (Ethernet)
                                   192.168.0.12 (Ethernet 2, alt)
```

Windows ICS shares WiFi internet to Ethernet 2 (Realtek Gaming GbE) which feeds the Phenom II directly.

## Persistent Configuration

### /etc/network/interfaces

```
auto lo
iface lo inet loopback

auto enp3s0
iface enp3s0 inet static
    address 192.168.137.100
    netmask 255.255.255.0
    gateway 192.168.137.1
    dns-nameservers 8.8.8.8 8.8.4.4
```

### SSH

- PermitRootLogin: yes
- Key-based auth: ed25519 from ASSFACE3000 (reneshizzle@ASSFACE3000)
- Key installed in /root/.ssh/authorized_keys
- Service: ssh (enabled)

### Services Enabled

- ssh
- networking

## Boot Sequence (Survives Reboot)

1. Power on Phenom II
2. BIOS POST (MSI monitor, Gigabyte board)
3. GRUB loads kernel 6.12.86+deb13-amd64
4. networking.service brings up enp3s0 at 192.168.137.100/24
5. ssh.service starts, accepts key-based auth
6. Windows ICS provides internet through gateway 192.168.137.1

## Connection from Windows (ASSFACE3000)

```powershell
ssh root@192.168.137.100
```

No password required (key-based auth).

## Installed Packages

- build-essential (gcc, make, etc.)
- linux-headers-6.12.86+deb13-amd64 (kernel headers for module compilation)
- msr-tools (rdmsr/wrmsr for MSR access)
- devmem2 (built from source at /usr/local/bin/devmem2)
- lm-sensors (sensors command, k10temp module loaded)
- openssh-server
- ethtool
- pciutils (lspci)
- firmware-atheros (for Netgear WNA1100 USB WiFi)

## Sensors Output

```
k10temp-pci-00c3
Adapter: PCI adapter
temp1:  +24.6C  (high = +70.0C, crit = +80.0C, hyst = +75.0C)

nouveau-pci-0100
Adapter: PCI adapter
fan1:    0 RPM
temp1:  +53.0C  (high = +95.0C, crit = +105.0C, emerg = +135.0C)
```

## lspci (Full)

```
00:00.0 Host bridge: AMD RD9x0/RX980 Host Bridge (rev 02)
00:00.2 IOMMU: AMD RD890S/RD9x0 IOMMU (rev 02)
00:02.0 PCI bridge: AMD RD890S/RD9x0 PCI to PCI bridge (GFX port 0)
00:04.0 PCI bridge: AMD RD890S/RD9x0 PCI to PCI bridge (GPP Port 0)
00:11.0 SATA controller: AMD SB7x0/SB8x0/SB9x0 SATA Controller [AHCI] (rev 40)
00:12.0 USB controller: AMD SB7x0/SB8x0/SB9x0 USB OHCI0 Controller
00:12.2 USB controller: AMD SB7x0/SB8x0/SB9x0 USB EHCI Controller
00:13.0 USB controller: AMD SB7x0/SB8x0/SB9x0 USB OHCI0 Controller
00:13.2 USB controller: AMD SB7x0/SB8x0/SB9x0 USB EHCI Controller
00:14.0 SMBus: AMD SBx0 SMBus Controller (rev 42)
00:14.1 IDE interface: AMD SB7x0/SB8x0/SB9x0 IDE Controller (rev 40)
00:14.3 ISA bridge: AMD SB7x0/SB8x0/SB9x0 LPC host controller (rev 40)
00:14.4 PCI bridge: AMD SBx0 PCI to PCI Bridge (rev 40)
00:14.5 USB controller: AMD SB7x0/SB8x0/SB9x0 USB OHCI2 Controller
00:16.0 USB controller: AMD SB7x0/SB8x0/SB9x0 USB OHCI0 Controller
00:16.2 USB controller: AMD SB7x0/SB8x0/SB9x0 USB EHCI Controller
00:18.0 Host bridge: AMD Family 10h Processor HyperTransport Configuration
00:18.1 Host bridge: AMD Family 10h Processor Address Map
00:18.2 Host bridge: AMD Family 10h Processor DRAM Controller
00:18.3 Host bridge: AMD Family 10h Processor Miscellaneous Control
00:18.4 Host bridge: AMD Family 10h Processor Link Control
01:00.0 VGA compatible controller: NVIDIA GeForce GTX 1050 (rev a1)
01:00.1 Audio device: NVIDIA GP107GL High Definition Audio Controller (rev a1)
02:00.0 USB controller: VIA VL805/806 xHCI USB 3.0 Controller (rev 01)
```

## Next Steps

- Kernel cmdline isolation (isolcpus, nohz_full, rcu_nocbs)
- Undervolt configuration via MSR
- Phase-master / PPU / readout core role assignment
- Agent governance connection setup

## Known Issues

- BIOS had LAN controller disabled by default (re-enabled manually)
- Debian 13 (Trixie) netinst but sources.list was pointing to bookworm (fixed)
- python3-pip has broken deps on trixie (python3-distutils conflict)
- devmem2 not in repos (built from source)
- k10temp-tools not a real package (use lm-sensors instead)
