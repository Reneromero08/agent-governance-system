# 50_phase_bm_cpu - Phenom II Bare Metal LAN Setup

## Target Hardware

| Component | Detail |
|-----------|--------|
| CPU | AMD Phenom II X6 (Thuban, Family 10h) |
| Motherboard | Gigabyte GA-970A-DS3P (AMD 970 chipset) |
| RAM | DDR3 1600 |
| GPU | NVIDIA GeForce GTX 1050 |
| NIC (onboard) | Realtek RTL8111F GbE (r8169 driver) |
| WiFi (USB) | Netgear WNA1100 (Atheros AR9271, ath9k_htc driver, firmware 1.4) |
| Boot Drive | Sabrent 128GB SATA SSD (hot-swap) |

## OS

| Component | Detail |
|-----------|--------|
| Distro | Debian 13 "Trixie" (netinst) |
| Kernel | 6.12.86+deb13-amd64 |
| Hostname | catcas |
| Authentication | Dedicated Ed25519 SSH key; passwords intentionally omitted |

## Network Architecture

```
                         +--- Beehive WiFi --- Internet
                         |    192.168.0.102/24 (DHCP at verification)
[Phenom II catcas] ------+
                         +--- Tailscale --- 100.94.228.48
                         |
                         +--- optional direct Ethernet maintenance link
                              192.168.137.100/24
```

The Netgear WNA1100 is the preferred internet interface. The direct Ethernet
link remains configured as a higher-metric fallback/maintenance path and may
be physically disconnected. WiFi-only Tailscale SSH was mechanically verified
with Ethernet reporting `NO-CARRIER`.

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
    metric 600
    dns-nameservers 8.8.8.8 8.8.4.4

allow-hotplug wlxe091f54f8afe
iface wlxe091f54f8afe inet dhcp
    wpa-conf /etc/wpa_supplicant/wpa_supplicant-beehive.conf
    metric 100
```

The WPA2 credential is stored on `catcas` as a derived PSK in
`/etc/wpa_supplicant/wpa_supplicant-beehive.conf` with mode `0600`. The
plaintext WiFi password is not recorded here.

### SSH

- PermitRootLogin: yes
- Key-based auth: dedicated Ed25519 key from the current Linux host
- Public key installed in `/root/.ssh/authorized_keys`
- Key fingerprint: `SHA256:NK3mLL+wEU/Aamr15ZntaDgyHA2t0iPOBR5DS70DKzg`
- Direct-link alias: `ssh catcas`
- Tailscale alias: `ssh catcas-ts`
- Service: ssh (enabled)

No private key or login password is stored in this report.

### Tailscale

- Node name: `catcas`
- Tailnet IPv4: `100.94.228.48`
- Tailnet IPv6: `fd7a:115c:a1e0::e338:e431`
- Service: `tailscaled` (active and enabled)
- WiFi-only Tailscale ping and SSH: verified

### Services Enabled

- ssh
- networking
- wpa_supplicant
- tailscaled

## Boot Sequence (Survives Reboot)

1. Power on Phenom II
2. BIOS POST (MSI monitor, Gigabyte board)
3. GRUB loads kernel 6.12.86+deb13-amd64
4. The WNA1100 loads `ath9k_htc` firmware and exposes `wlxe091f54f8afe`
5. networking.service associates with Beehive and obtains a DHCP lease
6. ssh.service starts and accepts dedicated key-based authentication
7. tailscaled reconnects `catcas` to the tailnet
8. WiFi is preferred at metric 100; Ethernet remains fallback at metric 600

## Connection from the Linux Host

```bash
ssh catcas-ts  # Tailscale; works without the Ethernet cable
ssh catcas     # Optional direct Ethernet maintenance link
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
- firmware-atheros
- firmware-ath9k-htc (AR9271 firmware required by ath9k_htc)
- wpasupplicant
- isc-dhcp-client
- curl
- tailscale 1.98.10 (official stable Debian Trixie repository)

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
- The WNA1100 fails to enumerate on the motherboard's legacy AMD EHCI/OHCI
  USB paths with descriptor errors `-32`/`-71`. It works reliably on the rear
  VIA VL805/806 xHCI USB 3.x port, where firmware 1.4 loads successfully.
- Boot logs warn that EHCI should load before OHCI; no remote controller reset
  or USB-controller reconfiguration was attempted.
