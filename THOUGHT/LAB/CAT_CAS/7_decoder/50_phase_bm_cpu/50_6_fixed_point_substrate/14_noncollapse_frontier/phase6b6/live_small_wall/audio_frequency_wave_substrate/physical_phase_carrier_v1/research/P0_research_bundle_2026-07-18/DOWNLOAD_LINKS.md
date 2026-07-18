# P0 research download links

Access audit date: **2026-07-18**

Use the official product page when a direct link fails. Dynamic product pages often require a browser-generated PDF or manual download.

## Core P0 component and instrument documents

### ADG1419_REV_A: ADG1419 data sheet and simulation models

- Role: fast source-route gate
- Exact part: `ADG1419BRMZ`
- [Direct download](https://www.analog.com/media/en/technical-documentation/data-sheets/ADG1419.pdf)
- [Official product/article page](https://www.analog.com/en/products/adg1419.html)
- Access mode: `direct_pdf`
- Why it matters: Pin truth table, break-before-make, transition time, off leakage/capacitance, charge injection, supply limits, and SPICE/IBIS models.

### ADR45XX_REV_G: ADR4520/25/30/33/40/50 family data sheet

- Role: 3.3 V witness/reference rail
- Exact part: `ADR4533BRZ`
- [Direct download](https://www.analog.com/media/en/technical-documentation/data-sheets/ADR4520_4525_4530_4533_4540_4550.pdf)
- [Official product/article page](https://www.analog.com/en/products/adr4533.html)
- Access mode: `direct_pdf`
- **Revision alert:** Update the registry only after retaining both the legacy expectation and the downloaded Rev. G hash.
- Why it matters: Reference pinout, output accuracy, noise, drive capability, bypassing, and exact revision.

### ADUM140D_REV_K: ADuM140D/E family data sheet

- Role: gate and relay digital isolation
- Exact part: `ADuM140D0BRZ`
- [Direct download](https://www.analog.com/media/en/technical-documentation/data-sheets/ADuM140D_140E_141D_141E_142D_142E.pdf)
- [Official product/article page](https://www.analog.com/en/products/adum140d.html)
- Access mode: `direct_pdf`
- Why it matters: Supply range, fail-safe state, pin map, propagation delay, bypass/layout, and isolation characteristics.

### ADXL354_REV_D: ADXL354/ADXL355 data sheet

- Role: vibration witness
- Exact part: `ADXL354CEZ`
- [Direct download](https://www.analog.com/media/en/technical-documentation/data-sheets/adxl354_adxl355.pdf)
- [Official product/article page](https://www.analog.com/en/products/adxl354.html)
- Access mode: `direct_pdf`
- Why it matters: Pin/power map, analog output scaling, filter behavior, noise, bandwidth, mounting and axis orientation.

### EPSON_FC135: FC-135 32.768 kHz tuning-fork crystal

- Role: mechanical phase carrier
- Exact part: `Q13FC1350000401`
- [Direct download](https://download.epsondevice.com/td/pdf/td_xtal_32khz/FC-135_Q13FC13500004_en.pdf)
- [Official product/article page](https://www.epsondevice.com/crystal/en/products/crystal-unit/fc135.html)
- [Fallback or asset page](https://support.epson.biz/td/api/doc_check.php?dl=brief_FC-135_en.pdf)
- Access mode: `direct_pdf`
- Legacy expected SHA-256: `7906f6bce8e15c8e4c570e31969ec27c1f4880917e28160529e898a0cbbf48b9`
- Legacy expected bytes: `161924`
- Why it matters: Primary carrier limits, order identity, load capacitance, ESR, drive level, and package constraints.

### KEMET_C0G_CURRENT: C0G multilayer ceramic capacitor product data

- Role: ADXL analog-output filter capacitors
- Exact part: `C0805C103J5GACTU`
- [Official product/article page](https://www.yageo.com/en/Chart/Download/pdf/C0805C103J5GACTU)
- [Fallback or asset page](https://www.digikey.com/en/products/detail/kemet/C0805C103J5GACTU/2211711)
- Access mode: `manual_product_page`
- Why it matters: Exact 10 nF C0G/NP0 filter capacitor identity, tolerance, voltage, package and temperature behavior.

### LITTELFUSE_0467_CURRENT: 467 series thin-film chip fuse data

- Role: isolated-supply primary fusing
- Exact part: `0467.500NR`
- [Official product/article page](https://www.littelfuse.com/products/fuses-overcurrent-protection/fuses/surface-mount-fuses/thin-film-chip-fuses/467/0467-500)
- Access mode: `manual_product_page`
- Why it matters: 0.5 A fuse identity, package, current-time behavior, voltage, interrupting rating, derating and reflow limits.

### MURATA_GJM_CURRENT: GJM high-Q C0G capacitor exact product data

- Role: matched electrical dummy carrier
- Exact part: `GJM1555C1H1R0BB01D`
- [Official product/article page](https://www.murata.com/en-us/products/productdetail?partno=GJM1555C1H1R0BB01D)
- [Fallback or asset page](https://psearch.en.murata.com/capacitor/partnumber/)
- Access mode: `manual_product_page`
- Why it matters: Exact 1.0 pF C0G dummy capacitance, tolerance, package, voltage and RF characteristics.

### MURATA_GRM_CURRENT: GRM X7R capacitor exact product data

- Role: local supply bypass capacitors
- Exact part: `GRM21BR71C104KA01L`
- [Official product/article page](https://www.murata.com/en-us/products/productdetail?partno=GRM21BR71C104KA01L)
- [Fallback or asset page](https://psearch.en.murata.com/capacitor/partnumber/)
- Access mode: `manual_product_page`
- Why it matters: 100 nF local bypass identity, voltage rating, X7R derating, package and reflow.

### MURATA_MEV1_CURRENT: MEV1D isolated DC-DC converter data

- Role: isolated ±5 V and relay supplies
- Exact part: `MEV1D0505SC`
- [Official product/article page](https://www.murata.com/en-us/products/productdetail?partno=MEV1D0505SC)
- Access mode: `manual_product_page`
- Why it matters: Input/output ratings, isolation, regulation, ripple, load requirements, capacitance across barrier and package pinout.

### NEXPERIA_1N4148: 1N4148W data sheet

- Role: relay clamp steering diode
- Exact part: `1N4148W,115`
- [Direct download](https://assets.nexperia.com/documents/data-sheet/1N4148W.pdf)
- [Official product/article page](https://www.nexperia.com/product/1N4148W)
- Access mode: `direct_pdf`
- Legacy expected SHA-256: `13f368ed2f370fe0a613fe1993183cce06717202bcd6a96b912bde017f5cea0e`
- Legacy expected bytes: `16777`
- Why it matters: Clamp current, reverse voltage, package, and pin orientation.

### NEXPERIA_2N7002: 2N7002PW data sheet

- Role: relay MOSFET driver
- Exact part: `2N7002PW,115`
- [Direct download](https://assets.nexperia.com/documents/data-sheet/2N7002PW.pdf)
- [Official product/article page](https://www.nexperia.com/product/2N7002PW)
- Access mode: `direct_pdf`
- Legacy expected SHA-256: `3ef49e87f0304160e534bb2750066bf596155b7d42578f3bc40a98955545e77e`
- Legacy expected bytes: `284517`
- Why it matters: Relay driver pinout, voltage/current ratings, switching, and package.

### NEXPERIA_BZT52: BZT52H series data sheet

- Role: relay release zener clamp
- Exact part: `BZT52H-C12,115`
- [Direct download](https://assets.nexperia.com/documents/data-sheet/BZT52H_SER.pdf)
- [Official product/article page](https://www.nexperia.com/product/BZT52H-C12)
- Access mode: `direct_pdf`
- Legacy expected SHA-256: `a5e54de45bdf1af712c2d9fdfb0fd092804cd83b26eb66e3656b12dea0c8b592`
- Legacy expected bytes: `336144`
- Why it matters: Zener clamp voltage, pulse behavior, package and thermal limits.

### OMRON_G6K_2026_06_01: G6K low-signal relay data sheet

- Role: physical series barriers and midpoint guard
- Exact part: `G6K-2F-Y DC5`
- [Official product/article page](https://components.omron.com/us-en/products/relays/G6K)
- [Fallback or asset page](https://components.omron.com/us-en/asset/54136)
- Access mode: `manual_product_page`
- Why it matters: Exact contact map, coil, operate/release time, bounce, insulation, mechanical life, package and ordering suffix.

### OPA810: OPA810 data sheet

- Role: high-impedance voltage sense buffer
- Exact part: `OPA810IDT`
- [Direct download](https://www.ti.com/lit/ds/symlink/opa810.pdf)
- [Official product/article page](https://www.ti.com/product/OPA810)
- Access mode: `direct_pdf`
- Legacy expected SHA-256: `74c61ac238989c94c1cf0d70da41bff6e167a590f86e06bae7cdd734d8fd26fa`
- Legacy expected bytes: `3406332`
- Why it matters: Input capacitance/leakage planning, unity-gain stability, supply, noise, output drive, package, and handling.

### SHT4X_2025: SHT4x data sheet

- Role: temperature and humidity custody
- Exact part: `SHT45-AD1B-R2`
- [Direct download](https://sensirion.com/media/documents/33FD6951/67EB9032/HT_DS_Datasheet_SHT4x_5.pdf)
- [Official product/article page](https://sensirion.com/products/catalog/SHT45)
- Access mode: `direct_pdf`
- **Revision alert:** Manifest retains product-page label 04/2025 and PDF cover Version 7.1 March 2025.
- Why it matters: Exact sensor variant, I2C address, command 0xFD, CRC polynomial, conversion formulas, handling and environmental limits.

### SIGLENT_DATASHEET: SDG1000X data sheet

- Role: source specifications
- Exact part: `SDG1032X`
- [Official product/article page](https://siglentna.com/SDG1000X/)
- [Fallback or asset page](https://www.siglent.com/my/products-overview/sdg1000x/)
- Access mode: `manual_product_page`
- Legacy expected SHA-256: `ca889ea73c85de7aef40d1faf2e85212ea6ed1d16435ae17c279a858a1d99d3a`
- Legacy expected bytes: `2572702`
- Why it matters: Amplitude, phase, output-impedance, channel-coherence, and accuracy limits.

### SIGLENT_MANUAL: SDG1000X user manual

- Role: source operation and phase setup
- Exact part: `SDG1032X`
- [Direct download](https://int.siglent.com/upload_file/user/SDG1000X_UserManual_UM0201X-E02D.pdf)
- [Official product/article page](https://siglentna.com/SDG1000X/)
- Access mode: `direct_pdf`
- Legacy expected SHA-256: `11c325f98fea514659be9790a001e90e445119584e31fd8d796b33e92d6e4bed`
- Legacy expected bytes: `2930139`
- Why it matters: Source topology, High-Z display mode versus physical 50-ohm output, dual-channel phase setup, and continuous-output operation.

### SIGLENT_PROGRAMMING: SDG programming guide

- Role: source remote-control/queryback
- Exact part: `SDG1032X / SDG1000X family`
- [Direct download](https://int.siglent.com/upload_file/user/SDG_ProgrammingGuide_PG02-E04D.pdf)
- [Official product/article page](https://siglentna.com/SDG1000X/)
- Access mode: `direct_pdf`
- Legacy expected SHA-256: `a27c841ef10ebeba8c437be88933079b358d80d55d20b0d3bbf032cbc8b7125d`
- Legacy expected bytes: `4240871`
- Why it matters: Defines command/queryback behavior needed to prove frequency, amplitude, phase, load mode, and output state.

### SPECTRUM_DATASHEET: DN2.59x digitizerNETBOX data sheet

- Role: four-channel simultaneous digitizer
- Exact part: `DN2.592-04`
- [Direct download](https://spectrum-instrumentation.com/dl/dn2_59x_datasheet_english.pdf)
- [Official product/article page](https://spectrum-instrumentation.com/products/details/DN2592-04.php)
- Access mode: `direct_pdf`
- Legacy expected SHA-256: `5bba0c74b950ac27e447bb25df70973ce19ff8a7a3e4d784378e25a9407d8925`
- Legacy expected bytes: `1060774`
- Why it matters: Confirms simultaneous channel architecture, resolution, sample rates, and input characteristics.

### SPECTRUM_MANUAL: DN.59x hardware manual

- Role: digitizer setup, input, clock, and SDK custody
- Exact part: `DN2.592-04`
- [Direct download](https://spectrum-instrumentation.com/dl/hardware_manual_dn2_59x_english.pdf)
- [Official product/article page](https://spectrum-instrumentation.com/products/details/DN2592-04.php)
- Access mode: `direct_pdf`
- Legacy expected SHA-256: `0cef0929de585c056ecc7605c570ba05c77b0f4fc6c414e393a2c1e578f6ca05`
- Legacy expected bytes: `13171754`
- Why it matters: Input loading, clocking, acquisition mode, channel simultaneity, native data, and SDK/export behavior.

### ST_UM2591_REV2: STM32G0 Nucleo-32 board user manual

- Role: controller board pinout and physical board identity
- Exact part: `NUCLEO-G031K8 / MB1455`
- [Direct download](https://www.st.com/resource/en/user_manual/um2591-stm32g0-nucleo32-board-mb1455-stmicroelectronics.pdf)
- [Official product/article page](https://www.st.com/en/evaluation-tools/nucleo-g031k8.html)
- Access mode: `direct_pdf`
- **Revision alert:** Current user manual is Rev. 2 and current product resources reference MB1455-G031K8-D01.
- Why it matters: Board connector pinout, board revision, power paths, ST-LINK behavior and physical dimensions.

### VISHAY_CRHV: CRHV high-voltage chip resistor data sheet

- Role: ultrahigh-value carrier bias return
- Exact part: `CRHV1206AF150MFKFB`
- [Direct download](https://www.vishay.com/docs/68002/crhv.pdf)
- [Official product/article page](https://www.vishay.com/en/product/68002/)
- Access mode: `direct_pdf`
- Legacy expected SHA-256: `8826283500cc095ea3ba24dcf3080bd9072ec90afa8cf7487d68b99bd11b8d6b`
- Legacy expected bytes: `117230`
- Why it matters: Bias resistance, leakage, voltage coefficient, package, and environmental behavior.

### VISHAY_TNPW: TNPW e3 thin-film resistor data sheet

- Role: precision low-value and timing resistors
- Exact part: `Multiple exact TNPW0805 values`
- [Direct download](https://www.vishay.com/docs/31006/tnpw_e3.pdf)
- [Official product/article page](https://www.vishay.com/en/product/28758/)
- Access mode: `direct_pdf`
- Legacy expected SHA-256: `42309eb5c95365801d46cbfdc020fb9171d756f5e9015b35a191b35d043890f9`
- Legacy expected bytes: `153883`
- Why it matters: Tolerance, temperature coefficient, voltage limits, package, and exact value availability.

## Official design resources

### ST_NUCLEO_G031K8_D01_RESOURCES: NUCLEO-G031K8 schematic, BOM, board design and manufacturing files

- Role: controller schematic and board-revision custody
- Exact part: `MB1455-G031K8-D01`
- [Official product/article page](https://www.st.com/en/evaluation-tools/nucleo-g031k8.html)
- Access mode: `manual_product_page`
- Why it matters: Allows independent pin/power-path verification rather than relying only on the user manual.

## Scientific background and methods

### GHZ_OPTOMECH_AIR_2011: GHz optomechanical resonators with high mechanical Q in air

- Role: future optomechanical carrier class
- [Direct download](https://arxiv.org/pdf/1109.4705)
- [Official product/article page](https://arxiv.org/abs/1109.4705)
- Access mode: `direct_pdf`
- Why it matters: Evidence for a future optical readout/drive carrier class, not the present electrical P0 experiment.

### QTF_PASSIVE_ELECTRICAL_DAMPING_2021: Passive Electrical Damping of a Quartz Tuning Fork

- Role: BVD loading and damping
- [Direct download](https://pmc.ncbi.nlm.nih.gov/articles/PMC8347380/pdf/sensors-21-05056.pdf)
- [Official product/article page](https://pmc.ncbi.nlm.nih.gov/articles/PMC8347380/)
- Access mode: `direct_pdf`
- Why it matters: Explains the BVD motional RLC plus parallel electrode capacitance and how electrical loading alters QTF dynamics.

### QTF_RESONANCE_TRACKING_2019: Quartz Tuning Fork Resonance Tracking and application

- Role: time-domain QTF resonance and ringdown methods
- [Direct download](https://pmc.ncbi.nlm.nih.gov/articles/PMC6960650/pdf/sensors-20-00206.pdf)
- [Official product/article page](https://pmc.ncbi.nlm.nih.gov/articles/PMC6960650/)
- Access mode: `direct_pdf`
- Why it matters: Directly relevant 32.768 kHz quartz tuning-fork characterization, transient/ringdown and BVD fitting.

### QTF_VOLTAGE_INDUCED_SHIFT_2014: Voltage Induced Frequency Shift on a Quartz Tuning Fork

- Role: drive/bias perturbation of 32.768 kHz QTF
- [Direct download](https://pmc.ncbi.nlm.nih.gov/articles/PMC4279570/pdf/sensors-14-24529.pdf)
- [Official product/article page](https://pmc.ncbi.nlm.nih.gov/articles/PMC4279570/)
- Access mode: `direct_pdf`
- Why it matters: Shows that applied voltage can perturb the eigenfrequency of a 32.768 kHz fork.

### QTF_VOLTAGE_MODE_READOUT_2023: Signal-to-Noise Ratio Analysis for Voltage-Mode Read-Out of Quartz Tuning Forks

- Role: QTF BVD/readout loading and SPICE model
- [Direct download](https://pmc.ncbi.nlm.nih.gov/articles/PMC10051664/pdf/sensors-23-03005.pdf)
- [Official product/article page](https://pmc.ncbi.nlm.nih.gov/articles/PMC10051664/)
- Access mode: `direct_pdf`
- Why it matters: Uses a 32,768 Hz BVD model and SPICE to show how input capacitance and load resistance shift readout response and SNR.

### SILICON_PHONONIC_SLAB_2011: Simultaneous high-Q confinement and selective direct piezoelectric excitation in a silicon phononic crystal slab resonator

- Role: future silicon phononic target
- [Official product/article page](https://doi.org/10.1016/j.sna.2011.03.014)
- Access mode: `citation_or_paywalled`
- Why it matters: Primary literature for the selected future silicon transposition class.

### THIN_FILM_QUARTZ_PHONONIC_2024: High-coherence thin-film quartz phononic resonators

- Role: future quartz/silicon phononic translation
- [Direct download](https://arxiv.org/pdf/2406.14660)
- [Official product/article page](https://arxiv.org/abs/2406.14660)
- Access mode: `direct_pdf`
- Why it matters: Demonstrates high-coherence piezoelectric phononic resonators and ringdown, relevant to the P1 translation target.

### UNIFIED_QTF_THEORY_2026: Unified Theory of Quartz Tuning Fork Resonators

- Role: new theoretical QTF electromechanical observability framework
- [Direct download](https://arxiv.org/pdf/2606.00681)
- [Official product/article page](https://arxiv.org/abs/2606.00681)
- Access mode: `direct_pdf`
- Why it matters: Recent first-principles electroelastic treatment of electrically observed QTF modes.

## Official simulation resources

### ADG1419_SIMULATION_MODELS: ADG1419 LTspice, SPICE macro and IBIS models

- Role: switch transient and feedthrough simulation
- Exact part: `ADG1419BRMZ`
- [Official product/article page](https://www.analog.com/en/products/adg1419.html)
- Access mode: `manual_product_page`
- Why it matters: Needed to move from direct synthetic waveform construction toward circuit-level source-off simulation.

### OPA810_SIMULATION_MODELS: OPA810 PSpice and TINA-TI models

- Role: sense-amplifier stability/noise simulation
- Exact part: `OPA810IDT`
- [Official product/article page](https://www.ti.com/product/OPA810)
- [Fallback or asset page](https://www.ti.com/tool/TINA-TI)
- Access mode: `manual_product_page`
- Why it matters: Needed to test stability, input loading, noise and transient recovery with the BVD carrier and switch network.
