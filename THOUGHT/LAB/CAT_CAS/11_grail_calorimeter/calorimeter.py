"""
Micro-Calorimeter Simulation Module
------------------------------------
Models a silicon die with physical thermal mass inside a micro-calorimeter,
tracking cumulative Landauer heat dissipation and temperature rise.

Physical constants:
  kB  = 1.380649e-23  J/K   (Boltzmann constant, exact per 2019 SI)
  ln2 = 0.693147...          (natural log of 2)
  Landauer limit per bit = kB * T * ln(2)

Silicon properties (bulk):
  Specific heat capacity: 712 J/(kg*K)
  Density: 2329 kg/m^3
"""

import numpy as np
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
kB = 1.380649e-23          # Boltzmann constant  (J / K)
LN2 = np.log(2)            # ln(2) ~ 0.6931


@dataclass
class SiliconDie:
    """
    Models a silicon chip with realistic thermal mass.

    Parameters
    ----------
    mass_kg : float
        Mass of the silicon die.  A typical small die (~5 mm x 5 mm x 0.5 mm)
        weighs about 29 mg.
    ambient_temp_K : float
        Ambient / heat-sink temperature (K).  Default 293.15 K (20 C).
    specific_heat : float
        Specific heat capacity of silicon (J / kg K).  Default 712.
    """
    mass_kg: float = 29e-6               # 29 mg
    ambient_temp_K: float = 293.15       # 20 C
    specific_heat: float = 712.0         # J/(kg*K) for Si

    # Derived
    thermal_mass_J_per_K: float = field(init=False)
    cumulative_energy_J: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.thermal_mass_J_per_K = self.mass_kg * self.specific_heat

    @property
    def temperature_K(self) -> float:
        """Current die temperature including heat rise above ambient."""
        return self.ambient_temp_K + self.delta_T_K

    @property
    def delta_T_K(self) -> float:
        """Temperature rise above ambient (K)."""
        return self.cumulative_energy_J / self.thermal_mass_J_per_K

    @property
    def delta_T_uK(self) -> float:
        """Temperature rise in micro-Kelvin."""
        return self.delta_T_K * 1e6

    def inject_heat(self, joules: float):
        """Inject energy into the die (raises temperature)."""
        self.cumulative_energy_J += joules

    def reset(self):
        """Reset the die to ambient temperature."""
        self.cumulative_energy_J = 0.0


class LandauerAccumulator:
    """
    Counts bits erased and computes cumulative Landauer energy.

    E_landauer = N_bits * kB * T * ln(2)
    """
    def __init__(self, temperature_K: float = 293.15):
        self.temperature_K = temperature_K
        self.total_bits_erased = 0
        self.total_energy_J = 0.0

    @property
    def energy_per_bit_J(self) -> float:
        return kB * self.temperature_K * LN2

    def record_erasure(self, bits_erased: int):
        """Record a batch of bit erasures and accumulate energy."""
        energy = bits_erased * self.energy_per_bit_J
        self.total_bits_erased += bits_erased
        self.total_energy_J += energy
        return energy


@dataclass
class CalorimetryReading:
    """Single calorimetry measurement point."""
    workload_name: str
    iteration_count: int
    bits_erased: int
    energy_J: float
    temperature_K: float
    delta_T_uK: float


class MicroCalorimeter:
    """
    Wraps a SiliconDie and LandauerAccumulator to provide a calorimetry
    measurement interface.
    """
    def __init__(self, label: str, die: SiliconDie = None):
        self.label = label
        self.die = die or SiliconDie()
        self.accumulator = LandauerAccumulator(self.die.ambient_temp_K)
        self.readings: list[CalorimetryReading] = []

    def run_workload(self, workload_name: str, bits_erased: int,
                     iteration_count: int) -> CalorimetryReading:
        """
        Record the result of running a workload and inject the corresponding
        Landauer heat into the silicon die.
        """
        energy = self.accumulator.record_erasure(bits_erased)
        self.die.inject_heat(energy)

        reading = CalorimetryReading(
            workload_name=workload_name,
            iteration_count=iteration_count,
            bits_erased=bits_erased,
            energy_J=self.accumulator.total_energy_J,
            temperature_K=self.die.temperature_K,
            delta_T_uK=self.die.delta_T_uK,
        )
        self.readings.append(reading)
        return reading

    def reset(self):
        """Reset the calorimeter for a fresh experiment."""
        self.die.reset()
        self.accumulator = LandauerAccumulator(self.die.ambient_temp_K)
        self.readings.clear()
