import numpy as np

class ReversibleCPU:
    """
    Simulates a CPU running strictly on reversible logic gates (XOR, NOT, Toffoli/AND_XOR).
    Tracks state transitions and verifies that no information is erased.
    """
    def __init__(self):
        # Register state: dict mapping bit name -> value (0 or 1)
        self.registers = {}
        # History of gates executed for the reverse pass
        self.gate_history = []

    def set_register(self, name: str, val: int):
        self.registers[name] = val & 1

    def get_register(self, name: str) -> int:
        return self.registers.get(name, 0)

    # Reversible Primitives
    def gate_xor(self, dest: str, src: str):
        """dest = dest ^ src (Reversible)"""
        val_dest = self.get_register(dest)
        val_src = self.get_register(src)
        self.set_register(dest, val_dest ^ val_src)
        self.gate_history.append(("XOR", dest, src))

    def gate_not(self, dest: str):
        """dest = ~dest (Reversible)"""
        val_dest = self.get_register(dest)
        self.set_register(dest, val_dest ^ 1)
        self.gate_history.append(("NOT", dest, None))

    def gate_and_xor(self, dest: str, src1: str, src2: str):
        """dest = dest ^ (src1 & src2) (Toffoli gate - Reversible)"""
        val_dest = self.get_register(dest)
        val_src1 = self.get_register(src1)
        val_src2 = self.get_register(src2)
        self.set_register(dest, val_dest ^ (val_src1 & val_src2))
        self.gate_history.append(("AND_XOR", dest, (src1, src2)))

    def run_reverse(self):
        """
        Executes all recorded gates in reverse order.
        Since all primitives are self-inverting, this perfectly unwinds the computation.
        """
        print("[Reversible CPU] Running reverse pass to clean registers...")
        reversed_history = list(reversed(self.gate_history))
        # Clear history so we don't double count
        self.gate_history = []
        
        for gate_type, dest, src in reversed_history:
            if gate_type == "XOR":
                # XOR is its own inverse
                val_dest = self.get_register(dest)
                val_src = self.get_register(src)
                self.set_register(dest, val_dest ^ val_src)
            elif gate_type == "NOT":
                # NOT is its own inverse
                val_dest = self.get_register(dest)
                self.set_register(dest, val_dest ^ 1)
            elif gate_type == "AND_XOR":
                # Toffoli is its own inverse
                val_dest = self.get_register(dest)
                val_src1 = self.get_register(src[0])
                val_src2 = self.get_register(src[1])
                self.set_register(dest, val_dest ^ (val_src1 & val_src2))


class IrreversibleCPU:
    """
    Simulates a standard CPU where operations overwrite registers,
    causing logical information erasure. Tracks total bits erased.
    """
    def __init__(self):
        self.registers = {}
        self.bits_erased = 0

    def set_register(self, name: str, val: int):
        # Overwriting a register with a different value represents information erasure
        # if the previous value is lost.
        prev = self.registers.get(name, 0)
        new_val = val & 1
        if prev != new_val:
            # We erased 1 bit of information (transition from prev -> new_val is lossy)
            self.bits_erased += 1
        self.registers[name] = new_val

    def get_register(self, name: str) -> int:
        return self.registers.get(name, 0)

    def write_overwrite(self, dest: str, val: int):
        """Direct write (non-reversible overwrite)."""
        self.set_register(dest, val)

    def discard_registers(self, names: list[str]):
        """Simulates cleaning up/discarding registers (resetting to 0)."""
        for name in names:
            if self.get_register(name) == 1:
                # Discarding a set bit erases 1 bit of information
                self.bits_erased += 1
            self.registers[name] = 0


# Physical constants for Landauer limit calculation
kB = 1.380649e-23  # Boltzmann constant (J/K)

def calculate_landauer_energy(bits_erased: int, temp_kelvin: float = 293.15) -> float:
    """Calculates the Landauer limit energy: E = N * kB * T * ln(2)"""
    return bits_erased * kB * temp_kelvin * np.log(2)
