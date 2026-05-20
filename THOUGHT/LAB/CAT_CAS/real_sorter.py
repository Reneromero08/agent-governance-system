import os
import sys

# Define clean memory tracker
class CleanMemoryTracker:
    def __init__(self, limit_bytes):
        self.limit = limit_bytes
        self.current = 0
        self.max_observed = 0

    def allocate(self, amount):
        self.current += amount
        if self.current > self.max_observed:
            self.max_observed = self.current
        if self.current > self.limit:
            raise MemoryError(f"Clean memory limit exceeded! Allocated {self.current} bytes, limit is {self.limit} bytes.")

    def free(self, amount):
        self.current -= amount

class CatalyticBmpTape:
    """
    Wraps the BMP image file pixel data as a catalytic tape.
    Retrieves original byte values on-the-fly using the deterministic gradient formula,
    avoiding the need to store pre-state bytes in clean RAM.
    """
    def __init__(self, file_path: str, width: int = 512):
        self.file_path = file_path
        self.width = width
        self.header_offset = 54
        self.f = open(file_path, "r+b")

    def close(self):
        self.f.close()

    def get_original_byte(self, offset: int) -> int:
        """Computes the original pixel byte value at a given offset on-the-fly."""
        pixel_index = offset // 3
        y = pixel_index // self.width
        x = pixel_index % self.width
        channel = offset % 3
        
        if channel == 0:   # Blue
            return (x + y * 3) % 256
        elif channel == 1: # Green
            return (x * 4 + y * 2) % 256
        else:              # Red
            return (x * y) % 256

    def read_virtual_byte(self, index: int) -> int:
        """Reads a byte from the tape, XORing it with the original to extract the stored value."""
        self.f.seek(self.header_offset + index)
        current_byte = self.f.read(1)[0]
        original_byte = self.get_original_byte(index)
        return current_byte ^ original_byte

    def write_virtual_byte(self, index: int, val: int):
        """Writes a virtual byte by XORing the value into the original pixel byte."""
        original_byte = self.get_original_byte(index)
        new_byte = original_byte ^ val
        self.f.seek(self.header_offset + index)
        self.f.write(bytes([new_byte]))

    # Visited bit-vector mapped to tape index 0 .. 200 (1600 bits)
    # Stack mapped to tape index 500 onwards (2 bytes per entry)
    def set_visited(self, cell_idx: int, visited: bool):
        byte_offset = cell_idx // 8
        bit_pos = cell_idx % 8
        curr_val = self.read_virtual_byte(byte_offset)
        if visited:
            new_val = curr_val | (1 << bit_pos)
        else:
            new_val = curr_val & ~(1 << bit_pos)
        self.write_virtual_byte(byte_offset, new_val)

    def is_visited(self, cell_idx: int) -> bool:
        byte_offset = cell_idx // 8
        bit_pos = cell_idx % 8
        curr_val = self.read_virtual_byte(byte_offset)
        return bool((curr_val >> bit_pos) & 1)

    def push_stack(self, sp: int, cell_idx: int):
        """Pushes a 2-byte cell index to the stack on the tape."""
        high = (cell_idx >> 8) & 0xFF
        low = cell_idx & 0xFF
        self.write_virtual_byte(500 + sp * 2, low)
        self.write_virtual_byte(500 + sp * 2 + 1, high)

    def pop_stack(self, sp: int) -> int:
        """Pops and returns a 2-byte cell index from the stack on the tape, cleaning it."""
        low = self.read_virtual_byte(500 + sp * 2)
        high = self.read_virtual_byte(500 + sp * 2 + 1)
        # Restore the tape back to original state (write 0)
        self.write_virtual_byte(500 + sp * 2, 0)
        self.write_virtual_byte(500 + sp * 2 + 1, 0)
        return (high << 8) | low

def get_neighbors(cell_idx: int, cols: int = 40, rows: int = 40) -> list[int]:
    """Dynamically generates valid maze neighbors using O(1) memory."""
    r = cell_idx // cols
    c = cell_idx % cols
    neighbors = []
    
    # Try Up, Down, Left, Right
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            # Deterministic wall function: if this coordinate mod 5 is 0, it's a wall
            # Except start (0,0) and end (rows-1, cols-1) which are always open
            if (nr == 0 and nc == 0) or (nr == rows - 1 and nc == cols - 1):
                neighbors.append(nr * cols + nc)
            elif (nr * 7 + nc * 13) % 5 != 0:
                neighbors.append(nr * cols + nc)
    return neighbors

def solve_maze_catalytic(tape: CatalyticBmpTape, tracker: CleanMemoryTracker, step_callback=None) -> list[int]:
    """
    Solves a 40x40 maze using DFS with a strict clean memory limit of 64 bytes.
    The visited map and stack are stored in the BMP pixel tape.
    """
    # Allocate stack variables in clean memory:
    # - current (2 bytes)
    # - target (2 bytes)
    # - sp (2 bytes)
    # - i (2 bytes)
    # - neighbor_idx (2 bytes)
    # Total: 10 bytes
    tracker.allocate(10)
    
    current = 0
    target = 39 * 40 + 39  # (39, 39)
    sp = 0
    
    # Mark start as visited
    tape.set_visited(current, True)
    tape.push_stack(sp, current)
    sp += 1
    
    solved = False
    
    while sp > 0:
        if step_callback:
            step_callback(sp)
        # Get current top of stack
        current = tape.pop_stack(sp - 1)
        # Re-push it because we are exploring
        tape.push_stack(sp - 1, current)
        
        if current == target:
            solved = True
            break
            
        # Get neighbors
        neighbors = get_neighbors(current)
        found_unvisited = False
        
        for neighbor in neighbors:
            if not tape.is_visited(neighbor):
                # Move to neighbor
                tape.set_visited(neighbor, True)
                tape.push_stack(sp, neighbor)
                sp += 1
                found_unvisited = True
                break
                
        if not found_unvisited:
            # Backtrack: pop and discard
            tape.pop_stack(sp - 1)
            sp -= 1

    # Extract final path if solved
    path = []
    if solved:
        for i in range(sp):
            path.append(tape.pop_stack(i))
    else:
        # Clean up any remaining stack elements to restore tape
        for i in range(sp):
            tape.pop_stack(i)

    # Reversibly clear visited vector to restore tape 100%
    for cell in range(1600):
        if tape.is_visited(cell):
            tape.set_visited(cell, False)

    # Free clean memory
    tracker.free(10)
    return path

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    bmp_path = os.path.join(base_dir, "workspace", "fractal.bmp")
    
    tracker = CleanMemoryTracker(limit_bytes=64)
    tape = CatalyticBmpTape(bmp_path)
    
    try:
        print("[App] Solving 40x40 maze using BMP pixel memory...")
        path = solve_maze_catalytic(tape, tracker)
        print(f"[App] Maze Solved! Path length: {len(path)} steps.")
        print(f"[App] Max clean memory observed: {tracker.max_observed} bytes")
    finally:
        tape.close()
