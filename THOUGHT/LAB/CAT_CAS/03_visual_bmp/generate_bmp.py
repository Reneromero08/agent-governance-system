import os
import struct

def generate_gradient_bmp(file_path: str, width: int = 512, height: int = 512):
    """Generates a beautiful 24-bit color gradient BMP image from scratch."""
    pixel_data_size = width * height * 3
    file_size = 54 + pixel_data_size
    
    # 1. BMP File Header (14 bytes)
    # Signature: 'BM'
    # File Size: file_size
    # Reserved: 0, 0
    # Offset to Pixel Data: 54
    file_header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
    
    # 2. DIB Header / BITMAPINFOHEADER (40 bytes)
    # Header Size: 40
    # Width: width
    # Height: height
    # Planes: 1
    # Bits per Pixel: 24 (BGR)
    # Compression: 0 (None)
    # Image Size: pixel_data_size
    # X pixels per meter: 2835
    # Y pixels per meter: 2835
    # Colors: 0
    # Important Colors: 0
    dib_header = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, pixel_data_size, 2835, 2835, 0, 0)
    
    # 3. Generate Pixel Data (BGR format, bottom-to-top, left-to-right)
    pixels = bytearray(pixel_data_size)
    idx = 0
    for y in range(height):
        for x in range(width):
            # Beautiful colorful mathematical gradient
            b = (x + y * 3) % 256
            g = (x * 4 + y * 2) % 256
            r = (x * y) % 256
            
            pixels[idx] = b
            pixels[idx + 1] = g
            pixels[idx + 2] = r
            idx += 3
            
    # Write to file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_header)
        f.write(dib_header)
        f.write(pixels)

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    target = os.path.join(base_dir, "workspace", "fractal.bmp")
    print(f"Generating BMP image at {target}...")
    generate_gradient_bmp(target)
    print("BMP image generated successfully.")
