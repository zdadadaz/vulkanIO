import os
from PIL import Image
import numpy as np

def generate_panning_sequence(input_path, output_dir, num_frames=30, width=1920, height=864):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and resize original image
    img = Image.open(input_path).convert('RGBA')
    img_resized = img.resize((width + num_frames, height), Image.Resampling.LANCZOS)
    
    img_data = np.array(img_resized)
    
    # Depth gradient from 0.0 to 1.0 (will be packed into 24-bit)
    y = np.linspace(0.001, 1.0, height)
    x = np.linspace(0.0, 1.0, width + num_frames)
    xv, yv = np.meshgrid(x, y)
    depth_base = yv.astype(np.float32)
    
    for i in range(num_frames):
        # Color Frame
        frame = img_data[:, i:i+width, :]
        frame_upside_down = frame[::-1, :, :]
        
        color_output_path = os.path.join(output_dir, f"color_input_0_{i:04d}.raw")
        with open(color_output_path, 'wb') as f:
            f.write(frame_upside_down.tobytes())
            
        # Depth Frame - Pack 24-bit into RGBA8888
        d_frame = depth_base[:, i:i+width]
        d_frame_upside_down = d_frame[::-1, :]
        
        # Packing: val = z * (2^24 - 1)
        z_int = (d_frame_upside_down * 16777215.0).astype(np.uint32)
        r = (z_int & 0xFF).astype(np.uint8)
        g = ((z_int >> 8) & 0xFF).astype(np.uint8)
        b = ((z_int >> 16) & 0xFF).astype(np.uint8)
        a = np.full_like(r, 255)
        
        # Stack to RGBA
        packed_depth = np.stack([r, g, b, a], axis=-1)
        
        depth_output_path = os.path.join(output_dir, f"depth_input_0_{i:04d}.raw")
        with open(depth_output_path, 'wb') as f:
            f.write(packed_depth.tobytes())
            
        if i % 10 == 0:
            print(f"Generated frames for index {i}")

if __name__ == "__main__":
    generate_panning_sequence("Lenna.jpg", "img")
