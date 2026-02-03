import os
import numpy as np

width = 1920
height = 864
num_frames = 150

def generate_frames():
    output_dir = "nvt_2026_01_23_11_43_31_45"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_frames):
        t = i / num_frames
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xv, yv = np.meshgrid(x, y)
        
        # Color
        r = (xv + t) % 1.0
        g = yv
        b = np.full_like(xv, 0.5)
        a = np.full_like(xv, 1.0)
        color = (np.dstack((r, g, b, a)) * 255).astype(np.uint8)
        
        # Depth
        depth = (np.full((height, width, 4), 128)).astype(np.uint8)
        
        # Normal
        normal = (np.full((height, width, 4), 128)).astype(np.uint8)
        
        # Albedo
        albedo = (np.full((height, width, 4), 200)).astype(np.uint8)
        
        # Motion Vector
        mv = (np.full((height, width, 4), 0)).astype(np.uint8)
        
        files = {
            f"color_input_0_{i:04d}.raw": color,
            f"depth_input_0_{i:04d}.raw": depth,
            f"normal_input_0_{i:04d}.raw": normal,
            f"albedo_0_{i:04d}.raw": albedo,
            f"mv_input_0_{i:04d}.raw": mv
        }
        
        for filename, data in files.items():
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(data.tobytes())
            print(f"Generated {filepath}")

if __name__ == "__main__":
    generate_frames()
