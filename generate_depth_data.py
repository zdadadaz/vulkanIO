import os
import numpy as np

width = 1920
height = 864
num_frames = 30

def generate_depth_frames():
    os.makedirs("img", exist_ok=True)
    for i in range(num_frames):
        # Create a simple scene: a sphere moving in depth
        t = i / num_frames
        
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1 * height / width, 1 * height / width, height)
        xv, yv = np.meshgrid(x, y)
        
        # Center of sphere moves
        cx = 0.5 * np.cos(2 * np.pi * t)
        cy = 0.3 * np.sin(2 * np.pi * t)
        cz = 2.0 # Fixed depth base
        
        # Distance to sphere center
        dist = np.sqrt((xv - cx)**2 + (yv - cy)**2)
        
        # Sphere radius
        radius = 0.4
        
        # Depth calculation
        # If inside sphere, depth is front surface
        # If outside, depth is background (e.g. 5.0)
        mask = dist < radius
        depth = np.full((height, width), 5.0, dtype=np.float32)
        
        # Front surface of sphere (semi-sphere)
        sphere_depth = cz - np.sqrt(np.maximum(0, radius**2 - dist**2))
        depth[mask] = sphere_depth[mask]
        
        filename = f"img/depth_input_0_{i:04d}.raw"
        with open(filename, 'wb') as f:
            f.write(depth.tobytes())
        
        print(f"Generated {filename}")

if __name__ == "__main__":
    generate_depth_frames()
