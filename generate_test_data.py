import os
import numpy as np

width = 1920
height = 864
num_frames = 10

def generate_frames():
    for i in range(num_frames):
        # Create a moving gradient or pattern
        t = i / num_frames
        
        # Create an array of shape (height, width, 4) for RGBA
        # We'll make a simple scrolling pattern
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xv, yv = np.meshgrid(x, y)
        
        r = (xv + t) % 1.0
        g = yv
        b = 0.5
        a = 1.0
        
        image = np.dstack((r, g, b, np.full_like(r, a))) * 255
        image = image.astype(np.uint8)
        
        filename = f"color_input_0_{i:04d}.raw"
        with open(filename, 'wb') as f:
            f.write(image.tobytes())
        
        print(f"Generated {filename}")

if __name__ == "__main__":
    generate_frames()
