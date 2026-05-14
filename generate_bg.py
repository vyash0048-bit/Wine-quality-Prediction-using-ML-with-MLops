import matplotlib.pyplot as plt
import numpy as np
import os

def create_background():
    # Set dimensions
    width, height = 1920, 1080
    
    # Create a meshgrid
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Generate an abstract wine-themed gradient
    # Deep wine red to dark purple/black
    Z = (1 - X) * (1 - Y) # Simple gradient
    
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Use a custom colormap: Deep Wine to Dark
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0.1, 0, 0.05), (0.5, 0, 0.1), (0.8, 0.1, 0.2)] # Black-ish, Deep Wine, Lighter Wine
    cm = LinearSegmentedColormap.from_list('wine', colors, N=256)
    
    # Add some noise/texture
    noise = np.random.normal(0, 0.05, (height, width))
    Z_noise = Z + noise
    
    ax.imshow(Z, cmap=cm, aspect='auto', interpolation='bicubic')
    ax.axis('off')
    
    # Ensure directory exists
    save_path = 'static/assets/img/generated_bg.jpg'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Image created at {save_path}")

if __name__ == "__main__":
    create_background()
