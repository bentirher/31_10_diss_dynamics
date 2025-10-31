import colorsys
from matplotlib.colors import to_rgb
import numpy as np

def generate_complementary_colors(n, palette_type='default', start_color='red'):

    start_rgb = to_rgb(start_color)
    
    start_h, start_s, start_v = colorsys.rgb_to_hsv(*start_rgb)

    colors = []

    hues = np.linspace(start_h, start_h + 1, n, endpoint=False) % 1  

    if palette_type == 'pastel':
        saturation = 0.4  
        value = 0.9       
    elif palette_type == 'neon':
        saturation = 1.0  
        value = 1.0      
    else: 
        saturation = 0.8  
        value = 0.9       

    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(h, saturation, value)

        colors.append((r, g, b))
    
    return colors