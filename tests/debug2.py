# Mikelsons, Karlis; Wang, Menghua; Jiang, Lide; Wang, Xiao-Long (2021), “Global land mask for satellite ocean color remote sensing”, Mendeley Data, V1, doi: 10.17632/9r93m9s7cw.1
import os
import sys; sys.path.append("../")
import buteo as beo
import numpy as np
import matplotlib.pyplot as plt


# def bresenham_line(x0, y0, x1, y1):
#     """Bresenham's Line Algorithm to generate points between start and end."""
#     points = []
#     dx = abs(x1 - x0)
#     dy = abs(y1 - y0)
#     sx = 1 if x0 < x1 else -1
#     sy = 1 if y0 < y1 else -1

#     if dx > dy:
#         err = dx / 2.0
#         while x0 != x1:
#             points.append((x0, y0))
#             err -= dy
#             if err < 0:
#                 y0 += sy
#                 err += dx
#             x0 += sx
#     else:
#         err = dy / 2.0
#         while y0 != y1:
#             points.append((x0, y0))
#             err -= dx
#             if err < 0:
#                 x0 += sx
#                 err += dy
#             y0 += sy
#     points.append((x0, y0))
#     return points

# def draw_curve(height, width):
#     """Draw a curve on an image of size (height, width) and return a binary mask."""
#     mask = np.ones((height, width), dtype=np.uint8)
    
#     # Random start and end points
#     y0, x0 = np.random.randint(0, height), np.random.randint(0, width)
#     y1, x1 = np.random.randint(0, height), np.random.randint(0, width)
    
#     # For simplicity, we'll use Bresenham's line algorithm to plot the curve
#     # This gives a straight line. To create an actual curve, you'd need to add more control points
#     # and interpolate between them.
#     points = bresenham_line(x0, y0, x1, y1)
#     for (x, y) in points:
#         mask[y, x] = 0
        
#     return mask[..., np.newaxis]


def quadratic_bezier(p0, p1, p2, t):
    """Calculate point on a quadratic Bezier curve defined by p0, p1, and p2 at t."""
    pt = (1-t)*((1-t)*p0 + t*p1) + t*((1-t)*p1 + t*p2)
    return tuple(map(int, pt))

def apply_thickness(mask, y, x, thickness):
    """Fill in pixels around the given point (y,x) in the mask based on thickness."""
    half_thickness = thickness // 2

    y_start = max(y - half_thickness, 0)
    y_end = min(y + half_thickness, mask.shape[0])

    x_start = max(x - half_thickness, 0)
    x_end = min(x + half_thickness, mask.shape[1])

    mask[y_start:y_end, x_start:x_end] = 0
    return mask

def draw_curve(height, width):
    """Draw a curve with random thickness on an image of size (height, width) and return a binary mask."""
    mask = np.ones((height, width), dtype=np.uint8)
    
    # Random start, control, end points, and thickness
    y0, x0 = np.random.randint(0, height), np.random.randint(0, width)
    yc, xc = np.random.randint(0, height), np.random.randint(0, width)
    y1, x1 = np.random.randint(0, height), np.random.randint(0, width)
    thickness = np.random.randint(1, 5)  # Change the range as desired
    
    p0 = np.array([y0, x0])
    p1 = np.array([yc, xc])
    p2 = np.array([y1, x1])
    
    for t in np.linspace(0, 1, max(height, width)):
        y, x = quadratic_bezier(p0, p1, p2, t)
        
        if 0 <= y < height and 0 <= x < width:
            mask = apply_thickness(mask, y, x, thickness)
            
    return mask[..., np.newaxis]


image_path = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/buteo/tests/features/test_image_rgb_8bit.tif"
arr = beo.raster_to_array(image_path, pixel_offsets=[900, 500, 100, 100], cast=np.float32)

for _ in range(10):
    arr2 = arr.copy()
    arr2[:, :, 0] = arr2[:, :, 0] * draw_curve(100, 100)[:, :, 0]
    arr2[:, :, 1] = arr2[:, :, 1] * draw_curve(100, 100)[:, :, 0]
    arr2[:, :, 2] = arr2[:, :, 2] * draw_curve(100, 100)[:, :, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(np.rint(arr).astype(np.uint8), vmin=0, vmax=255)
    ax1.set_title('Image 1')
    ax1.axis('off')

    ax2.imshow(np.rint(arr2).astype(np.uint8), vmin=0, vmax=255, cmap='gray')
    ax2.set_title('Image 2')
    ax2.axis('off')

    plt.show()

import pdb; pdb.set_trace()
