import os
import sys; sys.path.append("../")
import buteo as beo
import numpy as np

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/spatial_label_smoothing/data/"
FOLDER_DST = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/spatial_label_smoothing/visualisations/"

RADIUS = 2

data = os.path.join(FOLDER, "naestved_s2.tif")
labels = os.path.join(FOLDER, "naestved_label_lc.tif")

arr_label = beo.raster_to_array(labels, filled=True, fill_value=0, cast=np.uint8)
arr_data = beo.raster_to_array(data, filled=True, fill_value=0.0, cast=np.float32)

classes = [10, 30, 40, 50, 60, 80, 90]
classes_shape = np.array(classes).reshape(1, 1, -1)
classes_arr = (arr_label == classes_shape).astype(np.uint8)

# beo.array_to_raster(classes_arr, reference=labels, out_path=os.path.join(FOLDER_DST, "naestved_label_lc_hard.tif"))

# smoothing = 0.1
# global_smooth = ((1 - smoothing) * classes_arr) + (smoothing / len(classes))
# beo.array_to_raster(global_smooth, reference=labels, out_path=os.path.join(FOLDER_DST, "naestved_label_lc_smooth_global.tif"))

# smooth_half = beo.spatial_label_smoothing(arr_label, radius=RADIUS, method="half")
# beo.array_to_raster(smooth_half, reference=labels, out_path=os.path.join(FOLDER_DST, "naestved_label_lc_smooth_half.tif"))

# smooth_half = beo.spatial_label_smoothing(arr_label, radius=RADIUS, method="kernel")
# beo.array_to_raster(smooth_half, reference=labels, out_path=os.path.join(FOLDER_DST, "naestved_label_lc_smooth_kernel.tif"))

smooth_max = beo.spatial_label_smoothing(arr_label, radius=RADIUS, method="max")
beo.array_to_raster(smooth_max, reference=labels, out_path=os.path.join(FOLDER_DST, "naestved_label_lc_smooth_max.tif"))

# smooth_max = beo.spatial_label_smoothing(arr_label, radius=RADIUS, method=None)
# beo.array_to_raster(smooth_max, reference=labels, out_path=os.path.join(FOLDER_DST, "naestved_label_lc_smooth_none.tif"))

# SOBEL_POWER = 1.0
# sobel = beo.filter_edge_detection(arr_data, radius=RADIUS, scale=2)
# sobel = np.mean(sobel, axis=2, keepdims=True)
# sobel = np.power((sobel - np.min(sobel)) / ((np.max(sobel) - np.min(sobel) + 1e-7)), SOBEL_POWER)
# sobel = sobel.max() - sobel
# beo.array_to_raster(sobel, reference=labels, out_path=os.path.join(FOLDER_DST, "naestved_label_lc_sobel.tif"))

# smooth_half_sobel = beo.spatial_label_smoothing(arr_label, radius=RADIUS, method="half", variance=sobel)
# beo.array_to_raster(smooth_half_sobel, reference=labels, out_path=os.path.join(FOLDER_DST, "naestved_label_lc_smooth_half_sobel2.tif"))

# smooth_max_sobel = beo.spatial_label_smoothing(arr_label, radius=RADIUS, method="max", variance=sobel)
# beo.array_to_raster(smooth_max_sobel, reference=labels, out_path=os.path.join(FOLDER_DST, "naestved_label_lc_smooth_max_sobel.tif"))

# smooth_max_sobel = beo.spatial_label_smoothing(arr_label, radius=RADIUS, method=None, variance=sobel)
# beo.array_to_raster(smooth_max_sobel, reference=labels, out_path=os.path.join(FOLDER_DST, "naestved_label_lc_smooth_none_sobel.tif"))
