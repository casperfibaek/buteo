from sentinel_helper import readS2
from pansharpen import super_sample_red_edge
from mask_raster import mask_raster as mask_raster_func
import os
import time


before = time.time()


# First we ready the samples that require level three processing
S2_0513 = "D:\\pythonScripts\\yellow\\raster\\l3\\S2B_MSIL2A_20180513T104019_N0207_R008_T32VNJ_20180513T114818.SAFE"
S2_0515 = "D:\\pythonScripts\\yellow\\raster\\l3\\S2A_MSIL2A_20180515T103021_N0207_R108_T32VNJ_20180515T124152.SAFE"
S2_0516 = "D:\\pythonScripts\\yellow\\raster\\l3\\S2B_MSIL2A_20180516T105029_N0207_R051_T32VNJ_20180516T125357.SAFE"

S2_0513 = readS2(S2_0513)
S2_0515 = readS2(S2_0515)
S2_0516 = readS2(S2_0516)

S2_0513_scl = S2_0513['20m']['SCL']
S2_0513_B2 = S2_0513['10m']['B02']
S2_0513_B3 = S2_0513['10m']['B03']
S2_0513_B4 = S2_0513['10m']['B04']
S2_0513_B5 = S2_0513['20m']['B05']
S2_0513_B6 = S2_0513['20m']['B06']
S2_0513_B7 = S2_0513['20m']['B07']
S2_0513_B8A = S2_0513['20m']['B8A']
S2_0513_B8 = S2_0513['10m']['B08']
S2_0513_prefix = f"{S2_0513['meta']['tile']}_{S2_0513['meta']['sensingdate']}_"

# super_sample_red_edge(
#     S2_0513_B4,
#     S2_0513_B8,
#     B5=S2_0513_B5,
#     B6=S2_0513_B6,
#     B7=S2_0513_B7,
#     B8A=S2_0513_B8A,
#     out_folder=S2_0513['folders']['10m'],
#     mask_raster=S2_0513_scl,
#     prefix=S2_0513_prefix,
#     suffix='_10m_pan'
# )

S2_0515_scl = S2_0515['20m']['SCL']
S2_0515_B2 = S2_0515['10m']['B02']
S2_0515_B3 = S2_0515['10m']['B03']
S2_0515_B4 = S2_0515['10m']['B04']
S2_0515_B5 = S2_0515['20m']['B05']
S2_0515_B6 = S2_0515['20m']['B06']
S2_0515_B7 = S2_0515['20m']['B07']
S2_0515_B8A = S2_0515['20m']['B8A']
S2_0515_B8 = S2_0515['10m']['B08']
S2_0515_prefix = f"{S2_0515['meta']['tile']}_{S2_0515['meta']['sensingdate']}_"

# super_sample_red_edge(
#     S2_0515_B4,
#     S2_0515_B8,
#     B5=S2_0515_B5,
#     B6=S2_0515_B6,
#     B7=S2_0515_B7,
#     B8A=S2_0515_B8A,
#     out_folder=S2_0515['folders']['10m'],
#     mask_raster=S2_0515_scl,
#     prefix=S2_0515_prefix,
#     suffix='_10m_pan'
# )

S2_0516_scl = S2_0516['20m']['SCL']
S2_0516_B2 = S2_0516['10m']['B02']
S2_0516_B3 = S2_0516['10m']['B03']
S2_0516_B4 = S2_0516['10m']['B04']
S2_0516_B5 = S2_0516['20m']['B05']
S2_0516_B6 = S2_0516['20m']['B06']
S2_0516_B7 = S2_0516['20m']['B07']
S2_0516_B8A = S2_0516['20m']['B8A']
S2_0516_B8 = S2_0516['10m']['B08']
S2_0516_prefix = f"{S2_0516['meta']['tile']}_{S2_0516['meta']['sensingdate']}_"

# super_sample_red_edge(
#     S2_0516_B4,
#     S2_0516_B8,
#     B5=S2_0516_B5,
#     B6=S2_0516_B6,
#     B7=S2_0516_B7,
#     B8A=S2_0516_B8A,
#     out_folder=S2_0516['folders']['10m'],
#     mask_raster=S2_0516_scl,
#     prefix=S2_0516_prefix,
#     suffix='_10m_pan'
# )

mask_raster_func(
    S2_0513_B2,
    mask_raster=S2_0513_scl,
    out_raster=os.path.join(S2_0513['folders']['10m'], f'{S2_0513_prefix}B02_10m_mask.tif')
)

mask_raster_func(
    S2_0513_B3,
    mask_raster=S2_0513_scl,
    out_raster=os.path.join(S2_0513['folders']['10m'], f'{S2_0513_prefix}B03_10m_mask.tif')
)

mask_raster_func(
    S2_0513_B4,
    mask_raster=S2_0513_scl,
    out_raster=os.path.join(S2_0513['folders']['10m'], f'{S2_0513_prefix}B04_10m_mask.tif')
)

mask_raster_func(
    S2_0513_B8,
    mask_raster=S2_0513_scl,
    out_raster=os.path.join(S2_0513['folders']['10m'], f'{S2_0513_prefix}B08_10m_mask.tif')
)

mask_raster_func(
    S2_0515_B2,
    mask_raster=S2_0515_scl,
    out_raster=os.path.join(S2_0515['folders']['10m'], f'{S2_0515_prefix}B02_10m_mask.tif')
)

mask_raster_func(
    S2_0515_B3,
    mask_raster=S2_0515_scl,
    out_raster=os.path.join(S2_0515['folders']['10m'], f'{S2_0515_prefix}B03_10m_mask.tif')
)

mask_raster_func(
    S2_0515_B4,
    mask_raster=S2_0515_scl,
    out_raster=os.path.join(S2_0515['folders']['10m'], f'{S2_0515_prefix}B04_10m_mask.tif')
)

mask_raster_func(
    S2_0515_B8,
    mask_raster=S2_0515_scl,
    out_raster=os.path.join(S2_0515['folders']['10m'], f'{S2_0515_prefix}B08_10m_mask.tif')
)

mask_raster_func(
    S2_0516_B2,
    mask_raster=S2_0516_scl,
    out_raster=os.path.join(S2_0516['folders']['10m'], f'{S2_0516_prefix}B02_10m_mask.tif')
)

mask_raster_func(
    S2_0516_B3,
    mask_raster=S2_0516_scl,
    out_raster=os.path.join(S2_0516['folders']['10m'], f'{S2_0516_prefix}B03_10m_mask.tif')
)

mask_raster_func(
    S2_0516_B4,
    mask_raster=S2_0516_scl,
    out_raster=os.path.join(S2_0516['folders']['10m'], f'{S2_0516_prefix}B04_10m_mask.tif')
)

mask_raster_func(
    S2_0516_B8,
    mask_raster=S2_0516_scl,
    out_raster=os.path.join(S2_0516['folders']['10m'], f'{S2_0516_prefix}B08_10m_mask.tif')
)

after = time.time()

print(f'Execution took: {round(after - before, 2)}s')
