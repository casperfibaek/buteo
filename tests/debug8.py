import sys; sys.path.append("../")
import os
import buteo as beo


FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/glimmer_test/"

slave_img = os.path.join(FOLDER, "KX10_GIU_20220210_E23.16_N38.08_202200107446_L4A_B_LH_clipped_aligned.tif")
master_img = os.path.join(FOLDER, "KX10_GIU_20220405_E24.89_N38.02_202200092717_L4A_A_LH_clipped.tif")

beo.coregister_images_efolki(
    master_img,
    slave_img,
    out_path=os.path.join(FOLDER, "coregistered_efolki_fancy.tif"),
    iteration=5,
    radius=[32, 16],
    rank=5,
    levels=6,
    band_to_base_master=1,
    band_to_base_slave=1,
    mask=None,
    fill_value=0,
)

# beo.coregister_images_efolki(
#     master_img,
#     slave_img,
#     out_path=os.path.join(FOLDER, "coregistered_efolki.tif"),
#     iteration=4,
#     radius=[16, 8],
#     rank=4,
#     levels=5,
#     band_to_base_master=1,
#     band_to_base_slave=1,
#     mask=None,
#     fill_value=0,
# )
