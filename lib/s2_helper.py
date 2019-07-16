import os
import uuid
import time
import json
import shutil
import zipfile
import numpy as np
import numpy.ma as ma
from pprint import pprint
from glob import glob
from lib.orfeo_toolbox import pansharpen, concatenate_images, dimension_reduction, rescale, local_stats, split_images
from lib.raster_io import raster_to_array, array_to_raster
from lib.resample import resample


class Sentinel:
    _temp_folder_dir_path = os.path.abspath('./temp/')

    s2_spectral_profile = {
        'B04': {
            'mean': 664.75,      # Central wavelength
            'width': 31.0,       # Width of the spectral band
            'edge_top': 680.25,  # Highest edge on the spectrum
            'edge_bot': 633.75,  # Lowest edge on the spectrum
        },
        'B05': {
            'mean': 703.95,
            'width': 15.5,
            'edge_top': 711.7,
            'edge_bot': 696.2,
        },
        'B06': {
            'mean': 739.8,
            'width': 15.5,
            'edge_top': 747.3,
            'edge_bot': 732.3,
        },
        'B07': {
            'mean': 781.25,
            'width': 20.0,
            'edge_top': 791.25,
            'edge_bot': 771.25,
        },
        'B08': {
            'mean': 832.85,
            'width': 106.0,
            'edge_top': 885.85,
            'edge_bot': 779.85,
        },
    }

    metadata = {}
    data = {
        '10m': {},
        '20m': {},
        '60m': {},
        'custom': {},
    }
    folders = {
        'images': '',
        'qi': '',
        '10m': {},
        '20m': {},
        '60m': {},
        'custom': {},
    }

    def __str__(self):
        return str(pprint(self.metadata)) + str(pprint(self.folders))

    def __init__(self, url):
        self.path = os.path.abspath(url)
        self.exists = os.path.exists(self.path)

        if self.exists is not True:
            raise AttributeError('The input file does not exist')

        self.dir = os.path.dirname(self.path)
        self.base = os.path.basename(url).rsplit('.', 1)[0]
        self.filetype = os.path.basename(url).rsplit('.', 1)[1]
        self.zipped = True if self.filetype == 'zip' else False

        if self.zipped is True:
            zipped = zipfile.ZipFile(self.path)
            zippedFilename = zipped.namelist()[0]
            zipped.extractall(self.dir)
            zipped.close()
            self.path = os.path.join(self.dir, zippedFilename)
            self.filetype = os.path.basename(self.path).rsplit('.', 1)[1]

        # The arguments in the Sentinel string
        arguments = self.base.split('_')
        try:
            if 'S1' in arguments[0]:
                self.metadata['satellite'] = arguments[0]
                self.metadata['mode'] = arguments[1]
                self.metadata['producttype'] = arguments[2]
                self.metadata['processinglevel'] = arguments[3]
                self.metadata['sensingstart'] = arguments[4]
                self.metadata['sensingend'] = arguments[5]
                self.metadata['orbit'] = arguments[6]
                self.metadata['missionid'] = arguments[7]
                self.metadata['missionid'] = arguments[7]
            else:
                self.metadata['satellite'] = arguments[0]
                self.metadata['processinglevel'] = arguments[1]
                self.metadata['sensingdate'] = arguments[2]
                self.metadata['baseline'] = arguments[3]
                self.metadata['orbit'] = arguments[4]
                self.metadata['tile'] = arguments[5]
                self.metadata['basename'] = f"{arguments[5]}_{arguments[2]}"
        except:
            raise ValueError('Unable to parse input name structure. Has the string been changed?')

            if self.metadata['satellite'] != 'S2A' and self.metadata['satellite'] != 'S2B':
                if self.metadata['processinglevel'] != 'MSIL2A':
                    raise ValueError('Level 1 data currently not supported.')

        # GLOB the structure
        try:
            self.folders['qi'] = glob(f'{self.path}\\GRANULE\\*\\QI_DATA')[0]
            self.data['20m']['MSK_CLDPRB_20m'] = os.path.join(self.folders['qi'], 'MSK_CLDPRB_20m.jp2')
            self.data['20m']['MSK_SNWPRB_20m'] = os.path.join(self.folders['qi'], 'MSK_SNWPRB_20m.jp2')
            self.data['60m']['MSK_CLDPRB_60m'] = os.path.join(self.folders['qi'], 'MSK_CLDPRB_60m.jp2')
            self.data['60m']['MSK_SNWPRB_60m'] = os.path.join(self.folders['qi'], 'MSK_SNWPRB_60m.jp2')

            self.folders['images'] = glob(f'{self.path}\\GRANULE\\*\\IMG_DATA')[0]
            self.folders['10m'] = os.path.join(self.folders['images'], 'R10m')
            self.folders['20m'] = os.path.join(self.folders['images'], 'R20m')
            self.folders['60m'] = os.path.join(self.folders['images'], 'R60m')
            self.folders['custom'] = os.path.join(self.folders['images'], 'custom')

            for image in glob(f'{self.folders["10m"]}\\*'):
                image_base = os.path.basename(image).rsplit('.', 1)[0]
                image_filetype = os.path.basename(image).rsplit('.', 1)[1]

                if image_filetype != 'jp2':
                    continue

                image_base_split = image_base.split('_')
                image_name = image_base_split[2]
                self.data['10m'][image_name] = image

            for image in glob(f'{self.folders["20m"]}\\*'):
                image_base = os.path.basename(image).rsplit('.', 1)[0]
                image_filetype = os.path.basename(image).rsplit('.', 1)[1]

                if image_filetype != 'jp2':
                    continue

                image_base_split = image_base.split('_')
                image_name = image_base_split[2]
                self.data['20m'][image_name] = image

            for image in glob(f'{self.folders["60m"]}\\*'):
                image_base = os.path.basename(image).rsplit('.', 1)[0]
                image_filetype = os.path.basename(image).rsplit('.', 1)[1]

                if image_filetype != 'jp2':
                    continue

                image_base_split = image_base.split('_')
                image_name = image_base_split[2]
                self.data['60m'][image_name] = image

            if os.path.exists(self.folders['custom']):
                for image in glob(f'{self.folders["custom"]}\\*'):
                    image_base = os.path.basename(image).rsplit('.', 1)[0]
                    image_filetype = os.path.basename(image).rsplit('.', 1)[1]

                    if image_filetype == 'jp2' or image_filetype == 'tif':
                        self.data['custom'][image_base] = image

        except:
            raise ValueError('Unable the parse folder structure. Has the folder structure been changed?')

    def get_raw_image(self, name, resolution=10):
        if name in self.data[f'{resolution}m']:
            return self.data[f'{resolution}m'][name]
        else:
            return False

    def get_custom_image(self, name):
        basename = os.path.basename(name).rsplit('.', 1)[0]
        if basename in self.data['custom']:
            return self.data['custom'][basename]
        else:
            return False

    def image_exists(self, name, resolution=10):
        if name in self.data[f'{resolution}m']:
            return True
        elif name in self.data['custom']:
            return True
        else:
            return False

    def set_custom_image(self, name, path_or_dataframe):
        self.data['custom'][name] = path_or_dataframe

    def update_custom(self):
        if os.path.exists(self.folders['custom']) is True:
            for image in glob(f'{self.folders["custom"]}\\*'):
                image_base = os.path.basename(image).rsplit('.', 1)[0]
                image_filetype = os.path.basename(image).rsplit('.', 1)[1]

                if image_filetype != 'jp2' and image_filetype != 'tif':
                    continue

                image_base_split = image_base.split('_')
                image_name = image_base_split[2]
                self.data['custom'][image_name] = image

    def delete_custom(self):
        if os.path.exists(self.folders['custom']) is True:
            shutil.rmtree(self.folders['custom'])

    def resample_bands(self, bands=['B05', 'B06', 'B07', 'B8A']):
        if bands == 'all':
            bands = ['MSK_SNWPRB', 'MSK_CLDPRB', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SLC']

        if os.path.exists(self.folders['custom']) is False:
            os.mkdir(self.folders['custom'])

        reference = self.get_raw_image('B08', 10)

        for band in bands:
            image_name = f"{self.metadata['basename']}_{band}_rs_10m.tif"
            image_in_path = self.get_raw_image(band, 20)
            image_out_path = os.path.join(self.folders['custom'], image_name)
            resample(
                image_in_path,
                out_raster=image_out_path,
                reference_raster=reference,
            )
        self.set_custom_image(image_name, image_out_path)

    def _make_temp_dir(self):
        temp_folder_holder = os.path.abspath(self._temp_folder_dir_path)

        if os.path.exists(temp_folder_holder) is False:
            os.mkdir(temp_folder_holder)

        new_temp_folder = os.path.join(temp_folder_holder, uuid.uuid4().hex)
        os.mkdir(new_temp_folder)

        return new_temp_folder

    def super_sample_bands(self, bands=['B05', 'B06', 'B07', 'B8A']):

        try:
            temp_dir = self._make_temp_dir()

            B4 = self.get_raw_image('B04', 10)
            B8 = self.get_raw_image('B08', 10)

            B4_arr = raster_to_array(B4)
            B8_arr = raster_to_array(B8)

            if 'B8A' in bands:
                # Test if resampled version already exists
                band_potential_path = self.get_custom_image(f'{self.metadata["basename"]}_B8A_rs_10m')
                band_exists = os.path.exists(band_potential_path)
                if band_potential_path is not False and band_exists is True:
                    resampled_name = band_potential_path
                else:
                    resampled_name = os.path.join(temp_dir, f'{self.metadata["basename"]}_B8A_rs_10m.tif')
                    resample(self.get_raw_image('B8A', 20), reference_raster=B8, out_raster=resampled_name)

                pansharpened_name = os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_B8A_ss_10m.tif')

                pansharpen(B8, resampled_name, pansharpened_name, out_datatype='uint16')

            band_arrays = {}
            for band in bands:

                # Special case, will be handed at end of function.
                if band == 'B8A':
                    continue

                band_path = self.get_raw_image(band, 20)
                band_array = raster_to_array(band_path)

                B4_distance = self.s2_spectral_profile[band]['edge_bot'] - self.s2_spectral_profile['B04']['edge_top']
                B8_distance = self.s2_spectral_profile['B08']['edge_bot'] - self.s2_spectral_profile[band]['edge_top']

                distance_sum = B4_distance + B8_distance
                B4_weight = 1 - (B4_distance / distance_sum)
                B8_weight = 1 - (B8_distance / distance_sum)

                ratio = np.add(np.multiply(B4_arr, B4_weight), np.multiply(B8_arr, B8_weight))

                # Test if resampled version already exists
                band_potential_path = self.get_custom_image(f'{self.metadata["basename"]}_{band}_rs_10m')
                band_exists = os.path.exists(band_potential_path)
                if band_potential_path is not False and band_exists is True:
                    resampled_name = band_potential_path
                else:
                    resampled_name = os.path.join(temp_dir, f'{self.metadata["basename"]}_{band}_rs_10m.tif')
                    resample(band_path, reference_raster=B8, out_raster=resampled_name)

                ratio_name = os.path.join(temp_dir, f'{self.metadata["basename"]}_{band}_ratio_10m.tif')
                array_to_raster(ratio, reference_raster=B8, out_raster=ratio_name)

                pansharpened_name = os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_{band}_ss_10m.tif')

                pansharpen(ratio_name, resampled_name, pansharpened_name, out_datatype='uint16')

            resample(self.get_raw_image('B11', 20), reference_raster=B8, out_raster=os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_B11_ss_10m.tif'))
            resample(self.get_raw_image('B12', 20), reference_raster=B8, out_raster=os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_B12_ss_10m.tif'))

            shutil.copyfile(self.get_raw_image('B02', 10), os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_B02_ss_10m.tif'))
            shutil.copyfile(self.get_raw_image('B03', 10), os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_B03_ss_10m.tif'))
            shutil.copyfile(self.get_raw_image('B04', 10), os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_B04_ss_10m.tif'))
            shutil.copyfile(self.get_raw_image('B04', 10), os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_B08_ss_10m.tif'))
        finally:
            self.update_custom()
            shutil.rmtree(temp_dir)

    def principal_components(self):
        try:
            temp_dir = self._make_temp_dir()
            vis = [
                self.get_custom_image(f'{self.metadata["basename"]}_B02_ss_10m'),
                self.get_custom_image(f'{self.metadata["basename"]}_B03_ss_10m'),
                self.get_custom_image(f'{self.metadata["basename"]}_B04_ss_10m'),
                self.get_custom_image(f'{self.metadata["basename"]}_B08_ss_10m'),
            ]

            nir = [
                self.get_custom_image(f'{self.metadata["basename"]}_B05_ss_10m'),
                self.get_custom_image(f'{self.metadata["basename"]}_B06_ss_10m'),
                self.get_custom_image(f'{self.metadata["basename"]}_B07_ss_10m'),
                self.get_custom_image(f'{self.metadata["basename"]}_B08_ss_10m'),
                self.get_custom_image(f'{self.metadata["basename"]}_B8A_ss_10m'),
            ]

            swir = [
                self.get_custom_image(f'{self.metadata["basename"]}_B11_ss_10m'),
                self.get_custom_image(f'{self.metadata["basename"]}_B12_ss_10m'),
            ]

            vis_path = concatenate_images(vis, os.path.join(temp_dir, f'{self.metadata["basename"]}_vis_ss_10m.tif'), out_datatype='uint16')
            nir_path = concatenate_images(nir, os.path.join(temp_dir, f'{self.metadata["basename"]}_nir_ss_10m.tif'), out_datatype='uint16')
            swir_path = concatenate_images(swir, os.path.join(temp_dir, f'{self.metadata["basename"]}_swir_ss_10m.tif'), out_datatype='uint16')

            vis_pca = dimension_reduction(vis_path, os.path.join(temp_dir, f'{self.metadata["basename"]}_vis_pca_us_10m.tif'))
            nir_pca = dimension_reduction(nir_path, os.path.join(temp_dir, f'{self.metadata["basename"]}_nir_pca_us_10m.tif'))
            swir_pca = dimension_reduction(swir_path, os.path.join(temp_dir, f'{self.metadata["basename"]}_swir_pca_us_10m.tif'))

            rescale(vis_pca, os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_vis_pca_10m.tif'), options={
                'outmin': 0,
                'outmax': 65535,
            }, out_datatype='uint16')

            rescale(nir_pca, os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_nir_pca_10m.tif'), options={
                'outmin': 0,
                'outmax': 65535,
            }, out_datatype='uint16')

            rescale(swir_pca, os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_swir_pca_10m.tif'), options={
                'outmin': 0,
                'outmax': 65535,
            }, out_datatype='uint16')

        finally:
            self.update_custom()
            shutil.rmtree(temp_dir)

    def texture_variance(self):
        before = time.time()
        try:
            temp_dir = self._make_temp_dir()

            vis = self.get_custom_image(f'{self.metadata["basename"]}_vis_pca_10m')
            nir = self.get_custom_image(f'{self.metadata["basename"]}_nir_pca_10m')
            swir = self.get_custom_image(f'{self.metadata["basename"]}_swir_pca_10m')

            # vis_stats_3 = local_stats(vis, os.path.join(temp_dir, f'{self.metadata["basename"]}_vis_pca_stats_rad3_10m.tif'), options={'radius': 3}, band=2)
            # vis_stats_2 = local_stats(vis, os.path.join(temp_dir, f'{self.metadata["basename"]}_vis_pca_stats_rad2_10m.tif'), options={'radius': 2}, band=2)
            # vis_stats_1 = local_stats(vis, os.path.join(temp_dir, f'{self.metadata["basename"]}_vis_pca_stats_rad1_10m.tif'), options={'radius': 1}, band=2)

            # vis3_variance_arr = raster_to_array(os.path.join(temp_dir, f'{self.metadata["basename"]}_vis_pca_stats_rad3_10m.tif'))
            # vis2_variance_arr = raster_to_array(os.path.join(temp_dir, f'{self.metadata["basename"]}_vis_pca_stats_rad2_10m.tif'))
            # vis1_variance_arr = raster_to_array(os.path.join(temp_dir, f'{self.metadata["basename"]}_vis_pca_stats_rad1_10m.tif'))

            # with np.errstate(divide='ignore', invalid='ignore'):
            #     vis_mean_variance = np.true_divide(
            #         np.add(
            #             np.sqrt(vis3_variance_arr),
            #             np.sqrt(vis2_variance_arr),
            #             np.sqrt(vis1_variance_arr),
            #         ), 3)
            # array_to_raster(vis_mean_variance, reference_raster=vis, out_raster=os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_vis_pca_var_texture_10m.tif'))

            # nir_stats_3 = local_stats(nir, os.path.join(temp_dir, f'{self.metadata["basename"]}_nir_pca_stats_rad3_10m.tif'), options={'radius': 3}, band=2)
            # nir_stats_2 = local_stats(nir, os.path.join(temp_dir, f'{self.metadata["basename"]}_nir_pca_stats_rad2_10m.tif'), options={'radius': 2}, band=2)
            # nir_stats_1 = local_stats(nir, os.path.join(temp_dir, f'{self.metadata["basename"]}_nir_pca_stats_rad1_10m.tif'), options={'radius': 1}, band=2)

            # nir3_variance_arr = raster_to_array(os.path.join(temp_dir, f'{self.metadata["basename"]}_nir_pca_stats_rad3_10m.tif'))
            # nir2_variance_arr = raster_to_array(os.path.join(temp_dir, f'{self.metadata["basename"]}_nir_pca_stats_rad2_10m.tif'))
            # nir1_variance_arr = raster_to_array(os.path.join(temp_dir, f'{self.metadata["basename"]}_nir_pca_stats_rad1_10m.tif'))

            # with np.errstate(divide='ignore', invalid='ignore'):
            #     nir_mean_variance = np.true_divide(
            #         np.add(
            #             np.sqrt(nir3_variance_arr),
            #             np.sqrt(nir2_variance_arr),
            #             np.sqrt(nir1_variance_arr),
            #         ), 3)
            # array_to_raster(nir_mean_variance, reference_raster=nir, out_raster=os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_nir_pca_var_texture_10m.tif'))

            swir_arr = raster_to_array(swir).astype('uint32')
            swir_rast = array_to_raster(swir_arr, reference_raster=swir, out_raster=os.path.join(temp_dir, f'{self.metadata["basename"]}_swir_pca_uint32_10m.tif'))
            swir_stats_3 = local_stats(swir_rast, os.path.join(temp_dir, f'{self.metadata["basename"]}_swir_pca_uint32_stats_10m.tif'), options={'radius': 3}, band=2)
            # swir_stats_2 = local_stats(swir, os.path.join(temp_dir, f'{self.metadata["basename"]}_swir_pca_stats_rad2_10m.tif'), options={'radius': 2}, band=2)
            # swir_stats_1 = local_stats(swir, os.path.join(temp_dir, f'{self.metadata["basename"]}_swir_pca_stats_rad1_10m.tif'), options={'radius': 1}, band=2)

            swir3_variance_arr = raster_to_array(os.path.join(temp_dir, swir_stats_3))
            # swir2_variance_arr = raster_to_array(os.path.join(temp_dir, swir_stats_2))
            # swir1_variance_arr = raster_to_array(os.path.join(temp_dir, swir_stats_1))
            array_to_raster(swir3_variance_arr, reference_raster=swir, out_raster=os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_swir_pca_var_texture_10m_rad3.tif'))

            # with np.errstate(divide='ignore', invalid='ignore'):
            #     swir_mean_sqrt3 = np.sqrt(swir3_variance_arr)
            #     swir_mean_sqrt2 = np.sqrt(swir2_variance_arr)
            #     swir_mean_sqrt1 = np.sqrt(swir1_variance_arr)
            #     swir_mean_sqrt_sum = np.add(swir_mean_sqrt3, swir_mean_sqrt2)
            #     swir_mean_sqrt_sum = np.add(swir_mean_sqrt_sum, swir_mean_sqrt1)

            # swir_mean_variance = np.true_divide(swir_mean_sqrt_sum, 3).astype('float32')

            # swir_mean_variance = np.true_divide(
            #     np.add(
            #         np.sqrt(swir3_variance_arr),
            #         np.sqrt(swir2_variance_arr),
            #         np.sqrt(swir1_variance_arr),
            #     ), 3).astype('float32')
            # array_to_raster(swir_mean_variance, reference_raster=swir, out_raster=os.path.join(self.folders['custom'], f'{self.metadata["basename"]}_swir_pca_var_texture_10m.tif'))

        finally:
            self.update_custom()
            shutil.rmtree(temp_dir)
            print(f'execution took: {round(time.time() - before, 2)}s')

