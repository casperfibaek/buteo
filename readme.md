# Buteo - Facilitating EO-Supported Decision Support Systems

The Buteo-Toolbox is a series of packages that ease the creation of decision support systems. It is designed to run on Linux, but should run on MacOS/Windows as well. It contains the following packages:

* Rasters
* Statistics
* Sentinel/Landsat/Viirs/SRTM
* Learn
* Orfeo
* Snap
* Visualise
* Monitor

It is capable of downloading and processing a range of openEO data sources, integrate with ESA Snap and CNES Orfeo to wrangle the data. It has a series of recipes to apply machine learning to the data and a monitoring package to automatically download, process and upload imagery.

Dependencies:
* Orfeo
* Snap
* OpenCV
* SentinelSat
* Numpy
* Cython

The system is under active development and is not ready for public release. It is being developed by NIRAS and Aalborg University.


# Ubuntu setup
  * sudo apt-get update
  * sudo apt-get install otb-bin git build-essential manpages-dev libgfortran3
  * sudo apt-get upgrade
  * sudo apt full-upgrade
  * sudo apt autoremove


# Packages
  ## Anaconda
  * Get the newest link @ https://www.anaconda.com/distribution/ 
  * cd /tmp
  * curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
  * sudo bash Anaconda3-2019.10-Linux-x86_64.sh
  * source ~/.bashrc
  * cd ~
  * sudo chown cfi -R ./*
  * sudo chown cfi .conda/environments.txt
  * conda update conda
  * conda update --all
  * conda create --name green
  * pip install --upgrade pip
  * conda install -c conda-forge statsmodels scikit-image opencv shapely gdal geojson scikit-learn sqlalchemy sqlite imbalanced-learn pandas matplotlib geopandas pyshp psutil cython tensorflow
  * pip install tensorflow
  * pip install keras
  * conda update --all
  * pip install sentinelsat

  * add: OTB_MAX_RAM_HINT=24000 and GDAL_CACHEMAX=16000 to ~./bashrc with appropriate ram limits.





  ## Snappy
  * cd ~
  * curl -O http://step.esa.int/downloads/7.0/installers/esa-snap_sentinel_unix_7_0.sh
  * bash esa-snap_sentinel_unix_7_0.sh
  * Yes to all defaults, set path of python env to /home/cfi/anaconda3/envs/yellow/bin/python
  * echo 'alias gpt=~/esa_snap/bin/gpt' >> ~/.bashrc
  * echo '-Xmx16G' >> ~/esa_snap/bin/gpt.vmoptions
  * snap --nosplash --nogui --modules --update-all
  * IF SNAP IS NOT UPDATE, THIS STEP IS NECESARRY ON FOCAL 20.04 https://forum.step.esa.int/t/error-in-applying-ers-orbit-informations/23195/13
  * Yes to all defaults, set path of python env to /home/cfi/anaconda3/envs/green/bin/python
  * echo 'alias gpt=~/snap/bin/gpt' >> ~/.bashrc
  * echo '-Xmx32G' >> ~/snap/bin/gpt.vmoptions
  * ~/snap/bin/snap --nosplash --nogui --modules --update-all


  ## Git
  * cd ~
  * git clone https://github.com/casperfibaek/yellow.git
  * git config --global user.name "johndoe"
  * git config --global user.email johndoe@example.com
  * git config --global user.password fakefakefake
  * git config --global credential.helper store
  * git config --global credential.helper wincred


  # Test
  * otbApplicationLauncherCommandLine