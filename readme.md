# Hello!

This is the Yellow EO Toolbox. The following script and setup will be created as a docker image.

# Ubuntu setup
  * sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
  * sudo apt-get update
  * sudo apt-get install otb-bin git build-essential manpages-dev
  * sudo apt-get upgrade
  * sudo apt full-upgrade
  * sudo apt autoremove

# Packages
  ## Anaconda
  * Get the newest link @ https://www.anaconda.com/distribution/ 
  * cd /tmp
  * curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
  * sudo bash Anaconda3-2019.10-Linux-x86_64.sh
  * source ~/.bashrc
  * cd ~
  * sudo chown cfi -R ./*
  * sudo chown cfi .conda/environments.txt
  * conda update conda
  * conda update --all
  * conda create --name yellow python=3.8
  * conda activate yellow
  * conda config --env --add channels conda-forge
  * conda config --env --set channel_priority strict
  * conda install -c anaconda mkl numpy scipy scikit-learn cython numexpr
  * conda install -c conda-forge gdal matplotlib shapely opencv
  * conda update --all

  ## Git
  * cd ~
  * git clone https://github.com/casperfibaek/yellow.git
  * git config --global user.name "username"
  * git config --global user.email johndoe@example.com
  * git config --global user.password fakefakefake

  # Test
  * otbApplicationLauncherCommandLine