# Hello!

This is the Yellow EO Toolbox. The following script and setup will be created as a docker image.

# Ubuntu setup
  * sudo apt-get update
  * sudo apt-get upgrade
  * sudo apt full-upgrade
  * sudo apt autoremove

# Packages
  ## Anaconda
  * Get the newest link @ https://www.anaconda.com/distribution/ 
  * cd /tmp
  * curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
  * sudo bash Anaconda3-2019.10-Linux-x86_64.sh -u
  * source ~/.bashrc *to activate conda*
  * cd ~
  * sudo chown cfi -R ./*
  * sudo chown cfi .conda
  * sudo chown cfi .conda/environments.txt
  * conda update conda
  * conda update --all
  * conda create --name eo --clone root
  * conda activate eo
  * conda update --all *just to verify*

  ## Git
  * sudo apt-get install git
  * cd ~
  * 

  ## Orfeo-toolbox
  * otb setup
c