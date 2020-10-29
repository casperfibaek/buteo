## install python client
pip install google-api-python-client

#check client
 python -c "from oauth2client import crypt"

#if fails run
pip install pyCrypto

---OBS--- above command did not work for me, instead installed with conda ---OBS---

# install pyCrypto
conda install -c anaconda pycrypto

#

Setup
Download Python
Download pip
Run the below command from a command-line to download/install the Python API client
 pip install google-api-python-client
Run the below command from a command-line to ensure you have the proper crypto libraries installed
 python -c "from oauth2client import crypt"
If running this command results in an error message, you will need to download and install the proper crypto libraries. This can be accomplished by running the below command.

 pip install pyCrypto
Run the below command from a command-line to download/install the Earth Engine Python library
 pip install earthengine-api
Run the below command from a command-line to initialize the API and verify your account
 python -c "import ee; ee.Initialize()"
This will result in an error message due to the fact that Google still needs to verify your account with Earth Engine and it currently does not have the proper credentials. Therefore, run:

 earthengine authenticate
This will open your default web-browser (ensure that you’re currently logged into your Google account) and provide you with a unique key to verify your account. Copy and paste the key into the terminal when prompted for the key.

Run python so that you’re utilizing the Python Command Line Interface (CLI) and run the following commands to ensure that the Earth Engine Python API is properly installed
 python
 >>> import ee
 >>> ee.Initialize()
 >>> image = ee.Image('srtm90_v4')
 >>> print(image.getInfo())
If you see metadata printed to the terminal and there are no errors then the Python API for Earth Engine is properly installed and you are ready to use it. If you were stuck or ran into errors not outline in this tutorial, a more in-depth tutorial can be found here


ADDITIONAL MODULES

--- ipygee ---
for visualisation of Map

###
import ipygee as ui

Map = ui.Map()
Map.show()

###