"""
Build the documentation using pdoc (not pdoc3)
"""
import os
import shutil


# Clean documentation folder
shutil.rmtree('./docs/', ignore_errors=True)

os.mkdir("./docs/")
os.mkdir("./docs/buteo/")

# Build documentation
# os.system("pdoc ./buteo -o ./docs --docformat google --logo ./logo.png --favicon ./favicon.ico -t ./docs_assets/")
os.system("pdoc ./buteo -o ./docs --docformat google --logo https://casperfibaek.github.io/buteo/logo.png --favicon https://casperfibaek.github.io/buteo/favicon.ico -t ./docs_assets/")

shutil.copy2("./docs_assets/logo.png", "./docs/logo.png")
shutil.copy2("./docs_assets/favicon.ico", "./docs/favicon.ico")
