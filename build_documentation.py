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


# This is an ugly work-around to ensure that the logo and favicon
# are available on every subpage of the documentation.
# shutil.copy2("./docs_assets/logo.png", "./docs/buteo/vector/logo.png")
# shutil.copy2("./docs_assets/logo.png", "./docs/buteo/utils/logo.png")
# shutil.copy2("./docs_assets/logo.png", "./docs/buteo/raster/logo.png")
# shutil.copy2("./docs_assets/logo.png", "./docs/buteo/eo/logo.png")
# shutil.copy2("./docs_assets/logo.png", "./docs/buteo/ai/logo.png")
# shutil.copy2("./docs_assets/logo.png", "./docs/buteo/logo.png")
shutil.copy2("./docs_assets/logo.png", "./docs/logo.png")

shutil.copy2("./docs_assets/favicon.ico", "./docs/buteo/vector/favicon.ico")
# shutil.copy2("./docs_assets/favicon.ico", "./docs/buteo/utils/favicon.ico")
# shutil.copy2("./docs_assets/favicon.ico", "./docs/buteo/raster/favicon.ico")
# shutil.copy2("./docs_assets/favicon.ico", "./docs/buteo/eo/favicon.ico")
# shutil.copy2("./docs_assets/favicon.ico", "./docs/buteo/ai/favicon.ico")
# shutil.copy2("./docs_assets/favicon.ico", "./docs/buteo/favicon.ico")
# shutil.copy2("./docs_assets/favicon.ico", "./docs/favicon.ico")
