from setuptools import setup, find_packages


def readme():
    try:
        with open("README.md") as f:
            return f.read()
    except IOError:
        return ""


setup(
    name="Buteo",
    version="0.5.3",
    author="Casper Fibaek",
    author_email="casperfibaek@gmail.com",
    description="A pythonic way of working with raster data",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/casperfibaek/buteo",
    project_urls={
        "Bug Tracker": "https://github.com/casperfibaek/buteo/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Alpha",
    ],
    packages=find_packages(),
    zip_safe=True,
    install_requires=[
        # "gdal",
        # "numba",
    ],
    include_package_data=True,
)

# python -m build
# python -m twine upload --repository testpypi dist/*
# pdoc3 --html --output-dir docs --config show_source_code=False buteo