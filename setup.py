""" Build script for pip and conda package. """
from setuptools import setup, find_packages

VERSION = "0.9.36"

def readme():
    """ Generate readme file. """
    try:
        with open("./readme.md", encoding="utf8") as file:
            return file.read()
    except IOError:
        return ""


setup(
    name="Buteo",
    version=VERSION,
    author="Casper Fibaek",
    author_email="casperfibaek@gmail.com",
    description="Geospatial Analysis Meets AI",
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
    install_requires=[],
    include_package_data=True,
)
