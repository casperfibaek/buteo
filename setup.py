import setuptools


def readme():
    try:
        with open("README.md") as f:
            return f.read()
    except IOError:
        return ""


setuptools.setup(
    name="toolbox-creator-CFI",
    version="0.0.30",
    author="Casper Fibaek",
    author_email="casperfibaek@gmail.com",
    description="Create a toolbox for your functions",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/casperfibaek/toolbox-gui",
    project_urls={
        "Bug Tracker": "https://github.com/casperfibaek/toolbox-gui/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
    ],
    packages=["toolbox_creator"],
    install_requires=["PySimpleGUIQt==0.35.0"],
    include_package_data=True,
)
