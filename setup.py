
from setuptools import find_packages, setup

with open("README.md") as file:
    long_description = file.read()

setup(
    name="tlc",
    version="0.1",
    description="Trap limited conversion efficiency analysis for photovoltaic materials",
    url="https://github.com/WMD-group/TrapLimitedConversion",
    author="TLC Developers",
    author_email="a.walsh@imperial.ac.uk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
    ],
    keywords="solar cells, photovoltaics",
    test_suite="nose.collector", 
    packages=find_packages(),
    # Specify any non-python files to be distributed with the package
    package_data={'': ['*']},  # include all files
    include_package_data=True,
    install_requires=[
        "scipy",
        "numpy",
        "matplotlib",
        "pymatgen"
    ],
    data_files=["LICENSE"],
)

