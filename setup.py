"""
setup.py
========
Minimal setup file for editable / pip install.

Install (development):
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="caqc",
    author="Anand Kumar"
    author_email="anandambastha72@gmail.com"
    version="1.0.0",
    description="Climate-Aware Quantum Channel model for satellite QKD",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "outputs*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "xarray>=2023.1",
        "netCDF4>=1.6",
        "cdsapi>=0.6.1",
        "skyfield>=1.46",
        "matplotlib>=3.7",
        "streamlit>=1.33",
    ],
    entry_points={
        "console_scripts": [
            "caqc=cli.simulate:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
