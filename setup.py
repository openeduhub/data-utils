#!/usr/bin/env python3
from setuptools import setup

setup(
    name="data_utils",
    version="0.1.0",
    description="Easily get data from WLO data dumps",
    author="",
    author_email="",
    packages=["data_utils"],
    install_requires=[
        d for d in open("requirements.txt").readlines() if not d.startswith("--")
    ],
    package_dir={"": "."},
)
