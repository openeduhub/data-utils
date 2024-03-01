#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="data_utils",
    version="0.1.0",
    description="Easily get data from WLO data dumps",
    packages=find_packages(),
    install_requires=[
        d for d in open("requirements.txt").readlines() if not d.startswith("--")
    ],
    entry_points={
        "console_scripts": [
            "download-data = data_utils.cli:main",
            "find-test-data = data_utils.identify_test_data.cli:main",
        ]
    },
)
