#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="its_data",
    version="0.1.0",
    description="Easily get data from WLO data dumps",
    packages=find_packages(),
    install_requires=[
        d for d in open("requirements.txt").readlines() if not d.startswith("--")
    ],
    entry_points={
        "console_scripts": [
            "download-data = its_data.fetch_cli:main",
            "publish-data = its_data.default_pipelines.cli:main",
            "find-test-data = its_data.identify_test_data.cli:main",
        ]
    },
)
