# flake8: noqa
from setuptools import find_packages, setup
from typing import List
import os
import re


# inspired by https://github.com/kubeflow/pipelines/blob/master/sdk/python/setup.py
def find_version(*file_path_parts: str) -> str:
    """Get version from kfp.__init__.__version__."""

    file_path = os.path.join(os.path.dirname(__file__), *file_path_parts)

    with open(file_path, "r") as f:
        version_file_text = f.read()
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        version_file_text,
        re.M,
    )
    if version_match:
        return version_match.group(1)

    raise RuntimeError(f"Unable to find version string in file: {file_path}.")


setup(
    name="ml-pilot",
    version=find_version("ml_pilot", "__init__.py"),
    description="",
    long_description="",
    url="https://github.com/mmazurekgda/ml-pilot",
    author="Michał Mazurek",
    license="GPLv3+",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11.5",
    packages=find_packages(),
    install_requires=[
        "click",
        "pyfiglet",
        "pydantic",
        "pydantic-settings",
        "pydanclick",
    ]
)