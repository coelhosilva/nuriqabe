import setuptools
import os
import io

NAME = "nuriqabe"
VERSION = "0.0.1a"
DESCRIPTION = (
    "nuriQabe is a Python package for solving Nurikabe puzzles with Q-Learning."
)
EMAIL = "coelho@ita.br"
AUTHOR = "Lucas Coelho e Silva"
REQUIRES_PYTHON = ">=3.10.4"
LICENSE = "MIT"

package_root = os.path.abspath(os.path.dirname(__file__))
readme_filename = os.path.join(package_root, "README.md")

# long description
with open(readme_filename, "r") as fh:
    long_description = fh.read()

# requirements
try:
    with io.open(os.path.join(package_root, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().splitlines()
except FileNotFoundError:
    REQUIRED = []

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=LICENSE,
    packages=setuptools.find_packages(),
    install_requires=REQUIRED,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=REQUIRES_PYTHON,
    include_package_data=True,
)
