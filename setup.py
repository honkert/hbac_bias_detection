from setuptools import setup, find_packages
from distutils.core import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name="hbac_bias_detection",
    version="0.1.9",
    packages=find_packages(),
    # package_dir={'hbac_bias_detection': 'src'},
    #package_dir={'': 'hbac_bias_detection'},
    # other arguments omitted
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires = REQUIREMENTS
)
