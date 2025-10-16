from setuptools import setup, find_packages

setup(
    name="Exploratory_Data_Analysis_Suite",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "dependency-injector>=4.41.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "numpy>=1.20.0",
    ],
)
