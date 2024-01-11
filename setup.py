from setuptools import setup

setup(
    name="ml_recon",
    version="0.1",
    packages=["ml_recon"],
    install_requires=[
        'numpy', 
        'torch',
        'matplotlib',
        'einops',
    ],
)

