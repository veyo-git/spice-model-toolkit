from setuptools import setup, find_packages

setup(
    name="spice-model-toolkit",
    version="0.1.0",
    description="SPICE Model Parameter Extraction Toolkit — semiconductor device modeling, optimization, and sensitivity analysis",
    author="veyo-git",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "gradio>=4.0.0",
        "scikit-learn>=1.3.0",
        "SALib>=1.4.0",
        "torch>=2.0.0",
        "h5py>=3.9.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "test": ["pytest>=7.4.0"],
    },
)
