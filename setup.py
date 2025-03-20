from setuptools import setup, find_packages

setup(
    name="nn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
    ],
    author="Vishvak",
    author_email="vishvak.subramanyam@ucsf.edu",
    description="A neural network implementation for autoencoder and TF binding classification",
    keywords="neural network, machine learning, autoencoder",
    python_requires=">=3.7",
)