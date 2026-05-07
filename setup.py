from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="drift-detection-dissertation",
    version="1.0.0",
    author="Malik Adeel",
    author_email="malikanjum.adeel@gmail.com",
    description="Data Drift Detection in Healthcare - MSc Dissertation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MalikAdeel-Hull/MSc-Dissertation-Drift-Detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)