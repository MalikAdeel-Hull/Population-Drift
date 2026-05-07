from setuptools import setup, find_packages

setup(
    name="drift-detection",
    version="1.0.0",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'scipy',
    ],
    python_requires='>=3.7',
)