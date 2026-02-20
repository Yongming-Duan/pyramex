from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyramex",
    version="0.1.0",
    author="Xiao Long Xia 1",
    author_email="xiaolongxia@openclaw.cn",
    description="A Python Ramanome Analysis Toolkit for ML/DL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yongming-Duan/pyramex",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "ml": [
            "torch>=1.9.0",
            "tensorflow>=2.6.0",
        ],
        "gpu": [
            "cupy>=9.0",
            "numba>=0.53",
        ],
    },
)
