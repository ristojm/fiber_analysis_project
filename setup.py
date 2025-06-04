"""
Setup script for SEM Fiber Analysis System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README.md if it exists
readme_path = Path("README.md")
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "SEM Fiber Analysis System - Automated analysis of hollow fibers and filaments"

# Read requirements.txt if it exists
requirements_path = Path("requirements.txt")
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    # Fallback to basic requirements
    requirements = [
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0", 
        "scipy>=1.7.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0"
    ]

setup(
    name="sem-fiber-analysis",
    version="1.0.0",
    author="SEM Fiber Analysis Team",
    description="Automated analysis system for SEM images of hollow fibers and filaments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "ocr": ["pytesseract>=0.3.8", "easyocr>=1.6.0"],
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.8"],
    },
    entry_points={
        "console_scripts": [
            "fiber-analysis=modules.cli:main",  # Future CLI interface
        ],
    },
)