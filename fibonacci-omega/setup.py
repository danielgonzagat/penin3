"""Setup script for Fibonacci Engine (fallback for older pip versions)."""

from setuptools import setup, find_packages

setup(
    name="fibonacci-engine",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "click>=8.0.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "fib=fibonacci_engine.api.cli:cli",
        ],
    },
)
