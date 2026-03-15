"""Setup script for the ridgeback_patrol Isaac Lab external project."""

from setuptools import setup, find_packages

setup(
    name="ridgeback_patrol",
    version="0.1.0",
    description="RL training environment for indoor security patrol robot using Isaac Lab.",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)
