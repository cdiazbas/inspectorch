from setuptools import setup, find_packages
import subprocess


def get_version_info():
    """
    Returns the version string in PEP 440 format based on git tags.

    Format: 1.0.dev{commits}+{hash} for development versions,
    or just 1.0 for release versions.
    """
    try:
        output = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return f"1.0.dev0+g{output}"
    except Exception:
        return "1.0"


setup(
    name="inspectorch",
    version=get_version_info(),
    description="Efficient rare event exploration with normalizing flows",
    author="Carlos Jose Diaz Baso",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "nflows",
        "numpy",
        "matplotlib",
        "einops",
        "scipy",
    ],
)
