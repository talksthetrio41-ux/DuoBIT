from setuptools import find_packages, setup

setup(
    name="duobit",
    version="1.0.0",
    description="DUOBIT-EST: native 2-bit / ternary / binary LLM training without master weights, with compressed training state",
    author="Pratyush Bhardwaj",
    packages=find_packages(include=["duobit", "duobit.*"]),
    python_requires=">=3.8",
    install_requires=["torch>=2.0.0"],
)
