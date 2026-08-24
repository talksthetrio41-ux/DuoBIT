from setuptools import find_packages, setup

setup(
    name="duobit",
    version="0.3.0",
    description="DUOBIT-EST v3: native 2-bit / ternary / binary LLM training without master weights (PEFA)",
    author="Pratyush Bhardwaj",
    packages=find_packages(include=["duobit", "duobit.*"]),
    python_requires=">=3.8",
    install_requires=["torch>=2.0.0"],
)
