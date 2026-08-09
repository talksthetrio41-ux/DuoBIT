from setuptools import setup, find_packages

setup(
    name="duobit",
    version="0.1.0",
    description="DUOBIT-EST: Native 2-bit LLM training without master weights via error-compensated stochastic transitions",
    author="DUOBIT Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "einops>=0.6.0",
        "transformers>=4.30.0",
        "datasets",
        "accelerate",
    ],
)
