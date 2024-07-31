from setuptools import find_packages, setup

setup(
    name="KANama",
    version="1.8.3",
    description="KANama: marrying Kolmogorov–Arnold Networks with Meta's Llama model.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author_email="goekdenizguelmez@gmail.com",
    author="Gökdeniz Gülmez",
    url="https://github.com/Goekdeniz-Guelmez/KANama",
    license="MIT",
    install_requires=["torch", "dataclasses", "typing", "sentencepiece", "matplotlib", "transformers"],
    packages=find_packages(),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10"
    ],
)
