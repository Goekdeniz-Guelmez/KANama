import sys
from pathlib import Path
from setuptools import find_packages, setup

# Get the project root directory
root_dir = Path(__file__).parent

# Read the requirements from the requirements.txt file in the root directory
requirements_file = root_dir / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as fid:
        requirements = [l.strip() for l in fid.readlines()]
else:
    print("\n\n\n\nWarning: requirements.txt not found. Proceeding without dependencies.\n\n\n\n")
    requirements = []

# Setup configuration
setup(
    name="KANama",
    version="2.5.5",
    description="Fusing Kolmogorov–Arnold Networks with Meta's Llama model for next-level AI performance and versatility.",
    long_description=open(root_dir / "README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author_email="goekdenizguelmez@gmail.com",
    author="Gökdeniz Gülmez",
    url="https://github.com/Goekdeniz-Guelmez/KANama",
    license="MIT",
    install_requires=requirements,
    packages=find_packages(include=["KANama", "KANama.*"]),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10"
    ],
    include_package_data=True,
    # package_data={
    #     # Include any package-specific data files here
    # },
)