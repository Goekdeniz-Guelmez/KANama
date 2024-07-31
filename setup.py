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

# Import the version from the package
version = {}
version_file = root_dir / "KANama/model/version.py"
if version_file.exists():
    with open(version_file) as f:
        exec(f.read(), version)
else:
    raise FileNotFoundError(f"\n\n\n\nVersion file {version_file} not found\n\n\n\n")

# Setup configuration
setup(
    name="KANama",
    version=version['__version__'],
    description="KANama: marrying Kolmogorov–Arnold Networks with Meta's Llama model.",
    long_description=open(root_dir / "README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author_email="goekdenizguelmez@gmail.com",
    author="Gökdeniz Gülmez",
    url="https://github.com/Goekdeniz-Guelmez/KANama",
    license="MIT",
    install_requires=requirements,
    include_package_data=True,
    # packages=find_packages(include=["KANama", "KANama.*", "model", "train"]),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10"
    ]
)