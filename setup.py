import sys
from pathlib import Path
from setuptools import find_packages, setup

# Get the project root directory
root_dir = Path(__file__).parent

# Add the package directory to the Python path
package_dir = root_dir

requirements_file = root_dir / "requirements.txt"

if requirements_file.exists():
    with open(requirements_file) as fid:
        requirements = [l.strip() for l in fid.readlines()]
else:
    print("Warning: requirements.txt not found. Proceeding without dependencies.")


# Import the version from the package
version = {}
with open(str(package_dir / "version.py")) as f:
    exec(f.read(), version)

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
    install_requires=requirements_file,
    packages=find_packages(),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10"
    ],
)
