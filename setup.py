import re
from pathlib import Path

from setuptools import find_packages, setup

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / "README.md").read_text(encoding="utf-8")


def get_version():
    file = PARENT / "florence2/__init__.py"
    return re.search(
        r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding="utf-8"), re.M
    )[1]


setup(
    name="florence2",
    version=get_version(),
    author="Daniel Choi",
    author_email="danielsejong55@gmail.com",
    description="Unofficial Repo for Florence2",
    long_description=README,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=find_packages(),
    python_requires=">=3.8",
)
