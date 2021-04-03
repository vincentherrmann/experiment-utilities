import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="experiment_utilities",
    version="0.0.1",
    author="Vincent Herrmann",
    author_email="vincent.herrmann@idsia.ch",
    description="Utilities for running and logging deep learning experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)