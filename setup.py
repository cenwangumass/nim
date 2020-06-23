import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="nim",
    version="0.0.1",
    author="Cen Wang",
    author_email="cenwang@umass.edu",
    description="Open source implementation for NIM: Neural Input Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cenwangumass/nim",
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.19.0", "click>=7.1.2", "torch>=1.5.1"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
)
