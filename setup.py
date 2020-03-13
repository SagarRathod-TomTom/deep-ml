import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepml",
    version="0.0.1",
    author="Sagar Rathod",
    author_email="sagar100rathod@gmail.com",
    description="Library for training deep neural nets in Pytorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sagar-rathod/PytorchDeepML",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['torch>=1.0.0', 'torchvision>=0.2.1']
)