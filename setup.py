import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepml",
    version="1.0.0",
    author="Sagar Rathod",
    author_email="sagar100rathod@gmail.com",
    description="Library for training deep neural nets in Pytorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sagar-rathod/PytorchDeepML",
    packages=setuptools.find_packages(include="deepml"),
    package_data={"deepml": ["resources/fonts/*.ttf"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.6',
    install_requires=['torch>=1.0.0', 'torchvision>=0.2.1', 'tensorboard>=1.14.0', 'scikit-learn>=0.22',
                      "pandas>=0.25.3", "matplotlib>=2.2.5"]
)