# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

import setuptools


def get_long_description():
    with open('README.md') as f:
        long_description = f.read()
    return long_description


def get_version():
    version_path = "essmc2/version.py"
    with open(version_path) as f:
        exec(compile(f.read(), version_path, "exec"))
    return locals()['__version__']


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


required = get_requirements()

setuptools.setup(
    name="essmc2",
    version=get_version(),
    author="Alibaba Damo Academy",
    author_email="logistic1994@gmail.com",
    description="EssentialMC2: A Video Understanding Algorithm Framework.",
    keywords="compute vision, video, framework",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/alibaba/EssentialMC2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=required,
)
