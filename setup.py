#!/usr/bin/env python

from distutils.core import setup

setup(
    name="HyTra",
    version="0.1.1",
    author="Carsten Haubold",
    author_email="team@ilastik.org",
    description=("Tracking pipeline of the IAL lab at the University of Heidelberg"),
    license="BSD",
    keywords="cell tracking divisions ilastik",
    url="http://github.com/ilastik/hytra",
    packages=["hytra", "hytra.core", "hytra.util", "hytra.pluginsystem", "hytra.dvid"],
    package_data={"hytra": ["plugins/*/*"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
