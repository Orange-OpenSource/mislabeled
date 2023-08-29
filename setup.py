#! /usr/bin/env python
"""A template for scikit-learn compatible packages."""

import codecs

from setuptools import find_packages, setup


DISTNAME = "mislabeled"
DESCRIPTION = "Detect mislabeled examples in machine learning datasets"
with codecs.open("README.md", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "T. George"
MAINTAINER_EMAIL = "tfjgeorge@gmail.com"
URL = "https://github.com/scikit-learn-contrib/project-template"
LICENSE = "new BSD"
DOWNLOAD_URL = "https://github.com/scikit-learn-contrib/project-template"
USE_SCM_VERSION = {"local_scheme": "no-local-version"}
SETUP_REQUIRES = ["setuptools_scm"]
INSTALL_REQUIRES = ["numpy", "scipy", "scikit-learn"]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
]
EXTRAS_REQUIRE = {
    "tests": ["pytest", "pytest-cov"],
    "docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib"],
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    use_scm_version=USE_SCM_VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
