import codecs
import os
import re

from setuptools import setup


# read and find_version from pip/setup.py {{{
here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")
# }}}


setup(
    name="hypervector",
    version=find_version("hypervector.py"),
    description="Simple, general, pure Python vectors",
    long_description=read("README.rst"),
    author="Ben Burrill",
    author_email="bburrill98+hypervector@gmail.com",
    url="https://github.com/benburrill/hypervector",
    py_modules=["hypervector"],
    license="Public Domain",
    platforms="any",
    classifiers=["Programming Language :: Python :: 3",
                 "Operating System :: OS Independent",
                 "Topic :: Scientific/Engineering :: Mathematics",
                 "License :: Public Domain"]
)

# vim: foldmethod=marker
