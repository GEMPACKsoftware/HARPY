import os
from setuptools import setup, find_packages

def package_files(directory, append_to=None):
    directory = os.path.normpath(directory)
    if append_to is None:
        paths = []
    else:
        paths = append_to
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

test_files = package_files('harpy/tests/testdata')

setup(
    name='harpy3',
    version='0.3.1',
    description='Library for reading/modifying/writing header-array (HAR) files.',
    packages=find_packages(),
    url='https://github.com/GEMPACKsoftware/HARPY',
    license='GPLv3',
    python_requires=">=3.4",
    install_requires=['numpy'],
    author="Lyle Collins",
    author_email='Lyle.Collins@csiro.au',
    package_data={'harpy': test_files},
)
