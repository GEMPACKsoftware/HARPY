from setuptools import setup, find_packages

setup(
    name='HARPY',
    version='0.1',
    packages=['', 'harpy'],
    url='https://github.com/GEMPACKsoftware/HARPY',
    license='GPLv3',
    author='GEMPACK software',
    install_requires=['numpy'],
    author_email='florian.schiffmann@vu.edu.au',
    description='Python interface to work with HAR files and Headers'
)
