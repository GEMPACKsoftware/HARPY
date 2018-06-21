# HARPY
This project provides a python package to read and write HAR files produced by GEMPACK 11.4 or lower.

## NEW - Harpy is available on pip!

Harpy can now be installed using pip:

```
    pip install harpy3
```

**IMPORTANT:** Note the command above specifies ``harpy3``, **not** ``harpy`` (the package name ``harpy`` was already in use on PIP). However, consistent with previous versions of harpy, the API can be imported with ``import harpy``.

## NEW - Harpy has adopted major (backwards-incompatible) changes

Harpy has recently undergone major structural changes to make it easier to users to learn and use the library - this new version is 0.3 (the old is 0.01). As part of these changes, Harpy has moved from a Python 2 base to Python 3. The simpler interface comes with associated documentation, as well as a testing suite which (together) will make learning the Harpy API easier and help maintain/improve the code quality of Harpy. Please see the documentation section below for details.

The previous interface can be considered deprecated. Although bug-fixes to the old API will be considered, no guarantees of continued development  are given. Any development of the old-interface will be merged with the v001 branch. New development will occur on the master branch (as per the status quo). Similar to v0.01, v0.3 (with the Python 3-related changes) can be considered 'under development'. To use v0.01 can be done by one of these processes:

 1. Downloading the zip file of the `v0.01` branch on GitHub, and following the identical install process as before (regardless of whether you were using `v0.01` or `v0.3`).
 2. If using a git program with a graphical interface, switch to the  `v0.01` branch. This would likely be found by looking for a *checkout* option, and selecting a different branch - the `v0.01` branch. The remaining install process is identical as before (regardless of whether you were using `v0.01` or `v0.3`).
 3. Using a git clone of the harpy repository, switch to the `v0.01` branch using the command ``git checkout v0.01``. The remaining install process is identical to that of `v0.3`. The remaining install process is identical as before (regardless of whether you were using `v0.01` or `v0.3`).

## Documentation

The documentation for the HARPY library can be read by downloading the source code and opening, in a web browser, the file ``doc/build/html/index.html``.

## Dependencies and compatibility
HARPY v0.3 depends on version 3.4 of Python (or later).
The only additional package required to use this project is [numpy](www.numpy.org).
The documentation library *Sphinx* (and some sphinx extensions) is required to build new versions of the documentation.

## License
The code on github is released under the GPLv3. We know it is not the most common license for python sources. However, for users who just use it for data processessing and do not distribute their code the GPL should not cause any problems. If you want to distribute code including the Harpy package, it will have to be under the terms of the GPL as well.

There is a [Contributor License Agreement(CLA)](https://gist.github.com/floschiffmann/de59328612863e1566a37a3490f9c5fd) for people who want to participate in the project (you will be asked to agree upon the first pull request). This will allow us to distribute the code under a different license in cases where there is a good reason why GPL can not be used. 

## Citations

If you use this software in an academic context, a citation is requested:

.. [1] F. Schiffmann and L. D. Collins, *"Harpy v0.3: A Python API to interact with header array (HAR) files,"* Melbourne, Australia, 2018, [https://github.com/GEMPACKsoftware/HARPY](https://github.com/GEMPACKsoftware/HARPY).

## Who is behind the project and why is it on github?
The project was initiated by GEMPACK software development team. We felt the need to make the processing of HAR files easier from within other programming languages. Therefore we decided to write a python interface to read and write HAR files. After the hard part was done (reading Fortran binary files from python is not fun), the question arose how to release the software. In June of 2018, the CSIRO Energy Business Unit contributed major (backwards-incompatible) structural changes to improve the interface. The version immediately prior to these changes is v0.01, the version immediately after is v0.3.

We decided to make it an open source project as we lack the man power to properly maintain the package according to the standards we impose in GEMPACK:
* Ensure correctness of all provided features
* Ensure completeness, i.e. all features of GEMPACK are supported
* Keeping it synchronized with future versions of the GEMPACK software and HAR format changes 

We will still be actively involved in the HARPY project by maintaining it on github, improving and updating features, making bug fixes,... However, the GEMPACK software team is not directly responsible for the HARPY project so please abstain from sending bug reports or feature requests to the GEMPACK support.

## Bugs and feature requests
As stated above, GEMPACK software is not directly responsible for the HARPY project. It is open source. So please abstain from sending bug reports or feature requests to the GEMPACK support. Instead use the [github pages](https://github.com/GEMPACKsoftware/HARPY/issues/new) to notify the developers of any issues.

## Contribute
We appreciate any contribution, no matter whether it is just fixing typos, providing examples, documention, fixing bugs or providing new functionality. Any improvements to the structure or interface of the package which makes it easier for other users are most welcome as well.

Contributing to the project is easy (recommended steps):
* Create a [fork](https://github.com/GEMPACKsoftware/HARPY/edit/master/README.md#fork-destination-box) of this project
* Create a feature branch from the master
* Commit changes to the branch
* Push the branch back into your github project
* Open a pull request on github for the changes
* If there are issues, there will be a round of discussion and fixes
* Once everything is OK, we will merge the changes with the main project



