# HARPY
This project provides a python package to read and write HAR files produced by GEMPACK 11.4 or lower.
In addition a basic decoder for SL4 to HAR format is implemented.

## Current state of the project
The package is currently in alpha 001 stage. From our testing most provided features will work. However, not everything is properly tested and there are some pitfalls in using the library (e.g. Fortran and C memory order of numpy arrays). Even if your program runs without error, you should check that the result is correct. We do not give any warranty for the package.

## Dependencies and compatibility
HARPY can be used with python 2.7 as well as 3.x verions.
The only additional package required to use this project is [numpy](www.numpy.org).  

## License
The code on github is released under the GPLv3. We know it is not the most common license for python sources. However, for users who just use it for data processessing and do not distribute their code the GPL should not cause any problems. If you want to distribute code including the HARPY package, it will have to be under the terms of the GPL as well.

There is a Contributor License Agreement(CLA) for people who want to participate in the project (you will be asked to agree upon the first pull request). This will allow us to distribute the code under a different license in cases where there is a good reason why GPL can not be used. 


## Who is behind the project and why is it on github?
The project was initiated by GEMPACK software development team. We felt the need to make the processing of HAR files easier from within other programming languages. Therefore we decided to write a python interface to read and write HAR files. After the hard part was done (reading Fortran binary files from python is not fun), the question arose how to release the software.
We decided to make it an open source project as we lack the man power to properly maintain the package according to the standards we impose in GEMPACK:
* Ensure correctness of all provided features
* Ensure completeness, i.e. all features of GEMPACK are supported
* Keeping it synchronized with future versions of the GEMPACK software and HAR format changes 

We will still be actively involved in the HARPY project by maintaining it on github, improving and updating features, making bug fixes,... However, the GEMPACK software team is not directly responsible for the HARPY project so please abstain from sending bug reports or feature requests to the GEMPACK support.

## Bugs and feature requests
As stated above, GEMPACK software is not directly responsible for the HARPY project. It is open source. So please abstain from sending bug reports or feature requests to the GEMPACK support. Instead use the [github pages](https://github.com/GEMPACKsoftware/HARPY/issues/new) to notify the developers of any issues.

## Contribute
We appreciate any contribution, no matter whether it is just fixing typos, providing examples, documention, fixing bugs or providing new functionality. None of us is a dedicated python developer so any improvements to the structure or interface of the package which makes it easier for other users are most welcome as well. 

Contributing to the project is easy (recommended steps):
* Create a [fork](https://github.com/GEMPACKsoftware/HARPY/edit/master/README.md#fork-destination-box) of this project
* Create a feature branch from the master
* Commit changes to the branch
* Push the branch back into your github project
* Open a pull request on github for the changes
* If there are issues, there will be a round of discussion and fixes
* Once everything is OK, we will merge the changes with the main project

## TODO
* [ ] Verify, improve or fix the installer
* [ ] Document the source code
* [ ] Write documentation for users
* [ ] Build proper testing framework

