.. _sec_development:

=======================
Developer documentation
=======================

Kastore largely follows the same structure and development processes
as `tskit <https://tskit.readthedocs.io/>`__. Please see the
tskit `developer documentation <https://tskit.readthedocs.io/en/stable/development.html>`__
for details on project structure, code formatting, and more.

**********
Versioning
**********

We use the `semver convention <https://semver.org/>`_ for versioning.
Kastore contains implementations for different languages, and these are versioned
independently so that the semver semantics are meaningful. Released versions are
tagged on GitHub with a language prefix and the version number (i.e., ``py_0.2.2``);
any release artefacts are uploaded separately as appropriate.

***************
Release process
***************

The release process differs depending on the language that been affected. If multiple
languages have been updated, then the process should be followed for each language.

------
Python
------

To make a release first prepare a pull request that sets the correct version
number in ``kastore/_version.py`` and updates the Python CHANGELOG.rst,
ensuring that all significant changes since the last release have been listed.
Once this PR is merged, push a tag to github following PEP440 format::

    git tag -a MAJOR.MINOR.PATCH -m "Python version MAJOR.MINOR.PATCH"
    git push upstream --tags

This will trigger a build of the distribution artifacts for Python
on `Github Actions <https://github.com/tskit-dev/kastore/actions>`_. and deploy
them to the `test PyPI <https://test.pypi.org/project/kastore/>`_. Check
the release looks good there, then create a release on Github based on the tag you
pushed. Publishing this release will cause the github action to deploy to the
`production PyPI <https://pypi.org/project/kastore/>`_.

-----
C API
-----

If the C API has been updated, the ``KAS_VERSION_*`` macros should be set
appropriately, ensuring that the Changelog has been updated to record the
changes. After the commit including these changes has been merged, tag a
release on GitHub using the pattern ``C_MAJOR.MINOR.PATCH``.


