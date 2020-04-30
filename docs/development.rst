.. _sec_development:

=======================
Developer documentation
=======================

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
Once this PR is merged, create a release on GitHub with the pattern
``py_MAJOR.MINOR.PATCH``. Create the distribution artefacts for Python and
upload to PyPI.

-----
C API
-----

If the C API has been updated, the ``KAS_VERSION_*`` macros should be set
appropriately, ensuring that the Changelog has been updated to record the
changes. After the commit including these changes has been merged, tag a
release on GitHub using the pattern ``c_MAJOR.MINOR.PATCH``.


-----------------
Pre-commit checks
-----------------

You can install pre-commit checks with: ``pre-commit install``
Then on each commit a `pre-commit hook <https://pre-commit.com/>`_  will run
checks for code style and common problems.
Sometimes these will report "files were modified by this hook" ``git add``
and ``git commit --amend`` will update the commit with the automatically modified
version.
The modifications made are for consistency, code readability and designed to
minimise merge conflicts. They are guaranteed not to modify the functionality of the
code. To run the checks without committing use ``pre-commit run``. To bypass
the checks (to save or get feedback on work-in-progress) use ``git commit
--no-verify``