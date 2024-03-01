Contributing
============

Bug reports, feature suggestions, and other contributions are greatly
appreciated!  pyIntensityFeatures is a community-driven project and welcomes
both feedback and contributions.

Short version
-------------

* Submit bug reports, feature requests, and questions at
  [GitHub](https://github.com/aburrell/pyIntensityFeatures/issues)

* Make pull requests to the ``develop`` branch

Issues
------

Bug reports, questions, and feature requests should all be made as GitHub
Issues.  Templates are provided for each type of issue, to help you include
all the necessary information.

Questions
^^^^^^^^^

Not sure how something works?  Ask away!  The more information you provide, the
easier the question will be to answer.  You can also search the closed issues
for potential answers to your questions.

Bug reports
^^^^^^^^^^^

When [reporting a bug](https://github.com/aburrell/pyIntensityFeatures/issues)
please include:

* Your operating system name and version

* Any details about your local setup that might be helpful in troubleshooting

* Detailed steps to reproduce the bug

Feature requests and feedback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The best way to send feedback is to file an
[issue](https://github.com/aburrell/pyIntensityFeatures/issues).

If you are proposing a new feature or a change in something that already exists:

* Explain in detail how it would work.

* Keep the scope as narrow as possible, to make it easier to implement.

* Remember that this is a volunteer-driven project, and that code contributions
  are welcome :)

Development
-----------

To set up `pyIntensityFeatures` for local development:

1. [Fork pyIntensityFeatures on
    GitHub](https://github.com/aburrell/pyIntensityFeatures/fork).

2. Clone your fork locally:

  ```
    git clone git@github.com:your_name_here/pyIntensityFeatures.git
  ```

3. Create a branch for local development:

  `git checkout -b name-of-your-bugfix-or-feature`

   Now you can make your changes locally.

   Tests for custom functions should be added to the appropriately named file
   in ``pyIntensityFeatures/tests``. For example, custom functions for the grid
   utilities are tested in ``pyIntensityFeatures/tests/test_utils_grids.py``.
   If no test file exists, then you should create one.  This testing uses
   unittest, which will run tests on any Python file in the test directory that
   starts with ``test``.  Classes must begin with ``Test``, and methods must
   begin with ``test`` as well.

4. When you're done making changes, run all the checks to ensure that nothing
   is broken on your local system, as well as check for flake8 compliance:

   ```
    python -m unittest discover
   ```

5. You should also check for flake8 style compliance:

   ```
   flake8 . --count --select=D,E,F,H,W --show-source --statistics
   ```

   Note that pyIntensityFeatures tests for uniformity in docstring formatting.

6. Update/add documentation (in ``docs``).  Even if you don't think it's
   relevant, check to see if any existing examples have changed.

7. Add your name, affiliation, and ORCiD to the .zenodo.json file as an author

8. Commit your changes:
   ```
   git add .
   git commit -m "AAA: Brief description of your changes"
   ```
   Where AAA is a standard shorthand for the type of change (e.g., BUG or DOC).
   `pyIntensityFeatures` follows the [numpy development workflow](https://numpy.org/doc/stable/dev/development_workflow.html),
   see the discussion there for a full list of this shorthand notation.

9. Once you are happy with the local changes, push to GitHub:
   ```
   git push origin name-of-your-bugfix-or-feature
   ```
   Note that each push will trigger the Continuous Integration workflow.

10. Submit a pull request through the GitHub website. Pull requests should be
    made to the ``develop`` branch.  Note that automated tests will be run on
    GitHub Actions, but these must be initialized by a member of the
    pyIntensityFeatures team for first time contributors.


Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code, just
make a pull request. Pull requests should be made to the ``develop`` branch.

For merging, you should:

1. Include an example for use
2. Add a note to ``CHANGELOG.md`` about the changes
3. Update the author list in ``zenodo.json``, if applicable
4. Ensure that all checks passed (future checks will include GitHub Actions,
   Coveralls and ReadTheDocs)

If you don't have all the necessary Python versions available locally or have
trouble building all the testing environments, you can rely on GitHub Actions to
run the tests for each change you add in the pull request. Because testing here
will delay tests by other developers, please ensure that the code passes all
tests on your local system first.


Project Style Guidelines
------------------------

In general, pyIntensityFeatures follows PEP8 and numpydoc guidelines. unittest
runs the unit and integration tests, flake8 checks for style, and sphinx-build
performs documentation tests.  However, there are certain additional style
elements that have been adopted to ensure the project maintains a consistent
coding style. These include:

* Line breaks should occur before a binary operator (ignoring flake8 W503)
* Combine long strings using `join`
* Preferably break long lines on open parentheses rather than using `\`
* Use no more than 80 characters per line
* The pyIntensityFeatures logger is imported into each sub-module and provides
  status updates at the info and warning levels (as appropriate)
* Several dependent packages have common nicknames, including:
  * `import datetime as dt`
  * `import numpy as np`
  * `import pandas as pds`
  * `import xarray as xr`
* When incrementing a timestamp, use `dt.timedelta` instead of `pds.DateOffset`
  when possible to reduce program runtime
* All classes should have `__repr__` and `__str__` functions
* Avoid creating a try/except statement where except passes
* Provide testing class methods with informative failure statements and
  descriptive, one-line docstrings
* Block and inline comments should use proper English grammar and punctuation
  with the exception of single sentences in a block, which may then omit the
  final period. Both British and American spelling is permitted.
* When casting is necessary, use `np.int64` and `np.float64` to ensure operating
  system agnosticism
