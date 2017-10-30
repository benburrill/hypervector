hypervector
===========
| Simple, general, pure Python vectors
|
| ``hypervector.py`` defines an infinite-dimensional ``Vector`` type for all your vectoring needs.  Instances of ``Vector`` are immutable and come with a wide array of useful operations and features.
|
| ``hypervector`` is in the public domain.

A few explanatory examples
--------------------------
Hello world

.. code:: python

    >>> from hypervector import Vector
    >>> (Vector(1, 2, 1) + Vector(3, 0, 2)).xy
    Vector(4, 2)

King of infinite (vector) space

.. code:: python

    >>> vec = Vector(2, 4)
    >>> (vec[0], vec[1], vec[2], vec[1114111])
    (2, 4, 0, 0)

Cross-eyed

.. code:: python

    >>> vec_1, vec_2 = Vector(1, 2, 3), Vector(3, 2, 1)
    >>> Vector.cross(vec_1, vec_2)
    Vector(-4.0, 8.0, -4.0)
    >>> [Vector.dot(_, vec) for vec in (vec_1, vec_2)]
    [0.0, 0.0]
    >>> vec_3 = Vector(-1, 8, 3, 2)
    >>> Vector.cross(vec_1, vec_2, vec_3)
    Vector(8.0, -16.0, 8.0, 56.0)
    >>> [Vector.dot(_, vec) for vec in (vec_1, vec_2, vec_3)]
    [0.0, 0.0, 0.0]

Testing
-------
| ``hypervector`` uses `py.test`_ and `hypothesis`_, with `tox`_ as a test runner.
|
| The unit tests can be run with ``tox``.  If you run into any slow data generation errors, use
| ``tox -- --hypothesis-profile=allow-slow`` instead.

Why infinite dimensions?
------------------------
| Because generalizing things is fun and having infinite dimensions is somewhat easier and simpler than having an abstract Vector type with arbitrary (but finite) dimensions.
|
| It is worth noting that ``Vector`` plays a little fast and loose with the concept of infinite dimensions, as it has a concept of "relevant" dimensions which is exposed through iteration, string representations, and a few other places to a lesser extent.
| Similarly, there is no way to have a ``Vector`` with infinite non-zero components.

How fast is hypervector?
------------------------
| Dunno.  Probably pretty slow.  If you need speed, use numpy.

Alternatives
------------
| There are many other libraries with similar features to ``hypervector``.  Some notable examples:

 * `numpy <http://www.numpy.org/>`_
 * `pyeuclid <https://pypi.python.org/pypi/euclid3>`_ |--| Has (among other things) ``Vector2`` and ``Vector3`` types.  Also has packages for *Python* 2 and *Python* 3.

Links
-----
| Get `hypervector from PyPi`_: ``pip install hypervector``
| Report bugs and offer suggestions at the github `issues`_ page.

.. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. ..
.. Dependencies
.. _py.test: https://pytest.org/
.. _hypothesis: http://hypothesis.works/
.. _tox: https://pypi.python.org/pypi/tox

.. Links
.. _hypervector from PyPi: https://pypi.python.org/pypi/hypervector
.. _issues: https://github.com/benburrill/hypervector/issues

.. Definitions
.. |--| unicode:: U+2014 .. (em dash)
