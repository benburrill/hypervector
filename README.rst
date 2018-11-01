hypervector
===========
| Simple, general, pure Python vectors
|
| ``hypervector.py`` defines arbitrary-dimentional vector types for all your vectoring needs.  The vectors are immutable and come with many useful and well-tested vector operations and features.
|
| ``hypervector`` is in the public domain.

A few explanatory examples
--------------------------
Hello world, *NOW IN 3D!*

.. code:: pycon

    >>> from hypervector import Vector3
    >>> (Vector3(1, 2, 1) + Vector3(3, 0, 2)).zxy
    Vector3(3, 4, 2)

Higher dimensions

.. code:: pycon

    >>> from hypervector import Vector, Vector2, Vector3
    >>> Vector2 is Vector[2] and Vector3 is Vector[3]
    True
    >>> Vector[4](1, 2, 3, 4)
    Vector[4](1, 2, 3, 4)
    >>> Vector[5](1, 2, 3, 4)
    Vector[5](1, 2, 3, 4, 0)
    >>> Vector[10].zero
    Vector[10](0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

King of infinite (vector) space

.. code:: pycon

    >>> vec = Vector(2, 4)  # Dimensionless vectors are "infinite"
    >>> (vec[0], vec[1], vec[2], vec[1114111])
    (2, 4, 0, 0)

Cross-eyed

.. code:: pycon

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

How fast is hypervector?
------------------------
| Dunno.  Probably pretty slow.  If you need speed, use numpy.

Alternatives
------------
| There are many other libraries with similar features to ``hypervector``.  Some notable examples I have used:

* `numpy <http://www.numpy.org/>`_
* `pyeuclid <https://pypi.python.org/pypi/euclid3>`_ |--| Has unrelated ``Vector2`` and ``Vector3`` classes.  Also has two separate packages for *Python* 2 and *Python* 3.

Links
-----
| Get `hypervector from PyPI`_: ``pip install hypervector``
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
