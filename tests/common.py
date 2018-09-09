from hypervector import Vector as V
import hypothesis.strategies as st
import math

# for tests
from hypothesis import given
from pytest import raises


SIDEWAYS_INFINITY = 8


def numbers(*, min_value=-1e9, max_value=1e9):
    return st.one_of(
        st.integers(min_value=min_value, max_value=max_value),
        st.floats(min_value=min_value, max_value=max_value,
                  allow_nan=False, allow_infinity=False),
        st.fractions(min_value=min_value, max_value=max_value)
    )

def num_lists(elements=numbers(), **kwargs):
    return st.lists(elements, **kwargs)

def finite_vectors(*args, **kwargs):
    return num_lists(*args, **kwargs).map(
        lambda l: V[len(l)].from_iterable(l)
    )

def infinite_vectors(*args, **kwargs):
    return st.builds(
        V.from_iterable,
        num_lists(*args, **kwargs)
    )

def vectors(*args, **kwargs):
    return st.one_of(
        finite_vectors(*args, **kwargs),
        infinite_vectors(*args, **kwargs)
    )

def _sphere_lists(*, min_size=1, max_size=None):
    if min_size is not None:
        min_size -= 1
    if max_size is not None:
        max_size -= 1

    return st.lists(
        st.floats(allow_nan=False, allow_infinity=False),
        min_size=min_size, max_size=max_size
    )

# These normals strategies would not give us evenly distributed normals
# if the lists strategy was evenly distributed, but since lists already
# isn't, it doesn't really matter.
def finite_normals(*args, **kwargs):
    return _sphere_lists(*args, **kwargs).map(
        lambda l: V[len(l) + 1].from_spherical(1, l)
    )

def infinite_normals(*args, **kwargs):
    return st.builds(
        V.from_spherical, st.just(1),
        _sphere_lists(*args, **kwargs)
    )

def normals(*args, **kwargs):
    return st.one_of(
        finite_normals(*args, **kwargs),
        infinite_normals(*args, **kwargs)
    )

def finite_vector_types(**kwargs):
    # By using num_lists here, we make sure the Hypothesis is
    # comfortable generating the amount of data required to fill a
    # vector of the given size, and also reuse the default
    # parameters all other vector strategies use.

    return num_lists(**kwargs).map(
        lambda l: V[len(l)]
    )

def vector_types(**kwargs):
    return st.one_of(st.just(V), finite_vector_types(**kwargs))

def isclose(a, b, *, rel_tol=1e-9, abs_tol=1e-7):
    """ math.isclose with support for vectors and a bigger abs_tol """

    # Adapted from Modules/mathmodule.c
    if rel_tol < 0 or abs_tol < 0:
        raise ValueError("tolerances must be non-negative")

    # Deal with infinity
    # NOTE: Neither of the infinity checks will work on vectors that
    # have some infinite components but are not otherwise equal.  I'll
    # cross that bridge when (hopefully if) I come to it.
    if a == b:
        return True

    # TODO: wtf is this and why did I do it?  If they were close to
    # zero, they might still be close.
    ###################################################################
    # Deal with negative infinity.
    if a == -b:
        return False

    try:
        diff = abs(a - b)
        return (diff <= abs(rel_tol * b) or 
                diff <= abs(rel_tol * a) or
                diff <= abs_tol)
    except OverflowError:
        # Oh FFS
        return isclose(a/1e5, b/1e5, rel_tol=rel_tol, abs_tol=abs_tol)
