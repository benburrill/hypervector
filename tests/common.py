from hypervector import Vector as V
import hypothesis.strategies as st
import math

# for tests
from hypothesis import given
from pytest import raises


def numbers(*, min_value=-1e9, max_value=1e9):
    return st.one_of(
        st.integers(min_value=min_value, max_value=max_value),
        st.floats(min_value=min_value, max_value=max_value,
                  allow_nan=False, allow_infinity=False),
        st.fractions(min_value=min_value, max_value=max_value)
    )

def vectors(comp_types=numbers(), *, average_size=3, max_size=None):
    return st.builds(
        V.from_iterable,
        st.lists(comp_types, average_size=average_size, 
                 max_size=max_size)
    )

# This would not give us an even distribution if the lists were evenly
# distributed, but that doesn't really matter because they are not.
def normals(*, average_size=3, max_size=None):
    if average_size is not None:
        average_size -= 1
    if max_size is not None:
        max_size -= 1
    return st.builds(
        V.from_spherical, st.just(1),
        st.lists(st.floats(allow_nan=False, allow_infinity=False),
                 average_size=average_size, max_size=max_size)
    )

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
