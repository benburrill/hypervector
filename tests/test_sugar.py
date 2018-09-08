from .common import *

def test_unit_vecs():
    """ The Vector class should have unit vector attributes """
    assert V(1, 0, 0) == V.x == V.i == V.r
    assert V(0, 1, 0) == V.y == V.j == V.g
    assert V(0, 0, 1) == V.z == V.k == V.b
    assert V(0, 0, 0, 1) == V.w == V.a

def test_no_class_swizzling():
    """ The Vector class-attributes should NOT implement swizzling """
    with raises(AttributeError):
        V.xyz

def test_bad_class_attr():
    """
    Test that getting invalid unit vectors results in an AttributeError
    Also helps to ensure that the underscore is not used in a name group
    """
    with raises(AttributeError):
        V._

@given(st.lists(numbers()))
def test_get_index(list):
    """ Vectors should be subscript-able in a way that mimics lists """
    vec = V(list)
    for idx, val in enumerate(list):
        assert vec[idx] == val

@given(st.lists(numbers()))
def test_get_out_of_list_bounds(list):
    """ Unlike lists, vectors should have infinite components of 0 """
    vec = V(list)
    assert vec[len(list)] == 0
    assert vec[len(list) + 1114111] == 0

@given(vectors())
def test_get_neg_index(vec):
    """ Negative vector indices should be 0 """
    assert vec[-1] == 0

@given(st.lists(numbers()), st.integers(min_value=0),
                            st.integers(min_value=0),
                            st.integers(min_value=1))
def test_get_slice(list, start, stop, step):
    """ Vectors should be slice-able in a way that mimics lists"""
    vec = V(list)[start:stop:step]
    assert isinstance(vec, V)
    for idx, val in enumerate(list[start:stop:step]):
        assert vec[idx] == val

@given(vectors())
def test_get_neg_slices(vec):
    """ Slices with negatives should behave appropriately """
    assert vec[-3::1] == V()
    assert vec[1:-2] == vec[1:]
    with raises(ValueError):
        vec[:2:-1]

@given(vectors(), st.lists(st.integers()))
def test_get_multiple(vec, selection):
    """ Iterable indices should be used to select components """
    selected = vec[selection]
    for idx, requested in enumerate(selection):
        assert selected[idx] == vec[requested]

@given(vectors())
def test_get_component_attr(vec):
    """ Vector instances should have component attributes """
    assert vec[0] == vec.x == vec.i == vec.r
    assert vec[1] == vec.y == vec.j == vec.g
    assert vec[2] == vec.z == vec.k == vec.b
    assert vec[3] == vec.w          == vec.a

@given(vectors())
def test_get_swizzle_attr(vec):
    """ Vector instance-attributes should implement swizzling """
    assert vec[2, 0, 1, 2] == vec.zxyz == vec.kijk == vec.brgb

def test_no_swizzle_mixing():
    """ Components from different name groups should not swizzle """
    vec = V(1, 2, 3)
    with raises(AttributeError):
        vec.xjb

@given(vectors())
def test_underscore_swizzling(vec):
    """ Underscores should be usable as zeros in swizzling """
    correct = V(0, 0, vec[2], vec[1], 0, vec[0], vec[2], 0, 0)
    assert correct == vec.__zy_xz__ == vec.__kj_ik__ == vec.__bg_rb__

def test_no_individual_underscore():
    """ The underscore should not work alone """
    vec = V(1, 2, 3)
    with raises(AttributeError):
        vec._

@given(vectors())
def test_immutability(vec):
    """ Vector components should not be settable """
    with raises(TypeError):
        vec[0] = 42
    with raises(TypeError):
        del vec[0]
    with raises(TypeError):
        vec.x = 42
    with raises(TypeError):
        del vec.x

# @given(vectors(), st.integers(min_value=0, max_value=100))
# def test_left_shift(vec, shift):
#     """ The << operator should shift components over to the left """
#     assert vec << shift == vec[shift:]

# @given(vectors(), st.integers(min_value=0, max_value=100))
# def test_right_shift(vec, shift):
#     """ The >> operator should shift components over to the right """
#     assert vec >> shift == V(shift * [0], vec)

# @given(vectors(), st.integers(min_value=0, max_value=100))
# def test_neg_shift(vec, shift):
#     """ Negative shifts should do the inverse shift operation """
#     assert vec << shift == vec >> -shift
#     assert vec >> shift == vec << -shift
