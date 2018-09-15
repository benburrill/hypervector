from .common import *

@given(vector_types(min_size=4))
def test_unit_vecs(cls):
    """ The Vector class should have unit vector attributes """
    assert cls(1, 0, 0) == cls.x == cls.i == cls.r
    assert cls(0, 1, 0) == cls.y == cls.j == cls.g
    assert cls(0, 0, 1) == cls.z == cls.k == cls.b
    assert cls(0, 0, 0, 1) == cls.w == cls.a

@given(finite_vector_types(max_size=3))
def test_bad_unit_vecs(cls):
    with raises(AttributeError):
        cls.w

    with raises(AttributeError):
        cls.a

@given(vector_types())
def test_no_class_swizzling(cls):
    """ The Vector class-attributes should NOT implement swizzling """
    with raises(AttributeError):
        cls.xyz

@given(vector_types())
def test_bad_class_attr(cls):
    """
    Test that getting invalid unit vectors results in an AttributeError
    Also helps to ensure that the underscore is not used in a name group
    """
    with raises(AttributeError):
        cls._

@given(st.integers(min_value=0, max_value=100))
def test_finite_vector_types(dim):
    assert V[dim].dim == dim

@given(st.integers(max_value=-1))
def test_no_negative_dimensional_vectors(dim):
    with raises(TypeError):
        V[dim]

@given(num_lists())
def test_get_index(list):
    """ Vectors should be subscript-able in a way that mimics lists """
    inf_vec = V(list)
    fin_vec = V[len(list)](list)

    for idx, val in enumerate(list):
        assert inf_vec[idx] == fin_vec[idx] == val

@given(num_lists(), st.integers(min_value=0))
def test_get_out_of_list_bounds(list, how_far):
    """
    Infinite vectors should have infinite 0 components, but finite
    vectors are more like lists
    """
    inf_vec = V(list)
    fin_vec = V[len(list)](list)

    assert inf_vec[len(list) + how_far] == 0

    with raises(IndexError):
        fin_vec[len(list) + how_far]

@given(infinite_vectors(), st.integers(max_value=-1))
def test_get_infinite_vector_neg_index(vec, idx):
    """
    Looking up any negative index on an infinite vector should yield 0
    """
    assert vec[idx] == 0

@given(num_lists(), st.integers(max_value=-1))
def test_get_finite_vector_neg_index(list, idx):
    """ Negative indices on finite vectors should work like on lists """
    fin_vec = V[len(list)](list)

    try:
        val = list[idx]
    except IndexError:
        with raises(IndexError):
            fin_vec[idx]
    else:
        assert fin_vec[idx] == val

@given(num_lists(), st.integers(min_value=0),
                    st.integers(min_value=0),
                    st.integers(min_value=1))
def test_get_pos_slices(list, start, stop, step):
    """
    Vectors should be slice-able in a way that mimics lists for positive
    slices.
    """
    inf_vslice = V(list)[start:stop:step]
    fin_vslice = V[len(list)](list)[start:stop:step]
    lslice = list[start:stop:step]
    assert isinstance(inf_vslice, V[len(lslice)])
    assert isinstance(fin_vslice, V[len(lslice)])
    for idx, val in enumerate(lslice):
        assert inf_vslice[idx] == fin_vslice[idx] == val

# TODO: test negative slices for finite vectors
@given(infinite_vectors())
def test_get_neg_slices(vec):
    """
    Slices with negatives on infinite vectors should have special
    behavior.
    """

    assert vec[-3::1] == V()
    assert vec[1:-2] == vec[1:]
    with raises(ValueError):
        vec[:2:-1]

# TODO: finite
@given(infinite_vectors(), st.lists(st.integers()))
def test_get_multiple(vec, selection):
    """ Iterable indices should be used to select components """
    selected = vec[selection]
    for idx, requested in enumerate(selection):
        assert selected[idx] == vec[requested]

@given(st.one_of(infinite_vectors(), finite_vectors(min_size=4)))
def test_get_component_attr(vec):
    """ Vector instances should have component attributes """
    assert vec[0] == vec.x == vec.i == vec.r
    assert vec[1] == vec.y == vec.j == vec.g
    assert vec[2] == vec.z == vec.k == vec.b
    assert vec[3] == vec.w          == vec.a

@given(finite_vectors(max_size=3))
def test_finite_vectors_get_bad_component_attr(vec):
    with raises(AttributeError):
        vec.w

    with raises(AttributeError):
        vec.a

@given(vectors(min_size=3))
def test_get_swizzle_attr(vec):
    """ Vector instance-attributes should implement swizzling """
    assert vec[2, 0, 1, 2] == vec.zxyz == vec.kijk == vec.brgb

@given(vectors())
def test_no_swizzle_mixing(vec):
    """ Components from different name groups should not swizzle """
    with raises(AttributeError):
        vec.xjb

@given(vectors(min_size=3))
def test_underscore_swizzling(vec):
    """ Underscores should be usable as zeros in swizzling """
    correct = V[9](0, 0, vec[2], vec[1], 0, vec[0], vec[2], 0, 0)
    assert correct == vec.__zy_xz__ == vec.__kj_ik__ == vec.__bg_rb__

@given(vectors())
def test_no_individual_underscore(vec):
    """ The underscore should not work alone """
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
