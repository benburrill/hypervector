from .common import *

@given(vector_types())
def test_zero_property(cls):
    """ Vector.zero should be the 0-vector """
    assert not cls.zero

@given(vector_types())
def test_construct_zero(cls):
    """ Vector() should be the 0-vector """
    assert cls() == cls.zero

@given(vectors())
def test_construction_identity(vec):
    """ Vector(vec) should be the identity function """
    assert type(vec)(vec) == vec

@given(st.lists(numbers()))
def test_iterable_construction(list):
    """ Vector(iterable) should be equivalent to from_iterable """
    assert V(list) == V.from_iterable(list)

@given(st.lists(st.lists(numbers())))
def test_combine_iterables(list_of_lists):
    """ Vector(*iterables) should combine iterables """
    assert V(*list_of_lists) == V(sum(list_of_lists, []))

@given(st.lists(st.lists(numbers())), numbers())
def test_combine_scalars(list_of_lists, scalar):
    """ Scalar quantities should be merged along with the iterables """
    assert V(*list_of_lists, scalar) == V(*(list_of_lists + [scalar]))

@given(st.lists(st.lists(numbers())), vectors())
def test_add_final_vector(list_of_lists, vec):
    """ A final vector should combine with previous iterables """
    assert V(*list_of_lists, vec) == V(*list_of_lists, list(vec))

@given(st.lists(numbers()), infinite_vectors(), st.lists(numbers()))
def test_no_append_to_vector(begin, vec, end):
    """ It should be invalid to append further iterables to a vector """
    with raises(TypeError):
        V(begin, vec, end)

@given(st.lists(numbers()))
def test_from_iterable_using_iterator(list):
    """ from_iterable take an arbitrary iterable and return a vector """
    iterator = iter(list)
    vec = V.from_iterable(iterator)
    assert isinstance(vec, V)
    # Go through in reverse to ensure iterator is not misused
    for idx, item in reversed(tuple(enumerate(list))):
        assert vec[idx] == item

_mappings = st.dictionaries(
    st.integers(min_value=0, max_value=100), numbers()
)

@given(_mappings)
def test_from_mapping(mapping):
    """ Mapped components should correspond to vector components """
    vec = V.from_mapping(mapping)
    for key, value in mapping.items():
        assert vec[key] == value

@given(vector_types())
def test_no_negative_index_mapping(cls):
    """ Negative component maps should always be invalid """
    with raises(ValueError):
        cls.from_mapping({-1: 1})

@given(infinite_vectors(), _mappings)
def test_from_mapping_extend(vec, mapping):
    """ from_mapping should mask over the base argument """
    extended = V.from_mapping(mapping, base=vec)
    important = list(vec)
    for idx in range(max(len(important), max(mapping, default=0) + 1)):
        if idx in mapping:
            assert extended[idx] == mapping[idx]
        else:
            assert extended[idx] == vec[idx]

@given(numbers(), numbers(), numbers())
def test_keyword_component_names(x, y, z):
    """ Component kwargs should correspond to component attributes """
    assert V(x=x).x == x
    assert V(y=y).y == y
    assert V(z=z).z == z
    assert V(x=x, y=y, z=z) == V(i=x, j=y, k=z) == V(r=x, g=y, b=z)

def test_no_keyword_mixing():
    """ Component kwargs should not allow mixing of name groups """
    with raises(TypeError):
        V(x=42, j=10)

@given(vectors(), st.dictionaries(st.sampled_from("xyzw"), numbers()))
def test_keyword_extend(vec, mapping):
    """ Component kwargs should extend a vector like from_mapping """
    index_map = {"xyzw".index(key): value
                 for key, value in mapping.items()}
    assert V(vec, **mapping) == V.from_mapping(index_map, base=vec)

@given(vectors())
def test_from_spherical_heading(vec):
    """ vec.heading should be the inverse of Vector.from_spherical """
    assert isclose(vec, V.from_spherical(vec.mag, vec.heading))

@given(numbers(min_value=0), vectors())
def test_from_spherical_mag(mag, direction):
    """ from_spherical should return a vector with the given mag """
    assert isclose(V.from_spherical(mag, direction).mag, mag)

@given(numbers(min_value=0), numbers())
def test_from_polar_mag(mag, theta):
    """ from_polar should return a vector with the given mag """
    assert isclose(V.from_polar(mag, theta).mag, mag)

# TODO: should this restriction be lifted?
@given(finite_vectors(max_size=1))
def test_bad_heading2_projection(vec):
    """
    vec.heading2 should be invalid for 0 or 1 dimensional vectors since
    they cannot be projected in the xy-plane.
    """
    with raises(AttributeError):
        vec.heading2

@given(vectors(min_size=2, max_size=2))
def test_from_polar_heading2(vec):
    """ vec.heading2 should be the inverse of Vector.from_polar """
    assert isclose(vec, V.from_polar(vec.mag, vec.heading2))

@given(vectors())
def test_vectorize(vec):
    """ Vector.vectorize should make a function component-wise """
    vsin = V.vectorize(math.sin)
    sined = vsin(vec)
    assert isinstance(sined, V)
    for idx, value in enumerate(vec):
        assert sined[idx] == math.sin(value)
