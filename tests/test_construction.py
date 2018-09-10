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

@given(st.data())
def test_from_iterable_accepts_iterators_and_other_iterables(data):
    """
    from_iterable should take arbitrary iterables and return vector
    with components from the iterable.
    """

    cls = data.draw(vector_types(), label="cls")
    l = data.draw(num_lists(max_size=cls.dim), label="l")

    list_vec = cls.from_iterable(l)
    iter_vec = cls.from_iterable(iter(l))

    # Go through in reverse to ensure iterator is not misused
    for idx, item in reversed(tuple(enumerate(l))):
        assert list_vec[idx] == item
        assert iter_vec[idx] == item

@given(st.data())
def test_iterable_construction(data):
    """ Vector(iterable) should be equivalent to from_iterable """
    cls = data.draw(vector_types(), label="cls")
    l = data.draw(num_lists(max_size=cls.dim), label="l")

    assert cls(l) == cls.from_iterable(l)

@given(st.data())
def test_finite_vector_pads_missing_components(data):
    cls = data.draw(finite_vector_types(), label="cls")
    l = data.draw(num_lists(max_size=cls.dim), label="l")

    assert list(cls(l)) == l + [0] * (cls.dim - len(l))

@given(st.data())
def test_iterable_too_big_for_finite_vector(data):
    """
    If the iterable passed to a finite vector constructor (or
    from_iterable) is too large, an error should be raised.
    """

    cls = data.draw(finite_vector_types(), label="cls")
    l = data.draw(num_lists(min_size=cls.dim+1), label="l")

    with raises(TypeError):
        cls(l)

    with raises(TypeError):
        cls.from_iterable(l)

@given(vector_types(), st.lists(st.one_of(numbers(),
                                          num_lists(),
                                          finite_vectors())))
def test_combine_iterables_scalars(cls, arguments):
    """
    The vector constructor should combine any combination of finite
    vectors, iterables, and scalars from positional arguments together.
    """

    combined = []
    for arg in arguments:
        if isinstance(arg, V):
            # Explicit is better than implicit, especially in tests.
            combined.extend(iter(arg))
        elif isinstance(arg, list):
            combined.extend(arg)
        else:
            combined.append(arg)

    try:
        from_combined = cls(combined)
    except TypeError:
        with raises(TypeError):
            cls(*arguments)
    else:
        assert cls(*arguments) == from_combined

@given(num_lists(), infinite_vectors())
def test_final_inf_vec_in_inf_vec_constructor(begin, vec):
    """
    Infinite vectors can trail other arguments in the infinite
    vector constructor.
    """

    assert V(*begin, vec) == V(*begin, list(vec))

@given(finite_vector_types(), num_lists(), infinite_vectors())
def test_no_infinite_vector_in_finite_constructor(cls, begin, inf_vec):
    """
    Infinite vectors cannot be passed to finite vector constructors
    without setting truncate to True.
    """
    with raises(TypeError):
        cls(*begin, inf_vec)

@given(vector_types(), num_lists(), infinite_vectors(), num_lists(min_size=1))
def test_no_append_to_infinite_vector(cls, begin, inf_vec, end):
    """
    It should be invalid to append further items after an infinite
    vector in any vector type.
    """

    with raises(TypeError):
        cls(*begin, inf_vec, *end)

@given(vector_types(), num_lists())
def test_truncate_truncates_iterable_as_needed(cls, l):
    """
    Truncate should truncate data to fit in the vector.
    """

    truncated = l[:cls.dim]
    assert (cls(l, truncate=True) ==
            cls(truncated, truncate=False) ==
            cls(truncated))

@given(vector_types(), num_lists(), infinite_vectors(), num_lists())
def test_truncate_with_infinite_vector(cls, begin, inf_vec, end):
    """
    Data following a truncated infinite vector should never be expressed
    in the vector.

    Implied: truncate allows infinite vectors to be passed as arguments
    to finite vector constructors.
    """

    assert (cls(*begin, inf_vec, *end, truncate=True) ==
            cls(*begin, inf_vec, truncate=True))

_mappings = st.dictionaries(
    st.integers(min_value=0, max_value=100), numbers()
)

@given(st.data())
def test_from_mapping(data):
    """ Mapped components should correspond to vector components """
    cls = data.draw(vector_types(), label="cls")
    mapping = data.draw(component_mappings(cls.dim), label="mapping")

    vec = cls.from_mapping(mapping)
    for key, value in mapping.items():
        assert vec[key] == value

@given(vector_types())
def test_no_negative_index_mapping(cls):
    """ Negative component maps should always be invalid """
    with raises(ValueError):
        cls.from_mapping({-1: 1})

@given(st.data())
def test_from_mapping_extend(data):
    """ from_mapping should mask over the base argument """
    base_vec = data.draw(vectors(), label="base_vec")
    cls = type(base_vec)
    mapping = data.draw(component_mappings(cls.dim), label="mapping")

    extended = cls.from_mapping(mapping, base=base_vec)
    important = list(base_vec)
    for idx in range(max(len(important), max(mapping, default=-1) + 1)):
        if idx in mapping:
            assert extended[idx] == mapping[idx]
        else:
            assert extended[idx] == base_vec[idx]

@given(vector_types(min_size=3), numbers(), numbers(), numbers())
def test_keyword_component_names(cls, x, y, z):
    """ Component kwargs should correspond to component attributes """
    assert cls(x=x).x == x
    assert cls(y=y).y == y
    assert cls(z=z).z == z
    assert cls(x=x, y=y, z=z) == cls(i=x, j=y, k=z) == cls(r=x, g=y, b=z)

@given(vector_types())
def test_no_keyword_mixing(cls):
    """ Component kwargs should not allow mixing of name groups """
    with raises(TypeError):
        cls(x=42, j=10)

@given(vectors(min_size=4),
       st.dictionaries(st.sampled_from("xyzw"), numbers()))
def test_keyword_extend(vec, mapping):
    """ Component kwargs should extend a vector like from_mapping """
    index_map = {"xyzw".index(key): value
                 for key, value in mapping.items()}
    assert (type(vec)(vec, **mapping) ==
            type(vec).from_mapping(index_map, base=vec))

@given(vectors(min_size=2))
def test_from_spherical_heading(vec):
    """ vec.heading should be the inverse of Vector.from_spherical """
    assert isclose(vec, type(vec).from_spherical(vec.mag, vec.heading))

@given(numbers(min_value=0), vectors())
def test_from_spherical_mag(mag, direction):
    """ from_spherical should return a vector with the given mag """
    cls = V if direction.dim is None else V[direction.dim + 1]
    assert isclose(cls.from_spherical(mag, direction).mag, mag)

@given(vector_types(min_size=2), numbers(min_value=0), numbers())
def test_from_polar_mag(cls, mag, theta):
    """ from_polar should return a vector with the given mag """
    assert isclose(cls.from_polar(mag, theta).mag, mag)

@given(vectors(min_size=2, max_size=2))
def test_from_polar_heading2(vec):
    """ vec.heading2 should be the inverse of Vector.from_polar """
    assert isclose(vec, type(vec).from_polar(vec.mag, vec.heading2))

@given(vectors())
def test_vectorize(vec):
    """ Vector.vectorize should make a function component-wise """
    vsin = type(vec).vectorize(math.sin)
    sined = vsin(vec)
    assert isinstance(sined, type(vec))
    for idx, value in enumerate(vec):
        assert sined[idx] == math.sin(value)
