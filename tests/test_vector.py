from .common import *
import itertools

@given(vectors())
def test_identity_equality(vec):
    """ Vectors should be equal to themselves """
    assert vec == vec

@given(vectors())
def test_hash_equality(vec):
    """ Equal vectors should have equal hashes """
    new = type(vec)(vec)
    # It would be valid for Vector(vec) to just return vec if I somehow
    # wanted it to in the future, but it would break this test.
    if new is vec:
        raise RuntimeError("Unable to run test: must have two distinct "
                           "vectors of the same dimension and with "
                           "equal components")
    assert vec == new
    assert hash(vec) == hash(new)

@given(vectors())
def test_pos_identity(vec):
    """ Unary + should be the identity function """
    assert +vec == vec

@given(vectors())
def test_additive_identity(vec):
    """ The 0-vector should be an additive identity for vectors """
    assert vec + type(vec).zero == vec

@given(vectors())
def test_multiplicative_identity(vec):
    """ 1 should be a multiplicative identity for vectors """
    assert vec * 1 == vec

@given(vectors(), vectors())
def test_commutative_addition(vec_1, vec_2):
    """ Vector addition should be commutative """
    assert vec_1 + vec_2 == vec_2 + vec_1

@given(vectors(), numbers())
def test_add_scalar_fails(vector, scalar):
    with raises(TypeError):
        vector + scalar

@given(numbers(), vectors())
def test_commutative_multiplication(scalar, vec):
    """ Vector scaling should be commutative """
    assert scalar * vec == vec * scalar

@given(vectors(), vectors())
def test_commutative_dot(vec_1, vec_2):
    """ The dot product should be commutative """
    assert vec_1 @ vec_2 == vec_2 @ vec_1

@given(vectors(), numbers())
def test_dot_scalar_fails(vector, scalar):
    with raises(TypeError):
        vector @ scalar

@given(numbers(), vectors(), vectors())
def test_distributive(scalar, vec_1, vec_2):
    """ Vector scaling should obey the distributive property """
    assert isclose(scalar * (vec_1 + vec_2),
                   scalar * vec_1 + scalar * vec_2)

@given(vectors(), vectors(), vectors())
def test_dot_distributive(vec_1, vec_2, vec_3):
    """ The dot product should obey the distributive property """
    assert isclose(vec_1 @ (vec_2 + vec_3),
                   vec_1 @ vec_2 + vec_1 @ vec_3)

@given(vectors(), vectors(), vectors())
def test_associative(vec_1, vec_2, vec_3):
    """ Vector addition should obey the associative property """
    assert isclose(vec_1 + (vec_2 + vec_3), (vec_1 + vec_2) + vec_3)

@given(vectors(), vectors())
def test_add_negative(vec_1, vec_2):
    """ vec_1 + vec_2 and vec_1 + -vec_2 should be equivalent """
    assert isclose(vec_1 - vec_2, vec_1 + -vec_2)

@given(vectors(), numbers())
def test_mult_reciprocal(vec, divisor):
    """ vec / divisor and vec * (1 / divisor) should behave the same """
    try:
        mult_recip = vec * (1 / divisor)
    except ZeroDivisionError:
        with raises(ZeroDivisionError):
            vec / divisor
    else:
        assert isclose(mult_recip, vec / divisor)

@given(vectors(), numbers())
def test_floor_division(vec, divisor):
    """ The // operator should behave like the / operator floored """
    try:
        floored_division = type(vec).vectorize(math.floor)(vec / divisor)
    except ZeroDivisionError:
        with raises(ZeroDivisionError):
            vec // divisor
    else:
        assert floored_division == vec // divisor

@given(vectors(), vectors())
def test_vector_muldiv_fails(vec_1, vec_2):
    with raises(TypeError):
        vec_1 * vec_2

    with raises(TypeError):
        vec_1 / vec_2

    with raises(TypeError):
        vec_1 // vec_2

@given(numbers(), vectors())
def test_vector_divisor_fails(scalar, vec):
    with raises(TypeError):
        scalar / vec

    with raises(TypeError):
        scalar // vec

@given(vectors())
def test_repr(vec):
    """ repr(vec) be a Python expression representing vec """
    from fractions import Fraction
    from hypervector import Vector, Vector2, Vector3
    assert isclose(vec, eval(repr(vec)))

@given(vectors())
def test_str(vec):
    """ str(vec) should be a human-readable string """
    # I will make no assertions about the format other than it should be
    # different from the repr (implying greater readability than Python
    # code).
    assert str(vec) != repr(vec)

@given(vectors())
def test_angle_between_vec_and_origin(vec):
    """ The angle between any vector and the 0-vector should be 0 """
    assert vec.angle(type(vec).zero) == 0
    assert type(vec).zero.angle(vec) == 0

@given(vectors())
def test_strict_angle_between_vec_and_origin(vec):
    """ If strict is passed, ZeroDivisionError should be raised """
    with raises(ZeroDivisionError):
        vec.angle(type(vec).zero, strict=True)
    with raises(ZeroDivisionError):
        type(vec).zero.angle(vec, strict=True)

@given(vectors())
def test_mag_aliases(vec):
    """
    abs(vec) should be an alias for vec.mag
    Also, vec.mag_sq should be vec.mag ** 2
    """
    assert isclose(abs(vec), vec.mag)
    assert isclose(vec.mag_sq, vec.mag ** 2)

@given(numbers(), vectors())
def test_mag_scaling(scale, vec):
    """ Vector scaling should scale the magnitude proportionally """
    assert isclose(abs(scale * vec), abs(scale) * abs(vec))

@given(normals(), numbers(min_value=0))
def test_with_mag(vec, mag):
    """ vec.with_mag(mag) should have the magnitude mag """
    assert isclose(vec.with_mag(mag).mag, mag)

@given(vectors())
def test_unit_vs_with_mag(vec):
    """ vec.unit and vec.with_mag(1) should behave the same """
    try:
        mag_1 = vec.with_mag(1)
    except ZeroDivisionError:
        with raises(ZeroDivisionError):
            vec.unit
    else:
        assert isclose(mag_1, vec.unit)

@given(normals(), numbers(min_value=0), numbers(min_value=0))
def test_clamp_mag(vec, bound_1, bound_2):
    """ clamp_mag should return a vector with a magnitude in bounds """
    min_mag = min(bound_1, bound_2)
    max_mag = max(bound_1, bound_2)
    lim_mag = vec.clamp_mag(min_mag, max_mag).mag
    assert (min_mag <= lim_mag <= max_mag or
            isclose(lim_mag, min_mag) or isclose(lim_mag, max_mag))

@given(normals())
def test_clamp_mag_negative(vec):
    """ Negative magnitude clamp bounds should raise a ValueError """
    with raises(ValueError):
        vec.clamp_mag(-2, -1)

@given(normals())
def test_clamp_mag_wrong_order(vec):
    """ A ValueError should be raised if min_mag > max_mag """
    with raises(ValueError):
        vec.clamp_mag(2, 1)

@given(normals())
def test_normal_unit_vec(normal):
    """ normal.unit should be the identity function """
    assert isclose(normal, normal.unit)

@given(vectors(), vectors())
def test_dot_matmul(vec_1, vec_2):
    """ The matmul operator should be an alias for the dot product """
    assert isclose(vec_1.dot(vec_2), vec_1 @ vec_2)

@given(vectors(), normals())
def test_project_parallel(vec, onto):
    """ vec.project(onto) should be parallel to onto """
    theta = onto.angle(vec.project(onto))
    assert isclose(math.sin(theta), 0)

@given(vectors(), normals())
def test_project_right_triangle(vec, onto):
    """ vec.project(onto) should form the base of a right triangle """
    projection = vec.project(onto)
    assert isclose(projection.mag_sq + (vec - projection).mag_sq,
                   vec.mag_sq)

@given(vectors(), normals())
def test_project_dist(vec, onto):
    """
    vec.project(onto) should create a second right triangle at the
    perpendicular.
    Vector.dist should demonstrate the properties of this triangle.
    """
    projection = vec.project(onto)
    a = projection.dist(onto)
    b = vec.dist(projection)
    c = vec.dist(onto)
    assert isclose(a ** 2 + b ** 2, c ** 2)

@given(vectors(), normals())
def test_reflect_direction(vec, normal):
    """ The direction of a reflected vector should be reflected """
    reflected = vec.reflect(normal)
    # Use sin to allow reflections from behind
    assert isclose(math.sin(vec.angle(normal)),
                   math.sin(reflected.angle(normal)))

@given(vectors(), normals())
def test_reflect_mag(vec, normal):
    """ vec.reflect(normal) should have the same magnitude as vec """
    reflected = vec.reflect(normal)
    assert isclose(vec.mag, reflected.mag)

@given(st.data())
def test_cross_orthogonality(data):
    """ The cross product should be orthogonal to its operands """
    cls = data.draw(vector_types(min_size=1), label="cls")
    nvecs = cls.dim and cls.dim - 1
    vecs = data.draw(
        st.lists(vectors(), min_size=nvecs, max_size=nvecs),
        label="vecs"
    )

    ortho = cls.cross(*vecs).clamp_mag(1)
    for vec in vecs:
        assert isclose(ortho @ vec.clamp_mag(1), 0)

@given(finite_vector_types(), st.lists(vectors()))
def test_finite_cross_must_have_n_minus_1_operands(cls, vecs):
    assume(len(vecs) != cls.dim - 1)
    with raises(TypeError):
        cls.cross(*vecs)

@given(infinite_vectors(), numbers())
def test_cross_scalar_fails(vec, scalar):
    with raises(TypeError):
        vec.cross(scalar)

@given(vectors())
def test_bool(vec):
    """
    A Vector should be falsy when its magnitude is 0
    Otherwise, the vector should be truthy
    """
    if abs(vec) == 0:
        assert not vec
    else:
        assert vec

# Limit size of vectors so this test doesn't take too long
@given(vectors(max_size=5))
def test_round(vec):
    """
    round(vec) should return the closest grid vector
    A "grid vector" here means a point in the unit hypercube lattice
    that surrounds vec.

    This is a silly, largely pointless test.
    """
    orderings = itertools.product([math.floor, math.ceil],
                                  repeat=len(list(vec)))
    closest = min((type(vec)(func(val) for func, val in zip(funcs, vec))
                   for funcs in orderings),
                  key=vec.dist)
    assert (closest == round(vec) or
            # If round didn't chose the same vector, it should at least
            # have chosen a vector equally close to ours.
            isclose(vec.dist(closest), vec.dist(round(vec))))

@given(vectors())
def test_pickle(vec):
    """ Vector pickling should work """
    import pickle
    assert vec == pickle.loads(pickle.dumps(vec))

@given(vectors())
def test_iter(vec):
    """ iter() should yield all data needed to recreate the vector """
    assert vec.from_iterable(iter(vec)) == vec

@given(vectors())
def test_not_equal_to_iter(vec):
    """ Vectors should not be equal to their corresponding iterator """
    assert vec != iter(vec)

@given(finite_vectors())
def test_finite_iter_all(fin_vec):
    """ iter_all should be equivalent to iter for finite vectors """
    assert list(fin_vec.iter_all()) == list(iter(fin_vec))

@given(infinite_vectors())
def test_infinite_iter_all(inf_vec):
    """ iter_all should be infinite for infinite vectors """
    prefix = list(inf_vec)
    iterator = inf_vec.iter_all()
    assert list(itertools.islice(iterator, len(prefix))) == prefix
    assert (list(itertools.islice(iterator, SIDEWAYS_INFINITY)) == 
            [0] * SIDEWAYS_INFINITY)
