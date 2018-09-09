"""
hypervector -- Simple, general, pure Python vectors

Issue tracker: https://github.com/benburrill/hypervector/issues
"""

__version__ = "0.0.1"
__author__ = "Ben Burrill"
__license__ = "Public Domain"

__all__ = ["Vector", "Vector2", "Vector3"]


import math
import operator
import types
import functools
import itertools as it


def _crush(iterable):
    for item in iterable:
        try: yield from item
        except TypeError:
            yield item


def _zmap(fn, *iterables):
    return it.starmap(fn, it.zip_longest(*iterables, fillvalue=0))


def _zpad(iterable, *, to=None):
    count = 0
    for comp in iterable:
        yield comp
        count += 1
    if to is None:
        yield from it.repeat(0)
    else:
        yield from it.repeat(0, to - count)


def _clamp(val, min_val, max_val):
    return min(max(val, min_val), max_val)


# TODO: better name?
class _ClassyMethod(classmethod):
    """
    A combination between a classmethod and a normal method

    cls.meth(*args) -> meth(cls, *args)
    obj.meth(*args) -> meth(type(obj), obj, *args)
    """

    def __get__(self, obj, cls=None):
        method = super().__get__(obj, cls)

        # method called from class
        if obj is None:
            return method
        return types.MethodType(method, obj)


########################################################################


class VectorType(type):
    """
    Metaclass magic for vectors

    Types must provide a classmethod ``get_name_group(names)`` which
    returns a string representing the best name group to represent all
    component names in ``names``.  If no such name group exists,
    ``IndexError`` must be raised.
    """

    def __new__(mcls, name, bases, namespace, *, dim=None, _root=None):
        """
        Create a new Vector type, validating the class members

        Unless the keyword class argument ``dim`` is passed, the created
        vector type will be dimensionless.  ``dim`` is not inherited.
        """

        cls = super().__new__(mcls, name, bases, namespace)

        if None is not dim < 0:
            raise TypeError("Dimension must be positive")
        else:
            cls.__dim = dim
        
        if _root is None:
            cls._root = cls
            cls.__derived = {cls.__dim: cls}
        else:
            cls._root = _root

        for name in dir(cls):
            try:
                group = cls.get_name_group(name)
                raise AttributeError(
                    f"Attribute conflict: collision between name group "
                    f"{group!r} and class member {name!r}"
                )
            except IndexError: pass
        return cls

    @property
    def dim(cls):
        return cls.__dim

    @property
    def zero(cls):
        return cls()

    def _make_dim(cls, name, dim):
        try:
            return cls._root.__derived[dim]
        except KeyError:
            derived = type(cls)(name, (cls._root,), {
                "__slots__": []
            }, dim=dim, _root=cls._root)

            cls._root.__derived[dim] = derived
            return derived
            
    def __getitem__(cls, dim):
        return cls._make_dim(f"{cls._root.__name__}[{dim}]", dim)

    def __getattr__(cls, attr):
        """ Named unit vectors as attributes """

        if len(attr) == 1:
            try:
                return cls.from_mapping({
                    cls.get_name_group(attr).index(attr): 1
                })
            except IndexError: pass
        raise AttributeError(f"type object {cls.__name__!r} "
                             f"has no attribute {attr!r}")


class Vector(metaclass=VectorType):
    """ Infinite-dimensional vector type """

    __slots__ = ["__data", "__frozen"]

    # TODO: Not sure I want Vector2().rg to work.  Some name groups only
    # make sense for certain dimensions.  However, most ways I can think
    # of to add such restrictions to the name_groups are either ugly or
    # overcomplicated.
    name_groups = ["xyzw", "ijk", "rgba"]

    @classmethod
    def get_name_group(cls, names):
        """ Return the relevant name group string if one exists """
        names = set(names)
        for group in cls.name_groups:
            group = group[:cls.dim]
            if names.issubset(set(group)):
                return group
        raise IndexError(f"No matching name group for {names}")

    def __new__(cls, *args, truncate=False, **masks):
        """
        Construct a vector

        Vector() is the zero vector.

        Vectors, iterables, and scalar components can be combined as
        positional arguments.
        
        The truncate argument, when True, truncates the number of values
        if they exceed the vector's dimensionality.  Since infinite
        vectors have infinite values, they cannot be passed to a finite
        vector constructor unless truncate is True.

        Additionally, unless truncate is True, infinite vectors can only
        be used as the final argument because values that follow cannot
        be represented as they would follow an infinite sequence of 0s.

        Named components passed as keyword arguments are masked over any
        positional arguments.
        """

        arg_it, args = iter(args), []
        for arg in arg_it:
            args.append(arg)
            if isinstance(arg, cls._root) and arg.dim is None:
                if not truncate and cls.dim is not None:
                    raise TypeError("Infinite vector must be truncated")
                if not truncate and list(arg_it):
                    raise TypeError("Cannot append to infinite vector")
                break

        comp_it = _crush(args)
        
        if truncate:
            comp_it = it.islice(comp_it, cls.dim)

        if masks:
            try:
                group = cls.get_name_group(masks)
            except IndexError:
                raise TypeError(f"Invalid mask group: {set(masks)}")
            return cls.from_mapping({
                group.index(name): masks[name]
                for name in masks
            }, base=comp_it)
        else:
            return cls.from_iterable(comp_it)

    def __repr__(self):
        name = type(self).__name__
        return f"{name}({', '.join(map(repr, self))})"

    def __str__(self):
        # TODO: Maybe indicate infinite vectors?
        return f"<{', '.join(map(str, self))}>"

    @types.DynamicClassAttribute
    def dim(self):
        return type(self).dim

    @classmethod
    def from_iterable(cls, iterable):
        """
        Construct vector from an iterable

        Equivalent to ``Vector(iterable)``.  However, it is recommended
        to use the Vector constructor when constructing vectors from
        iterables.  ``from_iterable`` is mostly for internal use.
        """

        data = tuple(iterable)
        comps = len(data)

        if cls.dim is not None:
            if cls.dim > comps:
                data += (0,) * (cls.dim - comps)
            elif cls.dim < comps:
                raise TypeError(f"{cls.dim} dimensional vector cannot "
                                f"have {comps} components")

        self = object.__new__(cls)
        self.__data = data
        self.__frozen = True
        return self

    @classmethod
    def from_mapping(cls, mapping, *, base=()):
        """
        Construct vector from a mapping or an iterable of kv-pairs

        May optionally take a base iterable which the map writes over.
        """

        mapping = dict(mapping)
        dims = max(mapping, default=0) + 1
        base = list(base)
        data = base + [0] * (dims - len(base))

        for dim in range(dims):
            try:
                data[dim] = mapping.pop(dim)
            except KeyError: pass

        if mapping:
            raise ValueError(f"Invalid keys: {set(mapping.keys())}")

        return cls.from_iterable(data)

    @classmethod
    def from_spherical(cls, radius, direction):
        """
        Creates a vector from a hyperspherical direction vector, or
        iterable of angles.

        I based this off the N-Spherical coordinates from
        https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates,
        which may not give the result you expect if you are used to
        spherical coordinates.  However, this is easy to fix with
        ``Vector.from_spherical(...).yzx``
        """

        data = []
        coeff = radius

        for ang in direction:
            data.append(coeff * math.cos(ang))
            coeff *= math.sin(ang)

        data.append(coeff)
        return cls.from_iterable(data)

    @classmethod
    def from_polar(cls, radius, theta):
        """ Creates a vector from 2d polar coordinates """
        return cls.from_spherical(radius, [theta])

    @property
    def heading(self):
        """
        Returns a vector in hyperspherical space representing the
        direction of self.

        The ``heading`` property consistently returns a vector, even in
        the two-dimensional case (where it returns a Vector[1]).  If a
        scalar is desired, use ``heading2`` instead.

        Effectively the inverse of ``from_spherical``
        """

        comps = reversed(list(_zpad(self, to=2)))

        data = []
        opp = next(comps)
        for adj in comps:
            data.append(math.atan2(opp, adj))
            opp = math.hypot(opp, adj)

        if self.dim is None:
            cls = type(self)
        else:
            cls = type(self)[len(data)]

        return cls.from_iterable(reversed(data))

    @property
    def heading2(self):
        """
        Returns the angular measure of the 2d vector projected onto the
        xy plane by self.

        Effectively the inverse of ``from_polar``
        """

        return math.atan2(self.y, self.x)

    def angle(self, other, *, strict=False):
        """
        Calculates the angle between two vectors.

        By default, if one or more vectors have a magnitude of 0, the
        angle between them is considered to be 0.  If this behavior is
        not desired, pass in the keyword argument ``strict=True`` and
        the function will raise ``ZeroDivisionError``
        """

        try:
            cos_theta = self.dot(other) / abs(self) / abs(other)
            # Due to floating-point shenanigans, we sometimes end up
            # with a value slightly outside the domain of acos.  For
            # now, we just clamp it to the domain.  It feels a little
            # sketchy to do that, but it's probably fine.
            return math.acos(_clamp(cos_theta, -1, 1))
        except ZeroDivisionError:
            if not strict:
                return 0
            raise

    @property
    def mag_sq(self):
        """ Magnitude squared """
        return sum(comp ** 2 for comp in self)

    @property
    def mag(self):
        """ Magnitude """
        return self.mag_sq ** 0.5

    def __abs__(self):
        """ abs(vec): alias for ``vec.mag`` """
        return self.mag

    def dist(self, other):
        """ Distance between two coordinate vectors """
        return abs(self - other)

    @property
    def unit(self):
        """
        Return a scaled vector with magnitude 1 in the direction of self
        Equivalent to self.with_mag(1)
        """
        return self / abs(self)

    def with_mag(self, mag):
        """ Return a scaled vector with magnitude ``mag`` """
        if mag:
            return self * (mag / abs(self))
        # Return a zero vector if we can to avoid needlessly dividing.
        return type(self).zero

    def clamp_mag(self, min_mag, max_mag=None):
        """ Limit the magnitude of a vector between two bounds """

        if max_mag is None:
            max_mag = min_mag
            min_mag = 0
        if 0 <= min_mag <= max_mag:
            return self.with_mag(_clamp(abs(self), min_mag, max_mag))
        raise ValueError("Magnitude bounds must be such that "
                         "0 <= min_mag <= max_mag")

    @_ClassyMethod
    def cross(cls, *vecs):
        """
        Cross product

        This generalization of the cross product takes n-1 vectors and
        returns a vector orthogonal to all vectors with up to n non-zero
        components.
        """

        n = len(vecs) + 1

        if None is not cls.dim != n:
            raise TypeError(f"The {cls.dim} dimensional cross product "
                            f"takes {cls.dim - 1} vectors")

        if not all(isinstance(vec, cls._root) for vec in vecs):
            raise TypeError(f"Cross product operands must be instances "
                            f"of {cls._root.__name__!r}")
        
        # Matrix with n-1 column vectors from vecs.  The final column is
        # made up of corresponding unit vectors.  This particular
        # transpose is chosen because it prevents the vectors in that
        # last column from getting multiplied later on.
        mat = list(zip(*[it.islice(_zpad(vec), n) for vec in vecs],
                       [cls.from_mapping({c: 1}) for c in range(n)]))

        # The determinant of this matrix will be the cross product
        det = 1
        for idx in range(n):
            if not mat[idx][idx]:
                pivot = max(range(idx, n),
                            key=lambda row: abs(mat[row][idx]))
                mat[idx], mat[pivot] = mat[pivot], mat[idx]
                det = -det

            if not mat[idx][idx]:
                return cls.zero

            det *= mat[idx][idx]

            for row in range(idx + 1, n):
                scale = mat[row][idx] / mat[idx][idx]
                mat[row] = [b - scale * a
                            for a, b in zip(mat[idx], mat[row])]
        return det

    @_ClassyMethod
    def dot(cls, self, other):
        """ Dot product of two vectors """
        if not isinstance(other, cls._root):
            raise TypeError(f"Dot product operands must be instances "
                            f"of {cls._root.__name__!r}")
        
        return sum(_zmap(operator.mul, self, other))

    def __matmul__(self, other):
        """ vec_1 @ vec_2: alias for ``vec_1.dot(vec_2)`` """
        return self.dot(other)

    def project(self, onto):
        """ Project a vector onto the line defined by ``onto`` """
        return onto.with_mag(self.dot(onto.unit))

    def reflect(self, normal):
        """ Reflect a vector across a normal """
        return self - 2 * self.project(normal)

    def __bool__(self):
        """ bool(vec): False if ``vec`` is the zero vector """
        return any(self)

    def __eq__(self, other):
        """ Two vectors are equal if all their components are equal """
        # The first two conditions SHOULD always be equivalent to
        # type(self) == type(other), but this is more flexible to silly
        # metaclass bugs.
        return (isinstance(other, type(self)._root) and
                self.dim == other.dim and
                all(_zmap(operator.eq, self, other)))

    def __hash__(self):
        result = 0
        for comp in self:
            result ^= hash(comp) - hash(0)
        return result

    def __iter__(self):
        """
        Iterate over vector components

        ``__iter__`` yields only the relevant vector components in the
        case of infinite vectors -- the infinite components are
        truncated, but it is guaranteed to include all non-zero
        components.  Use ``iter_all`` if a representation of all vector
        components is desired for some reason.
        """

        return iter(self.__data)

    def iter_all(self):
        """
        For finite vectors, ``iter_all`` is equivalent to ``__iter__``.
        For infinite vectors, the iterator ``iter_all`` returns will be
        infinite, including all the infinite components of the vector.
        """

        yield from self
        if self.dim is None:
            yield from it.repeat(0)

    @staticmethod
    def _dimkey(vec):
        return (vec.dim is None, vec.dim)

    @classmethod
    def vectorize(cls, func):
        """
        The vectorized function returns a vector directly related to
        ``cls`` with the dimensions of the largest vector passed as an
        argument.

        When the vectorized function is passed an infinite vector,
        ``func(0, 0, ...)`` is assumed to be ``0``.
        """

        @functools.wraps(func)
        def vec_func(*vecs):
            if not all(isinstance(vec, cls._root) for vec in vecs):
                raise TypeError(f"All arguments must be instances of "
                                f"{cls._root.__name__!r}")
            container = cls[max(vecs, key=cls._dimkey).dim]
            return container.from_iterable(_zmap(func, *vecs))
        return vec_func

    def __add__(self, other):
        """ vec_1 + vec_1 """
        return self.vectorize(operator.add)(self, other)

    def __sub__(self, other):
        """ vec_1 - vec_2 """
        return self.vectorize(operator.sub)(self, other)

    def __mul__(self, other):
        """ vector * scalar """
        if isinstance(other, type(self)._root):
            return NotImplemented
        return self.from_iterable(comp * other for comp in self)

    def __rmul__(self, other):
        """ scalar * vector """
        return self.__mul__(other)

    def __truediv__(self, other):
        """ vector / scalar """
        if isinstance(other, type(self)._root):
            return NotImplemented

        # Ensure ZeroDivisionError is raised when necessary, even when
        # there is nothing to iterate over.  0 seems to be the best
        # choice of numerator, since in the case of infinite vectors we
        # are simulating infinite divisions of 0 / other
        if not self.__data:
            0 / other

        return self.from_iterable(comp / other for comp in self)

    def __floordiv__(self, other):
        """ vector // scalar """
        if isinstance(other, type(self)._root):
            return NotImplemented

        if not self.__data:
            0 / other

        return self.from_iterable(comp // other for comp in self)

    def __neg__(self):
        """ -vector """
        return self.vectorize(operator.neg)(self)

    def __pos__(self):
        """ +vector """
        return self.vectorize(operator.pos)(self)

    def __round__(self, n=None):
        """ Rounds the vector, snapping it to the grid defined by n """
        return self.vectorize(functools.partial(round, ndigits=n))(self)

    @classmethod
    def _translate_slice(cls, sl):
        """
        Translate a slice into __data space, taking into account
        infinite vectors.  Returns an additional boolean that is True if
        the slice represents an infinite number of values.
        """
        if cls.dim is not None:
            return sl, False
        if None is not sl.step < 0:
            raise ValueError("Infinite vector slice step cannot be "
                             "negative")
        if None is not sl.start < 0:
            return slice(0), sl.stop is None
        if None is not sl.stop < 0:
            return slice(sl.start, None, sl.step), True
        return sl, sl.stop is None

    def __getitem__(self, idx):
        """
        idx is an index, a slice, or an iterable of indices.

        The use of nested iterables or slice-iterable combinations as
        indices is undefined behavior.
        """

        try:
            indices = tuple(idx)
        except TypeError:
            if isinstance(idx, slice):
                sl, inf = self._translate_slice(idx)
                data = self.__data[sl]
                dim = len(data) if not inf else None
                return type(self)[dim].from_iterable(data)
            if self.dim is None and idx < 0:
                return 0
            try:
                return self.__data[idx]
            except IndexError:
                if self.dim is None:
                    return 0
                raise
        else:
            return type(self)[len(indices)].from_iterable(
                self[idx] for idx in indices
            )

    def __getattr__(self, attr):
        """
        Swizzle-style lookups

        In multi-character attributes, underscores may be used as
        placeholders for 0s.
        """

        try:
            if len(attr) == 1:
                return self[self.get_name_group(attr).index(attr)]

            group = self.get_name_group(attr.replace("_", ""))
            return type(self)[len(attr)].from_iterable(
                self[group.index(name)] if name != "_" else 0
                for name in attr
            )
        except IndexError:
            raise AttributeError(f"{type(self).__name__!r} object "
                                 f"has no attribute {attr!r}") from None

    def _check_frost(self):
        try:
            if self.__frozen:
                raise TypeError(f"{type(self).__name__!r} object is "
                                f"frozen and cannot be changed")
        except AttributeError: pass

    def __setattr__(self, attr, value):
        self._check_frost()
        return super().__setattr__(attr, value)

    def __delattr__(self, attr):
        self._check_frost()
        return super().__delattr__(attr)

    @classmethod
    def _make_vec(cls, dim, iterable):
        return cls[dim].from_iterable(iterable)

    def __reduce__(self):
        return (type(self)._root._make_vec, (self.dim, self.__data))


Vector2 = Vector._make_dim("Vector2", 2)
Vector3 = Vector._make_dim("Vector3", 3)
