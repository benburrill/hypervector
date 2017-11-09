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


def _zmap(fn, *args):
    return it.starmap(fn, it.zip_longest(*args, fillvalue=0))


def _limit(val, min_val, max_val):
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

    def __new__(mcls, name, bases, namespace, *, dim=None, **kwargs):
        """
        Create a new Vector type, validating the class members

        Unless the keyword class argument ``dim`` is passed, the created
        vector type will be dimensionless.  ``dim`` is not inherited.
        """

        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        if None is not dim < 0:
            raise TypeError("Dimension must be positive")
        else:
            cls.__dim = dim

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
    @functools.lru_cache()
    def zero(cls):
        return cls()

    @functools.lru_cache()
    def __getitem__(cls, dim):
        name = f"{cls.__name__}[{dim}]"

        if cls.dim is not None:
            raise TypeError(f"Can only create {name} from a "
                            f"dimensionless type")

        return type(cls)(name, (cls,), {}, dim=dim)

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

        Any combination of vectors, iterables, and scalar components can
        be combined as positional arguments, but there may only be one
        vector and it must be the last positional argument.

        Named components passed as keyword arguments are masked over any
        positional arguments.
        """

        arg_it, args = iter(args), []
        for arg in arg_it:
            args.append(arg)
            if isinstance(arg, Vector) and arg.dim is None:
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
        return f"<{', '.join(map(str, self))}>"

    @types.DynamicClassAttribute
    def dim(self):
        return type(self).dim

    @classmethod
    def from_iterable(cls, iterable):
        """
        Construct vector from an iterable

        Equivalent to Vector(iterable).  It is recommended to use the
        Vector constructor when constructing vectors from iterables.
        """

        data = tuple(iterable)
        comps = len(data)

        if cls.dim is not None:
            if cls.dim < comps:
                raise TypeError(f"{cls.dim} dimensional vector cannot "
                                f"have {comps} components")
            if cls.dim > comps:
                data += (0,) * (cls.dim - comps)

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

        Effectively the inverse of ``from_spherical``
        """

        comps = reversed(list(self.pad_iter(2)))

        data = []
        opp = next(comps)
        for adj in comps:
            data.append(math.atan2(opp, adj))
            opp = math.hypot(opp, adj)
        # TODO: what should we return for finite vectors?  This is
        # actually a bit problematic for Vector[0]().heading.  Should
        # just return a Vector?  Should we return a Vector[len(data)]?
        # Should the vector we return be dependent on whether dim is
        # None?  Should we not return a Vector at all and instead just
        # return a tuple or something?
        return self.from_iterable(reversed(data))

    @property
    def heading2(self):
        """
        Returns the angular measure of the 2d vector projected onto the
        xy plane by self.

        Effectively the inverse of ``from_polar``
        """

        return math.atan2(self.y, self.x)

    def angle_between(self, other, *, strict=False):
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
            # now, we just limit it to the domain.  It feels a little
            # sketchy to do that, but it's probably fine.
            return math.acos(_limit(cos_theta, -1, 1))
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

    def normalize(self):
        """
        Return a scaled vector with magnitude 1
        Equivalent to self.with_mag(1)
        """
        return self / abs(self)

    def with_mag(self, mag):
        """ Return a scaled vector with magnitude ``mag`` """
        if mag:
            return self * (mag / abs(self))
        # Return a zero vector if we can to avoid dividing by zero
        return type(self)()

    def limit_mag(self, min_mag, max_mag=None):
        """ Limit the magnitude of a vector between two bounds """

        if max_mag is None:
            max_mag = min_mag
            min_mag = 0
        if 0 <= min_mag <= max_mag:
            return self.with_mag(_limit(abs(self), min_mag, max_mag))
        raise ValueError("Magnitude limits must be such that "
                         "0 <= min_mag <= max_mag")

    def dot(self, other):
        """ Dot product of two vectors """
        return sum(self.vectorize(operator.mul)(self, other))

    def __matmul__(self, other):
        """ vec_1 @ vec_2: alias for ``vec_1.dot(vec_2)`` """
        return self.dot(other)

    def project(self, onto):
        """ Project a vector onto the line defined by ``onto`` """
        return onto.with_mag(self.dot(onto.normalize()))

    def reflect(self, normal):
        """ Reflect a vector across a normal """
        return self - 2 * self.project(normal)

    @_ClassyMethod
    def cross(cls, *vecs):
        """
        Cross product

        This generalization of the cross product takes n-1 vectors and
        returns a vector orthogonal to all vectors with up to n non-zero
        components.
        """

        n = len(vecs) + 1
        
        if n > cls.dim:
            raise TypeError(f"{cls.dim} dimensional cross product "
                            f"takes {cls.dim - 1} vectors")

        # Matrix with n-1 column vectors from vecs.  The final column is
        # made up of corresponding unit vectors.  This particular
        # transpose is chosen because it prevents the vectors in that
        # last column from getting multiplied later on.
        mat = list(zip(*[it.islice(vec.pad_iter(), n) for vec in vecs],
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
                return cls()

            det *= mat[idx][idx]

            for row in range(idx + 1, n):
                scale = mat[row][idx] / mat[idx][idx]
                mat[row] = [b - scale * a
                            for a, b in zip(mat[idx], mat[row])]
        return det

    def __bool__(self):
        """ bool(vec): False if ``vec`` is the zero vector """
        return any(self)

    def __eq__(self, other):
        """ Two vectors are equal if all their components are equal """
        return (isinstance(other, type(self)) and 
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

        ``__iter__`` is a method of convenience.  It yields the _data_
        associated with the vector.  It is not meant to represent the
        infinite vector components in their entirety, but is guaranteed
        to include all non-zero components.
        """

        return iter(self.__data)

    # TODO: how much sense does our iteration api, especially pad_iter,
    # make with the advent of finite vectors?  We could potentially make
    # it internal or redesign things.  Or just keep it.  Not sure.
    def pad_iter(self, num=None):
        """
        Padded version of ``__iter__``

        By default, returns an infinite iterator over all vector
        components.

        If ``num`` is passed, at least ``num`` items will be yielded.
        vec.pad_iter(0) is equivalent to iter(vec).
        """
        count = 0
        for comp in self:
            yield comp
            count += 1
        if num is None:
            yield from it.repeat(0)
        else:
            yield from it.repeat(0, num - count)

    @classmethod
    def vectorize(cls, func):
        """ func(0, 0, ...) is assumed to be 0 """
        @functools.wraps(func)
        def vec_func(*vecs):
            if all(isinstance(vec, Vector) and vec.mag == cls.mag for vec in vecs):
                return cls.from_iterable(_zmap(func, *vecs))
            raise TypeError(f"All arguments must be {cls.mag} "
                            f"dimensional vectors")
        return vec_func

    def __add__(self, other):
        """ vec_1 + vec_1 """
        return self.vectorize(operator.add)(self, other)

    def __sub__(self, other):
        """ vec_1 - vec_2 """
        return self.vectorize(operator.sub)(self, other)

    def __mul__(self, other):
        """ vector * scalar """
        if isinstance(other, Vector):
            return NotImplemented
        return self.from_iterable(comp * other for comp in self)

    def __rmul__(self, other):
        """ scalar * vector """
        return self * other

    def __truediv__(self, other):
        """ vector / scalar """
        if isinstance(other, Vector):
            return NotImplemented
        return self.from_iterable(
            comp / other for comp in self.pad_iter(1)
        )

    def __floordiv__(self, other):
        """ vector // scalar """
        if isinstance(other, Vector):
            return NotImplemented
        return self.from_iterable(
            comp // other for comp in self.pad_iter(1)
        )

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
    def _translate_slice(cls, idx):
        """ Translate a slice of infinite space into __data space """
        if cls.dim is not None:
            return idx
        if None is not idx.step < 0:
            raise ValueError("Infinite vector slice step cannot be "
                             "negative")
        if None is not idx.start < 0:
            return slice(0)
        if None is not idx.stop < 0:
            return slice(idx.start, None, idx.step)
        return idx

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
                idx = self._translate_slice(idx)
                return self.from_iterable(self.__data[idx])

            if self.dim is None and idx < 0:
                return 0
            try:
                return self.__data[idx]
            except IndexError:
                if self.dim is None:
                    return 0
                raise
        else:
            return self.from_iterable(self[idx] for idx in indices)

    def __getattr__(self, attr):
        """ Swizzle-style lookups """

        try:
            if len(attr) == 1:
                return self[self.get_name_group(attr).index(attr)]

            group = self.get_name_group(attr)
            return self[group.index(name) for name in attr]
        except IndexError:
            raise AttributeError(f"{type(attr).__name__!r} object "
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

    def __getnewargs__(self):
        return self.__data

    def __getstate__(self):
        return None


Vector2 = Vector[2]
Vector3 = Vector[3]
