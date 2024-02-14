import numpy as np


class FieldOfView:
    """
    Replacement for pgmlink's field of view by simply copy'n'pasting it and adjusting so that it works.
    FIXME: Could be made much nicer in python, using more Numpy functionality...
    """

    def __init__(self, lt, lx, ly, lz, ut, ux, uy, uz):
        """
        Initialize this field of view with lower (l) and upper (u) values for t,x,y,z.
        """
        self.__lowerBound = np.array([lt, lx, ly, lz])
        self.__upperBound = np.array([ut, ux, uy, uz])

    def __dot(self, v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

    def __norm(self, v):
        return np.linalg.norm(v)

    def __cross(self, a, b):
        res = np.zeros_like(a)
        res[0] = a[1] * b[2] - a[2] * b[1]
        res[1] = a[2] * b[0] - a[0] * b[2]
        res[2] = a[0] * b[1] - a[1] * b[0]
        return res

    def __hesse_normal(self, v1, v2):
        temp = self.__cross(v1, v2)
        n = self.__norm(temp)
        return temp / n

    def __abs_distance(self, p1, p2, p3, q):
        u = p2 - p1
        v = p3 - p1

        normal = self.__hesse_normal(u, v)
        w = q - p1
        return np.abs(self.__dot(w, normal))

    def spatial_distance_to_border(self, t, x, y, z, relative=False):
        """
        distance to 6 cuboid planes, in the 2D case where Z=0,
        we take the planes with Z upper bound set to 1.0
        and return the distances to the 4 corresponding planes
        """
        zub = 1.0  # 2D case
        vlen = 4

        if self.__upperBound[3] - self.__lowerBound[3] > 0:  # 3D case
            zub = self.__upperBound[3]
            vlen = 6

        # the eight corners of the fov cube
        c1 = np.array([self.__lowerBound[1], self.__lowerBound[2], self.__lowerBound[3]])
        c2 = np.array([self.__upperBound[1], self.__lowerBound[2], self.__lowerBound[3]])
        c3 = np.array([self.__upperBound[1], self.__upperBound[2], self.__lowerBound[3]])
        c4 = np.array([self.__lowerBound[1], self.__upperBound[2], self.__lowerBound[3]])

        c5 = np.array([self.__lowerBound[1], self.__lowerBound[2], zub])
        c6 = np.array([self.__upperBound[1], self.__lowerBound[2], zub])
        # c7 = np.array([self.__upperBound[1], self.__upperBound[2], zub; # unuse])
        c8 = np.array([self.__lowerBound[1], self.__upperBound[2], zub])

        # distances to the six faces of the cube
        ds = np.zeros(6)
        q = np.array([x, y, z])
        ds[0] = self.__abs_distance(c1, c2, c5, q)
        ds[1] = self.__abs_distance(c2, c3, c6, q)
        ds[2] = self.__abs_distance(c4, c3, c8, q)
        ds[3] = self.__abs_distance(c1, c4, c5, q)
        ds[4] = self.__abs_distance(c1, c2, c4, q)
        ds[5] = self.__abs_distance(c5, c6, c8, q)

        if relative:
            # normalize relative to radius of range
            ds[0] /= self.__upperBound[2] - self.__lowerBound[2]  # / 2)
            ds[1] /= self.__upperBound[1] - self.__lowerBound[1]  # / 2)
            ds[2] /= self.__upperBound[2] - self.__lowerBound[2]  # / 2)
            ds[3] /= self.__upperBound[1] - self.__lowerBound[1]  # / 2)
            ds[4] /= zub - self.__lowerBound[3]  # / 2)
            ds[5] /= zub - self.__lowerBound[3]  # / 2)
        # return *min_element(ds, ds+vlen)
        return np.min(ds[:vlen])

    def getUpperBound(self):
        return self.__upperBound

    def getLowerBound(self):
        return self.__lowerBound
