"""
Voronoi cell decomposition of the surface of the unit sphere.

Based on [Renka1997]_.

.. [Renka1997] **Renka, R. J.** Algorithm 772\\: STRIPACK\\: Delaunay
   triangulation and Voronoi diagram on the surface of a sphere.
   *ACM Transactions on Mathematical Software 23*, 3 (1997), 416--434.
"""
import copy
from itertools import ifilter, takewhile

import numpy
from numpy import tan, sqrt, arctan2, arctan
from numpy import sin, cos
from numpy import linalg
import scipy.spatial


def angular_to_cartesian(theta, phi):
    """
    Cartesian coordinates of a point on the unit sphere.
    Also works if *theta* and *phi* are same-sized arrays.
    """
    return array([sin(theta) * cos(phi),
                  sin(theta) * sin(phi),
                  cos(theta)])


def origin():
    """
    The origin in cartesian coordinates.
    """

    # indices: [axis, point]
    return numpy.zeros((3, 1))


def random_point():
    """
    A point in angular coordinates generated randomly
    from a distribution that is uniform over the surface of the unit sphere.
    """
    theta = numpy.arccos(2. * numpy.random.random() - 1)
    phi = 2 * numpy.pi * numpy.random.random()
    return theta, phi


def random_points(N, condition=None):
    """
    Random points in angular coordinates
    distributed uniformly over the surface of the unit sphere.

    :arg N: the number of required points
    :arg condition: the condition on :math:`(\\theta, \\phi)`
                    that the points must satisfy
    """

    def stream():
        """ An infinite stream of random points. """
        while True:
            yield random_point()

    if condition is None:
        # approve unconditionally
        indexed_points = enumerate(stream())
    else:
        indexed_points = enumerate(ifilter(condition, stream()))

    points = list(takewhile(lambda i, point: i < N, indexed_points))
    return (numpy.array([theta for _, (theta, _) in points]),
            numpy.array([phi for _, (_, phi) in points]))


def great_circle_distance(A, B):
    """
    The shortest distance between two points *A* and *B*
    expressed in cartesian coordinates
    measured along the surface of the unit sphere.
    For a unit sphere, this distance is just the angle the
    two points make at the origin. Accepts arrays of points.
    """
    AdotB = numpy.einsum('...i,...i', A, B)
    AcrossB = numpy.cross(A, B)
    last_axis = len(AcrossB.shape) - 1
    return arctan2(linalg.norm(AcrossB, axis=last_axis), AdotB)


def convex_hull(points):
    """
    The `convex hull <https://en.wikipedia.org/wiki/Convex_hull>`_
    of cartesian *points* on the unit sphere.
    """

    # SciPy requires the array of points to have indices [point, axis]
    # so we have to transpose our array
    return scipy.spatial.ConvexHull(points.T)


def is_origin_inside(points):
    """
    Whether the origin is inside the hull formed by the points.
    If not, they must all be on one hemisphere.
    We do not try to construct a reference frame based on the locations
    of such a problematic distribution of points.
    """

    # SciPy requires the array of points to have indices [point, axis]
    # so we have to transpose our arrays
    hull = scipy.spatial.Delaunay(points.T)
    return numpy.all(hull.find_simplex(origin().T) >= 0)


def Delaunay_triangulation(hull):
    """
    The `Delaunay triangulation
    <https://en.wikipedia.org/wiki/Delaunay_triangulation>`_ of the
    convex hull. This triangulation is dual to the
    Voronoi cell decomposition.
    """
    return hull.points[hull.simplices]


def Delaunay_circumcenters(triangles):
    """
    Circumcenters of the triangles from the Delaunay triangulation.
    """
    na = numpy.newaxis

    sign = numpy.sign(linalg.det(triangles))

    A = triangles[:, 1, :] - triangles[:, 0, :]
    B = triangles[:, 2, :] - triangles[:, 0, :]
    C = numpy.cross(A, B)

    return sign[:, na] * (C / linalg.norm(C, axis=1)[:, na])


def Delaunay_circumradii(triangles, centers):
    """
    Radii of the triangles from the Delaunay triangulation.
    Serves as a check.
    """
    N, _, _ = triangles.shape

    def radius(triangle, center):
        """
        Radius of the circumcircle as measured on the surface
        of the sphere.
        """
        A = great_circle_distance(triangle[0], center)
        B = great_circle_distance(triangle[1], center)
        C = great_circle_distance(triangle[2], center)

        assert numpy.allclose(A, B)
        assert numpy.allclose(B, C)
        assert numpy.allclose(C, A)

        return A

    return numpy.array([radius(triangles[i], centers[i]) for i in range(N)])


def simplices_around_vertices(hull):
    """
    Collect the simplices around the vertices. These
    are the combinatorial representation of the points
    and the triangles of the Delaunay triangulation.
    """
    # one list for each vertex
    # starting empty
    result = {i: [] for i in range(len(hull.vertices))}

    # collect simplices into the lists corresponding
    # to their member vertices
    for simplex in range(len(hull.simplices)):
        for vertex in hull.simplices[simplex]:
            result[vertex].append(simplex)

    return result


def Voronoi_cell(hull, centers, vertex, original_fan):
    """
    Form a Voronoi cell by connecting the circumcenters
    of the Delaunay triangles in order.

    :arg hull: the convex hull of the set of generator points
    :arg centers: the list of circumcenters of Delaunay triangles
    :arg vertex: the vertex around which the cell is to be formed
    :arg original_fan: the collection of simplices, that is, faces
                   around this *vertex*
    """
    # make a copy so that the original does not get mutated
    fan = copy.deepcopy(original_fan)

    # start by moving the center
    # of the first one in the list of adjacent faces
    # to the resulting polygon
    simplex = fan.pop(0)
    result = [centers[simplex]]

    # find the vertices of this face
    simplex_vertices = hull.simplices[simplex]

    # there should only be two other vertices on this face
    # other than our given vertex

    # pick one of them and mark it 'known'
    # the other one will be common to the next simplex to consider
    known_vertex, common_vertex = [x for x in simplex_vertices if x != vertex]

    while fan:
        # the collection of faces is not exhausted yet
        assert known_vertex in simplex_vertices

        known_vertex_index = list(simplex_vertices).index(known_vertex)

        # next simplex to consider
        # it is the simplex which is opposite to the known vertex
        simplex = hull.neighbors[simplex][known_vertex_index]

        assert simplex in fan

        # now move its center to our resulting polygon
        fan.remove(simplex)
        result.append(centers[simplex])

        # and repeat the process
        simplex_vertices = hull.simplices[simplex]
        known_vertex = common_vertex

        # of the three vertices of the simplex
        # one should be our given vertex
        # and one should already have been processed
        remaining = [x for x in hull.simplices[simplex]
                     if x != vertex and x != known_vertex]

        assert len(remaining) == 1
        common_vertex = remaining[0]

    return numpy.array(result)


def Voronoi_decomposition(points):
    """
    The `Voronoi decomposition
    <https://en.wikipedia.org/wiki/Voronoi_diagram>`_
    of the surface of the unit sphere
    around a set of points called generators.
    """
    assert points.shape[0] == 3

    # the convex hull of the points
    hull = convex_hull(points)

    # Delaunay triangulation of the hull
    triangles = Delaunay_triangulation(hull)

    # circumcenters of the triangles
    # these points are dual to the original points, that is, generators
    # in other words they represent faces formed by the generators
    centers = Delaunay_circumcenters(triangles)

    # collect the simplices, that is, faces, around
    # the original generator points
    fans = simplices_around_vertices(hull)

    # connect the circumcenters on these faces in order
    # and thus form the Voronoi cells
    return {vertex: Voronoi_cell(hull, centers, vertex, fan)
            for vertex, fan in fans.iteritems()}


def decompose_polygon(points):
    """
    Decomposes a polygon into triangles.
    """
    N, _ = points.shape

    for i in range(1, N - 1):
        yield numpy.array([points[0], points[i], points[i + 1]])


def triangle_area(triangle):
    """
    Area of a spherical triangle.
    """
    # sides of the triangle
    a = great_circle_distance(triangle[0], triangle[1])
    b = great_circle_distance(triangle[0], triangle[2])
    c = great_circle_distance(triangle[1], triangle[2])

    # it may happen that the triangle is degenerate
    # for the rare case where a fourth generator just
    # touches the circumcircle
    assert a >= 0.
    assert b >= 0.
    assert c >= 0.

    s = (a + b + c) / 2.

    # does not quite work for extra large polygons
    # where area is ambiguous
    try:
        return 4. * arctan(sqrt(tan(s / 2.) *
                                tan((s - a) / 2.) *
                                tan((s - b) / 2.) *
                                tan((s - c) / 2.)))
    except FloatingPointError:
        # floating point weirdness
        return 0.


def polygon_area(points):
    """
    The area of the polygon on the surface of the unit sphere
    formed by *points*.
    """
    def area(triangles):
        """
        Area of a spherical triangle. Vectorized version of
        :func:`triangle_area`.
        """
        # sides of the triangle
        sides = great_circle_distance(triangles,
                                      numpy.roll(triangles, 1, axis=1))

        assert numpy.all(sides >= 0.)

        # s = (a + b + c) / 2.
        s = (numpy.sum(sides, axis=1) / 2.)

        # tan(s / 2) * tan((s - a) / 2) * tan((s - b) / 2) * tan((s - c) / 2)
        product = (tan(s / 2.) *
                   numpy.prod(tan((s[:, numpy.newaxis] - sides) / 2.), axis=1))

        try:
            return 4. * arctan(sqrt(product))
        except FloatingPointError:
            # floating point weirdness

            def individual(prod):
                """
                Area of an individual triangle.
                """
                try:
                    return 4. * arctan(sqrt(prod))
                except FloatingPointError:
                    return 0.

            return numpy.array([individual(prod) for prod in product])

    triangles = numpy.array(list(decompose_polygon(points)))
    return area(triangles).sum()


def areas(cells):
    """
    The areas of the Voronoi cells. These should sum up to
    :math:`4\\pi`.
    """
    return numpy.array([polygon_area(cells[i])
                        for i in range(len(cells.keys()))])
