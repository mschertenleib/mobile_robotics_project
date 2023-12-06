from image_processing import *


class Edge(object):
    __slots__ = ['vertex', 'length']

    def __init__(self, vertex: int, length: float):
        self.vertex: int = vertex
        self.length: float = length


class Graph:
    SOURCE = -1
    TARGET = -2

    def __init__(self):
        self.vertices: list[np.ndarray] = []
        self.adjacency: list[list[Edge]] = []
        self.to_prev: list[np.ndarray] = []
        self.to_next: list[np.ndarray] = []


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """
    Checks for an intersection between the open segment ]a---b[ and closed segments [c[i]---d[i]]
    """

    # NOTE: the explicit computations of cross and dot products here are for performance reasons. Calling np.cross() is
    # about 2x slower than doing the product by hand, at least when done on arrays of 2D vectors.
    # See tests/benchmark_intersection.py

    ab = b - a
    cd = d - c
    ac = c - a
    cb_cd = (b[0] - c[:, 0]) * cd[:, 1] - (b[1] - c[:, 1]) * cd[:, 0]
    ca_cd = (a[0] - c[:, 0]) * cd[:, 1] - (a[1] - c[:, 1]) * cd[:, 0]
    ad_ab = (d[:, 0] - a[0]) * ab[1] - (d[:, 1] - a[1]) * ab[0]
    ac_ab = ac[:, 0] * ab[1] - ac[:, 1] * ab[0]

    # Detects intersections between non-colinear segments
    if np.any(
            (((cb_cd > 0) & (ca_cd < 0)) | ((cb_cd < 0) & (ca_cd > 0))) & (
                    ((ad_ab >= 0) & (ac_ab <= 0)) | ((ad_ab <= 0) & (ac_ab >= 0)))):
        return True

    # Detects intersections between colinear segments
    colinear_mask = (cb_cd == 0) & (ca_cd == 0) & (ad_ab == 0) & (ac_ab == 0)
    ac_dot_ab = ac[:, 0] * ab[0] + ac[:, 1] * ab[1]
    ab_dot_cd = ab[0] * cd[:, 0] + ab[1] * cd[:, 1]
    ab_dot_ab = ab[0] * ab[0] + ab[1] * ab[1]
    mask_intersect_dir_same = (ab_dot_cd > 0) & (ac_dot_ab + ab_dot_cd > 0) & (ac_dot_ab - ab_dot_ab < 0)
    mask_intersect_dir_opposite = (ab_dot_cd < 0) & (ac_dot_ab > 0) & (ac_dot_ab + ab_dot_cd - ab_dot_ab < 0)
    return np.any(colinear_mask & (mask_intersect_dir_same | mask_intersect_dir_opposite))


def segment_intersects_contours(pt1: np.ndarray, pt2: np.ndarray, contours: list[np.ndarray]) -> bool:
    """
    Checks for an intersection between the open segment ]pt1---pt2[ and all contours
    """
    for contour in contours:
        if segments_intersect(pt1, pt2, contour[:-1], contour[1:]):
            return True
        if segments_intersect(pt1, pt2, np.array([contour[-1]]), np.array([contour[0]])):
            return True
    return False


def extract_contours(obstacle_mask: np.ndarray, epsilon: float) -> list[list[np.ndarray]]:
    """
    Get a list of contour regions from the given obstacle_mask, using the given epsilon for polygon approximation.
    Each contour region is a list of contours, where the first one is the outline of a region of free space,
    and subsequent ones are outlines of obstacles enclosed within that region (or, equivalently, holes in the region).
    The orientation of the region outline is negative (left-hand rule), and the orientation of the outline of
    holes within the region is positive.
    """

    # Inverting the mask means we get the contours of free regions instead of the contours of obstacles
    inverted_mask = 1 - obstacle_mask

    # NOTE: we assume the orientation of retrieved contours is as explained above. This is not explicitly stated in the
    # OpenCV documentation, but seems to be the case
    raw_contours, raw_hierarchy = cv2.findContours(inverted_mask, mode=cv2.RETR_CCOMP,
                                                   method=cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = [cv2.approxPolyDP(contour, epsilon=epsilon, closed=True) for contour in raw_contours]

    # Group contours following their hierarchy
    # NOTE: this assumes a parent appears before its children in the hierarchy
    regions = [[] for _ in range(len(approx_contours))]
    for i in range(len(approx_contours)):
        # Only keep contour approximations that have at least 3 vertices
        if len(approx_contours[i]) >= 3:
            parent = raw_hierarchy[0][i][3]
            # If parent is -1, the contour is the top-level outline of its own region.
            # Else, it is the outline of a hole in its parent region.
            region = i if parent == -1 else parent
            regions[region].append(np.squeeze(approx_contours[i]))

    # Remove empty groups
    regions = [group for group in regions if len(group) > 0]

    return regions


def extract_convex_vertices(contours: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    vertices = []
    to_prev = []
    to_next = []
    for contour in contours:
        for i in range(len(contour)):
            to_prev_i = contour[i - 1 if i > 0 else len(contour) - 1] - contour[i]
            to_next_i = contour[(i + 1) % len(contour)] - contour[i]
            if np.cross(to_prev_i, to_next_i) <= 0:
                vertices.append(contour[i])
                to_prev.append(to_prev_i)
                to_next.append(to_next_i)
    return vertices, to_prev, to_next


def extract_static_adjacency(contours: list[np.ndarray], vertices: list[np.ndarray], to_prev: list[np.ndarray],
                             to_next: list[np.ndarray]) -> list[list[Edge]]:
    adjacency = [[] for _ in range(len(vertices))]
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):

            # Discard vertices for which the previous and next one lie on opposite sides of the edge
            edge = vertices[j] - vertices[i]
            sin_prev_i = np.cross(edge, to_prev[i])
            sin_next_i = np.cross(edge, to_next[i])
            if (sin_prev_i < 0 or sin_next_i < 0) and (sin_prev_i > 0 or sin_next_i > 0):
                continue
            sin_prev_j = np.cross(edge, to_prev[j])
            sin_next_j = np.cross(edge, to_next[j])
            if (sin_prev_j < 0 or sin_next_j < 0) and (sin_prev_j > 0 or sin_next_j > 0):
                continue

            # Discard edges that intersect contours, but not for edges between connected vertices
            is_j_prev_i = np.all(to_prev[i] == edge)
            is_j_next_i = np.all(to_next[i] == edge)
            is_i_prev_j = np.all(to_prev[j] == -edge)
            is_i_next_j = np.all(to_next[j] == -edge)
            are_pts_connected = (is_j_prev_i or is_j_next_i) and (is_i_prev_j or is_i_next_j)
            if not are_pts_connected and segment_intersects_contours(vertices[i], vertices[j], contours):
                continue

            edge_length = np.linalg.norm(edge)
            adjacency[i].append(Edge(vertex=j, length=edge_length))
            adjacency[j].append(Edge(vertex=i, length=edge_length))

    return adjacency


def extract_dynamic_edges(regions: list[list[np.ndarray]], vertices: list[np.ndarray], to_prev: list[np.ndarray],
                          to_next: list[np.ndarray], point: np.ndarray) -> list[Edge]:
    # NOTE: we need to check for an intersection with all contours from all regions, because we consider all vertices
    # of the graph, not only those in the same region as the point. If we wanted to consider only the latter (which we
    # could), we would need a way to extract the subset of vertices from the graph corresponding to our region.
    # Alternatively, and perhaps more intuitively, building one graph per region would give the same benefits and might
    # be interesting. This however is fundamentally just an optimization over what we already have here, and goes beyond
    # the scope of what is needed for this project.
    all_contours = [contour for region in regions for contour in region]

    edges = []
    for i in range(len(vertices)):

        # Discard vertices for which the previous and next one lie on opposite sides of the edge
        edge = vertices[i] - point
        sin_prev_i = np.cross(edge, to_prev[i])
        sin_next_i = np.cross(edge, to_next[i])
        if (sin_prev_i < 0 or sin_next_i < 0) and (sin_prev_i > 0 or sin_next_i > 0):
            continue

        # Discard edges that intersect contours, but not for edges between connected vertices
        is_pt_prev_i = np.all(to_prev[i] == -edge)
        is_pt_next_i = np.all(to_next[i] == -edge)
        are_pts_connected = is_pt_prev_i or is_pt_next_i
        if not are_pts_connected and segment_intersects_contours(point, vertices[i], all_contours):
            continue

        edge_length = np.linalg.norm(edge)
        edges.append(Edge(vertex=i, length=edge_length))

    return edges


def build_graph(regions: list[list[np.ndarray]]) -> Graph:
    graph = Graph()

    # For the current region being processed, keep track of the index of its first vertex in the graph
    region_first_vertex = 0

    # Process each region separately, because each region essentially has its own local visibility graph, isolated from
    # other regions. This also improves performance, by limiting the various computations on vertices/edges to only the
    # ones contained in the region.
    for region_contours in regions:
        vertices, to_prev, to_next = extract_convex_vertices(region_contours)
        adjacency = extract_static_adjacency(region_contours, vertices, to_prev, to_next)

        # NOTE: += on lists concatenates them, this is not an actual addition
        graph.vertices += vertices
        graph.to_prev += to_prev
        graph.to_next += to_next
        # Add the region's first vertex index to each vertex index in the adjacency list, before appending it to the
        # graph's global adjacency list
        graph.adjacency += [[Edge(edge.vertex + region_first_vertex, edge.length) for edge in edges] for edges in
                            adjacency]

        region_first_vertex += len(vertices)

    # Reserve source (robot) and target vertices at the end of the graph lists. This dynamic part of the graph is
    # updated separately, since this update is done much more frequently than the full graph build.
    graph.vertices += [np.array([]), np.array([])]
    graph.adjacency += [[], []]

    return graph


def update_graph(graph: Graph, regions: list[list[np.ndarray]], source: np.ndarray, target: np.ndarray) -> tuple[
    np.ndarray, np.ndarray]:
    """
    Updates the dynamic part of the graph. Also returns free_source and free_target, which are versions of source and
    target which are projected on obstacle boundaries, if they happen to be inside one.
    """

    free_source, source_region_index, source_contour_index, source_vertex_index = push_out(source, regions)
    free_target, target_region_index, target_contour_index, target_vertex_index = push_out(target, regions)

    # Remove old dynamic edges
    # NOTE: this relies on dynamic edges being the last ones in the adjacency lists
    for edge in graph.adjacency[Graph.SOURCE]:
        del graph.adjacency[edge.vertex][-1]
    for edge in graph.adjacency[Graph.TARGET]:
        del graph.adjacency[edge.vertex][-1]
    graph.adjacency[Graph.SOURCE] = []
    graph.adjacency[Graph.TARGET] = []

    # Extract new dynamic edges
    graph.adjacency[Graph.SOURCE] = extract_dynamic_edges(regions, graph.vertices[:-2], graph.to_prev, graph.to_next,
                                                          free_source)
    graph.adjacency[Graph.TARGET] = extract_dynamic_edges(regions, graph.vertices[:-2], graph.to_prev, graph.to_next,
                                                          free_target)
    for edge in graph.adjacency[Graph.SOURCE]:
        graph.adjacency[edge.vertex].append(Edge(vertex=Graph.SOURCE, length=edge.length))
    for edge in graph.adjacency[Graph.TARGET]:
        graph.adjacency[edge.vertex].append(Edge(vertex=Graph.TARGET, length=edge.length))

    # Add the direct edge from source to target, if applicable
    if should_add_source_to_target_edge(regions, free_source, source_region_index, source_contour_index,
                                        source_vertex_index, free_target, target_region_index, target_contour_index,
                                        target_vertex_index):
        edge_length = np.linalg.norm(free_target - free_source)
        graph.adjacency[Graph.SOURCE].append(Edge(vertex=Graph.TARGET, length=edge_length))
        graph.adjacency[Graph.TARGET].append(Edge(vertex=Graph.SOURCE, length=edge_length))

    graph.vertices[Graph.SOURCE] = free_source
    graph.vertices[Graph.TARGET] = free_target

    return free_source, free_target


def should_add_source_to_target_edge(regions: list[list[np.ndarray]], free_source: np.ndarray, source_region_index: int,
                                     source_contour_index: int, source_vertex_index: int, free_target: np.ndarray,
                                     target_region_index: int, target_contour_index: int,
                                     target_vertex_index: int) -> bool:
    # If the source and target do not lie within the same region, they obviously cannot be connected
    if source_region_index != target_region_index:
        return False

    region_index = source_region_index

    # If at least one of the two points has not been projected to a contour vertex, just do an intersection check
    if source_vertex_index < 0 or target_vertex_index < 0:
        return not segment_intersects_contours(free_source, free_target, regions[region_index])

    # If both source and target have been projected to contour vertices, we must perform the same checks
    # we would do in extract_static_adjacency(), or we might falsely include the edge if it goes through a contour
    # without intersecting it.

    source_contour = regions[region_index][source_contour_index]
    target_contour = regions[region_index][target_contour_index]
    source_vertex = source_contour[source_vertex_index]
    source_to_prev = source_contour[source_vertex_index - 1 if source_vertex_index > 0 else len(
        source_contour) - 1] - source_vertex
    source_to_next = source_contour[(source_vertex_index + 1) % len(source_contour)] - source_vertex
    target_vertex = target_contour[target_vertex_index]
    target_to_prev = target_contour[target_vertex_index - 1 if target_vertex_index > 0 else len(
        target_contour) - 1] - target_vertex
    target_to_next = target_contour[(target_vertex_index + 1) % len(target_contour)] - target_vertex
    edge = target_vertex - source_vertex

    # This detects if the edge goes towards the interior of the contour 'source' belongs to
    sin_prev_source = np.cross(edge, source_to_prev)
    sin_next_source = np.cross(edge, source_to_next)
    is_source_convex = np.cross(source_to_prev, source_to_next) <= 0
    if is_source_convex and sin_prev_source > 0 > sin_next_source:
        return False
    if not is_source_convex and not (sin_prev_source <= 0 <= sin_next_source):
        return False

    # This detects if the edge goes towards the interior of the contour 'target' belongs to
    sin_prev_target = np.cross(edge, target_to_prev)
    sin_next_target = np.cross(edge, target_to_next)
    is_target_convex = np.cross(target_to_prev, target_to_next) <= 0
    if is_target_convex and sin_prev_target < 0 < sin_next_target:
        return False
    if not is_target_convex and not (sin_prev_target >= 0 >= sin_next_target):
        return False

    # If source and target have been projected to connected vertices on the same contour, we must include the edge.
    # We must not perform an intersection check in this case, because it would return True.
    if source_contour_index == target_contour_index and (
            source_vertex_index == (target_vertex_index + 1) % len(target_contour) or target_vertex_index == (
            source_vertex_index + 1) % len(source_contour)):
        return True

    # If none of the previous checks returned, do an intersection check
    return not segment_intersects_contours(source_vertex, target_vertex, regions[region_index])


def draw_graph(img: np.ndarray, graph: Graph):
    for i in range(len(graph.adjacency)):
        for edge in graph.adjacency[i]:
            if edge.vertex > i or edge.vertex < 0:
                cv2.line(img, graph.vertices[i].astype(np.int32), graph.vertices[edge.vertex].astype(np.int32),
                         color=(0, 0, 0))


def draw_path(img: np.ndarray, graph: Graph, path: list[int], source: np.ndarray, free_source: np.ndarray,
              target: np.ndarray, free_target: np.ndarray):
    vertices = [graph.vertices[v].astype(np.int32) for v in path]
    cv2.polylines(img, [np.array(vertices)], isClosed=False, color=(192, 64, 64), thickness=2)
    for vertex in vertices:
        cv2.circle(img, vertex.astype(np.int32), color=(64, 64, 192), radius=6, thickness=-1)

    cv2.line(img, source.astype(np.int32), free_source.astype(np.int32), color=(0, 0, 0), thickness=2)
    cv2.circle(img, free_source.astype(np.int32), color=(0, 0, 0), radius=6, thickness=-1)
    cv2.circle(img, source.astype(np.int32), color=(0, 0, 0), radius=6, thickness=-1)

    cv2.line(img, target.astype(np.int32), free_target.astype(np.int32), color=(0, 0, 0), thickness=2)
    cv2.circle(img, free_target.astype(np.int32), color=(0, 0, 0), radius=6, thickness=-1)
    cv2.circle(img, target.astype(np.int32), color=(0, 0, 0), radius=6, thickness=-1)


def dijkstra(adjacency_list: list[list[Edge]], source: int, target: int) -> list[int]:
    """
    source: the starting vertex of the search
    target: the target vertex of the search
    """
    vertex_count = len(adjacency_list)
    assert vertex_count > 0

    if target < 0:
        target = len(adjacency_list) + target
    if source < 0:
        source = len(adjacency_list) + source

    dist = np.full(vertex_count, np.inf, dtype=float)
    prev = np.full(vertex_count, -1, dtype=int)
    visited = np.zeros(vertex_count, dtype=bool)
    unvisited_count = vertex_count
    dist[source] = 0.0

    while unvisited_count > 0:
        u = np.ma.MaskedArray(dist, visited).argmin()
        if u == target:
            return reconstruct_path(prev, source, target)

        visited[u] = True
        unvisited_count -= 1

        for edge in adjacency_list[u]:
            if not visited[edge.vertex]:
                alt = dist[u] + edge.length
                if alt < dist[edge.vertex]:
                    dist[edge.vertex] = alt
                    prev[edge.vertex] = u

    return []


def reconstruct_path(prev: np.ndarray, source: int, target: int) -> list[int]:
    if target == source:
        return [target]

    # If the target has no parent it is not reachable
    if prev[target] < 0:
        return []

    path = []
    v = target
    while v >= 0:
        path.append(v)
        v = prev[v]
    path.reverse()
    return path


def distance_to_polyline(point: np.ndarray, vertices: np.ndarray) -> float:
    """
    Returns the closest distance between the point and the polyline formed by vertices
    """

    min_distance = np.inf

    for i in range(vertices.shape[0] - 1):
        edge = vertices[i + 1] - vertices[i]
        to_vertex_1 = vertices[i] - point
        to_vertex_2 = vertices[i + 1] - point
        normal = np.float32([edge[1], -edge[0]])
        n_cross_v1 = np.cross(normal, to_vertex_1)
        n_cross_v2 = np.cross(normal, to_vertex_2)

        # If this is true, the projection of the point on the segment lies somewhere between the two vertices
        if (n_cross_v1 < 0 < n_cross_v2) or (n_cross_v1 > 0 > n_cross_v2):
            normal /= np.linalg.norm(normal)
            distance = np.abs(np.dot(to_vertex_1, normal))
            if distance < min_distance:
                min_distance = distance

        # Else the projection of the point on the segment lies on a vertex or beyond one of the vertices
        else:
            distance = np.linalg.norm(to_vertex_1)
            if distance < min_distance:
                min_distance = distance
            distance = np.linalg.norm(to_vertex_2)
            if distance < min_distance:
                min_distance = distance

    return min_distance


def distance_to_contours(point: np.ndarray, region_contours: list[np.ndarray]) -> tuple[float, int]:
    """
    Returns the signed distance from the point to the region contours, as well as the index of the closest contour.
    Distance is positive if the point is within the free space of the region, else negative.
    """

    distances = np.empty(len(region_contours))
    distances[0] = cv2.pointPolygonTest(region_contours[0], np.float32(point), measureDist=True)
    if distances[0] < 0:  # The point is outside the region
        return distances.item(0), 0

    for i in range(1, len(region_contours)):
        distances[i] = cv2.pointPolygonTest(region_contours[i], np.float32(point), measureDist=True)
        if distances[i] > 0:  # The point is inside an inner obstacle
            return -distances[i], i
        distances[i] = -distances[i]

    closest_contour = np.argmin(distances)
    return distances.item(closest_contour), closest_contour


def project(pt: np.ndarray, contour: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Projects pt on the contour, returning the point on the contour that is closest.
    If the closest point is a vertex of the contour, also returns its index. Else, returns a negative value.
    """

    closest_point = np.empty(2)
    vertex_index = -1
    min_distance = np.inf

    for i in range(len(contour)):
        index_vertex_1 = i
        index_vertex_2 = (i + 1) % len(contour)
        vertex_1 = contour[index_vertex_1]
        vertex_2 = contour[index_vertex_2]
        edge = vertex_2 - vertex_1
        to_vertex_1 = vertex_1 - pt
        to_vertex_2 = vertex_2 - pt
        exterior_normal = np.float32([edge[1], -edge[0]])
        n_cross_v1 = np.cross(exterior_normal, to_vertex_1)
        n_cross_v2 = np.cross(exterior_normal, to_vertex_2)

        # If this is true, the projection of the point on the segment lies somewhere between the two vertices
        if (n_cross_v1 < 0 < n_cross_v2) or (n_cross_v1 > 0 > n_cross_v2):
            exterior_normal /= np.linalg.norm(exterior_normal)
            distance = np.dot(to_vertex_1, exterior_normal)
            if np.abs(distance) < min_distance:
                # If the distance is positive, the point is on the interior side of this part of the contour. If the
                # distance is negative, the point is on the exterior side of this part of the contour, but it might
                # be inside the contour somewhere else!
                # Offset by a small distance towards the exterior normal, to avoid floating point issues (because we
                # cannot get a resulting point exactly on the segment, without the offset the point might be
                # accidentally still considered as inside the contour).
                closest_point = pt + (distance + 1e-3) * exterior_normal
                vertex_index = -1
                min_distance = np.abs(distance)

        # Else the projection of the point on the segment lies on a vertex or beyond one of the vertices
        else:
            distance = np.linalg.norm(to_vertex_1)
            if distance < min_distance:
                closest_point = vertex_1
                vertex_index = index_vertex_1
                min_distance = distance
            distance = np.linalg.norm(to_vertex_2)
            if distance < min_distance:
                closest_point = vertex_2
                vertex_index = index_vertex_2
                min_distance = distance

    return closest_point, vertex_index


def push_out(point: np.ndarray, regions: list[list[np.ndarray]]) -> tuple[np.ndarray, int, int, int]:
    """
    If the point is inside an obstacle, returns the projected point on the closest contour.
    If the point is in free space, returns it unchanged.
    Also returns: the region index, the index of the closest contour within that region, and the index of
    the vertex the projection lies on, if it is the case.
    If the point is in free space or its projection is between two vertices, that last index has a negative value.
    Note: the resulting point might be either floating point or integer.
    """

    distances = np.empty(len(regions), dtype=np.float32)
    contour_indices = np.empty(len(regions), dtype=np.int32)
    for i in range(len(regions)):
        distance, contour_index = distance_to_contours(point, regions[i])
        if distance >= 0:
            # We are in free space in this region, nothing to do
            return point, i, contour_index, -1

        # We are outside of this region or in a hole within it, so record how far we are from it, and which contour we
        # are the closest from
        distances[i] = distance
        contour_indices[i] = contour_index

    # If we got here, it means we are in an obstacle. We also know that all distances are negative.
    closest_region: int = np.argmax(distances)
    # Project the point on the recorded closest contour of the closest region
    contour_index: int = contour_indices.item(closest_region)
    projection, vertex_index = project(point, regions[closest_region][contour_index])
    return projection, closest_region, contour_index, vertex_index
