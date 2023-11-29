import cv2.typing

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

    colinear_mask = (cb_cd == 0) & (ca_cd == 0) & (ad_ab == 0) & (ac_ab == 0)
    ac_dot_ab = ac[:, 0] * ab[0] + ac[:, 1] * ab[1]
    ab_dot_cd = ab[0] * cd[:, 0] + ab[1] * cd[:, 1]
    ab_dot_ab = ab[0] * ab[0] + ab[1] * ab[1]
    mask_dir_same = (ab_dot_cd > 0) & (ac_dot_ab + ab_dot_cd > 0) & (ac_dot_ab - ab_dot_ab < 0)
    mask_dir_opposite = (ab_dot_cd < 0) & (ac_dot_ab > 0) & (ac_dot_ab + ab_dot_cd - ab_dot_ab < 0)
    return np.any(colinear_mask & (mask_dir_same | mask_dir_opposite))


def segment_intersects_contours(pt1: np.ndarray, pt2: np.ndarray, contours) -> bool:
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
    # TODO(performance): remove the clip if we KNOW the mask is binary
    inverted_mask = 1 - obstacle_mask.clip(0, 1)

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


def extract_convex_vertices(contours: list[np.ndarray]):
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


def extract_static_adjacency(contours: list[np.ndarray], vertices, to_prev, to_next):
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


def extract_dynamic_edges(contours: list[np.ndarray], vertices, to_prev, to_next, point):
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
        if not are_pts_connected and segment_intersects_contours(point, vertices[i], contours):
            continue

        edge_length = np.linalg.norm(edge)
        edges.append(Edge(vertex=i, length=edge_length))

    return edges


def build_graph(contours: list[np.ndarray]) -> Graph:
    vertices, to_prev, to_next = extract_convex_vertices(contours)
    adjacency = extract_static_adjacency(contours, vertices, to_prev, to_next)
    # Reserve source and target vertices
    vertices += [[], []]
    adjacency += [[], []]
    graph = Graph()
    graph.vertices = vertices
    graph.adjacency = adjacency
    graph.to_prev = to_prev
    graph.to_next = to_next
    return graph


def update_graph(graph: Graph, contours, source, free_source, target, free_target):
    """
    Update the dynamic part of the graph. The considered vertices to be added in the graph are source and target,
    but the visibility edges are extracted from free_source and free_target, which should be outside any obstacle.
    This allows for source and target to be in obstacles, while behaving as if they were outside.
    """

    # Remove old dynamic edges
    # NOTE: this relies on dynamic edges being the last ones in the neighbor lists
    for edge in graph.adjacency[Graph.SOURCE]:
        del graph.adjacency[edge.vertex][-1]
    for edge in graph.adjacency[Graph.TARGET]:
        del graph.adjacency[edge.vertex][-1]
    graph.adjacency[Graph.SOURCE] = []
    graph.adjacency[Graph.TARGET] = []

    # Extract new dynamic edges
    graph.adjacency[Graph.SOURCE] = extract_dynamic_edges(contours, graph.vertices[:-2], graph.to_prev, graph.to_next,
                                                          free_source)
    graph.adjacency[Graph.TARGET] = extract_dynamic_edges(contours, graph.vertices[:-2], graph.to_prev, graph.to_next,
                                                          free_target)
    for edge in graph.adjacency[Graph.SOURCE]:
        graph.adjacency[edge.vertex].append(Edge(vertex=Graph.SOURCE, length=edge.length))
    for edge in graph.adjacency[Graph.TARGET]:
        graph.adjacency[edge.vertex].append(Edge(vertex=Graph.TARGET, length=edge.length))

    # Add the direct edge from source to target
    if not segment_intersects_contours(source, target, contours):
        edge_length = np.linalg.norm(target - source)
        graph.adjacency[Graph.SOURCE].append(Edge(vertex=Graph.TARGET, length=edge_length))
        graph.adjacency[Graph.TARGET].append(Edge(vertex=Graph.SOURCE, length=edge_length))

    graph.vertices[Graph.SOURCE] = source
    graph.vertices[Graph.TARGET] = target


def draw_contour_orientations(img: np.ndarray, contours):
    """
    Draw positive orientation as green, negative as red;
    first vertex is black, last is white
    """
    for c in range(len(contours)):
        orientation = np.sign(cv2.contourArea(contours[c], oriented=True))
        color = (64, 192, 64) if orientation >= 0 else (64, 64, 192)
        cv2.drawContours(img, [contours[c]], contourIdx=-1, color=color, thickness=2)
        n_points = len(contours[c])
        for i in range(n_points):
            brightness = i / (n_points - 1) * 255
            cv2.circle(img, contours[c][i], color=(brightness, brightness, brightness), radius=5, thickness=-1)


def draw_graph(img: np.ndarray, graph: Graph):
    for i in range(len(graph.adjacency)):
        for edge in graph.adjacency[i]:
            if edge.vertex > i or edge.vertex < 0:
                cv2.line(img, graph.vertices[i].astype(np.int32), graph.vertices[edge.vertex].astype(np.int32),
                         color=(0, 0, 0))


def draw_path(img, graph, path, raw_source, free_source, raw_target, free_target):
    vertices = [graph.vertices[v].astype(np.int32) for v in path]
    cv2.polylines(img, [np.array(vertices)], isClosed=False, color=(192, 64, 64), thickness=2)

    cv2.line(img, raw_source, free_source.astype(np.int32), color=(0, 0, 0), thickness=2)
    cv2.circle(img, free_source.astype(np.int32), color=(192, 64, 64), radius=6, thickness=-1)
    cv2.circle(img, raw_source, color=(0, 0, 0), radius=6, thickness=-1)

    cv2.line(img, raw_target, free_target.astype(np.int32), color=(0, 0, 0), thickness=2)
    cv2.circle(img, free_target.astype(np.int32), color=(192, 64, 64), radius=6, thickness=-1)
    cv2.circle(img, raw_target, color=(0, 0, 0), radius=6, thickness=-1)


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


def reconstruct_path(prev, source: int, target: int) -> list[int]:
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


def project(pt: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Returns the point on the contour that is the closest to pt
    """

    closest_point = np.empty(2)
    min_distance = np.inf

    for i in range(len(contour)):
        pt1 = contour[i]
        pt2 = contour[(i + 1) % len(contour)]
        edge = pt2 - pt1
        to_pt1 = pt1 - pt
        to_pt2 = pt2 - pt
        exterior_normal = np.float32([edge[1], -edge[0]])
        n_cross_pt1 = np.cross(exterior_normal, to_pt1)
        n_cross_pt2 = np.cross(exterior_normal, to_pt2)

        # If this is true, the projection of the point on the segment lies somewhere between the two vertices
        if (n_cross_pt1 < 0 < n_cross_pt2) or (n_cross_pt1 > 0 > n_cross_pt2):
            exterior_normal /= np.linalg.norm(exterior_normal)
            distance = np.dot(to_pt1, exterior_normal)
            if np.abs(distance) < min_distance:
                # If the distance is positive, the point is on the interior side of this part of the contour. If the
                # distance is negative, the point is on the exterior side of this part of the contour, but it might
                # be inside the contour somewhere else!
                # Offset by a small distance towards the exterior normal, to avoid floating point issues (because we
                # cannot get a resulting point exactly on the segment, without the offset the point might be
                # accidentally still considered as inside the contour).
                closest_point = pt + (distance + 1e-3) * exterior_normal
                min_distance = np.abs(distance)

        # Else the projection of the point on the segment lies on or beyond one of the vertices
        else:
            distance = np.linalg.norm(to_pt1)
            if distance < min_distance:
                closest_point = pt1
                min_distance = distance
            distance = np.linalg.norm(to_pt2)
            if distance < min_distance:
                closest_point = pt2
                min_distance = distance

    return closest_point


def distance_to_contours(point: np.ndarray, region_contours: list[np.ndarray]):
    """
    Returns the signed distance from the point to the region contours, as well as the index of the closest contour
    Distance is positive if the point is within the free space of the region, else negative.
    """

    distances = np.empty(len(region_contours))
    distances[0] = cv2.pointPolygonTest(region_contours[0], np.float32(point), measureDist=True)
    if distances[0] < 0:  # The point is outside the region
        return distances[0], 0

    for i in range(1, len(region_contours)):
        distances[i] = cv2.pointPolygonTest(region_contours[i], np.float32(point), measureDist=True)
        if distances[i] > 0:  # The point is inside an inner obstacle
            return -distances[i], i
        distances[i] = -distances[i]

    closest_contour = np.argmin(distances)
    return distances[closest_contour], closest_contour


def push_out(point: np.ndarray, regions: list[list[np.ndarray]]) -> np.ndarray:
    distances = np.empty(len(regions), dtype=np.float32)
    contour_indices = np.empty(len(regions), dtype=np.int32)
    for i in range(len(regions)):
        distance, contour_index = distance_to_contours(point, regions[i])
        if distance >= 0:
            # We are in free space in this region, nothing to do
            return point

        # We are outside of this region or in a hole within it, record how far we are from it
        distances[i] = distance
        contour_indices[i] = contour_index

    # If we got here, it means we are in an obstacle. We also know that all distances are negative.
    closest_region = np.argmax(distances)
    # Project the point on the closest contour
    contour_index = contour_indices[closest_region]
    return project(point, regions[closest_region][contour_index])


raw_target = (120, 730)


def main():
    # Note: the minimum distance to any obstacle is 'kernel_size - approx_poly_epsilon'
    approx_poly_epsilon = 2
    raw_source = (200, 100)
    color_image = cv2.imread('../images/map_divided.png')
    obstacle_mask = get_obstacle_mask(color_image)

    regions = extract_contours(obstacle_mask, approx_poly_epsilon)

    all_contours = [contour for region in regions for contour in region]

    free_space = np.empty_like(color_image)
    free_space[:] = (64, 64, 192)
    cv2.drawContours(free_space, all_contours, contourIdx=-1, color=(255, 255, 255), thickness=-1)

    # TODO: take the regions into account when building the graph (for performance), but it is probably better
    #  to make only one Graph object
    graph = build_graph(all_contours)

    free_source = push_out(np.float32(raw_source), regions)

    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('main', color_image.shape[1], color_image.shape[0])

    while True:
        # FIXME: we do not detect intersection if source and target are on opposite vertices of the same contour
        free_target = push_out(np.float32(raw_target), regions)
        print(free_target.dtype)

        distance = -np.inf
        for region_contours in regions:
            dist, contour_index = distance_to_contours(free_target, region_contours)
            if dist > distance:
                distance = dist
        assert distance >= 0

        # TODO: take the regions into account when building the graph (for performance), but it is probably better
        #  to make only one Graph object
        update_graph(graph, all_contours, np.array(raw_source), free_source, np.array(raw_target), free_target)
        path = dijkstra(graph.adjacency, Graph.SOURCE, Graph.TARGET)

        img = cv2.addWeighted(color_image, 0.75, free_space, 0.25, 0.0)
        draw_contour_orientations(img, [contour for region in regions for contour in region])
        cv2.drawContours(img, all_contours, contourIdx=-1, color=(64, 64, 192))
        draw_graph(img, graph)
        draw_path(img, graph, path, raw_source, free_source, raw_target, free_target)

        cv2.namedWindow('main', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('main', mouse_callback)
        cv2.imshow('main', img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def mouse_callback(event, x, y, flags, param):
    global raw_target
    if event == cv2.EVENT_MOUSEMOVE:
        raw_target = (x, y)


def pathfinding_test():
    adjacency = [[Edge(1, 7), Edge(2, 9), Edge(5, 14)],
                 [Edge(0, 7), Edge(2, 10), Edge(3, 15)],
                 [Edge(0, 9), Edge(1, 10), Edge(3, 11), Edge(5, 2)],
                 [Edge(1, 15), Edge(2, 11), Edge(4, 6)],
                 [Edge(3, 6), Edge(5, 9)],
                 [Edge(0, 14), Edge(2, 2), Edge(4, 9)]]
    source = 0
    target = 4
    # Correct path for the above: [0, 2, 5, 4]
    path = dijkstra(adjacency, source, target)
    print(path)


if __name__ == '__main__':
    main()
    # pathfinding_test()
