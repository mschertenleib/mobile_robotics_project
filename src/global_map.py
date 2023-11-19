import cv2
import numpy as np


class Edge(object):
    __slots__ = ['vertex', 'length']

    def __init__(self, vertex, length):
        self.vertex: int = vertex
        self.length: float = length


class Graph:
    SOURCE = -1
    TARGET = -2

    def __init__(self):
        self.vertices: list[int] = []
        self.adjacency: list[list[Edge]] = []
        self.to_prev: list[np.ndarray] = []
        self.to_next: list[np.ndarray] = []


def image_info(img: np.ndarray):
    print(
        f'dtype: {img.dtype}, shape: {img.shape}, min: {img.min()}, max: {img.max()}')


def normalize(img: np.ndarray):
    range_values = img.max() - img.min()
    if range_values == 0:
        return img.astype(np.float32)
    else:
        return (img.astype(np.float32) - img.min()) / range_values


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """
    Checks for an intersection between the half-open segment ]a---b] and closed segments [c[i]---d[i]]
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
            (((cb_cd >= 0) & (ca_cd < 0)) | ((cb_cd <= 0) & (ca_cd > 0))) & (
                    ((ad_ab > 0) & (ac_ab < 0)) | ((ad_ab < 0) & (ac_ab > 0)))):
        return True

    colinear_mask = (cb_cd == 0) & (ca_cd == 0) & (ad_ab == 0) & (ac_ab == 0)
    ac_dot_ab = ac[:, 0] * ab[0] + ac[:, 1] * ab[1]
    ab_dot_cd = ab[0] * cd[:, 0] + ab[1] * cd[:, 1]
    ab_dot_ab = ab[0] * ab[0] + ab[1] * ab[1]
    mask_dir_same = (ab_dot_cd > 0) & (ac_dot_ab + ab_dot_cd > 0) & (ac_dot_ab - ab_dot_ab <= 0)
    mask_dir_opposite = (ab_dot_cd < 0) & (ac_dot_ab > 0) & (ac_dot_ab + ab_dot_cd - ab_dot_ab <= 0)
    return np.any(colinear_mask & (mask_dir_same | mask_dir_opposite))


def segment_intersects_contours(pt1: np.ndarray, pt2: np.ndarray, contours) -> bool:
    """
    Checks for an intersection between the half-open segment ]pt1---pt2] and all contours
    """
    for contour in contours:
        if segments_intersect(pt1, pt2, contour[:-1], contour[1:]):
            return True
        if segments_intersect(pt1, pt2, np.array([contour[-1]]), np.array([contour[0]])):
            return True
    return False


def _extract_convex_vertices(contours):
    vertices = []
    to_prev = []
    to_next = []
    for contour in contours:
        for i in range(len(contour)):
            to_prev_i = contour[i - 1 if i > 0 else len(contour) - 1] - contour[i]
            to_next_i = contour[i + 1 if i < len(contour) - 1 else 0] - contour[i]
            if np.cross(to_prev_i, to_next_i) >= 0:
                vertices.append(contour[i])
                to_prev.append(to_prev_i)
                to_next.append(to_next_i)
    return vertices, to_prev, to_next


def _extract_static_adjacency(contours, vertices, to_prev, to_next):
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
            are_ij_connected = (is_j_prev_i or is_j_next_i) and (is_i_prev_j or is_i_next_j)
            if not are_ij_connected and segment_intersects_contours(vertices[i], vertices[j], contours):
                continue

            edge_length = np.linalg.norm(edge)
            adjacency[i].append(Edge(vertex=j, length=edge_length))
            adjacency[j].append(Edge(vertex=i, length=edge_length))

    return adjacency


def _extract_dynamic_edges(contours, vertices, to_prev, to_next, point):
    # TODO: if the point is inside a contour, make it move out by adding a node on the contour and making it its only
    #  connection, but that implies the number of edges might increase by one sometimes
    edges = []
    for i in range(len(vertices)):

        # Discard vertices for which the previous and next one lie on opposite sides of the edge
        edge = vertices[i] - point
        sin_prev_i = np.cross(edge, to_prev[i])
        sin_next_i = np.cross(edge, to_next[i])
        if (sin_prev_i < 0 or sin_next_i < 0) and (sin_prev_i > 0 or sin_next_i > 0):
            continue

        # Discard edges that intersect contours
        if segment_intersects_contours(point, vertices[i], contours):
            continue

        edge_length = np.linalg.norm(edge)
        edges.append(Edge(vertex=i, length=edge_length))

    return edges


def build_graph(contours):
    vertices, to_prev, to_next = _extract_convex_vertices(contours)
    adjacency = _extract_static_adjacency(contours, vertices, to_prev, to_next)
    # Reserve source and target vertices
    vertices += [[], []]
    adjacency += [[], []]
    graph = Graph()
    graph.vertices = vertices
    graph.adjacency = adjacency
    graph.to_prev = to_prev
    graph.to_next = to_next
    return graph


def update_graph(graph, contours, source, target):
    # Remove old dynamic edges
    # NOTE: this relies on dynamic edges being the last ones in the neighbor lists
    for edge in graph.adjacency[Graph.SOURCE]:
        del graph.adjacency[edge.vertex][-1]
    for edge in graph.adjacency[Graph.TARGET]:
        del graph.adjacency[edge.vertex][-1]
    graph.adjacency[Graph.SOURCE] = []
    graph.adjacency[Graph.TARGET] = []

    # Extract new dynamic edges
    graph.adjacency[Graph.SOURCE] = _extract_dynamic_edges(contours, graph.vertices[:-2], graph.to_prev, graph.to_next,
                                                           source)
    graph.adjacency[Graph.TARGET] = _extract_dynamic_edges(contours, graph.vertices[:-2], graph.to_prev, graph.to_next,
                                                           target)
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


def draw_contour_orientations(img, contours, orientations):
    """
    Draw positive orientation as green, negative as red;
    first vertex is black, last is white
    """
    img[:] = (192, 192, 192)
    for c in range(len(contours)):
        color = (64, 192, 64) if orientations[c] >= 0 else (64, 64, 192)
        cv2.drawContours(img, [contours[c]], contourIdx=-1, color=color, thickness=3)
        n_points = len(contours[c])
        for i in range(n_points):
            brightness = i / (n_points - 1) * 255
            cv2.circle(img, contours[c][i][0], color=(brightness, brightness, brightness), radius=5, thickness=-1)


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
            break

        visited[u] = True
        unvisited_count -= 1

        for edge in adjacency_list[u]:
            if not visited[edge.vertex]:
                alt = dist[u] + edge.length
                if alt < dist[edge.vertex]:
                    dist[edge.vertex] = alt
                    prev[edge.vertex] = u

    return reconstruct_path(prev, source, target)


def reconstruct_path(prev, source: int, target: int) -> list[int]:
    if target == source:
        return [target]

    if prev[target] < 0:
        # print('Target is not reachable from source')
        return []

    path = []
    v = target
    while v >= 0:
        path.append(v)
        v = prev[v]
    path.reverse()
    return path


mouse_x, mouse_y = 0, 0
target_point = (120, 730)


def main():
    threshold = 200
    kernel_size = 50
    # Note: the minimum distance to any obstacle is 'kernel_size - approx_poly_epsilon'
    approx_poly_epsilon = 2
    source_point = (200, 100)
    original_img = cv2.imread('../map.png')
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    img = cv2.dilate(img, kernel)

    # Flood fill from the robot location, so we only get the contours that
    # are relevant. Note that this requires to know the position of the robot
    # a priori, and might not be suitable if the map has several disconnected
    # regions across which the robot might get kidnapped
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    assert img[source_point[::-1]] == 0, 'Flood fill seed point is not in free space'
    _, img, _, _ = cv2.floodFill(img, mask=mask,
                                 seedPoint=source_point, newVal=2)
    _, img = cv2.threshold(img, thresh=1, maxval=1, type=cv2.THRESH_BINARY_INV)

    # NOTE: using RETR_EXTERNAL means we would only get the outer contours,
    # which is basically equivalent to floodfilling the binary image from the
    # outside prior to calling findContours. However, this assumes our region
    # of interest is "outside", and would not work if we are "walled in",
    # which is actually very likely
    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_LIST,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.squeeze(cv2.approxPolyDP(contour, epsilon=approx_poly_epsilon, closed=True)) for contour in contours]

    # NOTE: orientation is positive for a clockwise contour, which is the opposite of the mathematical standard
    # (right-hand rule). Note however that the outer contour of a shape is always
    # counter-clockwise (hence orientation is negative), and the contour of a hole is always clockwise (hence
    # orientation is positive).
    orientations = [np.sign(cv2.contourArea(contour, oriented=True)) for contour in contours]

    # If we know that 'contours' only contains those in the region where the
    # robot is located, we can use them directly for computing the visibility
    # graph. Else, we must take into account the hierarchy.

    walkable = np.zeros(original_img.shape, dtype=np.uint8)
    walkable[:] = (192, 64, 64)
    cv2.drawContours(walkable, contours, contourIdx=-1, color=(255, 255, 255), thickness=-1)

    graph = build_graph(contours)

    print(f'Number of static convex vertices: {len(graph.vertices) - 2}')
    num_static_edges = sum(
        [len([edge for edge in graph.adjacency[i] if edge.vertex > i]) for i in range(len(graph.adjacency))])
    print(f'Number of static edges: {num_static_edges}')

    update_graph(graph, contours, np.array(source_point), np.array(target_point))

    num_edges_source = len(graph.adjacency[Graph.SOURCE])
    print(f'Number of edges from source: {num_edges_source}')
    num_edges_target = len(graph.adjacency[Graph.TARGET])
    print(f'Number of edges to target: {num_edges_target}')

    path = dijkstra(graph.adjacency, Graph.SOURCE, Graph.TARGET)

    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('main', img.shape[1], img.shape[0])
    cv2.setMouseCallback('main', on_click)
    while True:
        update_graph(graph, contours, np.array(source_point), np.array(target_point))
        path = dijkstra(graph.adjacency, Graph.SOURCE, Graph.TARGET)

        img = cv2.addWeighted(original_img, 0.75, walkable, 0.25, 0.0)
        cv2.drawContours(img, contours, contourIdx=-1, color=(64, 64, 192))
        for i in range(len(graph.adjacency)):
            for edge in graph.adjacency[i]:
                if edge.vertex > i or edge.vertex < 0:
                    cv2.line(img, graph.vertices[i], graph.vertices[edge.vertex], color=(0, 0, 0))
        for i in range(len(path) - 1):
            cv2.line(img, graph.vertices[path[i]], graph.vertices[path[i + 1]], color=(64, 64, 192), thickness=3)
        cv2.circle(img, source_point, color=(64, 192, 64), radius=6, thickness=-1)
        cv2.circle(img, target_point, color=(64, 64, 192), radius=6, thickness=-1)
        # draw_contour_orientations(img, contours, orientations)
        cv2.imshow('main', normalize(img))
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def on_click(event, x, y, flags, param):
    global mouse_x, mouse_y, target_point
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        target_point = (x, y)


def floodfill_background():
    img = cv2.imread('../map.png')
    img = cv2.cv2tColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('main', img.shape[1], img.shape[0])
    cv2.setMouseCallback('main', on_click)
    cv2.imshow('main', normalize(img))
    cv2.waitKey(0)

    # Not sure this would actually work well...
    global mouse_x, mouse_y
    seed_point = (mouse_x, mouse_y)
    _, img, mask, _ = cv2.floodFill(img, mask=np.array([], dtype=np.uint8), seedPoint=seed_point, newVal=0, loDiff=2,
                                    upDiff=2, flags=cv2.FLOODFILL_MASK_ONLY)
    # TODO: actually use the mask here
    # _, img = cv2.threshold(img, thresh=1, maxval=1, type=cv2.THRESH_BINARY)
    image_info(mask)
    cv2.imshow('main', np.where(mask[1:-1, 1:-1], img, 0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pathfinding_test():
    graph = [[Edge(1, 7), Edge(2, 9), Edge(5, 14)],
             [Edge(0, 7), Edge(2, 10), Edge(3, 15)],
             [Edge(0, 9), Edge(1, 10), Edge(3, 11), Edge(5, 2)],
             [Edge(1, 15), Edge(2, 11), Edge(4, 6)],
             [Edge(3, 6), Edge(5, 9)],
             [Edge(0, 14), Edge(2, 2), Edge(4, 9)]]
    source = 0
    target = 4
    # Correct path for the above: [0, 2, 5, 4]
    path = dijkstra(graph, source, target)
    print(path)


if __name__ == '__main__':
    main()
    # floodfill_background()
    # pathfinding_test()

    """
    a1 = np.array([3.7, -1.9])
    a2 = np.array([10.67, -5.09])
    b1 = np.array([3.7, -1.9])
    b2 = np.array([3.14, 7.5])
    print(intersect_segments_closed(a1, a2, b1, b2))
    """
