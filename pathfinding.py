import numpy as np


class Edge(object):
    __slots__ = ['vertex', 'length']

    def __init__(self, vertex, length):
        self.vertex = vertex
        self.length = length


def dijkstra(graph: list[list[Edge]], source: int, target: int):
    """
    graph: adjacency list
    source: the starting vertex of the search
    target: the target vertex of the search
    """
    vertex_count = len(graph)
    assert vertex_count > 0
    assert 0 <= source < vertex_count
    assert 0 <= target < vertex_count

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

        for edge in [edge for edge in graph[u] if not visited[edge.vertex]]:
            alt = dist[u] + edge.length
            if alt < dist[edge.vertex]:
                dist[edge.vertex] = alt
                prev[edge.vertex] = u

    return reconstruct_path(prev, source, target)


def reconstruct_path(prev, source: int, target: int) -> list[int]:
    if target == source:
        return [target]

    if prev[target] < 0:
        print('Target is not reachable from source')
        return []

    path = []
    v = target
    while v >= 0:
        path.append(v)
        v = prev[v]
    path.reverse()
    return path


if __name__ == '__main__':
    def main():
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


    main()
