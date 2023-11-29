from global_map import *


def main():
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
