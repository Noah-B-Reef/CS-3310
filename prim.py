# Return minimum spanning tree for a graph using Prim's algorithm given an Adjacency Matrix
def prim(graph,v0):
    # Initialize
    n = len(graph)
    visited = [False]*n
    visited[v0] = True
    edges = []
    totalweight = 0

    # Add edges
    for i in range(n-1):
        minweight = float('inf')
        for j in range(n):
            if visited[j]:
                for k in range(n):
                    if not visited[k] and graph[j][k] < minweight and graph[j][k] > 0:
                        minweight = graph[j][k]
                        edge = (j,k)
        edges.append(edge)
        totalweight += minweight
        visited[edge[1]] = True

    return edges, totalweight




def kruskal(graph):
    # Initialize
    n = len(graph)
    edges = []
    totalweight = 0

    # Add edges
    for i in range(n-1):
        minweight = float('inf')
        for j in range(n-1):
            for k in range(n-1):
                if graph[j][k] < minweight and graph[j][k] > 0:
                    minweight = graph[j][k]
                    edge = (j,k)
        edges.append(edge)
        totalweight += minweight
        graph[edge[0]][edge[1]] = float('inf')

    return edges, totalweight

graph = [[0,23,9,5,0,0,0,0,0,0],
         [23,0,0,0,31,0,0,0,0,0],
         [9,0,0,19,0,0,0,0,0,0],
         [5,0,19,0,13,0,0,6,0,0],
         [0,31,0,13,21,0,0,12,0],
         [0,0,0,0,21,0,0,0,0,18],
         [0,0,0,0,0,0,0,99,0,0],
         [0,0,0,6,0,0,99,0,3,0],
         [0,0,0,0,12,0,0,3,0,7],
         [0,0,0,0,0,18,0,0,7,0]]

print(kruskal(graph))
