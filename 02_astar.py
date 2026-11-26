import heapq

graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 3), ('E', 1)],
    'C': [('F', 2)],
    'D': [('G', 2)], 'E': [('G', 4)], 'F': [('G', 1)], 'G': []
}
heuristic = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 3, 'F': 1, 'G': 0}

def astar(start, goal):
    queue = [(heuristic[start], 0, start, [start])]
    visited_cost = {}
    while queue:
        f, cost, node, path = heapq.heappop(queue)
        if node == goal:
            return path, cost
        if node in visited_cost and visited_cost[node] <= cost:
            continue
        visited_cost[node] = cost
        for nxt, weight in graph[node]:
            new_cost = cost + weight
            heapq.heappush(queue, (new_cost + heuristic[nxt], new_cost, nxt, path + [nxt]))
    return None, float('inf')

path, cost = astar('A', 'G')
print("Path:", path)
print("Cost:", cost)
