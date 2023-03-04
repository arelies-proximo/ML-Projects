#astar
#bfs
#dfs

def astar(start, stopnode):
    open_set = set(start)
    close_set = set()

    g = {}
    parent = {}
    g[start] = 0
    parent[start] = start

    while len(open_set) > 0:

        n = None

        for v in open_set:
            if n==None or g[v] +hrst(v)  <  g[n] +hrst(n):
                n = v
        
        if n==stopnode or GraphNodes[n] == None:
            pass
        else:
            for m,weight in get_neighbors(n):
                if m not in open_set and m not in close_set:
                    open_set.add(m)
                    parent[m] = n
                    g[m] = weight + g[n]
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parent[m] = n
                        if m in close_set:
                            close_set.remove(m)
                            open_set.add(m)
                
        if n==None:
            print("Path does not exist")
            return
        
        if n==stopnode:
            path = []
            while parent[n] != n:
                path.append(n)
                n = parent[n]
            path.append(start)
            path.reverse()
            print("Path Found")
            print("Path: " , path)
            return
        
        open_set.remove(n)
        close_set.add(n)
    print("Path does not exist")


def get_neighbors(v):
    if v in GraphNodes:
        return GraphNodes[v]
    else:
        return None

def hrst(n):
    hr_st = {
        'S':14, 
        "B":12,
        'C':11,
        'D':6,
        'E':4,
        'F':11,
        'G':0 
    }
    return hr_st[n]

GraphNodes = {
    'S': [('B', 4), ('C', 3)],
    'B': [('F',5)],
    'C': [('E',10), ('D',7)],
    'D': [('E',2)],
    'E': [('G',5)],
    'F': [('G',16)],
    'G': [('G',0)]
}

astar('S', 'G')






def bfs(graph, root_node):
    queue = []
    visited = []

    queue.append(root_node)
    visited.append(root_node)

    while queue:
        s = queue.pop(0)
        print(s, end=' ')

        for i in graph[s]:
            if i not in visited:
                visited.append(i)
                queue.append(i)



def dfs(graph, root_node):
    stack = []
    visited = []

    stack.insert(0,root_node)
    visited.append(root_node)

    while stack:
        s =  stack.pop(0)
        print(s, end=' ')

        for i in graph[s]:    #OR for i in reversed(graph[s]):
                
            if i not in visited:
                stack.insert(0,i)
                visited.append(i)


graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E', 'F'],
    'C': ['G'],
    'D': [],
    'E': [],
    'F': ['H'],
    'G': ['I'],
    'H': [],
    'I': [],

}

