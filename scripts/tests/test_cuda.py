import networkx as nx
import time

# 1. Create a large random graph
G = nx.fast_gnp_random_graph(n=50000, p=0.0001)

# 2. Test CPU Performance (Standard NetworkX)
start = time.time()
pagerank_cpu = nx.pagerank(G)
print(f"CPU Time: {time.time() - start:.4f}s")

# 3. Test GPU Performance (using the 'cugraph' backend)
# You can specify the backend directly in the function call
start = time.time()
pagerank_gpu = nx.pagerank(G, backend="cugraph")
print(f"GPU Time: {time.time() - start:.4f}s")
