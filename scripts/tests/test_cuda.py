import networkx as nx
import time

# 1. Create a large random graph
G = nx.fast_gnp_random_graph(n=50000, p=0.0001)
nx.config.warnings_to_ignore.add("cache")
# 2. Test CPU Performance (Standard NetworkX)
start = time.time()
pagerank_cpu = nx.pagerank(G)
print(f"CPU Time: {time.time() - start:.4f}s")

# 3. Test GPU Performance (using the 'cugraph' backend)
# You can specify the backend directly in the function call
start = time.time()
pagerank_gpu = nx.pagerank(G, backend="cugraph")
print(f"GPU Time: {time.time() - start:.4f}s")


# 4. Test Louvain Community Detection (using the 'cugraph' backend)
start = time.time()
communities_cpu = nx.algorithms.community.louvain_communities(G)
print(f"Louvain CPU Time: {time.time() - start:.4f}s")
print(f"Number of communities detected: {len(communities_cpu)}")
print(f"Sizes of first 5 communities: {[len(c) for c in communities_cpu[:5]]}")
print(f"Modularity score: {nx.algorithms.community.modularity(G, communities_cpu):.4f}")

# 3. Test Leiden Community Detection (using the 'cugraph' backend)
start = time.time()
communities_gpu = nx.algorithms.community.leiden_communities(G, backend="cugraph")
print(f"Leiden GPU Time: {time.time() - start:.4f}s")
print(f"Number of communities detected: {len(communities_gpu)}")
print(f"Sizes of first 5 communities: {[len(c) for c in communities_gpu[:5]]}")
print(f"Modularity score: {nx.algorithms.community.modularity(G, communities_gpu):.4f}")
