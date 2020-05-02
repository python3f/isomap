from sklearn.neighbors    import kneighbors_graph
from sklearn.manifold     import MDS
from scipy.sparse.csgraph import csgraph_to_dense
from scipy.sparse.csgraph import floyd_warshall


"""Args:
X: input samples, array (num, dim)
n_components: dimension of output data
n_neighbours: neighborhood size

Returns:
Y: output samples, array (num, n_components)
"""
def Isomap(X, n_components=2, n_neighbours=10):
  NN = kneighbors_graph(X, n_neighbours, mode='distance')  # nearest neighbour matrix
  SP = floyd_warshall(NN, directed=False)                  # shortest path matrix (geodesic)
  EM = MDS(n_components, dissimilarity='precomputed')      # embedding
  Y  = EM.fit_transform(SP)
  return Y
