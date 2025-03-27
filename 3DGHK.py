"""The 3D Graph Hopper kernel as defined in :cite:``."""
# Author: >
# License: BSD 3 clause
import numpy as np
import torch
import time
from collections import defaultdict
from numbers import Real
from warnings import warn
from numpy.matlib import repmat

from grakel.kernels import Kernel
from grakel.graph import Graph
from grakel.graph import dijkstra
# Python 2/3 cross-compatibility import
from six.moves import filterfalse
from six.moves.collections_abc import Iterable
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import networkx as nx
from sklearn.metrics import accuracy_score

class GraphHopper(Kernel):
    """Graph Hopper Histogram kernel as found in :cite:`feragen2013scalable`.

    Parameters
    ----------
    kernel_type : str, tuple or function
        For `kernel_type` of **type**:
            + **str** can either be 'linear', 'gaussian', 'bridge'.
            + **tuple** can be of the form ('gaussian', mu) where mu is a number.
            + **function** can be a function that takes two tuples of np.arrays for each graph
              corresponding to the M matrix and the attribute matrix and returns a number.


    Attributes
    ----------
    metric_ : function
        The base metric applied between features.

    calculate_norm_ : bool
        Defines if the norm of the attributes will be calculated
        (in order to avoid recalculation when using it with e.g. gaussian).

    """

    _graph_format = "all"

    def __init__(self, n_jobs=None, normalize=False, verbose=False, kernel_type='linear'):
        """Initialize an Graph Hopper kernel."""
        super(GraphHopper, self).__init__(n_jobs=n_jobs,
                                          normalize=normalize,
                                          verbose=verbose)
        self.kernel_type = kernel_type
        self._initialized.update({"kernel_type": False})

    def initialize(self):
        """Initialize all transformer arguments, needing initialization."""
        super(GraphHopper, self).initialize()
        if not self._initialized["kernel_type"]:
            if type(self.kernel_type) is str:
                if self.kernel_type == "linear3dghk":
                    self.metric_ = linear3dghk_kernel
                    self.calculate_norm_ = False
                elif self.kernel_type == "cmg_kernel":
                    self.metric_ = cmk_kernel
                    self.calculate_norm_ = False
                else:
                    raise ValueError('Unsupported kernel with name "' + str(self.kernel_type) + '"')
            elif (type(self.kernel_type) is tuple and len(self.kernel_type) == 2 and
                    self.kernel_type[0] == "gaussian" and isinstance(self.kernel_type[1], Real)):
                self.metric_ = lambda x, y: gaussian_kernel(x, y, self.kernel_type[1])
                self.calculate_norm_ = True
           
            elif callable(self.kernel_type):
                self.metric_ = self._kernel_type
                self.calculate_norm_ = False
            else:
                raise TypeError('Unrecognized "kernel_type": can either be a str '
                                'from the supported: "linear", "gaussian", "bridge" '
                                'or tuple ("gaussian", mu) or a callable.')

    def parse_input(self, X):
        """Parse and check the given input for the Graph Hopper kernel.

        Parameters
        ----------
        X : iterable
            For the input to pass the test, we must have:
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format).

        Returns
        -------
        out : np.array, shape=(len(X), n_labels)
            A np array for frequency (cols) histograms for all Graphs (rows).

        """
        #s = time.time()
        if not isinstance(X, Iterable):
            raise TypeError('input must be an iterable\n')
        else:
            ni = 0
            diam = list()
            graphs = list()
            for (i, x) in enumerate(iter(X)):
                is_iter = False
                if isinstance(x, Iterable):
                    is_iter = True
                    x = list(x)

                if type(x) is Graph:
                    g = Graph(x.get_adjacency_matrix(),
                              x.get_labels(purpose="adjacency"),
                              {},
                              self._graph_format)
                elif is_iter and len(x) == 0 or len(x) >= 2:
                    if len(x) == 0:
                        warn('Ignoring empty element on index: '+str(i))
                        continue
                    elif len(x) >= 2:
                        g = Graph(x[0], x[1], {}, "adjacency")
                        g.change_format(self._graph_format)
                else:
                    raise TypeError('each element of X must be either a '
                                    'graph object or a list with at least '
                                    'a graph like object and node, ')

                spm, attr = g.build_shortest_path_matrix(labels="vertex") # attributes are the node labels
                #print("======================",attr,spm)
                nv = g.nv()
                try:
                    attributes = np.array([attr[j] for j in range(nv)])
                except TypeError:
                    raise TypeError('All attributes of a single graph should have the same dimension.')
                diam.append(int(np.max(spm[spm < float("Inf")])))
                graphs.append((g.get_adjacency_matrix(), nv, attributes))
                ni += 1
            #print("max diameter of graph", max(diam))
       

        if self._method_calling == 1:
            max_diam = self._max_diam = max(diam) + 1
        else:
       	    #print("max diam:",max(diam))
            max_diam = max(self._max_diam, max(diam) + 1)
	
        out = list()
        #print("number of graph =======>",ni)
        for i in range(ni):
            #print("for graph======>",i+1)
            AM, node_nr, attributes = graphs[i]                              #Adjacency matrix, number of nodes and attributes
            #print("attributes", attributes)
            #print("number of nodes for Graph",i+1,"is =========>",node_nr)
            des = np.zeros(shape=(node_nr, node_nr, max_diam), dtype=int)    # shape(4,4,2) implies 4, 4X2 zero matrix
            occ = np.zeros(shape=(node_nr, node_nr, max_diam), dtype=int)
            #if i ==0:
            	#print("number of nodes in graph 1 is ",node_nr)
            # Convert adjacency matrix to dictionary
            idx_i, idx_j = np.where(AM > 0)
            ed = defaultdict(dict)                                             #{0:{edge pairs},1:{}.....,c:{}}  c denotes the rows that are non-zero
            for (a, b) in filterfalse(lambda a: a[0] == a[1], zip(idx_i, idx_j)):
                ed[a][b] = AM[a, b]

            for j in range(node_nr):
                #if j == 2000:
                 #   print("in each graph we are looping over each node ",j)
                A = np.zeros(shape=AM.shape)

                # Single-source shortest path from node j
                D, p = dijkstra(ed, j)                                          #on dict ed and output D = dict = {node:short dist form source} p =predecessors

                D = np.array(list(D.get(k, float("Inf")) for k in range(node_nr)))
                p[j] = -1

                # Restrict to the connected component of node j
                conn_comp = np.where(D < float("Inf"))[0]

                # To-be DAG adjacency matrix of connected component of node j
                A_cc = A[conn_comp, :][:, conn_comp]                                 #adj matrix for connected components

                # Adjacency matrix of connected component of node j
                AM_cc = AM[conn_comp, :][:, conn_comp]
                D_cc = D[conn_comp]                                                  #dict. for connected component
                conn_comp_converter = np.zeros(shape=(A.shape[0], 1), dtype=int)
                for k in range(conn_comp.shape[0]):
                    conn_comp_converter[conn_comp[k]] = k
                conn_comp_converter = np.vstack([0, conn_comp_converter])
                p_cc = conn_comp_converter[np.array(list(p[k] for k in conn_comp)) + 1]

                # Number of nodes in connected component of node j
                conncomp_node_nr = A_cc.shape[0]
                for v in range(conncomp_node_nr):
                    if p_cc[v] > 0:
                        # Generate A_cc by adding directed edges of form (parent(v), v)
                        A_cc[p_cc[v], v] = 1

                    # Distance from v to j
                    v_dist = D_cc[v]

                    # All neighbors of v in the undirected graph
                    v_nbs = np.where(AM_cc[v, :] > 0)[0]

                    # Distances of neighbors of v to j
                    v_nbs_dists = D_cc[v_nbs]

                    # All neighbors of v in undirected graph who are
                    # one step closer to j than v is; i.e. SP-DAG parents
                    v_parents = v_nbs[v_nbs_dists == (v_dist - 1)]

                    # Add SP-DAG parents to A_cc
                    A_cc[v_parents, v] = 1

                # Computes the descendants & occurence vectors o_j(v), d_j(v)
                # for all v in the connected component
                occ_p, des_p = od_vectors_dag(A_cc, D_cc)                                   #occurence vector and decendants for parent

                if des_p.shape[0] == 1 and j == 0:
                    des[j, 0, 0] = des_p
                    occ[j, 0, 0] = occ_p
                else:
                    # Convert back to the indices of the original graph
                    for v in range(des_p.shape[0]):
                        for l in range(des_p.shape[1]):
                            des[j, conn_comp[v], l] = des_p[v, l]
                    # Convert back to the indices of the original graph
                    for v in range(occ_p.shape[0]):
                        for l in range(occ_p.shape[1]):
                            occ[j, conn_comp[v], l] = occ_p[v, l]

            M = np.zeros(shape=(node_nr, max_diam, max_diam))                               #this is delta cross delta matrix
            # j loops through choices of root
            for j in range(node_nr):
                des_mat_j_root = np.squeeze(des[j, :, :])
                occ_mat_j_root = np.squeeze(occ[j, :, :])
                # v loops through nodes
                for v in range(node_nr):
                    for a in range(max_diam):
                        for b in range(a, max_diam):
                            # M[v,:,:] is M[v]; a = node coordinate in path, b = path length
                            M[v, a, b] += des_mat_j_root[v, b - a]*occ_mat_j_root[v, a]

            if self.calculate_norm_:
                out.append((M, attributes, np.sum(attributes ** 2, axis=1)))
            else:
                out.append((M, attributes))
        #print(" output for the parser", out)
        #print(len(out[0][0][1]),out[0][0][1])
        #e = time.time()
        #print("time taken to parse the input",s-e)
        return out
 

    def pairwise_operation(self, x, y):
        """Graph Hopper kernel as proposed in :cite:`feragen2013scalable`.

        Parameters
        ----------
        x, y : tuple
            Extracted features from `parse_input`.

        Returns
        -------
        kernel : number
            The kernel value.

        """
        #print(x,len(x))
        xp, yp = x[0], y[0]
        #print(yp)
        #print("shape of xp",xp.shape) #(node_nr, max_dia,max_dia)
        #print("second value of x", x[1:])
        
        m = min(xp.shape[1], yp.shape[1])  #min number of columns
        #print("#min number of columns",m)
        m_sq = m**2
        if x[0].shape[1] > m:
            xp = xp[:, :m, :][:, :, :m]   #colums are restricted to m
        elif y[0].shape[1] > m:
            yp = yp[:, :m, :][:, :, :m]

        return self.metric_((xp.reshape(xp.shape[0], m_sq),) + x[1:],   
                            (yp.reshape(yp.shape[0], m_sq),) + y[1:])   #reshaped it to rows of x and cols = m_square


def linear_kernel(x, y):
    """Graph Hopper linear pairwise kernel as proposed in :cite:`feragen2013scalable`.

    Parameters
    ----------
    x, y : tuple
        Extracted features from `parse_input`.

    Returns
    -------
    kernel : number
        The kernel value.
        
    """
    M_i, NA_i = x
    M_j, NA_j = y
    weight_matrix = np.dot(M_i, M_j.T)
    NA_linear_kernel = np.outer(NA_i, NA_j.T)
    #print("dim for both",NA_i.shape,NA_j.shape)
    return np.dot(weight_matrix.flat, NA_linear_kernel.flat) 

def dist_btw_mat_vec(A,B):
    A_expanded = A.unsqueeze(1)
    B_expanded = B.unsqueeze(0)
    diff = A_expanded - B_expanded #elementwise difference
    squared_distance = torch.sum(diff**2, dim=-1) # eulidean distance along last axis
    euclidean_distance = torch.sqrt(squared_distance)
    return euclidean_distance

def simi(dict1,dict2):
    #print('dict1,dict2',dict1,dict2)
    keys1 = [k for k in dict1.keys()]
    keys2 = [k for k in dict2.keys()]
    value1 = np.array([v for v in dict1.values()])
    value2 = np.array([v for v in dict2.values()])
    #print('shape v1', value1.shape)
    #print('shape v2', value2.shape)
    
    h_key1=np.array([hash(element) for element in keys1])
    h_key2=np.array([hash(element) for element in keys2])
    
    delta_matrix = h_key1[:, np.newaxis] == h_key2
    #result = np.sum(delta_matrix)
    dot_prod = np.dot(value1, value2.T)
    
    #print(delta_matrix,dot_prod)
    matrix_prod = np.multiply(delta_matrix, dot_prod)
    
    
    result = np.sum(matrix_prod)
            
    return result
	
def cmk_kernel(x, y):
    M_i, NA_i = x
    M_j, NA_j = y

    # Convert NumPy arrays to PyTorch tensors and move to DEVICE
    M_i = torch.tensor(M_i, device=DEVICE, dtype=torch.float32)
    M_j = torch.tensor(M_j, device=DEVICE, dtype=torch.float32)

    # Compute the weight matrix using matrix multiplication
    weight_matrix = torch.matmul(M_i, M_j.T)

    batch_size_i = 32
    batch_size_j = 32  

    pair_sim = torch.empty((0, len(NA_j)), device=DEVICE, dtype=torch.float32)  # Preallocate tensor for efficiency

    # Process batches
    for start_i in range(0, len(NA_i), batch_size_i):
        end_i = min(start_i + batch_size_i, len(NA_i))
        batch_i = NA_i[start_i:end_i]

        batch_sim = torch.empty((end_i - start_i, 0), device=DEVICE, dtype=torch.float32)  # Row-wise storage

        for start_j in range(0, len(NA_j), batch_size_j):
            end_j = min(start_j + batch_size_j, len(NA_j))
            batch_j = NA_j[start_j:end_j]

            # Extract and convert point vectors
            points_i = torch.stack([torch.tensor(i['points'][1][4:], device=DEVICE, dtype=torch.float32) for i in batch_i])
            points_j = torch.stack([torch.tensor(j['points'][1][4:], device=DEVICE, dtype=torch.float32) for j in batch_j])

            # Compute dot product similarity
            dot_products = torch.matmul(points_i, points_j.T)

            # Create mask for matching labels
            mask = torch.tensor([[i['points'][1][:4] == j['points'][1][:4] for j in batch_j] for i in batch_i],
                                device=DEVICE, dtype=torch.float32)

            # Compute similarity
            similarity = dot_products * mask
            batch_sim = torch.cat([batch_sim, similarity], dim=1)  # Append results column-wise

        pair_sim = torch.cat([pair_sim, batch_sim], dim=0)  # Append results row-wise

    # Compute final similarity score
    s = torch.dot(weight_matrix.flatten(), pair_sim.flatten())

    return s.item()


def linear3dghk_kernel(x, y): 
    M_i, NA_i = x
    M_j, NA_j = y

    M_i = torch.tensor(M_i, device=DEVICE, dtype=torch.float32)
    M_j = torch.tensor(M_j, device=DEVICE, dtype=torch.float32)

    # Compute the weight matrix using matrix multiplication
    weight_matrix = torch.matmul(M_i, M_j.T)

    # Compute similarity matrix
    a = torch.tensor([[simi(node_i, node_j) for node_j in NA_j] for node_i in NA_i], 
                     device=DEVICE, dtype=torch.float32)

    return torch.dot(weight_matrix.flatten(), a.flatten())

def kernelmatrix2distmatrix(K):
    """Convert a Kernel Matrix to a Distance Matrix.

    Parameters
    ----------
    K : np.array, n_dim=2
        The kernel matrix.

    Returns
    -------
    D : np.array, n_dim=2
        The distance matrix.

    """
    diag_K = K.diagonal().reshape(K.shape[0], 1)
    return np.sqrt(diag_K + diag_K.T - 2*K)


def od_vectors_dag(G, shortestpath_dists):
    """Compute the set of occurrence and distance vectors for G.

    Defined in :cite:`feragen2013scalable`.

    Parameters
    ----------
    G : np.array, n_dim=2
        DAG induced from a gappy tree where the indexing of nodes gives a
        breadth first order of the corresponding original graph

    shortestpath_dists : np.array, n_dim=1
        Shortest path distances from the source node.

    Returns
    -------
    occ : np.array, n_dim=2
        n x d descendant matrix occ, where n: `G.shape[0]` loops through the
        nodes of G, and d: 'diameter of G'. The rows of the occ matrix will be
        padded with zeros on the right.

    des : np.array, n_dim=2
        n x d descendant matrix des, where n: `G.shape[0]` loops through the
        nodes of G, and d: 'diameter of G'. The rows of the des matrix will be
        padded with zeros on the right.

    """
    dag_size = G.shape[0]
    DAG_gen_vector = shortestpath_dists + 1

    # This only works when the DAG is a shortest path DAG on an unweighted graph
    gen_sorted = DAG_gen_vector.argsort()
    re_sorted = gen_sorted.argsort()
    sortedG = G[gen_sorted, :][:, gen_sorted]
    delta = int(np.max(DAG_gen_vector))

    # Initialize:
    # For a node v at generation i in the tree, give it the vector
    # [0 0 ... 1 ... 0] of length h_tree with the 1 at the ith place.
    occ = np.zeros(shape=(dag_size, delta), dtype=int)
    occ[0, 0] = 1

    # Initialize:
    # For a node v at generation i in the tree, give it the vector
    # [0 0 ... 1 ... 0] of length delta with the 1 at the ith place.
    des = np.zeros(shape=(dag_size, delta), dtype=int)
    des[:, 0] = np.ones(shape=(1, dag_size))

    for i in range(dag_size):
        edges_starting_at_ith = np.where(np.squeeze(sortedG[i, :]) == 1)[0]
        occ[edges_starting_at_ith, :] = occ[edges_starting_at_ith, :] + \
            repmat(np.hstack([0, occ[i, :-1]]), edges_starting_at_ith.shape[0], 1)

        # Now use message-passing from the bottom of the DAG to add up the
        # edges from each node. This is easy because the vertices in the DAG
        # are depth-first ordered in the original tree; thus, we can just start
        # from the end of the DAG matrix.
        edges_ending_at_ith_from_end = np.where(np.squeeze(sortedG[:, dag_size - i - 1]) == 1)[0]
        des[edges_ending_at_ith_from_end, :] = (
            des[edges_ending_at_ith_from_end, :] +
            repmat(np.hstack([0, des[dag_size - i - 1, :-1]]),
                   edges_ending_at_ith_from_end.shape[0], 1))

    return occ[re_sorted, :], des[re_sorted, :]
    
def paths(A):
    paths =[]
    for i in range(A.shape[0]):
        node_3path = []
        node = []
        node.append([i+1])
        nodes = [x for x in range(A.shape[0]) if x != i]
        nbd1 = [idx for idx in nodes if A[i][idx]>0]
        for j in nbd1:
            nbd2 = [idx for idx in nodes if A[j][idx]>0]
            for k in nbd2:
                nbd3 = [idx for idx in nodes if A[k][idx]>0 and idx!=j]
                for l in nbd3:
                    node_3path.append([i+1,j+1,k+1,l+1])

        if len(node_3path) !=0:
            paths.append(torch.tensor(node_3path))
        else:
            paths.append(torch.tensor(node))
    return paths
    
def node_feature(all_3hop_paths,graph):
    path_dict = {}
    for node in range(len(all_3hop_paths)):
        FV = {}
        for path in all_3hop_paths[node]:
            if len(path) ==4:
                adj_matrix, adj_len,atom_name = create_adjacency_matrix(path, graph)
                #print(atom_name)
                adj_exp = np.exp(adj_len)
                atom_name = atom_name.tolist()
    
                atom_name.append(adj_exp[0,1].item())
                atom_name.append(adj_exp[1,2].item())
                atom_name.append(adj_exp[2,3].item())
                atom_name.append(adj_exp[0,2].item())
                atom_name.append(adj_exp[1,3].item())
                atom_name.append(adj_exp[0,3].item())
                
                '''bond length similarity info is added'''
                simi_1 = np.multiply(adj_matrix, adj_len) #element wise multiplication
                ''' simillarity adj matrix with 6 non-zero enteries'''
            
                simi_2 = np.linalg.matrix_power(simi_1,2)
        
                for i in range(4):
                    for j in range(4):
                        if i!=j:
                            simi_2[i][j] = simi_2[i][j]/adj_len[i][j]
                            
                atom_name.append(simi_2[0,2].item())  
                atom_name.append(simi_2[1,3].item())
                '''bond angle similarity information'''
                
                simi_3 =np.dot(simi_2,simi_1)
                for i in range(4):
                    for j in range(4):
                        if i!=j:
                            simi_3[i][j] = simi_3[i][j]/adj_len[i][j] #need to check it is not symmtric matrix
                            
                torsion_sum = simi_3[0][3]+simi_3[3][0]  
        
                atom_name.append(torsion_sum.item())
                FV[tuple(atom_name[:4])] = atom_name[4:]
            else:
                atom_i_name = tuple([graph.atomic_num[path[0]-1].item(),0,0,0])
                
                FV[atom_i_name] = [1,0,0,0,0,0,0,0,0] #torch.tensor([1],dtype=torch.float32)
        path_dict[node] = FV
        
    return path_dict
    
def create_adjacency_matrix(path, graph):
    num_nodes = len(path)
    #print(path)
    matrix_list = [(0,1,0,0),(1,0,1,0),(0,1,0,1),(0,0,1,0)]
    matrix_tensor = torch.tensor(matrix_list, dtype=torch.float32)
    #print(matrix_tensor)
    
    adj_matrix_0 = torch.zeros((num_nodes, num_nodes))
    #print(adj_matrix_0)
    # Create an adjacency matrix for the given path
    atom_name = []
    for i in range(num_nodes):
        atom_i_name = graph.atomic_num[path[i]-1]
        #print(atom_i_name)
        atom_name.append(atom_i_name.item())
        for j in range(num_nodes):
            cor_atom_i = graph.p[path[i]-1]
            cor_atom_j = graph.p[path[j]-1]
            #print(cor_atom_i)
            distance = torch.norm(cor_atom_i - cor_atom_j)
            #print("distance",distance)
            adj_matrix_0[i,j] = distance
            adj_matrix_0[j,i] = distance
    #print(adj_matrix_0)
    atom_name = torch.tensor(atom_name)
    return matrix_tensor, adj_matrix_0,atom_name
    
def ker_input_3dghk(graphs_list):
    graphs = []
    r = len(graphs_list)
    for i in range(r):
        edge_index = graphs_list[i].edge_index
        atomic_numbers = graphs_list[i].atomic_num
        edge_list = [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.shape[1])]
        l = edge_index.shape[1]
        v =torch.ones(l, dtype=torch.float64)
        s = torch.sparse_coo_tensor(edge_index, v)
        Adj_matrix = s.to_dense()
        #num_nodes = len(atomic_numbers)
        all_3hop_paths = paths(Adj_matrix)
        #print("===============================",all_3hop_paths)
        FV = node_feature(all_3hop_paths,graphs_list[i])
        #edge_att = X.edge_attr
        lst = [{edge for edge in edge_list},FV,{edge: 0 for edge in edge_list}]
        #lst = [{edge for edge in edge_list},path_dict,{edge: x for edge, x in zip(edge_list,edge_att)}]
        graphs.append(lst)
    return graphs

def label_tensor(data):
    y = []
    for graph in data:
        graph_idx = graph.idx
        label = graph.y
        y.append(label)
    return y
    
def best_C_value(C_list, K_mat,X_train,X_val,y_tr,y_val):
    val_acc = []
    for i in C_list:
        clf = SVC(C=i)
        K_train = K_mat[X_train, :] #taking rows of train indicies and columns also
        K_train = K_train[:, X_train]
        K_val = K_mat[X_val, :] #taking rows of test indicies and columns also
        K_val = K_val[:, X_train]
        clf.fit(K_train, y_tr)
        Y_pred = clf.predict(K_val)
        acc_val = accuracy_score(y_val, Y_pred)
        val_acc.append(acc_val)
    best_c = val_acc.index(max(val_acc))
    return C_list[best_c]

def acc_on_test(best_c, K_mat,X_train,y_tr,X_test,y_test):
    clf = SVC(C = best_c)
    K_train = K_mat[X_train, :] #taking rows of train indicies and columns also
    K_train = K_train[:, X_train]
    K_test = K_mat[X_test, :] #taking rows of test indicies and columns also
    K_test = K_test[:, X_train]
    clf.fit(K_train, y_tr)
    Y_pred = clf.predict(K_test)
    acc_tst = accuracy_score(y_test, Y_pred)
    return acc_tst
    
def avg_acc_std(num,kernel_mat,X,y,data,C_list):
    acc_list = []
    for i in num:
        X_train, X_v, y_tr, y_v = train_test_split(X, y, test_size=0.3, stratify=y,shuffle = True,random_state=i)
        X_val, X_test, y_val, y_test = train_test_split(X_v, y_v, test_size=0.5, stratify=y_v,shuffle = True,random_state=i)
        best_c = best_C_value(C_list,kernel_mat,X_train,X_val,y_tr,y_val)
        acc_tst = acc_on_test(best_c, kernel_mat,X_train,y_tr,X_test,y_test)
        acc_list.append(acc_tst)
    print(f"Accuracy for each splits: ", acc_list)
    return np.mean(acc_list),np.std(acc_list)

''' 3-dimensional c-motif graph kernel'''
def three_hop_paths(A):
    paths =[]
    for i in range(A.shape[0]):
        node_3path = []
        node = []
        node.append([i+1])
        nodes = [x for x in range(A.shape[0]) if x != i]
        nbd1 = [idx for idx in nodes if A[i][idx]>0]
        for j in nbd1:
            nbd2 = [idx for idx in nodes if A[j][idx]>0]
            for k in nbd2:
                nbd3 = [idx for idx in nodes if A[k][idx]>0 and idx!=j]
                for l in nbd3:
                    node_3path.append([i+1,j+1,k+1,l+1])
                
        if len(node_3path) !=0:
            paths = paths + node_3path
        else:
            paths = paths +node
    return paths
    
def create_adj_matrix(path, graph):
    num_nodes = len(path)
    matrix_list = [(0,1,0,0),(1,0,1,0),(0,1,0,1),(0,0,1,0)]
    matrix_tensor = torch.tensor(matrix_list, dtype=torch.float32)
    
    adj_matrix_0 = torch.zeros((num_nodes, num_nodes))
    atom_name = []
    for i in range(num_nodes):
        atom_i_name = graph.atomic_num[path[i]-1]
        atom_name.append(atom_i_name.item())
        for j in range(num_nodes):
            cor_atom_i = graph.p[path[i]-1]
            cor_atom_j = graph.p[path[j]-1]
            distance = torch.norm(cor_atom_i - cor_atom_j)

            adj_matrix_0[i,j] = distance
            adj_matrix_0[j,i] = distance
    return matrix_tensor, adj_matrix_0,atom_name

def cMotif_graph(diff_paths,df):
    c_motif_graph= nx.Graph()
    cmg_feature = []
    # Iterate through c-motif and add nodes to the graph
    for idx, c_motif in enumerate(diff_paths):
        if len(c_motif) ==4:
            adj_matrix, adj_len,atom_name = create_adj_matrix(c_motif, df)
            adj_exp = np.exp(adj_len)
        
            atom_name.append(adj_exp[0,1].item())
            atom_name.append(adj_exp[1,2].item())
            atom_name.append(adj_exp[2,3].item())
            atom_name.append(adj_exp[0,2].item())
            atom_name.append(adj_exp[1,3].item())
            atom_name.append(adj_exp[0,3].item())
            
            '''bond length similarity info is added'''
            simi_1 = np.multiply(adj_matrix, adj_len) 
            ''' simillarity adj matrix with 6 non-zero enteries'''
        
            simi_2 = np.linalg.matrix_power(simi_1,2)
            for i in range(4):
                for j in range(4):
                    if i!=j:
                        simi_2[i][j] = simi_2[i][j]/adj_len[i][j]
                        
            atom_name.append(simi_2[0,2].item())  
            atom_name.append(simi_2[1,3].item())
            '''bond angle similarity information'''
            
            simi_3 =np.dot(simi_2,simi_1)
            for i in range(4):
                for j in range(4):
                    if i!=j:
                        simi_3[i][j] = simi_3[i][j]/adj_len[i][j] 
                        
            torsion_sum = simi_3[0][3]+simi_3[3][0]  
    
            atom_name.append(torsion_sum.item())
            #rint(atom_name)
            feature =list()
            feature.append(c_motif)
            feature.append(atom_name)
            c_motif_graph.add_node(idx, points=feature)
            cmg_feature.append(atom_name)
         
        else :
    
            atom_i_name = [df.atomic_num[c_motif[0]-1].item(),0,0,0]
            atom_i_name = atom_i_name + [1,0,0,0,0,0,0,0,0] #torch.tensor([1],dtype=torch.float32)
            feature =list()
            feature.append(c_motif)
            feature.append(atom_i_name)
        
            c_motif_graph.add_node(idx, points=feature)
            cmg_feature.append(atom_i_name)
            
            
            
    '''how to make a edge between any two cmotif'''
    for i in range(len(diff_paths)):
        for j in range(i + 1, len(diff_paths)):
            common_points = set(diff_paths[i]) & set(diff_paths[j])
            if common_points:
                c_motif_graph.add_edge(i, j, common_points=list(common_points))#undirected graph

    return c_motif_graph, cmg_feature

def ker_input_cmgk(graphs_list):
    graphs = []
    r = len(graphs_list)
    for i in range(r):
        edge_index = graphs_list[i].edge_index
        atomic_numbers = graphs_list[i].atomic_num
        edge_list = [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.shape[1])]
        l = edge_index.shape[1]
        v =torch.ones(l, dtype=torch.float64)
        s = torch.sparse_coo_tensor(edge_index, v)
        Adj_matrix = s.to_dense()
       
        all_3hop_paths = three_hop_paths(Adj_matrix)
        #print("===============================",all_3hop_paths)
        HG,FV = cMotif_graph(all_3hop_paths,graphs_list[i])
        node_dict = dict(HG.nodes.data())  
        edge_list = list(HG.edges())
        if len(edge_list) ==0:
            edge_list.append((0,1))
        iterable_list = [edge_list, node_dict]
        lst = [{edge for edge in edge_list},node_dict,{edge: 0 for edge in edge_list}]
        graphs.append(lst)
    return graphs
   

if __name__ == '__main__':
    import random
    from load_data import process_dataset
    import argparse
    # Create an argument parser for the installer of pynauty
    parser = argparse.ArgumentParser(
        description='Measuring classification accuracy '
                    ' on multiscale_laplacian_fast')

    parser.add_argument(
        '--dataset',
        help='choose the dataset you want the tests to be executed',
        type=str,
        default="1798"
    )

    parser.add_argument(
        '--full',
        help='fit_transform the full graph',
        action="store_true")

    mec = parser.add_mutually_exclusive_group()

    mec.add_argument(
        '--kernel_type',
        help='choose a linear3dghk or cmg_kernel',
        type=str,  
        choices=['linear3dghk', 'cmg_kernel'])

    
    parser.add_argument(
    '--seed',
    help='Set the random seed for reproducibility',
    type= int,
    default=42 )

    parser.add_argument(
    '--num_split',
    help='number of different splits ',
    type= int,
    default=20  
    )
    
    # Get the dataset name
    args = parser.parse_args()
    dataset_name = args.dataset
    kernel = args.kernel_type
    seed = args.seed
    split = args.num_split
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    active_molecules,inactive_molecules= process_dataset(dataset_name)
    random.seed(seed)
    idx = random.sample(range(len(inactive_molecules)), len(active_molecules))
    num = random.sample(range(1, 101),split)
    new_data_list = active_molecules + [inactive_molecules[i] for i in idx]
 
    start = time.time()
    
    gk = GraphHopper(normalize=True,kernel_type=kernel) 
    if kernel == 'linear3dghk':
    	input_3d = ker_input_3dghk(new_data_list)
    elif kernel=='cmg_kernel':
    	input_3d = ker_input_cmgk(new_data_list)
    else:
    	raise ValueError(f"Unknown kernel type: {kernel}") 
    	
    K_3d =gk.fit_transform(input_3d)
    X = [i for i in range(len(new_data_list))]
    y = np.array([i.y.item() for i in new_data_list])
    int = [-3 + i for i in range(6)]
    C = [10**i for i in int]
    print("=======================================================================================")
    acc, std = avg_acc_std(num,K_3d,X,y,new_data_list,C)
    print(f"accuracy for {kernel} on {dataset_name} dataset is: {acc} and std is: {std}")
    print("=======================================================================================")
    end = time.time()
    print("Total time for classification :", end - start)

