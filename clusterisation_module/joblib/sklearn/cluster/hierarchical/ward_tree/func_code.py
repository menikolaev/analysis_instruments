# first line: 89
def ward_tree(X, connectivity=None, n_components=None, copy=None,
              n_clusters=None):
    """Ward clustering based on a Feature matrix.

    Recursively merges the pair of clusters that minimally increases
    within-cluster variance.

    The inertia matrix uses a Heapq-based representation.

    This is the structured version, that takes into account some topological
    structure between samples.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        feature matrix  representing n_samples samples to be clustered

    connectivity : sparse matrix (optional).
        connectivity matrix. Defines for each sample the neighboring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is None, i.e, the Ward algorithm is unstructured.

    n_components : int (optional)
        Number of connected components. If None the number of connected
        components is estimated from the connectivity matrix.

    n_clusters : int (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. In this case, the
        complete tree is not computed, thus the 'children' output is of
        limited use, and the 'parents' output should rather be used.
        This option is valid only when specifying a connectivity matrix.

    Returns
    -------
    children : 2D array, shape (n_nodes, 2)
        The children of each non-leaf node. Values less than `n_samples` refer
        to leaves of the tree. A greater value `i` indicates a node with
        children `children[i - n_samples]`.

    n_components : int
        The number of connected components in the graph.

    n_leaves : int
        The number of leaves in the tree

    parents : 1D array, shape (n_nodes, ) or None
        The parent of each node. Only returned when a connectivity matrix
        is specified, elsewhere 'None' is returned.
    """
    if copy is not None:
        warnings.warn("The copy argument is deprecated and will be removed "
                      "in 0.16. The connectivity is now always copied.",
                      DeprecationWarning)

    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))
    n_samples, n_features = X.shape

    if connectivity is None:
        from scipy.cluster import hierarchy     # imports PIL

        if n_clusters is not None:
            warnings.warn('Partial build of the tree is implemented '
                          'only for structured clustering (i.e. with '
                          'explicit connectivity). The algorithm '
                          'will build the full tree and only '
                          'retain the lower branches required '
                          'for the specified number of clusters',
                          stacklevel=2)
        out = hierarchy.ward(X)
        children_ = out[:, :2].astype(np.intp)
        return children_, 1, n_samples, None

    connectivity = _fix_connectivity(X, connectivity,
                                     n_components=n_components)
    if n_clusters is None:
        n_nodes = 2 * n_samples - 1
    else:
        if n_clusters > n_samples:
            raise ValueError('Cannot provide more clusters than samples. '
                '%i n_clusters was asked, and there are %i samples.'
                % (n_clusters, n_samples))
        n_nodes = 2 * n_samples - n_clusters

    # create inertia matrix
    coord_row = []
    coord_col = []
    A = []
    for ind, row in enumerate(connectivity.rows):
        A.append(row)
        # We keep only the upper triangular for the moments
        # Generator expressions are faster than arrays on the following
        row = [i for i in row if i < ind]
        coord_row.extend(len(row) * [ind, ])
        coord_col.extend(row)

    coord_row = np.array(coord_row, dtype=np.intp, order='C')
    coord_col = np.array(coord_col, dtype=np.intp, order='C')

    # build moments as a list
    moments_1 = np.zeros(n_nodes, order='C')
    moments_1[:n_samples] = 1
    moments_2 = np.zeros((n_nodes, n_features), order='C')
    moments_2[:n_samples] = X
    inertia = np.empty(len(coord_row), dtype=np.float, order='C')
    _hierarchical.compute_ward_dist(moments_1, moments_2, coord_row, coord_col,
                                    inertia)
    inertia = list(six.moves.zip(inertia, coord_row, coord_col))
    heapify(inertia)

    # prepare the main fields
    parent = np.arange(n_nodes, dtype=np.intp)
    used_node = np.ones(n_nodes, dtype=bool)
    children = []

    not_visited = np.empty(n_nodes, dtype=np.int8, order='C')

    # recursive merge loop
    for k in range(n_samples, n_nodes):
        # identify the merge
        while True:
            inert, i, j = heappop(inertia)
            if used_node[i] and used_node[j]:
                break
        parent[i], parent[j] = k, k
        children.append((i, j))
        used_node[i] = used_node[j] = False

        # update the moments
        moments_1[k] = moments_1[i] + moments_1[j]
        moments_2[k] = moments_2[i] + moments_2[j]

        # update the structure matrix A and the inertia matrix
        coord_col = []
        not_visited.fill(1)
        not_visited[k] = 0
        _hierarchical._get_parents(A[i], coord_col, parent, not_visited)
        _hierarchical._get_parents(A[j], coord_col, parent, not_visited)
        # List comprehension is faster than a for loop
        [A[l].append(k) for l in coord_col]
        A.append(coord_col)
        coord_col = np.array(coord_col, dtype=np.intp, order='C')
        coord_row = np.empty(coord_col.shape, dtype=np.intp, order='C')
        coord_row.fill(k)
        n_additions = len(coord_row)
        ini = np.empty(n_additions, dtype=np.float, order='C')

        _hierarchical.compute_ward_dist(moments_1, moments_2,
                                        coord_row, coord_col, ini)

        # List comprehension is faster than a for loop
        [heappush(inertia, (ini[idx], k, coord_col[idx]))
            for idx in range(n_additions)]

    # Separate leaves in children (empty lists up to now)
    n_leaves = n_samples
    children = np.array(children)  # return numpy array for efficient caching

    return children, n_components, n_leaves, parent
