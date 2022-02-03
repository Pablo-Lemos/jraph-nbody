import jraph
import jax.numpy as jnp


def get_distances(X):
    nx = X.shape[0]
    return (X[:, None, :] - X[None, :, :])[jnp.tril_indices(nx, k=-1)]


def get_receivers_senders(nx):
    return jnp.tril_indices(nx, k=-1)


def numpy_to_graph(X, V, masses = None):
    nx = X.shape[0]
    receivers, senders = get_receivers_senders(nx)
    edges = get_distances(X)

    if masses is None:
        # Default all masses to one
        masses = jnp.ones(nx)
    elif isinstance(masses, (int, long, float)):
        masses = masses*jnp.ones(nx)
    else:
        assert len(masses) == nx, 'Wrong size for masses'
        assert isinstance(masses, jax.numpy.ndarray), "Masses must be a jax " \
                                                      "array"

    nodes = jnp.concatenate([masses.reshape([-1, 1]), X, V], axis=1)
    graph = jraph.GraphsTuple(nodes=nodes, senders=senders, receivers=receivers,
                              edges=edges, n_node=nx, n_edge=nx*(nx+1)//2, globals=None)
    return graph


def update_edge_dummy(edge, sender_node, receiver_node, globals_):
    return edge


def update_node_dummy(node, sender, receiver, globals_):
    return node