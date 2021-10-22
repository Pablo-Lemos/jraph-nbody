import jraph
import jax.numpy as jnp

def get_distances(X):
    nx = X.shape[0]
    return (X[:, None, :] - X[None, :, :])[jnp.tril_indices(nx, k=-1)]

def get_receivers_senders(nx):
    return jnp.tril_indices(nx, k=-1)

def numpy_to_graph(X, V):
    nx = X.shape[0]
    receivers, senders = get_receivers_senders(nx)
    edges = get_distances(X)
    masses = jnp.ones(nx)
    nodes = jnp.concatenate([masses.reshape([-1, 1]), X, V], axis=1)
    graph = jraph.GraphsTuple(nodes=nodes, senders=senders, receivers=receivers,
                              edges=edges, n_node=nx, n_edge=nx*(nx+1)//2, globals=None)
    return graph


def update_edge_dummy(edge, sender_node, receiver_node, globals_):
    return edge

def update_node_dummy(node, sender, receiver, globals_):
    return node

if __name__ == "__main__":
    import jax.random as jrnd
    seed = 0
    key = jrnd.PRNGKey(seed)

    # Create a simple graph with two bodies (one distance)
    X = jrnd.uniform(key, [2, 3])
    V = jnp.zeros([2, 3])
    print(numpy_to_graph(X, V))