from helper_functions import *


def force_newton(edge, sender_node, receiver_node, globals_):
    """Returns the update edge features."""
    return sender_node[:, :1] * receiver_node[:, :1] * edge / jnp.linalg.norm(
        edge, axis=-1, keepdims=True)


class Simulator:
    def __init__(self, x0, v0, force):
        # Initial position and velocity
        self.nParticles = x0.shape[0]
        self.graph = numpy_to_graph(x0, v0)
        self.force = force
        self.dt = None
        self.net = None
        self.net2 = None
        self.net3 = None

    def create_nets(self):
        self.net = jraph.GraphNetwork(update_edge_fn=self.force,
                                      update_node_fn=update_node_dummy, )

        self.net2 = jraph.GraphNetwork(update_edge_fn=update_edge_dummy,
                                       update_node_fn=self.get_velocity_position, )

        self.net3 = jraph.GraphNetwork(update_edge_fn=self.get_distance,
                                       update_node_fn=update_node_dummy, )

    def get_velocity_position(self, node, sender, receiver, globals_):
        sumForces = sender - receiver
        a = sumForces / node[:, :1]
        dv = a * self.dt
        node = node.at[:, 4:].set(node[:, 4:] + dv)
        node = node.at[:, 1:4].set(node[:, 1:4] + node[:, 4:] * self.dt)
        return node

    def get_distance(self, edge, sender_node, receiver_node, globals_):
        return receiver_node[:, 1:4] - sender_node[:, 1:4]

    def forward_step(self):
        if self.net is None:
            # On first step, create the nets
            self.create_nets()

        self.graph = self.net(self.graph)
        self.graph = self.net2(self.graph)
        self.graph = self.net3(self.graph)

    def simulate(self, nSteps, dt):
        self.dt = dt
        X = jnp.zeros([nSteps, self.nParticles, 3])
        V = jnp.zeros([nSteps, self.nParticles, 3])
        for i in range(nSteps):
            X = X.at[i].set(self.graph.nodes[:, 1:4])
            V = V.at[i].set(self.graph.nodes[:, 4:])
            self.forward_step()
        return X, V


if __name__ == "__main__":
    import jax.random as jrnd

    seed = 0
    key = jrnd.PRNGKey(seed)

    # Create a simple graph with two bodies (one distance)
    X0 = jrnd.uniform(key, [2, 3])
    V0 = jnp.zeros([2, 3])

    S = Simulator(X0, V0, force_newton)
    X, V = S.simulate(nsteps=10, dt=0.1)
