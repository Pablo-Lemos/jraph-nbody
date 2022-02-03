import jax.numpy as jnp
import jax.random as jrnd
from simulator import *


def generate_simulation(nParticles, nSteps, dt, seed, mass, dMass=0.1):
    key = jrnd.PRNGKey(seed)
    x0 = jrnd.uniform(key, [nParticles, 3])
    v0 = jnp.zeros([nParticles, 3])
    masses = mass * (1 + dMass * jrnd.normal(key, [nParticles]))
    S = Simulator(x0, v0, force_newton, masses)
    x, v = S.simulate(nSteps, dt)
    # Return only the last step
    return x[-1], v[-1]


def main(nSims=1000, nParticles=100, nSteps=100, dt=0.01, minMass=1,
         maxMass=10):

    G = 39.478  # AU3 yr-2 Msun-1
    masses = jnp.linspace(minMass, maxMass, nSims)
    data = jnp.zeros([nSims, nParticles, 6])
    for i, mass in enumerate(masses):
        if i%100 ==0 and i>0: print(f"Generated {i} simulations.")
        x, v = generate_simulation(nParticles, nSteps, dt, i, mass)
        data.at[i, :, :3].set(x)
        data.at[i, :, 3:].set(v)

if __name__ == "__main__":
    data = main()
    jnp.save("./data/sims.npy", data)

