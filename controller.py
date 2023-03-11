import jax
import jax.numpy as jnp
from trajax import optimizers

dt = 0.1
horizon = 100
goal_state = jnp.array([10.0, 0.0, 0.0])


@jax.jit
def system(state, action, timestep):
  """Classic (omnidirectional) wheeled robot system.
  Args:
    x: state, (3, ) array
    u: control, (2, ) array
    t: scalar time
  Returns:
    xdot: state time derivative, (3, )
  """
  x = state
  u = action
  
  c = jnp.cos(x[2])
  s = jnp.sin(x[2])
  
  g = jnp.array([[c, -s, 0],
                 [s, c, 0],
                 [0, 0, 1]])
  
  xdot = g @ u

  return xdot

def cost(x, u, t):
  err = x - goal_state
  stage_cost = 0.1 * jnp.dot(err, err) + 0.01 * jnp.dot(u, u)
  final_cost = 1000 * jnp.dot(err, err)
    
  return jnp.where(t == horizon, final_cost, stage_cost)

def dynamics(x, u, t):
  return x + dt * system(x, u, t)

x0 = jnp.array([0.0, 0.0, 0.1])

X, U, _, _, _, _, _ = optimizers.ilqr(
        cost,
        dynamics,
        x0,
        jnp.zeros((horizon, 3)),
        maxiter=1000
    ) 

import matplotlib.pyplot as plt

plt.plot(X)
plt.show()
plt.plot(U)
plt.show()
print(U)