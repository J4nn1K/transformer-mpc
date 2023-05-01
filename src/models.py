# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Tuple, Type

import flax.linen as nn
import jax.numpy as jnp
import jax

from trajax import optimizers

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer.
  """

  posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

  @nn.compact
  def __call__(self, inputs):
    """Applies the AddPositionEmbs module.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
    return inputs + pe


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            x)
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)(
            x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)

    return x + y


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  add_position_embedding: bool = True

  @nn.compact
  def __call__(self, x, *, train):
    """Applies Transformer model on the inputs.

    Args:
      x: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert x.ndim == 3  # (batch, len, emb)

    if self.add_position_embedding:
      x = AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input')(x)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads)(x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)

    return encoded


class OCSolver(nn.Module):
  """CFTOC Solver."""

  dt: float
  horizon: int
  cost_weights: dict
  u_des: list

  @nn.compact
  def __call__(self, inputs):
  
    x = inputs
    n, _ = x.shape
    
    P = x[:, :36].reshape((n, 6, 6))
    q = x[:, 36:].reshape((n, 6))
        
    u_opt_list = []
    for i in range(n):
    
      @jax.jit
      def system(x, u, t):
        """Classic (omnidirectional) wheeled robot system.
        Args:
          x: state, (3, ) array
          u: control, (3, ) array
          t: scalar time
        Returns:
          xdot: state time derivative, (3, )
        """
        c = jnp.cos(x[2])
        s = jnp.sin(x[2])
        A = jnp.array([[c, -s, 0],
                       [s, c, 0],
                       [0, 0, 1]])
        xdot = A @ u
        return xdot

      def dynamics(x, u, t):
        return x + self.dt * system(x, u, t)  

      def cost(x, u, t):
        w = self.cost_weights
        
        u_err = jnp.array(self.u_des) - u
        
        xu = jnp.concatenate([x,u])
        
        stage_cost = w['reference'] * jnp.dot(u_err, u_err)
        stage_cost += w['learned'] * jnp.matmul(jnp.matmul(jnp.matmul(xu.T, P[i].T), P[i]), xu)
        stage_cost += w['learned'] * jnp.matmul(q[i].T, xu)
        
        # stage_cost = jnp.dot(jnp.concatenate([x,u]), jnp.dot(P[i], jnp.concatenate([x,u]))) + jnp.dot(q[i], jnp.concatenate([x,u])) + jnp.dot(u_x_err, u_x_err) + jnp.dot(u,u)
        # stage_cost = jnp.dot(jnp.concatenate([x,u]), jnp.dot(P[i], jnp.concatenate([x,u]))) + jnp.dot(q[i], jnp.concatenate([x,u])) + 100*jnp.dot(u_x_err, u_x_err) + jnp.dot(u,u)
        
        # final_cost = 0
        # return jnp.where(t == horizon, final_cost, stage_cost)
        
        return stage_cost

      x0 = jnp.array([0., 0., 0.])
      u0 = jnp.zeros((self.horizon, 3))
      _, U, _, _, _, _, _ = optimizers.ilqr(
          cost,
          dynamics,
          x0,
          u0,
          maxiter=1000
      ) 
      
      # print(U[:,0])
      
      u_opt_list.append(U)    
    
    return jnp.array(u_opt_list)


class MPCTransformer(nn.Module):
  """MPCTransformer."""

  patches: Any
  transformer: Any
  solver: Any
  hidden_size: int
  num_output: int
  head_bias_init: float = 0.
  encoder: Type[nn.Module] = Encoder
  oc_solver: Type[nn.Module] = OCSolver
  model_name: Optional[str] = None  

  @nn.compact
  def __call__(self, inputs, *, train):

    x = inputs
    n, h, w, c = x.shape

    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(features=self.hidden_size,
                kernel_size=self.patches,
                strides=self.patches,
                padding='VALID',
                name='embedding')(x)
    # Here, x is a grid of embeddings.

    # (Possibly partial) Transformer.
    if self.transformer is not None:
      n, h, w, c = x.shape
      x = jnp.reshape(x, [n, h * w, c])

      x = self.encoder(name='Transformer', **self.transformer)(x, train=train)
      
    # Final embedding of a single token 
    x = x[:, -1]
    
    
    # Linear layer
    # x = x[:, 0]
    x = nn.Dense(features=self.num_output,
                 name='head')(x)
    # x = nn.tanh(x)
    
    x = self.oc_solver(name='Solver', **self.solver)(x)

    return x