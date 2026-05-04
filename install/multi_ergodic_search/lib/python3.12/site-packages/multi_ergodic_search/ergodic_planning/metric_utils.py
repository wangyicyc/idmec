import jax.numpy as jnp

from jax import vmap, jit


@jit
def gaussian_function(x, mean, sigma):
    # coeff = mvn.pdf(x=sigma, mean=0, cov=sigma)
    # sigma - standard deviation
    # 2d gaussian
    # return jnp.exp(-jnp.sum((x - mean) ** 2) / (1.0 * sigma**2))
    # return jnp.exp(-jnp.sum((x - mean) ** 2) / (1.0 * sigma**2)) * jnp.exp(1.0)
    return jnp.exp(-jnp.sum((x - mean) ** 2) / (2.0 * sigma**2))


@jit
def sigmoid_function(x, mean, sigma, rho=1.0):
    return 1 / (1 + jnp.exp(rho * (jnp.sum((x - mean) ** 2) / sigma**2 - 1)))


def gaussian_sum_topology(xti, xtj, index_pair: list[tuple[int, int]], _nx, pij_R):
    summer = 0.0
    # coeff = mvn.pdf(x=pij_R, mean=0, cov=pij_R)
    for i, j in index_pair:
        xi = xti[(i * _nx) : (i * _nx) + 2]
        xj = xtj[(j * _nx) : (j * _nx) + 2]
        # _delta_ij = xi - xj
        # summer += (
        #     mvn.pdf(x=xj, mean=xi, cov=jnp.array([[pij_R, 0.0], [0.0, pij_R]])) / coeff
        # )
        summer += gaussian_function(x=xj, mean=xi, sigma=pij_R)
    return summer / len(index_pair)


def sigmoid_sum_topology(
    xti, xtj, index_pair: list[tuple[int, int]], _nx, pij_R, pij_alpha=1.0
):
    summer = 0.0
    pij_K = 1.0
    for i, j in index_pair:
        xi = xti[(i * _nx) : (i * _nx) + 2]
        xj = xtj[(j * _nx) : (j * _nx) + 2]
        summer += pij_K * sigmoid_function(x=xj, mean=xi, sigma=pij_R, rho=pij_alpha)
    return summer / len(index_pair)


def pair_connection_doubleInt(
    traj: jnp.ndarray,
    _func_pair: callable,
    robot_pair: list[tuple[int, int]],
    beta_future,
    _nx: int,
    period_num: int,
    traj_length: int,
):

    traj_trunc = traj[:traj_length]
    x_traj_reshape = traj_trunc.reshape(period_num, -1, traj_trunc.shape[-1])
    sub_tsteps = x_traj_reshape[0].shape[0]
    idxs = jnp.arange(period_num, dtype=jnp.int32)
    _connection_probability = []
    # 如果增加机器人，则需要修改此处
    for pair in robot_pair:
        _connection_probability.append(
            vmap(
                lambda idx: jnp.sum(
                    vmap(vmap(_func_pair, in_axes=(None, 0)), in_axes=(0, None))(
                        x_traj_reshape[idx][:, (pair[0] * _nx) : (pair[0] * _nx) + 2],
                        x_traj_reshape[idx][:, (pair[1] * _nx) : (pair[1] * _nx) + 2],
                    )
                )
                / (sub_tsteps**2)
            )(idxs)
        )
    connection_probability = jnp.concatenate(_connection_probability)
    # connection_probability = jnp.sum(jnp.stack(_connection_probability), axis=0)
    return connection_probability