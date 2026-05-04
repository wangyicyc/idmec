def rk4(dxdt, dt, xt, **kwargs):
    k1 = dxdt(xt, **kwargs)
    k2 = dxdt(xt + 0.5 * dt * k1, **kwargs)
    k3 = dxdt(xt + 0.5 * dt * k2, **kwargs)
    k4 = dxdt(xt + dt * k3, **kwargs)

    xt_new = xt + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0
    return xt_new


def euler(dxdt, dt, xt, **kwargs):
    return xt + dxdt(xt, **kwargs) * dt


if __name__ == "__main__":

    def dxdt(x, u, r):
        return x * u * r

    a = rk4(dxdt, dt=0.1, xt=1, u=2, r=3)
    print(a)
