import torch


def meshgrid_old(x, y=None):
    if y is None:
        y = x
    x = torch.tensor(x)
    y = torch.tensor(y)
    m, n = x.size(0), y.size(0)
    grid_x = x[None].expand(n, m).contiguous()
    grid_y = y[:, None].expand(n, m).contiguous()
    return grid_x, grid_y


def meshgrid(x, y):
    x = torch.tensor(x)
    y = torch.tensor(y)
    x_exp_shape = tuple(1 for _ in y.shape) + x.shape
    y_exp_shape = y.shape + tuple(1 for _ in x.shape)

    xgrid = x.reshape(x_exp_shape).repeat(y_exp_shape)
    ygrid = y.reshape(y_exp_shape).repeat(x_exp_shape)
    new_shape = y.shape + x.shape
    xgrid = xgrid.reshape(new_shape)
    ygrid = ygrid.reshape(new_shape)

    return xgrid, ygrid
