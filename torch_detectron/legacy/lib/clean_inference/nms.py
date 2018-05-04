def _load_C_extensions():
    import os.path
    from torch.utils.cpp_extension import load as load_ext

    this_dir = os.path.dirname(os.path.abspath(__file__))
    source = [
        'nms.cpp',
    ]
    source = [os.path.join(this_dir, s) for s in source]
    return load_ext('nms_modules', source, extra_cflags=['-O3'],)

_C = _load_C_extensions()

nms = _C.nms


if __name__ == '__main__':
    import torch
    from utils.cython_nms import nms as cython_nms
    N = 1000
    t = 0.1
    a = torch.rand(N, 4)
    a[:, 2:] = a[:, :2] + torch.rand(N, 2) / 2
    a *= 100
    b = torch.rand(N)
    aa = torch.cat((a, b[:, None]), 1).numpy()

    keep1 = nms(a, b, t).long()
    keep2 = torch.from_numpy(cython_nms(aa, t)).long()
    print(keep1.equal(keep2))
    from IPython import embed; embed()
