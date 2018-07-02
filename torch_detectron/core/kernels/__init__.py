import os.path

from torch.utils.cpp_extension import load as load_ext

def _load_C_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    print(this_dir)
    extra_include_paths = [this_dir]
    sources = ['compute_flow.cpp', 'compute_flow.cu']
    sources = [os.path.join(this_dir, s) for s in sources]
    return load_ext('detectron_modules', sources, extra_include_paths=extra_include_paths)

_C = _load_C_extensions()
