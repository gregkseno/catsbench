from catsbench.stylegan2.torch_utils import persistence

@persistence.import_hook
def _remap_legacy_imports(meta):
    s = meta.module_src
    s = s.replace('from torch_utils', 'from catsbench.stylegan2.torch_utils')
    s = s.replace('import torch_utils', 'import catsbench.stylegan2.torch_utils as torch_utils')
    s = s.replace('import dnnlib', 'from catsbench.stylegan2 import dnnlib as dnnlib')
    meta.module_src = s
    return meta