# from https://github.com/slundberg/shap/issues/121
IN_IPYNB = None

def in_ipynb():
  global IN_IPYNB
  if IN_IPYNB is not None:
    return IN_IPYNB

  try:
    cfg = get_ipython().config
    if type(get_ipython()).__module__.startswith("ipykernel."):
    # if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
      # print ('Running in ipython notebook env.')
      IN_IPYNB = True
      return True
    else:
      return False
  except NameError:
    # print ('NOT Running in ipython notebook env.')
    return False

if in_ipynb():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
