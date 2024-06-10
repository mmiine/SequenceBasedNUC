from pathlib import Path
import glob

def increment_dir(dir, comment=''):
    # Increments a directory runs/expl --> runs/exp2_comment
    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    d = sorted(glob.glob(dir + '*'))  # directories
    if len(d):
        n = max([int(x[len(dir):x.find('_')] if '_' in x else None) for x in d]) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')
