from typing import List
from yaml import safe_load, safe_dump
import os
import sys

def compare_results(run_dirs: List[str] = None, run_dir_spec: str = None):
    assert not all((run_dirs is not None, run_dir_spec is not None)), \
        f'error, `run_dirs` and `run_dir_spec` cannot be set both'
    if run_dir_spec is not None:
        run_dirs = [each for each in os.popen(f'echo {run_dir_spec}').read().split() if each]
        if run_dirs[0].find('*') > 0:
            raise ValueError(f"error run_dir_spec '{run_dir_spec}'")

    print(f'found run_dirs: {run_dirs}')
    for each in run_dirs: 
        with open(os.path.join(each, 'stat_results.yaml'), 'r') as f:
            stats_yaml = safe_load(f)
            print(f'run name: {os.path.split(each)[1]}')
            for key in ['uas', 'las']:
                stats_yaml.pop(key)
            safe_dump(stats_yaml, sys.stdout)

if __name__ == '__main__':
    compare_results(run_dir_spec='./models/04-12*')