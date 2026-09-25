from contextlib import redirect_stdout
import csv
from datetime import datetime
from inspect import getfullargspec
from pathlib import Path
import shutil
import sys

from cluster_utils import finalize_job, initialize_job
from sklearn.model_selection import ParameterGrid
from smart_settings.param_classes import recursive_objectify


def run(params, path):
    raise NotImplementedError("'run' function not found!")

def main(globals_):
    """General main function.

    If no command line arguments are passed, will call the "run" function. If
    one command line argument is passed, this will be first interpreted as the 
    path to a config.yaml file. If this fails, it will be interpreted as the
    name of the function that is supposed to be run. With the option
    `--parameter-dict`, a python dictionary can also be passed (as a string)
    instead of a config.yaml file.
    The following config keys have a special meaning: `function` (the function
    that will be called; if not specified, `run` is used), `conf` (a config
    dict whose values will take precedent over those in the config file),
    `name`, `run`, and `working_dir`. Temporary runs should not have `name`
    set, normal runs should be named. If not set, `run` will be set to a
    timestamp. Then `name` will be set to `name`-`run` (or just `run` for
    temporary runs). The output path is set to `tmp/` for temporary runs and
    to `dat/runs/name` for named runs, unless `working_dir` is set, in which
    case the environment is assumed to be a cluster and this directory is used.
    Standard output is redirected to a `log.txt` file in the output path,
    unless the run is temporary (unnamed) or `interactive` is explicitly set.
    """
    now = datetime.now()
    globals_ = globals() | globals_

    # Read hyperparameters (either params from file or single function name)
    conf = {}
    try:
        params = initialize_job(sys.argv + ['--parameter-dict', '{}']
            if len(sys.argv) == 1 else [], verbose=False)
    except FileNotFoundError:
        conf['function'] = sys.argv[1]
        sys.argv = [sys.argv[0]]
        params = initialize_job(
            sys.argv + ['--parameter-dict', '{}'], verbose=False)
    param_grid = ParameterGrid(params.get('param_grid', {}))
    params = dictify(params)
    params = recursive_objectify(params, make_immutable=False)
    params.update(conf | dict(params.get('conf') or {}))

    # Configure working directory and job name
    named = 'name' in params
    cluster = 'working_dir' in params
    run_ = str(params.get('run', '')) or now.strftime("%Y%m%d%H%M%S")
    name = (n + '-' if (n := params.get('name')) else '') + run_
    params.name = name
    path = Path(params.working_dir) if cluster else \
        Path.cwd() / 'dat/runs' / name if named else Path.cwd() / 'tmp'
    if named and not cluster:
        if path.exists() and input(f"Path {path} already exists. "
                "Delete everything? (y/N) ").lower() == 'y':
            shutil.rmtree(path)
        path.mkdir(exist_ok=True)
    if named:  # Remove partial checkpoints
        for cp in path.glob('checkpoint.*tmp*'):
            shutil.rmtree(cp)
    if cluster and (metrics := path / 'metrics.csv').exists():
        with open(metrics) as f:
            metrics = next(csv.DictReader(f))
        finalize_job(metrics, params)  # type: ignore
        return

    # Run experiment
    metrics = None
    print(f"Using path {path}.")
    interactive = params.interactive if 'interactive' in params else not named
    with (open(path / 'log.txt', 'a' if named else 'w') if not interactive 
            else sys.stdout) as f, redirect_stdout(f):
        metrics = {}
        function = globals_[fun] if (fun := params.get('function')) \
            else globals_['run']
        vars = getfullargspec(function)[0]
        for i, grid_params in enumerate(param_grid):
            params_ = recursive_objectify(
                params | dictify(grid_params) | {'grid_id': i})
            print(now.strftime("%Y-%m-%d %H:%M:%S") + '\n' + str(params_),
                flush=True)
            metrics[tuple(grid_params.items())] = function(**(
                ({'path': path} if 'path' in vars else {})
                | ({'params': params_} if 'params' in vars else {})
                | {v: params_[v] for v in vars if v in params_}))

    if cluster:
        finalize_job(metrics or {}, params)

def dictify(d):
    if not isinstance(d, dict): return d
    d_ = {}
    for k, v in d.items():
        v = dictify(v)
        if '.' in k:
            k, r = k.split('.', 1)
            v = dictify({r: v})
        d_[k] = v if k not in d_ else v | d_[k]
    return d_
