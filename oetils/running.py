from datetime import datetime
import sys
from pathlib import Path
import shutil
import csv
from contextlib import redirect_stdout
from inspect import getfullargspec

from cluster_utils import finalize_job, initialize_job
from smart_settings.param_classes import recursive_objectify


def run(params, path):
    raise NotImplementedError("'run' function not found!")

def main(globals):
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
    globals = globals() | globals

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
        print(now.strftime("%Y-%m-%d %H:%M:%S") + '\n' + str(params),
            flush=True)
        function = globals[fun] if (fun := params.get('function')) \
            else globals['run']
        vars_ = getfullargspec(function)[0]
        metrics = function(**(({'path': path} if 'path' in vars_ else {})
            | ({'params': params} if 'params' in vars_ else {})
            | {var: params[var] for var in vars_ if var in params}))

    if cluster:
        finalize_job(metrics or {}, params)


if __name__ == "__main__":
    main()

