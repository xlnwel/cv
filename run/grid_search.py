import time
from datetime import datetime
from copy import deepcopy
from multiprocessing import Process

from utility.yaml_op import load_args
from utility.debug_tools import assert_colorize


class GridSearch:
    def __init__(self, args_file, train_func, dir_prefix=''):
        args = load_args(args_file)
        self.args = args
        self.train_func = train_func

        # add date to root directory
        now = datetime.now()
        if dir_prefix:
            dir_prefix = f'-{dir_prefix}'
        dir_fn = lambda filename: (f'logs/'
                                    f'{now.month:02d}{now.day:02d}-'
                                    f'{now.hour:02d}{now.minute:02d}'
                                    f'{dir_prefix}/'
                                    f'{filename}')
        dirs = ['model_root_dir', 'log_root_dir']
        for root_dir in dirs:
            self.args[root_dir] = dir_fn(self.args[root_dir])

        self.processes = []

    def __call__(self, **kwargs):
        if kwargs == {}:
            # if no argument is passed in, run the default setting
            self.train_func(self.args)        
        else:
            # do grid search
            self.args['model_name'] = 'GS'
            self._change_args(**kwargs)
            [p.join() for p in self.processes]

    def _change_args(self, **kwargs):
        if kwargs == {}:
            # basic case
            old_model_name = self.args['model_name']
            # arguments should be deep copied here, 
            # otherwise args will be reset if sub-process runs after
            p = Process(target=self.train_func,
                        args=(deepcopy(self.args), ))
            self.args['model_name'] = old_model_name
            p.start()
            time.sleep(1)   # ensure sub-processs start in order
            self.processes.append(p)
        else:
            # recursive case
            kwargs_copy = deepcopy(kwargs)
            key, value = self._popitem(kwargs_copy)

            valid_args = None
            if key in self.args:
                assert_colorize(valid_args is None, f'Conflict: found {key} in both {valid_args} and {self.args}!')
                valid_args = self.args

            err_msg = lambda k, v: f'Invalid Argument: {k}={v}'
            assert_colorize(valid_args is not None, err_msg(key, value))
            if isinstance(value, dict) and len(value) != 0:
                # For simplicity, we do not further consider the case when value is a dict of dicts here
                k, v = self._popitem(value)
                assert_colorize(k in valid_args[key], err_msg(k, v))
                if len(value) != 0:
                    # if there is still something left in value, put value back into kwargs
                    kwargs_copy[key] = value
                self._safe_call(f'-{key}', lambda: self._recursive_trial(valid_args[key], k, v, kwargs_copy))
            else:
                self._recursive_trial(valid_args, key, value, kwargs_copy)

    # helper functions for self._change_args
    def _popitem(self, kwargs):
        assert_colorize(isinstance(kwargs, dict))
        while len(kwargs) != 0:
            k, v = kwargs.popitem()
            if not isinstance(v, list) and not isinstance(v, dict):
                v = [v]
            if len(v) != 0:
                break
        return deepcopy(k), deepcopy(v)

    def _recursive_trial(self, arg, key, value, kwargs):
        assert_colorize(isinstance(value, list), f'Expect value of type list, not {type(value)}: {value}')
        for v in value:
            arg[key] = v
            self._safe_call(f'-{key}={v}', lambda: self._change_args(**kwargs))

    def _safe_call(self, append_name, func):
        old_model_name = self.args['model_name']
        self.args['model_name'] += append_name
        func()
        self.args['model_name'] = old_model_name
