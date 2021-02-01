"""
Author: Jason
"""
import time
from functools import wraps
import beepy as libBeepy


def formatT(startTime, endTime):
    return f'({endTime - startTime: .6f} s)'.rjust(12)


initTime = 0


def tic():
    global initTime
    initTime = time.time()


def toc(name='', show=True):
    if show:
        print(f'{name}: {formatT(initTime, time.time())}')


output = print


def obtainParamsFromKwargs(kw, params):
    newKwargs = dict()
    for key, value in kw.items():
        if key in params:
            params[key] = value
        else:
            newKwargs[key] = value
    return params, newKwargs


def timing(fun):
    @wraps(fun)
    def decoratedFun(*args, **kwargs):
        params = {'timerShow': True, 'timerBeforeRun': None,
                  'timerPrefix': '', 'timerSuffix': '', 'timerBeep': False}
        params, kwargs = obtainParamsFromKwargs(kwargs, params)
        show, beep = params['timerShow'], params['timerBeep']
        beforeRun, prefix, suffix = params['timerBeforeRun'], params['timerPrefix'], params['timerSuffix']

        if beforeRun is not None and show:
            output(beforeRun)

        startTime = time.time()
        retVal = fun(*args, **kwargs)
        t = formatT(startTime, time.time())

        if show:
            output(f'{prefix}{t} -> [{fun.__name__[0].upper() + fun.__name__[1:]}]{suffix}')
        if beep:
            libBeepy.beep()
        return retVal

    return decoratedFun


class Timer:
    def __init__(self, name='', show=True):
        self.show = show
        self.name = name

    def __enter__(self):
        self.initTime = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.show is True:
            t = formatT(self.initTime, time.time())
            print(f'{t} -> {self.name}')


def beepy(fun):
    @wraps(fun)
    def decoratedFun(*args, **kwargs):
        retVal = fun(*args, **kwargs)
        libBeepy.beep()
        return retVal

    return decoratedFun
