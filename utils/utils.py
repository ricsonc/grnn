import os
import copy
import psutil
import numpy as np
import math

def nyi():
    raise Exception('not yet implemented')


def ensure(path):
    if not os.path.exists(path):
        os.makedirs(path)

def exchange_scope(name, scope, oldscope):
    head, tail = name.split(scope)
    assert head == ''
    return oldscope + tail


def onmatrix():
    return 'Linux compute' in os.popen('uname -a').read()


def iscontainer(obj):
    return isinstance(obj, list) or isinstance(obj, dict) or isinstance(obj, tuple)

#an alias
iscollection = iscontainer

def strip_container(container, fn=lambda x: None):
    assert iscontainer(container), 'not a container'

    if isinstance(container, list) or isinstance(container, tuple):
        return [(strip_container(obj, fn) if iscontainer(obj) else fn(obj))
                for obj in container]
    else:
        return {k: (strip_container(v, fn) if iscontainer(v) else fn(v))
                for (k, v) in list(container.items())}


def memory_consumption():
    #print map(lambda x: x/1000000000.0, list(psutil.Process(os.getpid()).memory_info()))
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
    #return psutil.virtual_memory().used / 1000000000.0 #oof


 

def check_numerics(stuff):
    if isinstance(stuff, dict):
        for k in stuff:
            if not check_numerics(stuff[k]):
                raise Exception('not finite %s').with_traceback(k)
        return True
    elif isinstance(stuff, list) or isinstance(stuff, tuple):
        for x in stuff:
            check_numerics(x)
    else:
        return np.isfinite(stuff).all()

def apply_recursively(collection, f):

    def dict_valmap(g, dct):
        return {k:g(v) for (k,v) in dct.items()}

    def f_prime(x):
        if iscollection(x):
            return apply_recursively(x, f)
        else:
            return f(x)
    
    if not iscollection(collection):
        return f(collection)
    else:
        map_fn = dict_valmap if isinstance(collection, dict) else map
        return type(collection)(map_fn(f_prime, collection))

def map_if_list(x, fn):
    if isinstance(x, list):
        return list(map(fn, x))
    return fn(x)
    

def degrees(x):
    return x * 180/np.pi

def radians(x):
    return x * np.pi/180
