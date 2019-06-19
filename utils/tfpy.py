import tensorflow as tf
import numpy as np
import time
import time

def py_func(func, t):
    shape = t.shape
    out = tf.py_func(func, [t], t.dtype)
    out.set_shape(shape)
    return out


def py_func0(func, t):
    func_ = lambda t: [func(t), np.zeros_like(t)][1]
    return tf.cast(py_func(func_, t), t.dtype) + t


def stop_execution(t, msg=''):
    def f(t):
        print(msg)
        exit()
        return t
    return tf.py_func(f, [t], t.dtype)


def print_val(t, msg='', delay = 0.0):
    def f(A):
        if delay:
            time.sleep(delay) #good for putting in off-sets
        print(msg if msg else t.name)
        print(A)
        return 0
    return py_func0(f, t)

def st(t):
    def f(A):
        import ipdb
        ipdb.set_trace()
        return 0
    return py_func0(f, t)

def print_msg(t, msg):
    def f(A):
        print(msg, int(time.time()))
    return py_func0(f, t)

def print_shape(t, msg=''):
    def f(A):
        print(np.shape(A), msg)
        return A
    return py_func(f, t)


def inject_callback(t, callback_fn):  # i suspect this will be super nice for debugging
    _shape = t.get_shape()

    def f(T):
        callback_fn(T)
        return T

    out = tf.py_func(f, [t], t.dtype) * np.float32(0.0) + t
    out.set_shape(_shape)
    return out


def print_exec(t, msg=''):
    def f(T):
        if not msg:
            print('%s was run!' % T.name)
        else:
            print(msg)
        return T
    return tf.py_func(f, [t], t.dtype)


def check_shape(t, shape):
    def f(A):
        actual_shape = np.shape(A)
        if actual_shape != shape:
            msg = 'tensor %s has shape %s instead of %s' % (t.name, actual_shape, shape)
            raise Exception(msg)
        return A
    return tf.py_func(f, [t], t.dtype)


def checkrange(t, _max=10.0, _min=0.0, msg='', stop=False):
    if not const.check_losses:
        return t

    if msg == '':
        msg = 'range (%f, %f) was violated' % (_min, _max)

    def __f(t):
        tmax = np.max(t)
        tmin = np.min(t)
        if tmax > _max or tmin < _min:
            print(msg)
            if stop:
                exit()
        return t

    return tf.py_func(__f, [t], t.dtype)


def summarize_tensor(t, msg=''):
    def __f(t):
        tmax = np.max(t)
        tmin = np.min(t)
        tmean = np.mean(t)
        tstd = np.std(t)
        print(msg if msg else (t.name if hasattr(t, 'name') else 'summarize'))
        print('shape', t.shape)
        print('max', tmax)
        print('min', tmin)
        print('mean', tmean)
        print('std', tstd)
        return np.zeros(1, dtype=t.dtype)[0]
    return py_func(__f, t) + t  # a trick to keep gradients flowing

def save_pc(pc, name):
    if name in save_pc.seen:
        return
    np.savez_compressed('test/%s' % name, pc=pc)
    print('saved', name)
    save_pc.seen[name] = True


save_pc.seen = {}


def tf_save_pc(pc, name):
    return inject_callback(pc, lambda t: save_pc(t, name))
