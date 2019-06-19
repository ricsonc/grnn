#!/usr/bin/env python

import random
import time
import psutil
import pwd
import os
from . import utils

USERNAME = pwd.getpwuid(os.getuid()).pw_name
SIGNAME = 'aff_send'

def print_process(process):
    print('====')
    print('PID:', process.pid)
    print('USER:', process.username())
    print('CMDLINE:', process.cmdline())
    print('AFFINITY:', process.cpu_affinity())

def is_signal(process):
    if process.username() != USERNAME:
        return False

    cmd = process.cmdline()
    if (len(cmd) == 2) and (SIGNAME in cmd[1]) and ('python' in cmd[0]):
        return True
    return False

def get_signals():
    return list(filter(is_signal, psutil.process_iter()))

def affinity_of_signals(signals):
    total_affinity = []
    for signal in signals:
        affinity = signal.cpu_affinity()
        total_affinity.extend(affinity)
    return list(set(total_affinity))

def me():
    return psutil.Process(os.getpid())

def go():
    if not utils.onmatrix():
        return
    signals = get_signals()
    assert signals, 'no signals found'
    aff = affinity_of_signals(get_signals())
    print('setting affinity to', aff)
    me_ = me()
    me_.cpu_affinity(aff)
        
