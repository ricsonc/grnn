#!/usr/bin/env python3

import tensorflow as tf
import constants as const

import operators
import models
import exports

import sys
import os
import utils

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':

    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    if len(sys.argv) == 3:
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

    if const.eager:
        tf.enable_eager_execution()

    if const.opname == 'reconstruction':
        model = models.MultiViewReconstruction()
        vis = exports.MultiViewReconstructionVis()
    elif const.opname == 'query':
        model = models.MultiViewQuery()
        vis = exports.MultiViewQueryVis()
    elif const.opname == 'gqn2d':
        model = models.GQN2D()
        vis = exports.MultiViewQueryVis()
    elif const.opname == 'gqntower':
        model = models.GQN2Dtower()
        vis = exports.MultiViewQueryVis()
    elif const.opname == 'gqn3d':
        model = models.GQN3D()
        vis = exports.MultiViewQueryVis()
    elif const.opname == 'gqntest':
        model = models.TestGQN()
        vis = None
    elif const.opname == 'mnist':
        model = models.MnistAE()
        vis = exports.AEVis()
    else:
        raise Exception('unknown model')

    
    op_cls = operators.GenerateViews if const.generate_views else operators.ModalOperator
    test_modes = ['test']
    if const.generate_views:
        test_modes = ['test'] * const.GEN_NUM_VIEWS
    
    operator = op_cls(
        model,
        ['train'], ['valt', 'valv'], test_modes,
        savename=const.save_name,
        loadname=const.load_name,
        tb = exports.TB(valmodename='valv', ignoremodes=['valt']),
        vis = vis,
        evaluator = exports.Evaluator(['loss']),
    )

    operator.go(const.mode)
