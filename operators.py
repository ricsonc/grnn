import tensorflow as tf
from tensorflow.python import debug
import constants as const
import utils
import os
import models
import exports
from time import time, sleep
from os import path
import random
from tensorflow.python.client import timeline
import inputs

import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

from ipdb import set_trace as st

class SessionOperator(object):
    def __init__(self):

        if not const.eager:
            config = tf.ConfigProto()
            if const.DEBUG_PLACEMENT:
                config.log_device_placement = True
            self.sess = tf.Session(config=config)
            K.set_session(self.sess)
            self.run = self.sess.run
        else:
            self.sess = None

    def save(self):
        utils.utils.nyi()

    def load(self):
        return 0

    def setup(self):
        T1 = time()
        print('finished graph creation in %f seconds' % (time() - const.T0))

        if not const.eager:
            self.run(tf.global_variables_initializer())
            self.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

            #must come after the queue runners
            if const.DEBUG_NAN:
                self.sess = debug.LocalCLIDebugWrapperSession(self.sess)
                self.sess.add_tensor_filter("has_inf_or_nan", debug.has_inf_or_nan)

        self.step = self.load() #*0

        #it's in another graph
        # if const.generate_views: #not sure why this is necessary....
        #     #self.run(tf.variables_initializer(inputs.foo_counters))
        #     self.run(inputs.foo_counters)
        
        if not const.eager:
            tf.get_default_graph().finalize()
            
        print('finished graph initialization in %f seconds' % (time() - T1))

    def go(self, mode):
        self.setup()
        if mode == 'train':
            self.train()
        elif mode == 'test':
            #tf.logging.set_verbosity(tf.logging.FATAL)
            #prevents end of iterator error print outs
            self.test()

    def test(self):
        utils.utils.nyi()

    def train(self):
        utils.utils.nyi()


class ModelOperator(SessionOperator):
    def __init__(self, model,
                 savename=None, loadname=None,
                 vis=None, tb=None, evaluator=None):
        self.model = model
        self.savename = savename
        self.loadname = loadname
        self.vis = vis
        self.tb = tb
        self.evaluator = evaluator
        self.run_metadata = tf.RunMetadata()
        super(ModelOperator, self).__init__()

    def load(self):
        if not self.loadname:
            return 0
        else:
            return self.model.load(self.sess, self.loadname)

    def save(self):
        if not self.savename:
            return
        self.model.save(self.sess, self.savename, self.step)

    def fd_for_mode(self, mode):
        input_collection_to_number = {'train': 0, 'val': 1, 'test': 2}
        data_name = self.model.get_data_name(mode)
        fd = {self.model.data_selector: input_collection_to_number[data_name]}

        if self.model.data_selector is None:
            return {}
        else:
            return fd

    def run_steps(self, modes, same_batch = True):
        if const.DEBUG_SPEED:
            print('====')
            print('running', modes)
            t0 = time()

        stuff = []
        for mode in modes:

            if const.SKIP_RUN:
                print('skipping run')
                continue 

            if const.DEBUG_SPEED:
                print('running mode:', mode)

            stuff_ = self.model.run(mode, self.sess)
            stuff.append(stuff_)
                
        if const.DEBUG_SPEED:
            t1 = time()
            print('time: %f' % (t1 - t0))
            print('====')

        return stuff

    def train(self):
        print('STARTING TRAIN')
        
        if const.DEBUG_MEMORY:
            #need to write to log, since leak means process would be killed
            utils.utils.ensure('memory_log')
            f = open('memory_log/%s.log' % const.exp_name, 'w')

        for step in range(self.step, const.NB_STEPS):
            self.step = step
            print('step %d' % step)
            if const.DEBUG_MEMORY:
                m = utils.utils.memory_consumption()
                print('memory consumption is', m)
                f.write(str(m)+'\n')
                f.flush()
                os.fsync(f.fileno())

            if not(step % const.savep):
                print('saving')
                self.save()
            self.train_step(step)
            if not(step % const.valp):
                self.val_step(step)
            

    def test(self):
        step = 0
        #while 1:
        for _ in range(1000):
            step += 1
            if not self.test_step(step):
                break
            print('test step %d' % step)
            
        if self.evaluator:
            self.evaluator.finish()            
        
    def train_step(self, step):
        utils.utils.nyi()

    def val_step(self, step):
        utils.utils.nyi()

    def test_step(self, step):
        utils.utils.nyi()        


class ModalOperator(ModelOperator):

    def __init__(self, model, train_modes, val_modes, test_modes,
                 savename=None, loadname=None,
                 vis=None, tb=None, evaluator=None):

        if not isinstance(train_modes, list):
            train_modes = [train_modes]
        if not isinstance(val_modes, list):
            val_modes = [val_modes]
        if not isinstance(test_modes, list):
            test_modes = [test_modes]

        self.train_modes = train_modes
        self.val_modes = val_modes
        self.test_modes = test_modes
        
        super(ModalOperator, self).__init__(
            model, savename=savename, loadname=loadname, vis=vis, tb=tb, evaluator=evaluator
        )

        if const.DEBUG_FULL_TRACE:
            self.graph_writer = tf.summary.FileWriter(path.join(const.tb_dir, 'graph'),
                                                      self.sess.graph)

    def train_step(self, step):
        train_stuffs = self.run_steps(self.train_modes, same_batch = True)

        if const.SKIP_EXPORT or const.SKIP_TRAIN_EXPORT:
            print('skipping exports')
            return

        if const.DEBUG_SPEED:
            print('processing outputs')
        for mode, train_stuff in zip(self.train_modes, train_stuffs):
            if not train_stuff:
                continue
            if 'summary' in train_stuff:
                self.tb.process(train_stuff['summary'], mode, step)

    def val_step(self, step):
        val_stuffs = self.run_steps(self.val_modes, same_batch = False)

        if const.SKIP_EXPORT or const.SKIP_VAL_EXPORT:
            print('skipping exports')
            return

        if const.DEBUG_SPEED:
            print('processing outputs')

        for mode, val_stuff in zip(self.val_modes, val_stuffs):
            if not val_stuff:
                return
            if 'vis' in val_stuff and self.vis:
                self.vis.process(val_stuff['vis'], mode, step)
            if 'summary' in val_stuff and self.tb:
                self.tb.process(val_stuff['summary'], mode, step)

    def test_step(self, step):
        assert len(self.test_modes) == 1, "can't have multiple test modes"
        
        try:
            test_stuff = self.run_steps(self.test_modes)[0]
        except tf.errors.OutOfRangeError:
           return False

        if 'evaluator' in test_stuff and self.evaluator:
            self.evaluator.process(test_stuff['evaluator'], None, None)
        if 'vis' in test_stuff and self.vis:
            self.vis.process(test_stuff['vis'], self.test_modes[0], step)
        if 'summary' in test_stuff and self.tb:
            self.tb.process(test_stuff['summary'], self.test_modes[0], step)
        return True


class GenerateViews(ModalOperator):
    def test_step(self, step):
        try:
            test_stuffs = self.run_steps(self.test_modes)
        except tf.errors.OutOfRangeError:
           return False

        visualizations = [test_stuff['vis']['pred_views'][0] for test_stuff in test_stuffs]
        self.vis.process(test_stuffs[0]['vis'], self.test_modes[0], step)
        self.vis.process({'gen_views': visualizations}, self.test_modes[0], step)
        
        if False: #plot immediately
            
            #just for visualization purposes
            def chunks(l, n):
                """Yield successive n-sized chunks from l."""
                for i in range(0, len(l), n):
                    yield l[i:i + n]

            import numpy as np

            row_size = const.AZIMUTH_GRANULARITY if (const.ELEV_GRANULARITY > 1) else 12
        
            rows = list(chunks(visualizations, row_size))
            rows = [np.concatenate(row, axis = 1) for row in rows]
            total = np.concatenate(rows, axis = 0)

            import matplotlib.pyplot as plt
            plt.imshow(total)
            plt.show()

        return True
        
       
