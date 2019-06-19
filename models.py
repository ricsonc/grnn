import inputs
import nets
import constants as const
import os.path as path
from fig import Config
import utils
import tensorflow as tf
from pprint import pprint
import glob
import os
import sys
from ipdb import set_trace as st


class Mode:
    def __init__(self, data_name, ops_name):
        self.data_name = data_name
        self.ops_name = ops_name

        
class Model:
    def __init__(self, net, modes = None):
        self.net = net
        self.data_selector = self.net.data_selector
        
        self.modes = modes
        if modes is None:
            self.modes = {
                'train': Mode('train', 'train_run'),
                'valt': Mode('train', 'val_run'),
                'valv': Mode('val', 'val_run'),
                'test': Mode('test', 'test_run'),
            }
            

        if not const.eager:
            self.net.go() #build the graph here
        
    def get_ops(self, mode):
        if const.eager:
            self.net.go(self.index_for_mode(mode), is_training=self.get_is_training(mode))
        return getattr(self.net, self.modes[mode].ops_name)
    
    def get_data_name(self, mode):
        return self.modes[mode].data_name

    def index_for_mode(self, mode):
        input_collection_to_number = {'train': 0, 'val': 1, 'test': 2}
        data_name = self.get_data_name(mode)
        return input_collection_to_number[data_name]

    def get_is_training(self, mode):
        if self.modes[mode].ops_name == "train_run":
            is_training = True
        else:
            is_training = False
        return is_training
    
    def fd_for_mode(self, mode):
        assert not const.eager
        if self.data_selector is None:
            return None
        else:
            is_training = self.get_is_training(mode)
            return {self.data_selector: self.index_for_mode(mode),\
                    self.net.is_training:is_training}

    def run(self, mode, sess = None):
        ops = self.get_ops(mode)
        if const.eager:
            return utils.utils.apply_recursively(ops, utils.tfutil.read_eager_tensor)
        else:
            return sess.run(ops, feed_dict = self.fd_for_mode(mode))


class PersistentModel(Model):
    def __init__(self, model_, ckpt_dir, modes = None):
        self.ckpt_dir = ckpt_dir
        utils.utils.ensure(ckpt_dir)
        super(PersistentModel, self).__init__(model_, modes)

        self.initsaver()

    def initsaver(self):
        self.savers = {}
        parts = self.net.weights
        for partname in parts:
            partweights = parts[partname][1]
            if partweights:
                self.savers[partname] = tf.train.Saver(partweights)

    def save(self, sess, name, step):
        config = Config(name, step)
        parts = self.net.weights
        savepath = path.join(self.ckpt_dir, name)
        utils.utils.ensure(savepath)
        for partname in parts:
            partpath = path.join(savepath, partname)
            utils.utils.ensure(partpath)
            partscope, weights = parts[partname]

            if not weights: #nothing to do
                continue
            
            partsavepath = path.join(partpath, 'X')

            saver = self.savers[partname]
            saver.save(sess, partsavepath, global_step=step)

            config.add(partname, partscope, partpath)
        config.save()
        #exit()

    def load(self, sess, name):
        config = Config(name)
        config.load()
        parts = self.net.weights

        for partname in config.dct:
            partscope, partpath = config.dct[partname]

            if partname not in parts:
                raise Exception("cannot load, part %s not in model" % partpath)

            ckpt = tf.train.get_checkpoint_state(partpath)
            if not ckpt:
                raise Exception("checkpoint not found? (1)")
            elif not ckpt.model_checkpoint_path:
                raise Exception("checkpoint not found? (2)")
            loadpath = ckpt.model_checkpoint_path

            scope, weights = parts[partname]

            if not weights: #nothing to do
                continue

            weights = {utils.utils.exchange_scope(weight.op.name, scope, partscope): weight
                       for weight in weights}

            saver = tf.train.Saver(weights)
            saver.restore(sess, loadpath)
        return config.step

    
class MultiViewReconstruction(PersistentModel):
    def __init__(self):
        input_ = inputs.MultiViewReconstructionInput()
        net = nets.MultiViewReconstructionNet(input_)
        super().__init__(net, const.ckpt_dir)

        
class MultiViewQuery(PersistentModel):
    def __init__(self):
        input_ = inputs.MultiViewReconstructionInput()
        net = nets.MultiViewQueryNet(input_)
        
        super().__init__(net, const.ckpt_dir)

class GQNBaseModel(PersistentModel):
    def __init__(self, net_cls):
        if const.DEEPMIND_DATA:
            input_ = inputs.GQNInput()
        else:
            input_ = inputs.GQNShapenet()
        net = net_cls(input_)
        super().__init__(net, const.ckpt_dir)

        if const.generate_views:
            self.modes['gen'] = Mode('test', 'gen_run')
        
class GQN2D(GQNBaseModel):
    def __init__(self):
        return super().__init__(nets.GQN2D)

class GQN2Dtower(GQNBaseModel):
    def __init__(self):
        return super().__init__(nets.GQN2Dtower)
        
class GQN3D(GQNBaseModel):
    def __init__(self):
        return super().__init__(nets.GQN3D)

class TestGQN(GQNBaseModel):
    def __init__(self):
        return super().__init__(nets.TestInput)

class MnistAE(PersistentModel):
    def __init__(self):
        input_ = inputs.MNISTInput()
        net = nets.MnistAE(input_)
        super().__init__(net, const.ckpt_dir)

