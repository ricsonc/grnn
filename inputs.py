import tensorflow as tf
import tensorflow.contrib.slim as slim
import constants as const
import utils
import os.path as path
import numpy as np
import random
from munch import Munch
import ipdb
st = ipdb.set_trace
from tensorflow.examples.tutorials.mnist import input_data

'''
Note on the input system:
all input classes should inherit from Input and define a self.data method

the appropriate dataset (train/val/test) is selected as follows:
1. each mode (defined in models.py) has an associated data_name (train, val, or test)
2. each mode name maps to an index (0, 1, 2) respeectively

then one of the following will happen, depending on whether the graph is eager or not

Graph Mode:
3. the go method of the net class (in nets.py) is called ONCE to construct the graph
4. the data_selector placeholder attribute of the net class is used to pass the index
5. the data_selector is usually hooked up to q_ph of the input class
6. the data method of the input class selects the right data using q_ph and a case statement

Eager Mode:
3. the go method is called many times, once per iteration. an index is passed in
4. the go method usually calls something like self.prepare_data(index)
5. prepare_data calls the data method of the input class, passing in index
6. the data method selects the right data using a python conditional statement

- in graph mode, the index passed into data is ignored, so feel free to ignore it in nets.py
- in eager mode, q_ph is ignored

it is difficult to have a unified input pipeline for both eager and graph mode because
1. we can't call/construct input tensors once per iteration in graph mode 
2. we can't use placeholders in eager mode

'''

class Input:
    '''
    the only function that every input class MUST define
    index specifies whether the data returned should be from the
    train, test, or validation set

    return a (dictionary of) tf tensors
    '''
    
    def data(self, index = None):
        raise NotImplementedError

class TFDataInput(Input):
    '''
    if you inherit from this class, you should define attributes
    self.train_data, self.test_data, self.val_data, which are objects
    with member functions .get_next(), which when called will return
    a *tf tensor* (or a dictionary/list of tensors) containing the data

    this supposed to be used with the tf.data API (in particular, the
    make_one_shot_iterator function)
    '''
    
    def __init__(self):
        #q_ph explained in self.data_default
        if not const.eager:
            self.q_ph = tf.placeholder(dtype = tf.int32, shape = ())
        else:
            self.q_ph = None
    
    def data(self, index = None):
        #use index to grab data from train, val, or test set
        if const.eager:
            assert index is not None
            return self.data_eager(index)
        else:
            return self.data_default()
            
    def data_eager(self, index):
        return self.data_for_selector(tf.constant(index))

    def data_default(self):
        #in non-eager mode, we feed in data to this placeholder, else we leave it empty
        return self.data_for_selector(self.q_ph)
            
    def data_for_selector(self, selector):

        #case is buggy for eager mode...
        if const.eager:
            return [self.train_data.get_next,
                    self.val_data.get_next,
                    self.test_data.get_next][selector.numpy()]()
        else:
            rval = tf.case({
                tf.equal(selector, 0) : lambda: self.train_data.get_next(),
                tf.equal(selector, 1) : lambda: self.val_data.get_next(),
                tf.equal(selector, 2) : lambda: self.test_data.get_next(),
            }, exclusive = True)
            utils.utils.apply_recursively(rval, utils.tfutil.set_batch_size)
            return rval

#for debugging
class MNISTInput(Input):
    def __init__(self):
        super().__init__()
        self.q_ph = None
        
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        mnist_train = mnist.train.images #55k x 784
        mnist_val = mnist.validation.images #5k
        mnist_test = mnist.test.images #10k

        def foo(x):
            x = tf.constant(np.reshape(x, (-1, 28, 28, 1)), dtype = np.float32)
            x = self.make_batch(x)
            x = Munch(img = x)
            return x

        self.train = foo(mnist_train)
        self.val = foo(mnist_val)
        self.test = foo(mnist_test)

    def make_batch(self, x):
        N = x.shape[0]
        indices = tf.random_uniform(shape = (const.BS,), minval = 0, maxval = N, dtype = tf.int32)
        return tf.gather(x, indices)

    def data(self, index = None):
        print('warning, only using training data')
        return self.train
    
class ShapenetInput(TFDataInput):
    def __init__(self):
        super().__init__()
                
        self.train_data = self.make_data(const.train_file, True)
        self.val_data = self.make_data(const.val_file, False)
        self.test_data = self.make_data(const.test_file, False, True)

    def make_queue(self, fn, shuffle, istest = False):
        #this is not used
        assert False
        with open(path.join('setup/lists', fn), 'r') as f:
            fns = f.readlines()
        fns = [path.join(const.data_dir, fn.strip()) for fn in fns]
        if istest:
            return tf.train.string_input_producer(fns, shuffle=shuffle, num_epochs = 1)
        else:
            return tf.train.string_input_producer(fns, shuffle=shuffle)

    def make_data(self, fn, shuffle, istest = False):
        with open(path.join('setup/lists', fn), 'r') as f:
            fns = f.readlines()
        fns = [path.join(const.data_dir, fn.strip()) for fn in fns]

        # if const.generate_views:
        #     fns2 = []
        #     for fn in fns:
        #         for i in range(const.GEN_NUM_VIEWS):
        #             fns2.append(fn)
        #     fns = fns2
        
        nrots = 1 if istest else 8
        
        data = tf.data.TFRecordDataset(fns, compression_type = 'GZIP')

        if const.generate_views:
            data = data.map(self.decode, num_parallel_calls = 1)
            data = data.map(self.normalize_batch2(const.GEN_NUM_VIEWS), num_parallel_calls = 1)
            data = data.apply(tf.contrib.data.unbatch())
            data = data.batch(1)
            iterator = data.make_one_shot_iterator()
            return iterator
        
        data = data.map(self.decode, num_parallel_calls = 8)
        data = data.map(self.normalize_batch(nrots), num_parallel_calls = nrots)
        data = data.apply(tf.contrib.data.unbatch())
        if not istest:
            data = data.shuffle(256)
            data = data.repeat()
        data = data.batch(const.BS)
        if not istest:
            data = data.prefetch(4)
        iterator = data.make_one_shot_iterator()
        return iterator

class MultiViewInput(ShapenetInput):

    def decode(self, example):

        feature_names = ['images', 'zmaps']
        if not const.LIMITED_DATASET:
            feature_names.extend(['segs', 'voxel', 'obj1', 'obj2'])
        if const.MASKED_DATASET:
            feature_names.append('valids')
        
        stuff = tf.parse_single_example(
            example,
            features={k: tf.FixedLenFeature([], tf.string) for k in feature_names}
        )

        N = const.VV*const.HV

        images = tf.decode_raw(stuff['images'], tf.float32)
        images = tf.reshape(images, (N, const.Hdata, const.Wdata, 4))

        masks = tf.slice(images, [0, 0, 0, 3], [-1, -1, -1, 1])
        images = tf.slice(images, [0, 0, 0, 0], [-1, -1, -1, 3])

        zmaps = tf.decode_raw(stuff['zmaps'], tf.float32)
        zmaps = tf.reshape(zmaps, (N, const.Hdata, const.Wdata, 1))

        rvals = [images, masks, zmaps]

        if not const.LIMITED_DATASET:
            segs = tf.decode_raw(stuff['segs'], tf.float32)
            segs = tf.reshape(segs, (N, const.Hdata, const.Wdata, 2))

            voxel = tf.decode_raw(stuff['voxel'], tf.float32)
            voxel = tf.reshape(voxel, (128, 128, 128))

            obj1 = tf.decode_raw(stuff['obj1'], tf.float32)
            obj1 = tf.reshape(obj1, (128, 128, 128))

            obj2 = tf.decode_raw(stuff['obj2'], tf.float32)
            obj2 = tf.reshape(obj2, (128, 128, 128))

            rvals.extend([segs, voxel, obj1, obj2])
            
        if const.MASKED_DATASET:
            valids = tf.decode_raw(stuff['valids'], tf.uint8)
            valids = tf.reshape(valids, (N,))
            
            rvals.append(valids)

        return rvals

    def normalize(self, *args):
        raise NotImplementedError

    def normalize_batch(self, n):
        #n is the number of random rotations to pick
        def normalize_(*args):
            outputs = [self.normalize(*args) for i in range(n)]

            batched_output = Munch({k: [] for k in outputs[0]})
            for output in outputs:
                for k, v in output.items():
                    batched_output[k].append(v)                    
            for k, v in batched_output.items():
                if isinstance(v[0], tuple):
                    batched_output[k] = zip(*v)
                    batched_output[k] = tuple(map(lambda w: tf.stack(w, axis = 0), batched_output[k]))
                else:
                    batched_output[k] = tf.stack(v, axis = 0)
                    
            return batched_output
        
        return normalize_

    def normalize_batch2(self, n):
        assert const.generate_views
        #n is the number of random rotations to pick
        def normalize_(*args):
            outputs = [self.normalize(*args) for i in range(n)]

            for i in range(n):
                elev_index = i / const.AZIMUTH_GRANULARITY
                azimuth_index = i % const.AZIMUTH_GRANULARITY

                outputs[i].phis = outputs[i].phis[:-1] + (elev_index,)
                outputs[i].thetas = outputs[i].thetas[:-1] + (azimuth_index,)

            batched_output = Munch({k: [] for k in outputs[0]})
            for output in outputs:
                for k, v in output.items():
                    batched_output[k].append(v)                    
            for k, v in batched_output.items():
                if isinstance(v[0], tuple):
                    batched_output[k] = zip(*v)
                    batched_output[k] = tuple(map(lambda w: tf.stack(w, axis = 0), batched_output[k]))
                else:
                    batched_output[k] = tf.stack(v, axis = 0)
                    
            return batched_output
        
        return normalize_

foo_counters = []
    
class MultiViewReconstructionInput(MultiViewInput):

    def normalize(self, images, masks, zmaps, segs, voxel, obj1, obj2):
    
        def extract_view_(images, masks, zmaps, segs, idx):
            offset_tensor = tf.stack([idx, 0, 0, 0])
            size_tensor = [1, -1, -1, -1]
            extract = lambda x: tf.slice(x, offset_tensor, size_tensor)
            return extract(images), extract(masks), extract(zmaps), extract(segs)

        random.seed(0)
        def extract_rand_view(images, masks, zmaps, segs, seed = 0):
            if not const.FIX_VIEW:
                phi_idx = utils.tfutil.randidx(const.VV, seed = 123 + seed)
                theta_idx = utils.tfutil.randidx(const.HV, seed = 456 + seed)
            else:
                phi_idx = tf.constant(random.randint(0, const.VV-1), dtype = tf.int32)
                theta_idx = tf.constant(random.randint(0, const.HV-1), dtype = tf.int32)
            idx = phi_idx * const.HV + theta_idx
            return extract_view_(images, masks, zmaps, segs, idx), (phi_idx, theta_idx, idx)

        random.seed(0)
        views, idxs = list(zip(*[
            extract_rand_view(images, masks, zmaps, segs, seed = random.randint(0,100))
            for _ in range(const.NUM_VIEWS + const.NUM_PREDS) 
        ]))

        images, masks, zmaps, segs = list(zip(*views))
        phi_idx, theta_idx, idx = list(zip(*idxs))

        images = [img/255.0 for img in images]
        segs = [seg/255.0 for seg in segs]        
        masks = [mask/255.0 for mask in masks]
        zmaps = [zmap*2.0 for zmap in zmaps]

        if const.RANDOMIZE_BG:
            bg = utils.tfutil.tf_random_bg(1, darker=True)
            for i, (image, mask) in enumerate(zip(images, masks)):
                images[i] = image + bg * (1.0-mask)

        images = [tf.reshape(img, (const.Hdata, const.Wdata, 3)) for img in images]
        masks = [tf.reshape(mask, (const.Hdata, const.Wdata, 1)) for mask in masks]
        zmaps = [tf.reshape(zmap, (const.Hdata, const.Wdata, 1)) for zmap in zmaps]
        segs = [tf.reshape(seg, (const.Hdata, const.Wdata, 2)) for seg in segs]        

        images = tuple(images)
        masks = tuple(masks)
        zmaps = tuple(zmaps)
        segs = tuple(segs)

        if const.H != const.Hdata:
            def resize(x):
                return tuple(
                    tf.unstack(
                        tf.image.resize_nearest_neighbor(
                            tf.stack(x, axis = 0),
                            [const.H, const.W]
                        )
                    )
                )
            
            images = resize(images)
            masks = resize(masks)
            zmaps = resize(zmaps)
            segs = resize(segs)

        names = ['images', 'masks', 'segs', 'depths', 'phis', 'thetas', 'voxel', 'obj1', 'obj2']
        stuff = [images, masks, segs, zmaps, phi_idx, theta_idx, voxel, obj1, obj2]

        return Munch(zip(names, stuff))

class MultiViewReconstructionInput2(MultiViewInput):

    def normalize(self, images, masks, zmaps, *args):
        #called once per dataset
        assert len(args) == int(const.MASKED_DATASET)
        
        if const.MASKED_DATASET:
            valids = args[0]

        def extract_view_(images, masks, zmaps, idx):
            offset_tensor = tf.stack([idx, 0, 0, 0])
            size_tensor = [1, -1, -1, -1]
            extract = lambda x: tf.slice(x, offset_tensor, size_tensor)
            return extract(images), extract(masks), extract(zmaps)

        def extract_rand_view(images, masks, zmaps, seed = 0):
            if not const.FIX_VIEW:

                if not const.MASKED_DATASET:
                    phi_idx = utils.tfutil.randidx(const.VV, seed = 123 + seed)
                    theta_idx = utils.tfutil.randidx(const.HV, seed = 456 + seed)
                else:
                    phi_theta_idx = utils.tfutil.randidx(const.HV*const.VV, mask = valids, seed = seed)
                    phi_idx = tf.mod(phi_theta_idx, const.VV)
                    theta_idx = tf.div(phi_theta_idx, const.VV)
            else:
                if const.MASKED_DATASET:
                    raise Exception('this case is not yet implemented...')
                
                phi_idx = tf.constant(random.randint(0, const.VV-1), dtype = tf.int32)
                if const.ARITH_MODE:
                    phi_idx = tf.constant(0, dtype = tf.int32)
                theta_idx = tf.constant(random.randint(0, const.HV-1), dtype = tf.int32)
            idx = phi_idx * const.HV + theta_idx
            return extract_view_(images, masks, zmaps, idx), (phi_idx, theta_idx, idx)

        random.seed(2)
        views, idxs = list(zip(*[
            extract_rand_view(images, masks, zmaps, seed = random.randint(0,100))
            for _ in range(const.NUM_VIEWS + const.NUM_PREDS) 
        ]))

        images, masks, zmaps = list(zip(*views))
        phi_idx, theta_idx, idx = list(zip(*idxs))

        images = [img/255.0 for img in images]
        masks = [mask/255.0 for mask in masks]
        zmaps = [zmap*2.0 for zmap in zmaps]

        if const.RANDOMIZE_BG:
            bg = utils.tfutil.tf_random_bg(1, darker=True)
            for i, (image, mask) in enumerate(zip(images, masks)):
                images[i] = image + bg * (1.0-mask)

        images = [tf.reshape(img, (const.Hdata, const.Wdata, 3)) for img in images]
        masks = [tf.reshape(mask, (const.Hdata, const.Wdata, 1)) for mask in masks]
        zmaps = [tf.reshape(zmap, (const.Hdata, const.Wdata, 1)) for zmap in zmaps]

        images = tuple(images)
        masks = tuple(masks)
        zmaps = tuple(zmaps)

        if const.H != const.Hdata:
            def resize(x):
                return tuple(
                    tf.unstack(
                        tf.image.resize_nearest_neighbor(
                            tf.stack(x, axis = 0),
                            [const.H, const.W]
                        )
                    )
                )
            
            images = resize(images)
            masks = resize(masks)
            zmaps = resize(zmaps)

        # ####
        # if const.generate_views:
        #     assert const.BS == 1

        #     counter = tf.Variable(0, dtype = tf.int32)
        #     foo_counters.append(tf.variables_initializer([counter]))
            
        #     increment_op = tf.assign_add(counter, 1)
        #     counter_mod = tf.mod(counter, const.GEN_NUM_VIEWS)

        #     # printing is a bit weird due to the async ??
        #     counter_mod = utils.tfpy.print_val(counter_mod, 'counter is') 
            
        #     elev_index = (counter_mod-1) / const.AZIMUTH_GRANULARITY
        #     azimuth_index = tf.mod((counter_mod-1), const.AZIMUTH_GRANULARITY)

        #     elev_index = tf.cast(elev_index, tf.float32)
        #     azimuth_index = tf.cast(azimuth_index, tf.float32)
            
        #     with tf.control_dependencies([increment_op]):
        #         phi_idx = phi_idx[:3] + (elev_index,)
        #         theta_idx = theta_idx[:3] + (azimuth_index,)
            
        names = ['images', 'masks', 'depths', 'phis', 'thetas']
        stuff = [images, masks, zmaps, phi_idx, theta_idx]

        return Munch(zip(names, stuff))

class GQNInput(Input):
    
    def __init__(self):
        #self.q_ph = None
        self.q_ph = tf.placeholder(dtype = tf.int32, shape = ())
        
        from gqn_inputs import DataReader
        kwargs = {
            'dataset': const.GQN_DATA_NAME,
            'context_size': const.NUM_VIEWS,
            'root': 'gqn-dataset'
        }

        self.train_data_reader = DataReader(mode = 'train', **kwargs)
        self.test_data_reader = DataReader(mode = 'test', **kwargs)

        assert const.NUM_PREDS == 1
        
    def data(self):
        #self.q_ph = utils.tfpy.print_val(self.q_ph, 'qph')

        rval = tf.case({
            tf.equal(self.q_ph, 0): lambda: self.train_data_reader.read(batch_size=const.BS),
            tf.equal(self.q_ph, 1): lambda: self.test_data_reader.read(batch_size=const.BS),
            tf.equal(self.q_ph, 2): lambda: self.test_data_reader.read(batch_size=const.BS)
        }, exclusive = True)
        
        rval = self.munch(rval)

        if const.generate_views:
            assert const.BS == 1

            counter = tf.Variable(0, dtype = tf.int32)
            increment_op = tf.assign_add(counter, 1)
            counter_mod = tf.mod(counter, const.GEN_NUM_VIEWS)

            # printing is a bit weird due to the async ??
            counter_mod = utils.tfpy.print_val(counter_mod, 'counter is') 
            
            elev_index = (counter_mod-1) / const.AZIMUTH_GRANULARITY
            azimuth_index = tf.mod((counter_mod-1), const.AZIMUTH_GRANULARITY)

            elev_index = tf.cast(elev_index, tf.float32)
            azimuth_index = tf.cast(azimuth_index, tf.float32)
            
            azimuth = azimuth_index * (360/const.AZIMUTH_GRANULARITY) 
            azimuth = azimuth + tf.cast(azimuth > 180, tf.float32) * (-360)
            azimuth /= 180 / np.pi

            if const.ELEV_GRANULARITY == 1:
                elev = const.MIN_ELEV
            else:
                elev_step = (const.MAX_ELEV-const.MIN_ELEV) / (const.ELEV_GRANULARITY - 1)
                elev = (const.MIN_ELEV + elev_index * elev_step) / (180 / np.pi)
                
            
            with tf.control_dependencies([increment_op]):
                #phi, theta is the correct order here
                query_cam = tf.expand_dims(tf.stack([elev, azimuth], axis = 0), axis = 0)
                
            rval.query.query_camera = query_cam
        
        return rval

    def munch(self, task):
        return Munch(
            query = self.munchq(task.query),
            target = task.target
        )

    def munchq(self, query):
        return Munch(
            context = self.munchc(query.context),
            query_camera = query.query_camera
        )

    def munchc(self, context):
        return Munch(
            frames = tf.unstack(context.frames, axis = 1),
            cameras = tf.unstack(context.cameras, axis = 1)
        )


class GQNShapenet:
    #put shapenet in the form of gqn dataset

    def __init__(self):
        self.child = MultiViewReconstructionInput2()
        self.q_ph = self.child.q_ph

    def data(self):
        data = self.child.data_for_selector(self.q_ph)
        data = Munch(query = self.make_query(data),
                     target = self.make_target(data))
        return data

    def make_query(self, data):
        return Munch(context = self.make_context(data),
                     query_camera = self.make_query_camera(data))
    
    def make_context(self, data):
        return Munch(frames = self.make_frames(data),
                     cameras = self.make_cameras(data))

    def make_target(self, data):
        return data.images[-1]

    def make_frames(self, data):
        return data.images[:-1]
        
    def make_cameras(self, data):
        return [self.yp_for_pt(p, t) for (p,t) in zip(data.phis[:-1], data.thetas[:-1])]

    def make_query_camera(self, data):
        return self.yp_for_pt(data.phis[-1], data.thetas[-1])

    def yp_for_pt(self, phi, theta):
        phi = tf.cast(phi, tf.float32) * const.VDELTA + const.MINV
        theta = tf.cast(theta, tf.float32) * const.HDELTA + const.MINH
        phi = phi * np.pi/180.
        theta = theta * np.pi/180.
        return tf.stack([phi, theta], axis = 1)

