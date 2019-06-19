import tensorflow as tf
import tensorflow.contrib.slim as slim
import constants as const
import utils
import os.path as path
import numpy as np
import inputs
from tensorflow import summary as summ
from munch import Munch
import ipdb
import math

st = ipdb.set_trace

class Net:
    def __init__(self, input_):
        self.weights = {}
        self.input = input_

        #this is used to select whether to pull data from train, val, or test set
        self.data_selector = input_.q_ph
        if not const.eager:
            self.is_training = tf.placeholder(dtype = tf.bool, shape = (), name="is_training")
        else:
            self.is_training = None

    def add_weights(self, name):
        self.weights[name] = utils.tfutil.current_scope_and_vars()
        
    def optimize(self, fn):
        self.optimizer = tf.train.AdamOptimizer(const.lr, const.mom)
        self.opt = utils.tfutil.make_opt_op(self.optimizer, fn)

    def go(self, index = None, is_training=None):
        #index is passed to the data_selector, to control which data is used
        self.optimize(lambda: self.go_up_to_loss(index, is_training))
        self.assemble()

    def go_up_to_loss(self, index = None, is_training=None):
        #should save the loss to self.loss_
        #and also return the loss
        raise NotImplementedError

    def build_vis(self):
        #should save a Munch dictionary of values to self.vis
        raise NotImplementedError
    
    def assemble(self):
        #define all the summaries, visualizations, etc
        
        with tf.name_scope('summary'):
            summ.scalar('loss', self.loss_) #just to keep it consistent

        if not const.eager:
            self.summary = tf.summary.merge_all()
        else:
            self.summary = None

        self.evaluator = Munch(loss = self.loss_)
        self.build_vis()

        #these are the tensors which will be run for a single train, test, val step
        self.test_run = Munch(evaluator = self.evaluator, vis = self.vis, summary = self.summary)
        self.train_run = Munch(opt = self.opt, summary = self.summary)
        self.val_run = Munch(loss = self.loss_, summary = self.summary, vis = self.vis)
    
class TestInput:
    def __init__(self, inputs):
        self.weights = {}
        self.data_selector = inputs.q_ph
        
        data = inputs.data()

        def foo(x):
            total = 0
            if isinstance(x, dict):
                for val in x.values():
                    total += foo(val)
            elif isinstance(x, tuple):
                for y in x:
                    total += foo(y)
            else:
                total += tf.reduce_sum(x)
            return total

        bar = foo(data)
        
        self.test_run = Munch(bar = bar)
        self.val_run = Munch(bar = bar)
        self.train_run = Munch(bar = bar)

    def go(self):
        pass

        
class MnistAE(Net):

    def go_up_to_loss(self, index = None, is_training=None):
        if const.eager:
            self.is_training = tf.constant(is_training, tf.bool)
        self.prepare_data(index)

        if const.MNIST_CONVLSTM:
            net_fn = utils.nets.MnistAEconvlstm
        else:
            net_fn = utils.nets.MnistAE
            
        self.pred = net_fn(self.img)

        self.loss_ = utils.losses.binary_ce_loss(self.pred, self.img)
        
        if const.MNIST_CONVLSTM_STOCHASTIC:
            self.loss_ += self.pred.loss #kl
        
        return self.loss_
        
    def prepare_data(self, index):
        data = self.input.data(index)
        self.__dict__.update(data) #so we can do self.images, etc.
        
    def build_vis(self):
        self.vis = {
            'in': self.img,
            'out': self.pred
        }

class MultiViewNet(Net):

    def go_up_to_loss(self, index = None, is_training=None):
        # awkward, but necessary in order to record gradients using tf.eager
        if const.eager:
            self.is_training = tf.constant(is_training, tf.bool)
        
        self.prepare_data(index)
        self.pred_aux_input = self.construct_pred_aux_input()
        self.pred_main_input = self.construct_pred_main_input()
        self.predict()
        self.reproject()
        self.loss()
        return self.loss_

    def prepare_data(self, index):
        data = self.input.data(index)
        self.__dict__.update(data) #so we can do self.images, etc.
        
        #for convenience:
        self.phis_oh = [tf.one_hot(phi, depth = const.VV) for phi in self.phis]
        self.thetas_oh = [tf.one_hot(theta, depth = const.HV) for theta in self.thetas]

    def predict(self):
        with tf.variable_scope('3DED'):
            self.predict_(self.pred_main_input, self.pred_aux_input)

    def get_views_for_prediction(self):
        #order *must* be mask, depth, norms, else this will fail!
        # subtract 4 from the depth so that it is (-1,1)
        # (the shapenet data is all centered with camera at distance of 4)
        self.depths_ = [d-const.radius for d in self.depths]
        return [
            self.masks[:const.NUM_VIEWS],
            self.depths_[:const.NUM_VIEWS],
            self.images[:const.NUM_VIEWS],
        ]
            
    def construct_pred_main_input(self):
        pred_inputs = self.get_views_for_prediction()
        
        pred_inputs = list(zip(*pred_inputs))
        pred_inputs = [tf.concat(x, axis = 3) for x in pred_inputs]

        pred_inputs = self.preunprojection(pred_inputs)

        def stack_unproject_unstack(_pred_inputs):
            _pred_inputs = tf.stack(_pred_inputs, axis = 0)
            _pred_inputs = tf.map_fn(
                lambda x: utils.nets.unproject(x, self.__class__.resize_unproject),
                _pred_inputs, parallel_iterations = 1
            )
            _pred_inputs = tf.unstack(_pred_inputs, axis = 0)
            return _pred_inputs
        
        if isinstance(pred_inputs[0], list):
            rval = [stack_unproject_unstack(inp) for inp in pred_inputs]
        else:
            rval = stack_unproject_unstack(pred_inputs)
        return rval

    def preunprojection(self, pred_inputs):
        return pred_inputs

    def construct_pred_aux_input(self):
        self.poses = [
            tf.concat([phi, theta], axis = 1) for phi, theta in
            zip(self.phis_oh[:const.NUM_VIEWS], self.thetas_oh[:const.NUM_VIEWS])
        ]
        if const.MULTI_UNPROJ:
            return tf.concat(self.poses, axis = 1)
        else:
            return self.poses[0]

    def aggregate_inputs(self, inputs):
        if const.AGGREGATION_METHOD == 'stack':
            return tf.concat(inputs, axis = -1)
        elif const.AGGREGATION_METHOD == 'gru':
            return utils.nets.gru_aggregator(inputs)
        else:
            raise Exception('unknown aggregation method')

    def bin2theta(self, bin):
        return tf.cast(bin, tf.float32) * const.HDELTA + const.MINH

    def bin2phi(self, bin):
        return tf.cast(bin, tf.float32) * const.VDELTA + const.MINV
    
    def i2theta(self, idx):
        return self.bin2theta(self.thetas[idx])

    def i2phi(self, idx):
        return self.bin2phi(self.phis[idx])

    def translate_views_single(self, vid1, vid2, vox):
        return self.translate_views_multi([vid1], [vid2], [vox])[0]

    def translate_views_multi(self, vid1s, vid2s, voxs):
        """
        # num_views x batch x D X H X W X C
        rotates the 5d tensor `vox` from one viewpoint to another
        vid1s: indices of the bin corresponding to the input view
        vid2s: indices of the bin corresponding to the output view
        """
        # num_views x 8
        dthetas = [self.i2theta(vid2) - self.i2theta(vid1) for (vid2, vid1) in zip(vid2s, vid1s)]
        phi1s = list(map(self.i2phi, vid1s))
        phi2s = list(map(self.i2phi, vid2s))

        dthetas = tf.stack(dthetas, 0)
        phi1s = tf.stack(phi1s, 0)
        phi2s = tf.stack(phi2s, 0)
        
        voxs = tf.stack(voxs, 0)
        
        f = lambda x: utils.voxel.translate_given_angles(*x)
        out = tf.map_fn(f, [dthetas, phi1s, phi2s, voxs], dtype = tf.float32)
        return tf.unstack(out, axis = 0)
    
class MultiViewReconstructionNet(MultiViewNet):

    resize_unproject = True

    def predict_(self, pred_main_inputs, pred_aux_input):
        
        pred_main_inputs_ = [pred_main_inputs[0]]

        pred_main_inputs__ = self.translate_views_multi(
            list(range(1, const.NUM_VIEWS)),
            [0]*(const.NUM_VIEWS-1),
            pred_main_inputs[1:]
        )

        pred_main_inputs_.extend(pred_main_inputs__)
        #[8x32x32x32x32, 8x16x16x16x64, 8x8x8x8x12, 8x4x4x4x256]
        pred_main_input = self.aggregate_inputs(pred_main_inputs_)

        net_out = utils.nets.voxel_net_3d(
            pred_main_input, aux = pred_aux_input, outsize = const.S, d0 = 16
        )
        
        self.pred_voxel = net_out.pred
        self.pred_logit = net_out.logit
        self.pred_features = net_out.features

        if const.DEBUG_UNPROJECT:
            #visualize raw outlines...
            if const.MULTI_UNPROJ:
                self.pred_voxel = tf.reduce_max(pred_main_input, axis = 4, keep_dims = True)
            else:
                self.pred_voxel = pred_main_input

        if const.FAKE_NET:
            self.pred_voxel = tf.nn.tanh(tf.Variable(
                np.zeros(self.pred_voxel.get_shape()),
            dtype = tf.float32)) * 0.5 + 0.5


    def flatten(self, voxels):
        pred_depth = utils.voxel.voxel2depth_aligned(voxels)
        pred_mask = utils.voxel.voxel2mask_aligned(voxels)
        
        #replace bg with grey
        hard_mask = tf.cast(pred_mask > 0.5, tf.float32)
        pred_depth *= hard_mask
        pred_depth += const.RADIUS * (1.0 - hard_mask)

        pred_depth = tf.image.resize_images(pred_depth, (const.H, const.W))
        pred_mask = tf.image.resize_images(pred_mask, (const.H, const.W))
        return pred_depth, pred_mask


    def reproject(self): #this part is unrelated to the consistency...
        
        def rot_mat_for_angles_(invert_rot = False):
            return utils.voxel.get_transform_matrix_tf(
                theta = self.i2theta(0), 
                phi = self.i2phi(0), 
                invert_rot = invert_rot
            )

        world2cam_rot_mat = rot_mat_for_angles_()
        cam2world_rot_mat = rot_mat_for_angles_(invert_rot = True)
        
        #let's compute the oriented gt_voxel
        gt_voxel = tf.expand_dims(self.voxel, axis = 4)
        gt_voxel = utils.voxel.transformer_preprocess(gt_voxel)
        #simply rotate, but don't project!
        gt_voxel = utils.voxel.rotate_voxel(gt_voxel, world2cam_rot_mat)
        
        self.gt_voxel = gt_voxel

        #used later
        obj1 = tf.expand_dims(self.obj1, axis = 4)
        obj1 = utils.voxel.transformer_preprocess(obj1)
        self.obj1 = utils.voxel.rotate_voxel(obj1, world2cam_rot_mat)
        obj2 = tf.expand_dims(self.obj2, axis = 4)
        obj2 = utils.voxel.transformer_preprocess(obj2)
        self.obj2 = utils.voxel.rotate_voxel(obj2, world2cam_rot_mat)
        
        if not const.DEBUG_VOXPROJ:
            voxels_to_reproject = self.pred_voxel
        else:
            voxels_to_reproject = self.gt_voxel

        to_be_projected_and_postprocessed = [voxels_to_reproject]

        to_be_projected_and_postprocessed.extend(
            self.translate_views_multi(
                [0] * (const.NUM_VIEWS + const.NUM_PREDS - 1),
                list(range(1, const.NUM_VIEWS + const.NUM_PREDS)),
                tf.tile(
                    tf.expand_dims(voxels_to_reproject, axis = 0),
                    [const.NUM_VIEWS + const.NUM_PREDS - 1, 1, 1, 1, 1, 1]
                )
            )
        )
        projected_voxels = tf.map_fn(
            utils.voxel.project_and_postprocess,
            tf.stack(to_be_projected_and_postprocessed, axis = 0),
            parallel_iterations = 1
        )
        projected_voxels = tf.unstack(projected_voxels)


        self.unoriented = utils.voxel.rotate_voxel(voxels_to_reproject, cam2world_rot_mat)

        self.pred_depths, self.pred_masks = list(zip(*list(map(self.flatten, projected_voxels))))
        self.projected_voxels = projected_voxels

        
    def loss(self):

        if const.S == 64:
            self.gt_voxel = utils.tfutil.pool3d(self.gt_voxel)
        elif const.S == 32:
            self.gt_voxel = utils.tfutil.pool3d(utils.tfutil.pool3d(self.gt_voxel))
        
        loss = utils.losses.binary_ce_loss(self.pred_voxel, self.gt_voxel)
        if const.DEBUG_LOSSES:
            loss = utils.tfpy.print_val(loss, 'ce_loss')
        
        if const.DEBUG_VOXPROJ or const.DEBUG_UNPROJECT:
            z = tf.Variable(0.0)
            loss = z-z
        
        self.loss_ = loss

    def build_vis(self):
        self.vis = Munch(
            images = tf.concat(self.images, axis = 2),
            depths = tf.concat(self.depths, axis = 2),
            masks = tf.concat(self.masks, axis = 2),
            pred_masks = tf.concat(self.pred_masks, axis = 2),
            pred_depths = tf.concat(self.pred_depths, axis = 2),
            pred_vox = self.unoriented,
        )

        if hasattr(self, 'seg_obj1'):
            self.vis.seg_obj1 = self.seg_obj1
            self.vis.seg_obj2 = self.seg_obj2


class MultiViewQueryNet(MultiViewNet):

    resize_unproject = False

    def get_views_for_prediction(self):
        #no depth, mask, or seg
        return [self.images[:const.NUM_VIEWS]]
    
    def preunprojection(self, pred_inputs):
        with tf.variable_scope('2Dencoder'):
            return utils.tfutil.concat_apply_split(
                pred_inputs,
                utils.nets.encoder2D
            )
    
    def predict_(self, pred_main_inputs, pred_aux_input):
        pred_main_inputs_ = [
            self.translate_views_multi(
                list(range(0, const.NUM_VIEWS)),
                [0]*(const.NUM_VIEWS),
                x,
            )
            for x in pred_main_inputs
        ]

        pred_main_input = self.aggregate_inputs(pred_main_inputs_)

        # 8x 4x 4 x4 x 256, 8x8x8x8x128, 8x16x16x16x64, 8x32x32x32x32
        assert pred_aux_input is None
        self.feature_tensors = utils.nets.encoder_decoder3D(
            pred_main_input, aux = pred_aux_input,
        )

    def construct_pred_aux_input(self):
        return None

    def aggregate_inputs(self, inputs):
        assert const.AGGREGATION_METHOD == 'average'
        
        n = 1.0/float(len(inputs[0]))
        return [sum(input)*n for input in inputs]

    def reproject(self):
        # rotate scene to desired view point
        oriented_features = [
            self.translate_views_multi(
                [0] * const.NUM_PREDS,
                list(range(const.NUM_VIEWS, const.NUM_VIEWS + const.NUM_PREDS)),
                tf.tile(
                    tf.expand_dims(feature, axis = 0),
                    [const.NUM_PREDS, 1, 1, 1, 1, 1]
                )
            )
            for feature in self.feature_tensors
        ]
        # a NxM list of lists, where N is the number of feature scales and M is
        # the number of target views
        #concatenating before projection mysteriously fails???
        oriented_features = [
            utils.voxel.transformer_postprocess(
                tf.concat(
                    [
                        utils.voxel.project_voxel(feature)
                        for feature in features
                    ],
                    axis = 0
                )
            )
            for features in oriented_features
        ]

        with tf.variable_scope('2Ddecoder'):
            pred_views = utils.nets.decoder2D(oriented_features)
        self.pred_views = tf.split(pred_views, const.NUM_PREDS, axis = 0)
        
        
    def build_vis(self):
        self.vis = Munch(
            input_views = tf.concat(self.images[:const.NUM_VIEWS], axis = 2),
            query_views = tf.concat(self.queried_views, axis = 2),
            pred_views = tf.concat(self.pred_views, axis = 2),
            dump = {'occ': utils.nets.foo} if utils.nets.foo else {}
        )
        
    def loss(self):
        
        self.queried_views = self.images[const.NUM_VIEWS:]

        loss = sum(utils.losses.l2loss(pred, query, strict = True)
                   for (pred, query) in zip(self.pred_views, self.queried_views))
        loss /= const.NUM_PREDS

        #z = tf.Variable(0.0)
        #loss += z-z
        
        if const.DEBUG_LOSSES:
            loss = utils.tfpy.print_val(loss, 'l2_loss')
        self.loss_ = loss

class GQNBase(Net):

    def go_up_to_loss(self, index = None, is_training=None):
        if const.eager:
            self.is_training = tf.constant(is_training, tf.bool)
        self.setup_data()

        with tf.variable_scope('main'):
            self.predict()
            self.add_weights('main_weights')
            
        self.loss()
        return self.loss_

    def setup_data(self):
        data = self.input.data()
        self.__dict__.update(data)
        phis, thetas = zip(*[
            tf.unstack(cam, axis = 1)
            for cam in self.query.context.cameras
        ])

        #convert to degrees
        self.thetas = list(map(utils.utils.degrees, thetas))
        self.phis = list(map(utils.utils.degrees, phis))

        query_phi, query_theta = tf.unstack(self.query.query_camera, axis = 1)
        self.query_theta = utils.utils.degrees(query_theta)
        self.query_phi = utils.utils.degrees(query_phi)

    def predict(self):
        raise NotImplementedError

    def loss(self):
        #x = self.pred_view
        #loss = (tf.reduce_mean(x[:,:32,:32,:]) + tf.reduce_mean(x[:,32:,32:,:]) -
        #        tf.reduce_mean(x[:,32:,:32,:]) - tf.reduce_mean(x[:,:32,32:,:]))
        #loss = tf.reduce_mean(x[:,16:48,16:48,:]) - tf.reduce_mean(x)

        if const.LOSS_FN == 'L1':
            loss = utils.losses.l1loss(self.pred_view, self.target)
        elif const.LOSS_FN == 'CE':
            loss = utils.losses.binary_ce_loss(self.pred_view, self.target)

        if const.DEBUG_LOSSES:
            loss = utils.tfpy.print_val(loss, 'recon-loss')
            
        if const.GQN3D_CONVLSTM_STOCHASTIC:
            loss += utils.tfpy.print_val(self.pred_view.loss, 'kl-loss')

        if const.EMBEDDING_LOSS:
            embed_loss = utils.losses.embedding_loss(self.embed2d, self.embed3d)
            loss += utils.tfpy.print_val(const.embed_loss_coeff * embed_loss, 'embed-loss')
        
        self.loss_ = loss

    def build_vis(self):
        self.vis = Munch(
            input_views = tf.concat(self.query.context.frames, axis = 2),
            query_views = self.target,
            pred_views = self.pred_view
        )
        if const.ARITH_MODE:
            #actually i want the views on a vertical axis
            #and batch on the horizontal axis!

            if False:
                input_views = tf.concat(self.query.context.frames, axis = 2)
                input_views = tf.concat(tf.unstack(input_views), axis = 0)
                input_views = tf.expand_dims(input_views, axis = 0)
                input_views = tf.tile(input_views, (const.BS, 1, 1, 1))

            else: #for figure making
                input_views = self.query.context.frames[0]
                #i don't actually want the last view
                input_views = tf.unstack(input_views)
                input_views = [input_views[2], input_views[0], input_views[1]]
                input_views = tf.concat(input_views, axis = 1)
                input_views = tf.expand_dims(input_views, axis = 0)
                input_views = tf.tile(input_views, (const.BS, 1, 1, 1))

            query_views = tf.expand_dims(self.target[3], axis = 0) #view 3!!!
            query_views = tf.tile(query_views, (const.BS, 1, 1, 1))
            
            self.vis = Munch(
                input_views = input_views,
                query_views = query_views,
                pred_views = self.pred_view
            )


class GQN_with_2dencoder(GQNBase):
    '''shares some fns w/ gqn3d and gqn2d'''
    
    def get_inputs2Denc(self):
        return self.query.context.frames

    def get_outputs2Denc(self, inputs):
        with tf.variable_scope('2Dencoder'):
            f = lambda x: utils.nets.encoder2D(x,self.is_training)
            return utils.tfutil.concat_apply_split(
                inputs,
                f
                #utils.nets.encoder2D(*,self.is_training)
            )
        
    def aggregate(self, features):
        n = 1.0/float(len(features[0]))
        return [sum(feature)*n for feature in features]
        
class GQN3D(GQN_with_2dencoder):

    def predict(self):
        inputs2Denc = self.get_inputs2Denc()
        outputs2Denc = self.get_outputs2Denc(inputs2Denc)
        
        inputs3D = self.get_inputs3D(outputs2Denc)
        outputs3D = self.get_outputs3D(inputs3D)

        if const.ARITH_MODE:
            outputs3D = self.do_arithmetic(outputs3D)
        
        inputs2Ddec = self.get_inputs2Ddec(outputs3D)
        outputs2Ddec = self.get_outputs2Ddec(inputs2Ddec)

        self.pred_view = outputs2Ddec.pred_view
        if const.EMBEDDING_LOSS:
            self.embed2d = utils.nets.embedding_network(self.target)
            self.embed3d = outputs2Ddec.embedding

        self.tensors_to_dump = {
            'i2e': inputs2Denc,
            'o2e': outputs2Denc,
            'i3': inputs3D,
            'o3': outputs3D,
            'p3': self.__todump,
            'i2d': inputs2Ddec,
            'o2d': outputs2Ddec,
            'tgt': self.target,
        }

    def do_arithmetic(self, features):
        #0 contains [0,-,-]
        #1 contains [-,-,2]
        #2 contains [0,1,-]
        #3 contains [-,1,2]

        #testing #2 - #0 + #1 = #3
        
        def arith(feature):
            assert const.BS == 4
            feature =  tf.expand_dims(feature[2] - feature[0] + feature[1], axis = 0)
            return tf.tile(feature, (const.BS, 1, 1, 1, 1))
        
        return [arith(feature) for feature in features]
                           
    def get_inputs3D(self, inputs):
        unprojected_features = self.unproject_inputs(inputs)
        aligned_features = self.align_to_first(unprojected_features) # 4 scales, each in 3 views
        return self.aggregate(aligned_features)
        
    def get_outputs3D(self, inputs):
        with tf.variable_scope('3DED'):
            return utils.nets.encoder_decoder3D(inputs, self.is_training)
    
    def get_inputs2Ddec(self, inputs):
        aligned_inputs = self.align_to_query(inputs) #4 scales
        projected_inputs = self.project_inputs(aligned_inputs)

        self.__todump = projected_inputs #should also be postprocessed
        
        with tf.variable_scope('depthchannel_net'):
            return [utils.nets.depth_channel_net_v2(feat)
                    for feat in projected_inputs]

    def get_outputs2Ddec(self, inputs):
        with tf.variable_scope('2Ddecoder'):
            if const.GQN3D_CONVLSTM:
                return self.convlstm_decoder(inputs)
            else:
                raise Exception('need to update this with pred_view and embed attributes')
                return utils.nets.decoder2D(inputs, False)

    def convlstm_decoder(self, inputs):
        #we get feature maps of different resolution as input
        #downscale last and concat with second last

        inputs = [utils.tfutil.poolorunpool(x, 16) for x in inputs]
        net = tf.concat(inputs, axis = -1)
        net = slim.conv2d(net, 256, [3, 3])

        dims = 3+const.EMBEDDING_LOSS * const.embedding_size
        out, extra = utils.gqn_network.make_lstmConv(
            net,
            None,
            self.target,
            [['convLSTM', const.CONVLSTM_DIM, dims, const.CONVLSTM_STEPS, const.CONVLSTM_DIM]], 
            stochastic = const.GQN3D_CONVLSTM_STOCHASTIC,
            weight_decay = 1E-5,
            is_training = const.mode == 'train' or const.force_batchnorm_trainmode,
            reuse = False,
            output_debug = False,
        )

        out_img = utils.tfutil.tanh01(out[:,:,:,:3])
        embedding = out[:,:,:,3:] if const.EMBEDDING_LOSS else tf.constant(0.0, dtype = tf.float32)

        return Munch(pred_view = out_img, embedding = embedding, kl = extra['kl_loss'])

    def unproject_inputs(self, inputs):
        
        def stack_unproject_unstack(_inputs):
            _inputs = tf.stack(_inputs, axis = 0)
            _inputs = tf.map_fn(
                lambda x: utils.nets.unproject(x, False),
                _inputs, parallel_iterations = 1
            )
            _inputs = tf.unstack(_inputs, axis = 0)
            return _inputs
        
        return [stack_unproject_unstack(inp) for inp in inputs]

    def project_inputs(self, inputs):
        return [
            utils.voxel.transformer_postprocess(
                utils.voxel.project_voxel(feature)
            )
            for feature in inputs
        ]

    def translate_multiple(self, dthetas, phi1s, phi2s, voxs):
        dthetas = tf.stack(dthetas, axis = 0)
        phi1s = tf.stack(phi1s, 0)
        phi2s = tf.stack(phi2s, 0)
        voxs = tf.stack(voxs, 0)

        f = lambda x: utils.voxel.translate_given_angles(*x)
        out = tf.map_fn(f, [dthetas, phi1s, phi2s, voxs], dtype = tf.float32)
        return tf.unstack(out, axis = 0)
    
    def align_to_first(self, features):
        return [self.align_to_first_single(feature) for feature in features]

    def align_to_query(self, features):
        return [self.align_to_query_single(feature) for feature in features]
    
    def align_to_first_single(self, feature):
        #3 features from different views
        dthetas = [self.thetas[0] - theta for theta in self.thetas]
        phi1s = self.phis
        phi2s = [self.phis[0] for _ in self.phis]
        return self.translate_multiple(dthetas, phi1s, phi2s, feature)
    
    def align_to_query_single(self, feature):
        #a single feature from view 0
        dthetas = [self.query_theta - self.thetas[0]]
        phi1s = [self.phis[0]]
        phi2s = [self.query_phi]
        return self.translate_multiple(dthetas, phi1s, phi2s, [feature])[0]

    def build_vis(self):
        super().build_vis()
        self.vis.dump = self.tensors_to_dump

        if const.EMBEDDING_LOSS:
            self.vis.embed = tf.concat([self.embed2d, self.embed3d], axis = 2)
        
class GQN2D(GQN_with_2dencoder):
    
    def predict(self):
        self.tensors_to_dump = {}
        
        inputs2Denc = self.get_inputs2Denc()
        outputs2Denc = self.get_outputs2Denc(inputs2Denc)
        encoded = self.aggregate(outputs2Denc)
        encoded = self.add_query(encoded)
        decoded = utils.nets.decoder2D(encoded[::-1], False)
        self.pred_view = decoded

    def add_query(self, encoded):
        pose_info = tf.concat(self.poses + [self.query_pose], axis = 1)
        encoded = [utils.tfutil.add_feat_to_img(feat, pose_info) for feat in encoded]

        import tensorflow.contrib.slim as slim
        encoded = [
            slim.conv2d(feat, dims, 1, activation_fn = None)
            for (feat, dims) in zip(encoded, [64, 128, 256, 512])
        ]

        return encoded

    def setup_data(self):
        super().setup_data()
        
        thetas_r = list(map(utils.utils.radians, self.thetas))
        phis_r = list(map(utils.utils.radians, self.phis))
        query_theta_r = utils.utils.radians(self.query_theta)
        query_phi_r = utils.utils.radians(self.query_phi)

        foo = lambda theta, phi: [tf.cos(theta), tf.sin(theta), tf.cos(phi), tf.sin(phi)]
        bar = lambda theta, phi: tf.stack(foo(theta, phi), axis = 1)
        self.poses = [bar(*x) for x in zip(thetas_r, phis_r)]
        self.query_pose = bar(query_theta_r, query_phi_r)

class GQN2Dtower(GQNBase):

    def predict(self):

        #encoder
        images = list(self.query.context.frames)#+ [self.target]
        cam_posrot = list(self.query.context.cameras)
        encoded = utils.gqn_network.gqn2d_encoder(
            images, cam_posrot, is_training = const.mode == 'train' or const.force_batchnorm_trainmode
        )
        encoded = sum(encoded)
        if const.ARITH_MODE:
            encoded = self.do_arithmetic(encoded)        
        
        #decoder
        camera = tf.reshape(self.query.query_camera, (const.BS, 1, 1, 2))
        
        out, extra = utils.gqn_network.make_lstmConv(
            encoded,
            camera,
            self.target,
            [['convLSTM', const.CONVLSTM_DIM, 3, const.CONVLSTM_STEPS, const.CONVLSTM_DIM]], 
            stochastic = False,
            weight_decay = 1E-5,
            is_training = self.is_training,#const.mode == 'train' or const.force_batchnorm_trainmode,
            reuse = False,
            output_debug = False,
        )

        out = utils.tfutil.tanh01(out)
        out.loss = extra['kl_loss']

        self.pred_view = out
        

    def do_arithmetic(self, features):
        assert const.BS == 4
        feature = tf.expand_dims(features[2] - features[0] + features[1], axis = 0)
        return tf.tile(feature, (const.BS, 1, 1, 1))
