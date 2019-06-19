import sys
import math
import time

from options import OptionGroup as OG
import options

opname = 'reconstruction'
eager = False

LIMITED_DATASET = True
MASKED_DATASET = False

ARITH_MODE = False

# unlikely to change constants:
NUM_VIEWS = 3
NUM_PREDS = 1

MULTI_UNPROJ = True
AGGREGATION_METHOD = 'stack'
AUX_POSE = True
DUMP_TENSOR = False

PHI_IDX = None
CATS = 57 #number of categories
eps = 1E-8
#V = 18  # number of views for multiview -- we are keeping phi frozen for now

Hdata = 128
Wdata = 128

inject_summaries = False
summ_grads = False

HV = 18
VV = 3
MINH = 0
MAXH = 360 #exclusive
MINV = 0
MAXV = 30 #exclusive

H = 256
W = 256

fov = 30.0
radius = 4.0

bn_decay = 0.9

# since the scene size is always exactly 1
# if the actual scene size is larger or smaller
# you should scale the radius instead
# a scene twice as big is equivalent to having
# a radius half as large

#this should probably never be changed for an reason
SCENE_SIZE = 1.0

S = 128  # cube size
BS = 2
SS = 16
NS = BS * SS
NB_STEPS = 1000000

ORIENT = True
STNET = False


ARCH = 'unproj'
#options: 'unproj, marr'

NET3DARCH = 'marr' #or '3x3, marr'
USE_OUTLINE = True
USE_MESHGRID = True
USE_LOCAL_BIAS = False #set to false later

INPUT_RGB = False
INPUT_POSE = False
VOXNET_LATENT = 512


# test/train mode
mode = 'train'

# input constants
train_file = 'all'
val_file = 'all'
test_file = 'all'

# optimizer consts
lr = 1E-4
mom = 0.9

# validation period
valp = 1000
savep = 20000

# important directories
vis_dir = 'vis'
tb_dir = 'log'
data_dir = 'data'
ckpt_dir = 'ckpt'
ckpt_cfg_dir = 'ckpt_cfg'

# debug flags
FAKE_NET = False
REPROJ_SINGLE = False
ADD_FEEDBACK = False
VALIDATE_INPUT = False
DEBUG_MODE = False
DEBUG_32 = False
DEBUG_HISTS = False
DEBUG_PLACEMENT = False
DEBUG_VOXPROJ = False
DEBUG_VOXNET = False
DEBUG_REPROJ = False
DEBUG_EXPORTS = True
DEBUG_SPEED = True
DEBUG_NET = False
DEBUG_RP = False
DEBUG_FULL_TRACE = False
DEBUG_NODE_TRACE = False
DEBUG_NAN = False
DEBUG_CACHE = False
DEBUG_LOSSES = True
DEBUG_MEMORY = False
DEBUG_UNPROJECT = False

SKIP_RUN = False
SKIP_TRAIN_EXPORT = False
SKIP_VAL_EXPORT = False
SKIP_EXPORT = False

USE_GRAVITY_LOSS = False

FIX_VIEW = False
STOP_PRED_DELTA = True
STOP_REPROJ_MASK_GRADIENTS = False

USE_TIMELINE = False

rpvx_unsup = False
force_batchnorm_trainmode = False
force_batchnorm_testmode = False

RANDOMIZE_BG = True

MNIST_CONVLSTM = False
MNIST_CONVLSTM_STOCHASTIC = False

GQN3D_CONVLSTM = False
GQN3D_CONVLSTM_STOCHASTIC = False
LOSS_FN = 'L1'
CONVLSTM_DIM = 128
CONVLSTM_STEPS = 4

DEEPMIND_DATA = False
GQN_DATA_NAME = 'shepard_metzler_7_parts' # or rooms_ring_camera

#some stuff related to embeddings
embed_loss_coeff = 1.0
EMBEDDING_LOSS = False
embedding_size = 32
embedding_layers = 4

####

T0 = time.time()

exp_name = sys.argv[1].strip() if len(sys.argv) >= 2 else ''
load_name = ''
save_name = exp_name

options.data_options('doubledata', 'double', add_suffix = True)
options.data_options('doubledebugdata', 'double_single', add_suffix = False)
options.data_options('4_data', '4', add_suffix = True)
options.data_options('multi_data', 'multi', add_suffix = True)
options.data_options('arith_data', 'arith', add_suffix = True)
options.data_options('house_data', 'house', add_suffix = True)

#basicall we run the model for 1000 steps with a lower decay value so that update ops does its job
OG('fixbn', bn_decay = 0.99, savep = 1000)
OG('rename', savep = 1)

OG('doublemug',
   'doubledata',
   MULTI_UNPROJ = True, lr = 5E-4, BS = 2, valp = 1, H = 128, W = 128, data_dir = "double_tfrs/",
   MINV = 20, MAXV = 80, #VDELTA = 20
)

OG('doublemug_debug',
   'doublemug', 'doubledebugdata',
   DEBUG_VOXPROJ = True
)

OG('doublemug_train',
   'doublemug',
   valp = 100, savep = 10000, BS = 2
)

OG('doublemug_small',
   'doublemug_train',
   S = 64,
)

OG('doublemug_small_debug',
   'doublemug_small', 'doubledebugdata',
   DEBUG_VOXPROJ = False, DEBUG_UNPROJECT = True, valp = 50, BS = 4,
)

#what is voxproj vs unproj?
# works fine for
# no debug voxproj (single data) +
# debug voxproj? 
# no debug voxproj (all data) +
# debug unproj ??? seems fishy -- not sure if it works
OG('doublemug_small2_debug',
   'doublemug_small_debug',
   S = 32, H = 64, W = 64,
)

OG('doublemug_train_gru',
   'doublemug_train',
   AGGREGATION_METHOD = 'gru', BS = 1
)

#works w/ depth/mask
#works w/o depth/mask
OG('querytask',
   'doublemug_train_gru',
   opname = 'query',
   RANDOMIZE_BG = False, AGGREGATION_METHOD = 'average', BS = 2, lr = 1E-4,
   USE_OUTLINE = False, USE_MESHGRID = False, AUX_POSE = False
)

OG('querytask_debug',
   'querytask', 'doubledebugdata',
   lr = 1E-4
)

OG('size64',
   H = 64, W = 64, BS = 8,
)

#works
OG('querytask_debug64',
   'querytask_debug', 'size64'
)

#not sure if works w/ depth/mask
#doesn't work w/o depth/mask
#wait... this works!
OG('querytask64',
   'querytask', 'size64'
)

OG('querytask_eager',
   'querytask_debug', 
   eager = True, BS = 1, NUM_VIEWS = 2
)

OG('gqnbase',
   'querytask',
   NUM_PREDS = 1, H = 64, W = 64, S = None, savep = 10000, 
)

OG('gqn2d',
   'gqnbase', 
   opname = 'gqn2d', BS = 8, 
)

OG('gqntower',
   'gqnbase',
   opname = 'gqntower',
   BS = 8, CONVLSTM_DIM = 128, CONVLSTM_STEPS = 4, LOSS_FN = 'CE',
)


OG('gqntower2',
   'gqntower', load_name = 'gqntower'
)

OG('gqntower2_fixbn',
   'gqntower2', 'fixbn', load_name = 'gqntower2'
)

OG('gqntower2_eval',
   'gqntower2', load_name = 'gqntower2', mode = 'test', BS = 1,
)


OG('gqntower_debug',
   'gqntower', 'doubledebugdata'
)
   

OG('gqn3d',
   'gqnbase',
   opname = 'gqn3d', BS = 8
)

OG('gqn3d_ce', 'gqn3d', LOSS_FN='CE')

#converges to 0 quickly
OG('gqn3d_convlstm',
   'gqn3d', #'doubledebugdata',
   GQN3D_CONVLSTM = True,
   LOSS_FN = 'CE',
   CONVLSTM_DIM = 128,
   CONVLSTM_STEPS = 4,
)

OG('gqn3d_convlstm_stoch',
   'gqn3d_convlstm',
   GQN3D_CONVLSTM_STOCHASTIC = True,
)

OG('gqn3d_convlstm_big',
   'gqn3d_convlstm',
   CONVLSTM_DIM = 256,
   CONVLSTM_STEPS = 6,
)

#pretrain from 4 views
OG('grnn_shapenet_1view',
   'gqn3d_convlstm_big', load_name = 'gqn3d_convlstm_big',
   NUM_VIEWS=1,
)

#does not work
OG('gqn3d_cameratest1', 'gqn3d_convlstm_big', radius = 2.0)
#does not work
OG('gqn3d_cameratest2', 'gqn3d_convlstm_big', radius = 2.0, fov = 60.0)
#this works
OG('gqn3d_cameratest3', 'gqn3d_convlstm_big', fov = 15.0)
#ok what about this?
OG('gqn3d_cameratest4', 'gqn3d_convlstm_big', radius = 3.0)

OG('gqn3d_convlstm_big_eval',
   'gqn3d_convlstm_big', load_name = 'gqn3d_convlstm_big',
   mode = 'test', BS = 1
)
OG('gqn3d_convlstm_big_eval6', 'gqn3d_convlstm_big_eval')

OG('gqn3d_convlstm_4obj_eval',
   'gqn3d_convlstm_big', '4_data',
   load_name = 'gqn3d_convlstm_big',
   mode = 'test', BS = 1, data_dir = '4_tfrs', 
)

OG('gqntower_4obj_eval',
   'gqntower', '4_data',
   load_name = 'gqntower',
   mode = 'test', BS = 1, data_dir = '4_tfrs', 
)

#suncg
OG('gqn3d_suncg_base', 'gqn3d_convlstm_big', 'house_data', load_name = 'gqn3d_convlstm_big', data_dir = 'house_tfrs',
   HV = 8, VV = 3, MINV=20, MAXV=80, fov=30
)

OG('gqn3d_suncg_r4', 'gqn3d_suncg_base', radius = 4)
OG('gqn3d_suncg_r3', 'gqn3d_suncg_base', radius = 3)
OG('gqn3d_suncg_r2', 'gqn3d_suncg_base', radius = 2)
OG('gqn3d_suncg_r1', 'gqn3d_suncg_base', radius = 1)
OG('gqn3d_suncg_r0.5', 'gqn3d_suncg_base', radius = 0.5)

OG('gqn3d_suncg_r2_gru', 'gqn3d_suncg_base', radius = 2, AGGREGATION_METHOD='gru')

OG('gqn3d_suncg_masked', 'gqn3d_suncg_base', radius = 2, MASKED_DATASET = True)

OG('gqn3d_multi', 'gqn3d_convlstm_big', 'multi_data', load_name = 'gqn3d_convlstm_big', data_dir = 'multi_tfrs')

OG('gqn3d_multi2', 'gqn3d_multi')

OG('arith', 'gqn3d_multi', 'arith_data',
   data_dir = 'arith_tfrs', ARITH_MODE = True, load_name = 'gqn3d_multi',
   opname = 'gqn3d', BS = 4, #important!!,
   mode = 'test', FIX_VIEW = True,
)

OG('arith_redo', 'arith')
OG('arith_redo2', 'arith', force_batchnorm_trainmode = True)
OG('arith_redo3', 'arith_redo2')

OG('arith2', 'gqntower', 'arith_data',
   data_dir = 'arith_tfrs', ARITH_MODE = True, load_name = 'gqntower',
   opname = 'gqntower', BS = 4, #important!!,
   mode = 'test', FIX_VIEW = True,
)

OG('gqntower_eval',
   'gqntower', load_name = 'gqntower',
   mode = 'test', BS = 1
)

OG('gqn3d_deepmind',
   'gqn3d_convlstm_big',
   DEEPMIND_DATA = True, 
)

OG('gqn3d_rooms',
   'gqn3d_deepmind', GQN_DATA_NAME = 'rooms_ring_camera'
)

# OG('gqn3d_rooms2', 'gqn3d_rooms', radius = 2.0, savep=50000)
# OG('gqn3d_rooms3', 'gqn3d_rooms', load_name = 'gqn3d_rooms', fov = 20.0, savep=2000)


#also run gqn3d_rooms for a long time, and compare on test set
# OG('gqn3d_rooms4', 'gqn3d_rooms', fov = 60.0, radius = 1.2, savep=5000)
OG('gqn3d_rooms6', 'gqn3d_rooms', fov = 60.0, radius = 0.95, savep=5000)
OG('gqn3d_rooms6_resume', 'gqn3d_rooms6', load_name='gqn3d_rooms6')
OG('gqn3d_rooms6_eval', 'gqn3d_rooms6', load_name='gqn3d_rooms6_resume', BS = 1, mode='test')
#note that at a distance of 1 -- the far end of the scene, we have a coverage of siez 1
#we can also try the more typical radius of 2, at the risk of not covering the scene properly
   
OG('gqn3d_rooms_fixbn', 'gqn3d_rooms6', 'fixbn', load_name='gqn3d_rooms6_resume')

# OG('gqn3d_rooms5', 'gqn3d_rooms', fov = 60.0, radius = 2.0, savep=5000)

#.....
# OG('gqn3d_rooms6_embed', 'gqn3d_rooms5', EMBEDDING_LOSS = True)
# OG('gqn3d_rooms_embed', 'gqn3d_rooms6_embed', load_name = 'gqn3d_rooms6_embed')
# OG('gqn3d_rooms_embed2', 'gqn3d_rooms_embed', load_name = 'gqn3d_rooms_embed')

OG('gqn3d_deepmind2',
   'gqn3d_deepmind', load_name = 'gqn3d_deepmind'
)

OG('gqntower_deepmind',
   'gqntower',
   DEEPMIND_DATA = True, 
)

OG('gqntower_room',
   'gqntower_deepmind', GQN_DATA_NAME = 'rooms_ring_camera'
)

OG('gqntower_room_fixbn', 'gqntower_room', 'fixbn', load_name='gqntower_room')

OG('gqntower_room4', 'gqntower_room', BS = 32, CONVLSTM_STEPS = 12)

OG('gqntower_room_eval', 'gqntower_room', load_name = 'gqntower_room3', BS = 1, mode = 'test')

OG('gqntower_deepmind2',
   'gqntower_deepmind', load_name = 'gqntower_deepmind',
   DEEPMIND_DATA = True, 
)

OG('gqntower_deepmind2_fixbn', 'gqntower_deepmind2', 'fixbn', load_name='gqntower_deepmind2')

OG('gqn3d_deepmind_eval', 'gqn3d_deepmind', load_name = 'gqn3d_deepmind2', mode = 'test', BS = 1)
OG('gqntower_deepmind_eval', 'gqntower_deepmind', load_name = 'gqntower_deepmind2', mode = 'test', BS = 1)

OG('sm_fix_bn', 'gqn3d_deepmind', 'fixbn', load_name='gqn3d_deepmind2')

OG('gqntest',
   'gqnbase',
   opname = 'gqntest', DEEPMIND_DATA = True,
)


########

OG('gqn3dv2', 'gqn3d', lr = 5E-4)
OG('gqn3dv3', 'gqn3d', lr = 2E-5)


OG('gqn3d_debug',
   'gqn3d', 'doubledebugdata',
   FIX_VIEW = False
)

OG('mnist',
   opname = 'mnist', BS = 64, valp = 200,
)

OG('mnist_convlstm',
   'mnist',
   MNIST_CONVLSTM = True,
)

OG('mnist_convlstm_stoch',
   'mnist_convlstm',
   MNIST_CONVLSTM_STOCHASTIC = True,
)

#########

generate_views = False
ELEV_GRANULARITY = 1
AZIMUTH_GRANULARITY = 24*4
MIN_ELEV = 0
MAX_ELEV = 80
GEN_FRAMERATE = 24
# generate all possible output views

#rooms dataset
OG('gqn3d_rooms_gen', 'gqn3d_rooms6_eval', generate_views = True)
OG('gqn3d_rooms_gen_fixbn', 'gqn3d_rooms_fixbn', generate_views = True,
   BS = 1, mode = 'test', load_name = 'gqn3d_rooms_fixbn')

OG('gqntower_rooms_gen', 'gqntower_room_eval', generate_views = True)

#SM shapes
OG('gqn3d_sm_gen', 'gqn3d_deepmind_eval', generate_views = True, ELEV_GRANULARITY = 3, MIN_ELEV = -30, MAX_ELEV = 30)
OG('gqntower_sm_gen', 'gqntower_deepmind_eval', generate_views = True, ELEV_GRANULARITY = 3, MIN_ELEV = -30, MAX_ELEV = 30)

#shapenet shapes
#these dont' seem to work yet...
OG('gqn3d_shapenet_gen', 'gqn3d_convlstm_big_eval', generate_views = True,
   AZIMUTH_GRANULARITY = 18, ELEV_GRANULARITY = 3, MIN_ELEV = 0, MAX_ELEV = 40,
   GEN_FRAMERATE = 8)

OG('gqntower_shapenet_gen', 'gqntower2_eval', generate_views = True,
   AZIMUTH_GRANULARITY = 18, ELEV_GRANULARITY = 3, MIN_ELEV = 0, MAX_ELEV = 40,
   GEN_FRAMERATE = 8)

#...
OG('gqn3d_shapenet4_gen', 'gqn3d_convlstm_4obj_eval', generate_views = True,
   AZIMUTH_GRANULARITY = 18, ELEV_GRANULARITY = 3, MIN_ELEV = 0, MAX_ELEV = 40,
   GEN_FRAMERATE = 8)

OG('gqntower_shapenet4_gen', 'gqntower_4obj_eval', generate_views = True,
   AZIMUTH_GRANULARITY = 18, ELEV_GRANULARITY = 3, MIN_ELEV = 0, MAX_ELEV = 40,
   GEN_FRAMERATE = 8)

#suncg gen views
OG('gqn3d_suncg_gen', 'gqn3d_suncg_masked', generate_views = True,
   AZIMUTH_GRANULARITY = 18, ELEV_GRANULARITY = 3, MIN_ELEV = 0, MAX_ELEV = 40,
   GEN_FRAMERATE = 8,
   mode='test', BS=1, load_name='gqn3d_suncg_masked')

# 1 view training

OG('shapenet_1view', 'gqn3d_convlstm_big', load_name = 'gqn3d_convlstm_big', NUM_VIEWS=1)
OG('metzler_1view', 'gqn3d_deepmind', load_name = 'gqn3d_deepmind', NUM_VIEWS=1)
OG('room_1view', 'gqn3d_rooms6', load_name = 'gqn3d_rooms6_resume', NUM_VIEWS=1)

OG('shapenet_fixbn', 'gqn3d_convlstm_big', 'fixbn', load_name = 'gqn3d_convlstm_big')
OG('arith_fixbn', 'gqn3d_multi', 'fixbn', load_name = 'shapenet_fixbn')

# 1 view gen

OG('shapenet_1view_gen', 'shapenet_1view', load_name='shapenet_1view',
   mode = 'test', BS = 1,
   generate_views = True, AZIMUTH_GRANULARITY = 18, ELEV_GRANULARITY = 3,
   MIN_ELEV = 0, MAX_ELEV = 40, GEN_FRAMERATE = 8)

OG('shapenet4_1view_gen', 'shapenet_1view', '4_data', load_name='shapenet_1view',
   mode = 'test', BS = 1, data_dir = '4_tfrs', 
   generate_views = True, AZIMUTH_GRANULARITY = 18, ELEV_GRANULARITY = 3,
   MIN_ELEV = 0, MAX_ELEV = 40, GEN_FRAMERATE = 8)

OG('metzler_1view_gen', 'metzler_1view', load_name = 'metzler_1view',
   mode = 'test', BS = 1,   
   generate_views = True, ELEV_GRANULARITY = 3, MIN_ELEV = -30, MAX_ELEV = 30)

OG('room_1view_gen', 'room_1view', load_name='room_1view',
   mode = 'test', BS = 1,   
   generate_views = True)
   

OG('tower_shapenet_trained', 'gqntower2_fixbn', 'rename', load_name='gqntower2_fixbn')
OG('tower_metzler_trained', 'gqntower_deepmind2_fixbn', 'rename', load_name='gqntower_deepmind2_fixbn')
OG('tower_rooms_trained', 'gqntower_room_fixbn', 'rename', load_name='gqntower_room_fixbn')

OG('grnn_shapenet_trained', 'shapenet_fixbn', 'rename', load_name='shapenet_fixbn')
OG('grnn_metzler_trained', 'sm_fix_bn', 'rename', load_name='sm_fix_bn')
OG('grnn_rooms_trained', 'gqn3d_rooms_fixbn', 'rename', load_name='gqn3d_rooms_fixbn')


# START READING HERE #

#TRAINING

OG('tower_metzler_train', 'gqntower_deepmind', load_name='')
OG('tower_rooms_train', 'gqntower_room', load_name='')
OG('tower_shapenet_train', 'gqntower', load_name='')

OG('grnn_metzler_train', 'gqn3d_deepmind', load_name='')
OG('grnn_rooms_train', 'gqn3d_rooms6', load_name='')
OG('grnn_shapenet_train', 'gqn3d_convlstm_big', load_name='')

# EVAL and VISUALIZATION

OG('tower_metzler_eval', 'gqntower_deepmind_eval', load_name='tower_metzler_train')
OG('tower_metzler_gen', 'gqntower_sm_gen', load_name='tower_metzler_train')

OG('tower_rooms_eval', 'gqntower_room_eval', load_name='tower_rooms_train')
OG('tower_rooms_gen', 'gqntower_rooms_gen', load_name='tower_rooms_train')

OG('tower_shapenet_eval', 'gqntower2_eval', load_name='tower_shapenet_train')
OG('tower_shapenet_gen', 'gqntower_shapenet_gen', load_name='tower_shapenet_train')

OG('tower_shapenet4_eval', 'gqntower_4obj_eval', load_name='tower_shapenet_train')
OG('tower_shapenet4_gen', 'gqntower_shapenet4_gen', load_name='tower_shapenet_train')

OG('tower_arith_eval', 'arith2', load_name='tower_shapenet_train') 

##

OG('grnn_metzler_eval', 'gqn3d_deepmind_eval', load_name='grnn_metzler_train')
OG('grnn_metzler_gen', 'gqn3d_sm_gen', load_name='grnn_metzler_train')

OG('grnn_rooms_eval', 'gqn3d_rooms6_eval', load_name='grnn_rooms_train')
OG('grnn_rooms_gen', 'gqn3d_rooms_gen', load_name='grnn_rooms_train')

OG('grnn_shapenet_eval', 'gqn3d_convlstm_big_eval', load_name='grnn_shapenet_train')
OG('grnn_shapenet_gen', 'gqn3d_shapenet_gen', load_name='grnn_shapenet_train')

OG('grnn_shapenet4_eval', 'gqn3d_convlstm_4obj_eval', load_name='grnn_shapenet_train')
OG('grnn_shapenet4_gen', 'gqn3d_shapenet4_gen', load_name='grnn_shapenet_train')

OG('grnn_arith_eval', 'arith', load_name='grnn_shapenet_train')

##

OG('grnn_metzler_eval_1v', 'grnn_metzler_eval', NUM_VIEWS=1)
OG('grnn_metzler_gen_1v', 'grnn_metzler_gen', NUM_VIEWS=1)

OG('grnn_rooms_eval_1v', 'grnn_rooms_eval', NUM_VIEWS=1)
OG('grnn_rooms_gen_1v', 'grnn_rooms_gen', NUM_VIEWS=1)

OG('grnn_shapenet_eval_1v', 'grnn_shapenet_eval', NUM_VIEWS=1)
OG('grnn_shapenet_gen_1v', 'grnn_shapenet_gen', NUM_VIEWS=1)

OG('grnn_shapenet4_eval_1v', 'grnn_shapenet4_eval', NUM_VIEWS=1)
OG('grnn_shapenet4_gen_1v', 'grnn_shapenet4_gen', NUM_VIEWS=1)

####

def _verify_(key, value):
    #print(key, '<-', value)
    print('{0:20} <--  {1}'.format(key, value))
    assert key in globals(), ('%s is new variable' % key)

if exp_name not in options._options_:
    print('*' * 10 + ' WARNING -- no option group active ' + '*' * 10)
else:
    print('running experiment', exp_name)
    for key, value in options.get(exp_name).items():
        _verify_(key, value)
        globals()[key] = value

#stuff which must be computed afterwards, because it is a function of the constants defined above


HDELTA = (MAXH-MINH) / HV #20
VDELTA = (MAXV-MINV) / VV #10

# camera stuffs
fx = W / 2.0 * 1.0 / math.tan(fov * math.pi / 180 / 2)
fy = fx
focal_length = fx / (W / 2.0) #NO SCALING

x0 = W / 2.0
y0 = H / 2.0

#scene stuff
near = radius - SCENE_SIZE
far = radius + SCENE_SIZE

#other
GEN_NUM_VIEWS = ELEV_GRANULARITY * AZIMUTH_GRANULARITY
