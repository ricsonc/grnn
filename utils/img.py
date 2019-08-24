import random as random
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from scipy.misc import imsave, toimage
import constants as const
import math

try:
    import matplotlib.pyplot as plt
except:
    print('failed to import matplotlib')


def add_ch(im2):
    h, w, _ = np.shape(im2)
    zs = np.zeros((h, w, 1))
    return np.concatenate([im2, zs], axis=2)


def triple_ch(img1):
    return np.concatenate([img1, img1, img1], axis=2)


def make3ch(im):
    _shape = np.shape(im)
    if len(_shape) == 2:
        im = np.reshape(im, (_shape[0], _shape[1], 1))
        c = 1
    elif len(_shape) == 3:
        c = _shape[2]
    else:
        raise Exception('unsupported shape').with_traceback(_shape)
    if c == 1:
        return triple_ch(im)
    elif c == 2:
        return add_ch(im)
    elif c == 3:
        return im
    else:
        raise Exception('unsupported number of channels %d' % c)


def imsave01(name, img, _min=0.0, _max=1.0):
    img -= _min
    img /= (_max - _min)
    img *= 255.99
    img = img.astype(np.int64)
    #imsave(name, img)
    toimage(img, cmin=0, cmax=255).save(name)


def imsavegrid(pth, img):
    h, w, _ = np.shape(img)
    img = np.reshape(img, (h, w, const.V, -1))
    c = img.shape[-1]
    numcols = int(math.sqrt(const.V))
    numrows = (const.V - 1) / numcols + 1  # updiv
    canvas = np.zeros((numrows * const.H, numcols * const.W, c)).astype(img.dtype)
    for i in range(const.V):
        part = img[:, :, i, :]
        #row = i%numrows
        #col = i/numrows
        col = i % numcols
        row = i / numcols
        canvas[row * const.H:(row + 1) * const.H, col * const.W:(col + 1) * const.W, :] = part

    if c == 1:
        canvas = flatimg(canvas)
    if c == 2:
        canvas = add_ch(canvas)

    if img.dtype == np.int64:
        toimage(canvas, cmin=0, cmax=255).save(pth)
    else:
        #imsave01(pth, canvas)
        imsave(pth, canvas)


def imsavegrid01(pth, img, _min=0.0, _max=1.0):
    img -= _min
    img /= (_max - _min)
    img *= 255.99
    img = img.astype(np.int64)
    imsavegrid(pth, img)


def imsave01clip(name, img, _min=0.0, _max=1.0):
    img = np.clip(img, _min, _max)
    imsave01(name, img, _min, _max)


def flatimg(img):
    h, w, c = img.shape
    assert c == 1
    return np.reshape(img, (h, w))

def text_on_img_(img, text, c='w'):
    # only works on uint8 3 channel images!
    assert c in 'bw'
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    loc = (0, 0)
    color = (0, 0, 0) if c == 'b' else (255, 255, 255)
    draw.text(loc, text, color, font=font)
    return np.asarray(img)


def img_to_rgb_256(img):
    shape = img.shape
    rank = len(shape)
    if rank == 2:
        img = np.stack([img, img, img], axis=2)
        return img_to_rgb_256(img)
    elif rank == 3:
        h, w, d = img.shape
        if d == 1:
            img = np.concatenate([img, img, img], axis=2)
            return img_to_rgb_256(img)
        elif d == 2:
            img = add_ch(img)
            return img_to_rgb_256(img)
        elif d == 3:
            if img.dtype.kind not in 'ui':  # flow...
                if np.min(img) >= 0.0 and np.max(img) <= 1.0:
                    img *= 255.99
                    img = img.astype(np.uint8)
                    return img_to_rgb_256(img)
                else:
                    img -= np.min(img)
                    img /= np.max(img)
                    return img_to_rgb_256(img)
            else:
                assert np.min(img) >= 0 and np.max(img) <= 255
                img = img.astype(np.uint8)
                return img
        elif d == 4:
            img = img[:, :, 0:3]
            return img_to_rgb_256(img)
        else:
            raise Exception('unsupported img depth')
    elif rank == 4:
        bs = img.shape[0]
        assert bs == 1
        return img_to_rgb_256(img[0])
    else:
        raise Exception('unsupported rank')


def select_color(img):
    region = img[0:12, 0:100, :]
    if np.mean(region) > 127:
        return 'b'
    else:
        return 'w'


def text_on_img(img, text, c=None):
    img = img_to_rgb_256(img)
    if c is None:
        c = select_color(img)
    return text_on_img_(img, text, c)


def imsave_namedgrid(pth, imgdict):
    # imgdict maps img names -> img
    n = len(imgdict)
    images = []
    for imgname, img in imgdict:
        images.append(text_on_img(img, imgname))

    numcols = int(math.sqrt(n))
    numrows = (n - 1) / numcols + 1  # updiv
    canvas = np.zeros((numrows * const.H, numcols * const.W, 3), dtype=np.uint8)
    for i in range(n):
        col = i % numcols
        row = i / numcols
        canvas[row * const.H:(row + 1) * const.H, col * const.W:(col + 1) * const.W, :] = images[i]
    imsave(pth, canvas)


def plot_flow(flow):
    print('plotting flow!')

    n = 256

    X, Y = np.mgrid[0:n, 0:n]
    U = flow[0, :, :, 0]
    V = flow[0, :, :, 1]
    theta = np.arctan2(U, V)

    fig = plt.figure(figsize=(1.0, 1.0), dpi=2560)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.quiver(X, Y, U, V, theta, units='x', scale=1.0, alpha=0.5)
    plt.xlim(-1, n)
    plt.xticks(())
    plt.ylim(-1, n)
    plt.yticks(())

    i = random.randint(0, 9)
    fig.savefig('test/flow%d' % i, bbox_inches=0, pad_inches=0)
