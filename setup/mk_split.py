#!/usr/bin/env python
from os import listdir
from os.path import join, isdir
import random

IN_DIR = '/home/ricson/data/shapenet_tfrs'
names = listdir(IN_DIR)

random.seed(0)
random.shuffle(names)

N = len(names)


def write_names(names, fn):
    with open(join('lists', fn), 'w') as f:
        for name in names:
            f.write(name)
            f.write('\n')


# write_names([], 'empty')
# write_names(names[:N / 2], 'half1')
# write_names(names[N / 2:], 'half2')
# write_names(names[:3 * N / 4], 'q123')
# write_names(names[3 * N / 4:], 'q4')
# write_names(names[3 * N / 4: 7 * N / 8], 'val_')
# write_names(names[7 * N / 8: ], 'test_')
# write_names(names[0:1], 'one')
# write_names(names[0:2], 'two')
# write_names(names[0:10], 'ten')
# write_names(names, 'all')

# for i in range(10):
#     write_names(names[i + 10:i + 11], 'single_%d' % i)


    
def split_by_category():
    listcompletedir = lambda x: [join(x,y) for y in listdir(x)]
    listonlydir = lambda x: list(filter(isdir, listcompletedir(x)))
    CATEGORIES = sorted(listonlydir('/home/ricson/data/ShapeNetCore.v1/'))

    path2cat = {}
    
    for i, cat in enumerate(CATEGORIES):
        paths = listdir(cat)
        for path in paths:
            path2cat[path] = i


    synset2catno = {}
    for i, cat in enumerate(CATEGORIES):
        synset = cat.split('/')[-1]
        synset2catno[synset] = i
            
    #partition the categories
    cats = list(range(len(CATEGORIES)))
    random.seed(0)
    random.shuffle(cats)
    #there's 57
    train_cats = cats[0:40]
    val_cats = cats[40:50]
    test_cats = cats[50:57]

    train_names = []
    val_names = []
    test_names = []

    #cat 20 is chairs
    cat_names = [[] for _ in CATEGORIES]
    
    for name in names:
        name_cat = path2cat[name]
        cat_names[name_cat].append(name)
        if name_cat in train_cats:
            train_names.append(name)
        elif name_cat in test_cats:
            test_names.append(name)
        elif name_cat in val_cats:
            val_names.append(name)
        else:
            print(name)
            assert False, 'bad'

    print('-=====-')
    print(len(train_names))
    print(len(val_names))
    print(len(test_names))
    print('-=====-')

    car_synsets = ['02958343']
    hh_obj_synsets = ['02801938', '02876657', '02942699', '02954340',
                      '03046257', '03085013', '03261776', '03513137',
                      '03624134', '03593526', '03636649', '03642806',
                      '03761084', '03797390', '03991062']
    mug_synsets = ['03797390']
    
    car_hh = car_synsets+hh_obj_synsets

    def write_synset_names(synset_names, name):
        catnos = [synset2catno[synset] for synset in synset_names]
        names = []
        for catno in catnos:
            names.extend(cat_names[catno])

        #duh..
        random.seed(0)
        random.shuffle(names)
            
        write_names(names, name)
        l = 3*len(names)/4
        write_names(names[:l], '%s_train' % name)
        write_names(names[l:], '%s_val' % name)
        
    write_synset_names(car_synsets, 'car')
    write_synset_names(hh_obj_synsets, 'hhobj')
    write_synset_names(car_hh, 'carhh')
    write_synset_names(mug_synsets, 'mug')

    write_names(train_names, 'train')
    write_names(val_names, 'val')
    write_names(test_names, 'test')
    for i, cat_name in enumerate(cat_names):
        write_names(cat_name, 'cat_%d' % i)
        cat_len = 3*len(cat_name)/4
        write_names(cat_name[cat_len:], 'cat_%d_0' % i)
        write_names(cat_name[:cat_len], 'cat_%d_1' % i)

    for i in range(10):
        write_names([cat_names[20][i]], 'chair_ft_%d'% i)
    #-4 is interesting asymmetric chair

def split_double_mugs():
    write_names(list(map(str, list(range(100)))), 'double_train')
    write_names(list(map(str, list(range(100, 107)))), 'double_val')
    write_names(list(map(str, list(range(100, 107)))), 'double_test')

def split_mix4():
    write_names(list(map(str, list(range(300)))), 'double_train')
    write_names(list(map(str, list(range(300, 332)))), 'double_val')
    write_names(list(map(str, list(range(300, 332)))), 'double_test')

    
def split_4obj():
    names = listdir('/home/ricson/data/4obj')
    #write_names(list(map(str, list(range(300)))), '4_train')
    #write_names(list(map(str, list(range(300, 332)))), '4_val')
    #write_names(list(map(str, list(range(150)))), '4_test')
    write_names(names, '4_test')

def split_multi():
    write_names(list(map(str, list(range(400)))), 'multi_train')
    write_names(list(map(str, list(range(400, 450)))), 'multi_val')
    write_names(list(map(str, list(range(450, 500)))), 'multi_test')

def split_arith():
    write_names(list(map(str, list(range(40)))), 'arith_train')
    write_names(list(map(str, list(range(40)))), 'arith_val')
    write_names(list(map(str, list(range(40)))), 'arith_test')

def split_house():
    write_names(list(map(str, list(range(0, 300)))), 'house_train')
    write_names(list(map(str, list(range(300, 350)))), 'house_val')
    write_names(list(map(str, list(range(350, 450)))), 'house_test')
    
if __name__ == '__main__':
    #split_by_category()
    #split_double_mugs()
    #split_mix4()
    #split_4obj()
    #split_multi()
    #split_arith()
    split_house()
