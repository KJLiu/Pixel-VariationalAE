import numpy

import os
import urllib
import gzip
import cPickle as pickle

N_CHANNELS = 3

def cifar10_generator_drop_train(data, batch_size, all_of_it, remove_digit):
    images, targets = data
    targets = targets.astype('int32')

    # remove digit :
    positions_without = numpy.in1d(targets, numpy.setdiff1d(range(10), remove_digit))
    
    images = images[positions_without, :]
    images = images.reshape(-1, N_CHANNELS, 32, 32)

    nonlocal_data = {'images': images, 'targets': targets}
    
    def get_epoch():
        images = nonlocal_data['images']

        rng_state = numpy.random.get_state()
        shuffle_inds = numpy.random.permutation(images.shape[0])
        images = images[shuffle_inds, :]

        if not all_of_it:
            # get shape, and replicate last few examples so total # is multiple of batch_size:
            num_examples = images.shape[0]
            number_to_replicate = batch_size - num_examples%batch_size

            # pick random examples:
            rep_positions = numpy.random.permutation(num_examples)[:number_to_replicate]

            # append:
            images = numpy.concatenate((images, images[rep_positions, :]), axis = 0)

            assert(images.shape[0]%batch_size == 0)

            image_batches = images.reshape(-1, batch_size, N_CHANNELS, 32, 32)
            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]),)
        else:
            yield numpy.copy(images)

    return get_epoch

def cifar10_generator_drop_dev(data, batch_size, all_of_it, remove_digit):
    images, targets = data
    targets = targets.astype('int32')

    # remove digit :
    positions_without = numpy.in1d(targets, numpy.setdiff1d(range(10), remove_digit))
    
    images = images[positions_without, :]
    images = images.reshape(-1, N_CHANNELS, 32, 32)

    nonlocal_data = {'images': images, 'targets': targets}
    
    def get_epoch():

        images = nonlocal_data['images']

        rng_state = numpy.random.get_state()
        shuffle_inds = numpy.random.permutation(images.shape[0])
        images = images[shuffle_inds, :]

        if not all_of_it:
            # get shape, and replicate last few examples so total # is multiple of batch_size:
            num_examples = images.shape[0]
            number_to_replicate = batch_size - num_examples%batch_size

            # pick random examples:
            rep_positions = numpy.random.permutation(num_examples)[:number_to_replicate]

            # append:
            images = numpy.concatenate((images, images[rep_positions, :]), axis = 0)

            assert(images.shape[0]%batch_size == 0)

            image_batches = images.reshape(-1, batch_size, N_CHANNELS, 32, 32)
            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]),)
        else:
            yield numpy.copy(images)

    return get_epoch


def cifar10_generator_only(data, batch_size, all_of_it, remove_digit):
    images, targets = data

    # remove digit :
    positions_with = numpy.in1d(targets, remove_digit)
    images = images[positions_with, :]
    images = images.reshape(-1, N_CHANNELS, 32, 32)
    
    nonlocal_data = {'images': images, 'targets': targets}

    def get_epoch():

        images = nonlocal_data['images']

        rng_state = numpy.random.get_state()
        shuffle_inds = numpy.random.permutation(images.shape[0])
        images = images[shuffle_inds, :]

        if not all_of_it:
            # get shape, and replicate last few examples so total # is multiple of batch_size:
            num_examples = images.shape[0]
            number_to_replicate = batch_size - num_examples%batch_size

            # pick random examples:
            rep_positions = numpy.random.permutation(num_examples)[:number_to_replicate]

            # append:
            images = numpy.concatenate((images, images[rep_positions, :]), axis = 0)

            assert(images.shape[0]%batch_size == 0)

            image_batches = images.reshape(-1, batch_size, N_CHANNELS, 32, 32)
            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]),)
        else:
            yield numpy.copy(images)

    return get_epoch

def cifar10_generator_test(data, batch_size, all_of_it, remove_digit):
    images, targets = data

    # remove digit :
    positions_without = numpy.in1d(targets, numpy.setdiff1d(range(10), remove_digit))
    
    images = images[positions_without, :]
    images = images.reshape(-1, N_CHANNELS, 32, 32)

    nonlocal_data = {'images': images, 'targets': targets}
    
    def get_epoch():

        images = nonlocal_data['images']

        rng_state = numpy.random.get_state()
        shuffle_inds = numpy.random.permutation(images.shape[0])
        images = images[shuffle_inds, :]

        if not all_of_it:
            # get shape, and replicate last few examples so total # is multiple of batch_size:
            num_examples = images.shape[0]
            number_to_replicate = batch_size - num_examples%batch_size

            # pick random examples:
            rep_positions = numpy.random.permutation(num_examples)[:number_to_replicate]

            # append:
            images = numpy.concatenate((images, images[rep_positions, :]), axis = 0)
            image_batches = images.reshape(-1, batch_size, N_CHANNELS, 32, 32)
            for i in xrange(len(image_batches)):
                yield (numpy.copy(image_batches[i]),)
        else:
            yield numpy.copy(images)

    return get_epoch


import os
import numpy as np

CIFAR_DIR = '/u/ahmedfar/Projects/dist_shift/error-detection/Vision/'

def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo)   #, encoding='latin1')
    fo.close()
    return dict

def load_data(dataset):
    xs = []
    ys = []
    if dataset == 'CIFAR-10':
        for j in range(5):
            d = unpickle(CIFAR_DIR + 'cifar-10-batches-py/data_batch_'+str(j+1))
            x = d['data']
            y = d['labels']
            xs.append(x)
            ys.append(y)

        d = unpickle(CIFAR_DIR + 'cifar-10-batches-py/test_batch')
        xs.append(d['data'])
        ys.append(d['labels'])
    if dataset == 'CIFAR-100':
        d = unpickle(CIFAR_DIR + 'cifar-100-python/train')
        x = d['data']
        y = d['fine_labels']
        xs.append(x)
        ys.append(y)

        d = unpickle(CIFAR_DIR + 'cifar-100-python/test')
        xs.append(d['data'])
        ys.append(d['fine_labels'])

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    #pixel_mean = np.mean(x[0:50000],axis=0)
    #x -= pixel_mean

    ## create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_train_flip = X_train[:,:,:,::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train,X_train_flip),axis=0)
    Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict(
        X_train=X_train.astype('int32'),
        Y_train=Y_train.astype('int32'),
        X_test=X_test.astype('int32'),
        Y_test=Y_test.astype('int32'),)

def load(batch_size, test_batch_size, remove_digit):
    cifar10_data = load_data('CIFAR-10')
    trX = cifar10_data['X_train']
    trY = cifar10_data['Y_train']
    teX = cifar10_data['X_test']
    teY = cifar10_data['Y_test']

    shuffle_inds = np.random.permutation(trX.shape[0])
    trX = trX[shuffle_inds]
    trY = trY[shuffle_inds]

    valX = trX[int(0.8*trX.shape[0]):]
    valY = trY[int(0.8*trX.shape[0]):]
    trY  = trY[:int(0.8*trX.shape[0])]
    trX  = trX[:int(0.8*trX.shape[0])]

    train_data = (trX, trY)
    dev_data = (valX, valY)
    test_data = (teX, teY)

    return (
        cifar10_generator_drop_train(train_data, batch_size, False, remove_digit),
        cifar10_generator_drop_dev(dev_data, batch_size, False, remove_digit),
        cifar10_generator_only(test_data, test_batch_size, True, remove_digit),
        cifar10_generator_test(test_data, test_batch_size, True, remove_digit), 
    )


