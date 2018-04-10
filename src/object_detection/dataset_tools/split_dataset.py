import logging
import os
import random
from shutil import copy2, Error, copystat

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw Liana dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output split dataset.')
flags.DEFINE_float('ratio', 0.75, 'Train set ratio to the size of the dataset.')
FLAGS = flags.FLAGS


def copytree(src, names, type, dst, symlinks=False):
    os.makedirs(dst)
    errors = []
    for name in names:
        name += type
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if symlinks and os.path.islink(srcname):
                linkto = os.readlink(srcname)
                os.symlink(linkto, dstname)
            elif os.path.isdir(srcname):
                copytree(srcname, dstname, symlinks)
            else:
                copy2(srcname, dstname)
            # XXX What about devices, sockets etc.?
        except OSError as why:
            errors.append((srcname, dstname, str(why)))
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except Error as err:
            errors.extend(err.args[0])
    try:
        copystat(src, dst)
    except OSError as why:
        # can't copy file access times on Windows
        if why.winerror is None:
            errors.extend((src, dst, str(why)))
    if errors:
        raise Error(errors)


def main(_):
    data_dir = FLAGS.data_dir
    output_dir = FLAGS.output_dir
    ratio = FLAGS.ratio

    output_dir_train_images = os.path.join(output_dir, 'train', 'images')
    output_dir_train_groundtruth = os.path.join(output_dir, 'train', 'groundtruth')
    output_dir_eval_images = os.path.join(output_dir, 'eval', 'images')
    output_dir_eval_groundtruth = os.path.join(output_dir, 'eval', 'groundtruth')

    logging.info('Reading from Liana dataset.')
    image_dir = os.path.join(data_dir, 'images')
    gndtruth_dir = os.path.join(data_dir, 'groundtruth')

    examples_list = []
    for name in os.listdir(gndtruth_dir):
        if os.path.isdir(name) or name.find('.xml') < 0:
            continue
        name_parts = name.split('.')
        if name_parts[1] == 'xml':
            examples_list.append(name_parts[0])

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(ratio * num_examples)
    train_examples = examples_list[:num_train]
    eval_examples = examples_list[num_train:]
    logging.info('%d training and %d evaluation examples.',
                 len(train_examples), len(eval_examples))

    copytree(image_dir, train_examples, '.jpg', output_dir_train_images)
    copytree(gndtruth_dir, train_examples, '.xml', output_dir_train_groundtruth)
    copytree(image_dir, eval_examples, '.jpg', output_dir_eval_images)
    copytree(gndtruth_dir, eval_examples, '.xml', output_dir_eval_groundtruth)


if __name__ == '__main__':
    tf.app.run()
