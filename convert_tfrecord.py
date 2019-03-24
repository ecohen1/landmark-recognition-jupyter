# -*- coding: utf-8 -*-
# !/usr/bin/python

# NOTE: this will take a lot of network bandwidth, so it's preferable to run on an AWS instance

import sys, os, multiprocessing, csv
from urllib import request, error
from PIL import Image
from io import BytesIO
import boto3
import uuid
import pickle
from random import shuffle
import tempfile
import matplotlib.image as mpimg
import tensorflow as tf
import base64

# global vars
tfrecord_batch_size = 500
should_compress = True
n_processes = None # defaults to n_cpus
test_mode = False
dest_dir = "tf_data_test/" if test_mode else "tf_data/"
tfrecord_temp_dir = "tfrecord_temp/"

# S3 setup
s3_resource = boto3.resource('s3')
bucket_name = 'landmark-data-12345'
bucket = s3_resource.Bucket(bucket_name)

# find a list of already-completed tfrecords
tfrecord_filekeys = [ obj.key for obj in list(bucket.objects.filter(Prefix=dest_dir).all()) ]
tfrecord_indices = [ int(tfrecord_filekey.split('/')[1].split('.')[0]) for tfrecord_filekey in tfrecord_filekeys ]
print('found', len(tfrecord_indices), 'pre-existing tfrecord files')
if len(tfrecord_indices) > 0:
    print('example existing tfrecord_index:', tfrecord_indices[0])    

def create_and_upload_tfrecord_from_batch(enumerated_batch):
    index, tfrecord_batch = enumerated_batch
    if index in tfrecord_indices:
        print('already completed tfrecord', index, '- skipping')
    else:
        try:
            examples = []
            for filekey in tfrecord_batch:
                print('downloading', filekey)
                
                try:
                    id_string = filekey.split('/')[-1].split('.')[0]
                    if should_compress:
                        jpeg_string, shape, label = load_jpeg_string_from_filekey(filekey)
                        print('converting', filekey, 'to example')
                        example = create_example_from_jpeg_string(jpeg_string, shape, label, id_string)
                    else:
                        image, label = load_image_from_filekey(filekey)
                        print('converting', filekey, 'to example')
                        example = create_example_from_image_data(image, label, id_string)
                    
                    examples.append(example)
                except Exception as e:
                    print('ERROR:', e)

            tfrecord_filename = save_examples_as_tfrecord(examples, index)
            
            upload_tfrecord(tfrecord_filename)
            
            delete_files(tfrecord_filename)

        except KeyboardInterrupt:
            raise

def delete_files(*args):
    for arg in args:
        if isinstance(arg, str):
            if os.path.exists(arg):
                os.remove(arg)
        elif isinstance(arg, list):
            for subarg in arg:
                if os.path.exists(subarg):
                    os.remove(subarg)

def upload_tfrecord(filename):
    dest_filekey = dest_dir + filename
    bucket.upload_file(filename, dest_filekey)

def save_examples_as_tfrecord(examples, index):
    print('saving as tfrecord')
    filename = tfrecord_temp_dir + str(index) + '.tfrecord'
    writer = tf.python_io.TFRecordWriter(filename)

    for example in examples:
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

    print('saved as', filename)    

    return filename

def hex_to_int_list(hex_string):
    if len(hex_string) != 16:
        print('hex string is not 16 digits long:', hex_string)
        raise
        
    hex_string1 = int(hex_string[:8], 16)
    hex_string2 = int(hex_string[8:], 16)
    return [ hex_string1, hex_string2 ]

def create_example_from_image_data(image_data, label, id_string):
    # convert id_string from hex to int
    id_int_list = hex_to_int_list(id_string)

    # Create a feature
    feature = {
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data.flatten().tobytes()])),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image_data.shape)),
        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=id_int_list)),
    }
    
    # Create an example protocol buffer
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_example_from_jpeg_string(jpeg_string, shape, label, id_string):
    # convert id_string from hex to int
    id_int_list = hex_to_int_list(id_string)

    # Create a feature
    feature = {
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[jpeg_string])),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=shape)),
        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=id_int_list)),
    }
    
    # Create an example protocol buffer
    return tf.train.Example(features=tf.train.Features(feature=feature))

def load_s3_file_indirect(filekey, as_string=False):
    bucket_object = bucket.Object(filekey)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    with open(tmp.name, 'wb') as f:
        bucket_object.download_fileobj(f)
        img = mpimg.imread(tmp.name)
    if as_string:
        with open(tmp.name, 'rb') as download:
            result = (download.read(), img.shape)
    else:
        result = (img, img.shape)
    os.remove(tmp.name)
    return result        

def load_image_from_filekey(filekey):
    image, _ = load_s3_file_indirect(filekey, False)
    label = '-1' if test_mode else filekey.split('/')[2]
    return image, label

def load_jpeg_string_from_filekey(filekey):
    image, shape = load_s3_file_indirect(filekey, True)
    label = '-1' if test_mode else filekey.split('/')[2]
    return image, shape, label

def get_batches_from_filekeys(filekeys):
    print('num filekeys:', len(filekeys))

    n_batches = len(filekeys) // tfrecord_batch_size

    batches = [ filekeys[i*tfrecord_batch_size:i*tfrecord_batch_size+tfrecord_batch_size] for i in range(n_batches) ]
    batches.append(filekeys[n_batches*tfrecord_batch_size:]) # add any remaining keys
    
    print('num batches:', len(batches))
    print('num filekeys across batches:', sum([ len(batch) for batch in batches ]))
    
    return batches

def generate_or_load_filekeys():
    pickle_path = 'pickles/tfrecord_filekeys_test' if test_mode else 'pickles/tfrecord_filekeys'
    if os.path.isfile(pickle_path):
        print('loading data from files')
    
        file = open(pickle_path, 'rb')
        filekeys = pickle.load(file)
        file.close()
    
    else:
        print('data pickles dont exist, generating')
    
        prefix = "data/test/" if test_mode else "data/train/"
        filekeys = [ obj.key for obj in list(bucket.objects.filter(Prefix=prefix).all()) ]
        shuffle(filekeys)

        file = open(pickle_path, 'wb')
        pickle.dump(filekeys, file)
        file.close()

    return filekeys

def run():
    # redo these methods (slash rename tfrecord files) if spot instance closed
    filekeys = generate_or_load_filekeys()
    tfrecord_batches = get_batches_from_filekeys(filekeys)

    print(len(tfrecord_batches), 'batches to generate tfrecords for')

    # add indices to name the tfrecord files
    tfrecord_batches = list(enumerate(tfrecord_batches, 1))

    pool = multiprocessing.Pool(processes=n_processes)
    pool.map(create_and_upload_tfrecord_from_batch, tfrecord_batches)
    pool.close()
    pool.terminate()


if __name__ == '__main__':
    run()
