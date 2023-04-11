import tensorflow as tf
import argparse


def parseArgs():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--trainBatchSize', type=int, default=1028)
    argparser.add_argument('--testBatchSize', type=int, default=256)
    argparser.add_argument('--trainDataPath', type=str, default="data/census/train_data.csv")
    argparser.add_argument('--testDataPath', type=str, default="data/census/test_data.csv")
    argparser.add_argument('--trainRecordPath', type=str, default="data/census/tfrecords/train.tfrecords")
    argparser.add_argument('--testRecordPath', type=str, default="data/census/tfrecords/test.tfrecords")
    argparser.add_argument('--nTasks', type=int, default=2)
    argparser.add_argument('--taskWeights', type=str, default="1,1")
    argparser.add_argument('--nExperts', type=int, default=3)
    argparser.add_argument('--inputDim', type=int, default=499)
    argparser.add_argument('--expertDim', type=int, default=256)
    argparser.add_argument('--hiddenDim', type=int, default=128)
    argparser.add_argument('--printSteps', type=int, default=10)
    argparser.add_argument('--prepareRecords', type=int, default=1)

    return argparser.parse_args()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, tf.Tensor): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def buildSample(fea, matrital, income):
    data = {
        'fea': _bytes_feature(tf.io.serialize_tensor(fea)),
        'marital': _int64_feature(matrital),
        'income': _int64_feature(income)
    }
    example = tf.train.Example(features=tf.train.Features(feature=data))

    return example


def parseSample(sample):
    # use the same structure as `buildSample`
    data = {
        'fea':tf.io.FixedLenFeature([], tf.string),
        'marital' : tf.io.FixedLenFeature([], tf.int64),
        'income': tf.io.FixedLenFeature([], tf.int64),
    }

    sample = tf.io.parse_single_example(sample, data)

    fea = tf.io.parse_tensor(sample["fea"], out_type=tf.float32)
    marital = sample['marital']
    income = sample['income']

    return fea, marital, income


def parseLine(line):
    fields = line.strip().split(",")
    return int(fields[0]), int(fields[1]), [float(x) for x in fields[2:]]


def prepareDataset(args):
    print("preprare dataset")
    # save as tfrecord to disk, or use `tf.data.Dataset.from_tensor_slices` in memory
    with tf.io.TFRecordWriter(args.trainRecordPath) as writer:
        with open(args.trainDataPath, "r") as f:
            for line in f.readlines():
                marital, income, fea = parseLine(line)
                example = buildSample(fea, marital, income)
                writer.write(example.SerializeToString())

    with tf.io.TFRecordWriter(args.testRecordPath) as writer:
        with open(args.testDataPath, "r") as f:
            for line in f.readlines():
                marital, income, fea = parseLine(line)
                example = buildSample(fea, marital, income)
                writer.write(example.SerializeToString())