from __future__ import absolute_import
from __future__ import division
import os
import numpy as np
import math
import heapq # for retrieval topK
from multiprocessing import Pool
from multiprocessing import cpu_count
import tensorflow as tf
import argparse
import logging
from time import time
from time import strftime
from time import localtime
from Dataset import Dataset
from saver import GMFSaver

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

_user_input = None
_item_input = None
_labels = None
_batch_size = None
_index = None
_model = None
_sess = None
_dataset = None
_K = None
_DictList = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run MF-BPR.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--model', nargs='?', default='GMF',
                        help='Choose model: GMF')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0,0]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--task', nargs='?', default='',
                        help='Add the task name for launching experiments')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--evaluate', type=int, default=0,
                        help='0: 1 positve sample and 99 negtive samples, 1: global ranking')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    return parser.parse_args()

# data sampling and shuffling

# input: dataset(Mat, List, Rating, Negatives), batch_choice, num_negatives
# output: [_user_input_list, _item_input_list, _labels_list]
def sampling(args, dataset, num_negatives):
    _user_input, _item_input, _labels = [], [], []
    sample_dict = {}
    num_users, num_items =  dataset.trainMatrix.shape
    for (u, i) in dataset.trainMatrix.keys():
        # positive instance
        item_pair = []
        _user_input.append(u)
        item_pair.append(i)
        _labels.append(1)
        # negative instances
        j = np.random.randint(num_items)
        while dataset.trainMatrix.has_key((u, j)):
            j = np.random.randint(num_items)
        item_pair.append(j)
        _item_input.append(item_pair)
    return _user_input, _item_input, _labels

def shuffle(samples, batch_size, dataset = None):
    global _user_input
    global _item_input
    global _labels
    global _batch_size
    global _index
    _user_input, _item_input, _labels = samples
    _batch_size = batch_size
    _index = range(len(_labels))
    np.random.shuffle(_index)
    num_batch = len(_labels) // _batch_size
    pool = Pool(cpu_count())
    res = pool.map(_get_train_batch, range(num_batch))
    pool.close()
    pool.join()
    user_list = [r[0] for r in res]
    item_list = [r[1] for r in res]
    labels_list = [r[2] for r in res]
    return user_list, item_list, labels_list

def _get_train_batch(i):
    user_batch, item_batch, labels_batch = [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input[_index[idx]])
        labels_batch.append(_labels[_index[idx]])
    return np.array(user_batch), np.array(item_batch), np.array(labels_batch)

# prediction model
class GMF:
    def __init__(self, num_users, num_items, args):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.evaluate = args.evaluate

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = "user_input")
            self.item_input = tf.placeholder(tf.int32, shape = [None, None], name = "item_input")
            self.labels = tf.placeholder(tf.float32, shape = [None, 1], name = "labels")  #(b,1)
    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_P', dtype=tf.float32)  #(users, embedding_size)
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_Q', dtype=tf.float32)  #(items, embedding_size)
            #self.h = tf.Variable(tf.ones([self.embedding_size, 1]), name='h', dtype=tf.float32)  #how to initialize it  (embedding_size, 1)
            self.h = tf.Variable(tf.random_uniform([self.embedding_size, 1], minval = -tf.sqrt(3/self.embedding_size),
                                                   maxval = tf.sqrt(3/self.embedding_size)), name = 'h')
            # self.h = tf.constant(1.0, tf.float32, [self.embedding_size, 1], name = "h")
    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1) #(b, embedding_size)
            return self.embedding_p, self.embedding_q,\
                tf.matmul(self.embedding_p*self.embedding_q, self.h)  #(b, embedding_size) * (embedding_size, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.p1, self.q1, self.output = self._create_inference(tf.expand_dims(self.item_input[:, 0], axis = 1))
            self.p2, self.q2, self.output_neg = self._create_inference(tf.expand_dims(self.item_input[:, -1], axis = 1))
            self.result = self.output - self.output_neg
            self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) \
                                + self.lambda_bilinear * tf.reduce_sum(tf.square(self.p1)) \
                                + self.lambda_bilinear * tf.reduce_sum(tf.square(self.p2)) \
                                + self.gamma_bilinear * tf.reduce_sum(tf.square(self.q1))
    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()

# training
def training(model, dataset, args, saver = None): # saver is an object to save pq
    with tf.Session() as sess:
        # pretrain nor not
        sess.run(tf.global_variables_initializer())
        logging.info("initialized")
        print "initialized"

        # initialize for Evaluate
        eval_feed_dicts = init_eval_model(model, dataset)

        # train by epoch
        for epoch_count in range(args.epochs):

            # initialize for training batches

            batch_begin = time()
            samples, sample_dict = sampling(args, dataset, args.num_neg)
            batches = shuffle(samples, args.batch_size)
            batch_time = time() - batch_begin

            train_begin = time()
            training_batch(model, sess, batches)
            train_time = time() - train_begin

            if epoch_count % args.verbose == 0:
                loss_begin = time()
                train_loss = training_loss(model, sess, batches)
                loss_time = time() - loss_begin

                eval_begin = time()
                hr, ndcg, auc = evaluate(model, sess, dataset, eval_feed_dicts)
                eval_time = time() - eval_begin

                logging.info(
                    "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f AUC = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                        epoch_count, batch_time, train_time, hr, ndcg, auc, eval_time, train_loss, loss_time))
                print "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f AUC = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                        epoch_count, batch_time, train_time, hr, ndcg, auc, eval_time, train_loss, loss_time)

        if saver != None:
            saver.save(model, sess)

# input: batch_index (shuffled), model, sess, batches
# do: train the model optimizer
def training_batch(model, sess, batches):
    user_input, item_input, labels = batches
    for i in range(len(labels)):
        feed_dict = {model.user_input: user_input[i][:, None],
                     model.item_input: item_input[i],
                     model.labels: labels[i][:, None]}
        sess.run(model.optimizer, feed_dict)

# input: model, sess, batches
# output: training_loss
def training_loss(model, sess, batches):
    train_loss = 0.0
    num_batch = len(batches[1])
    user_input, item_input, labels = batches
    for i in range(len(labels)):
        feed_dict = {model.user_input: user_input[i][:, None],
                     model.item_input: item_input[i],
                     model.labels: labels[i][:, None]}
        loss = sess.run(model.loss, feed_dict)
        train_loss += loss
        return train_loss / num_batch


def init_eval_model(model, dataset):
    global _dataset
    global _model
    _dataset = dataset
    _model = model

    pool = Pool(cpu_count())
    feed_dicts = pool.map(_evaluate_input, range(_dataset.num_users))
    pool.close()
    pool.join()

    print("already load the evaluate model...")
    return feed_dicts

def _evaluate_input(user):
    # generate items_list
    item_input = []
    test_item = _dataset.testRatings[user][1]
    train_items = _dataset.trainList[user]
    for j in range(_dataset.num_items):
        if j != test_item and j not in train_items:
            item_input.append(j)
    item_input.append(test_item)
    user_input = np.full(len(item_input), user, dtype='int32')[:, None]
    item_input = np.array(item_input)[:,None]
    return user_input, item_input

def evaluate(model, sess, dataset, feed_dicts):
    global _model
    global _K
    global _sess
    global _dataset
    global _feed_dicts
    _dataset = dataset
    _model = model
    _sess = sess
    _K = 100
    _feed_dicts = feed_dicts

    pool = Pool(cpu_count())
    res = pool.map(_eval_by_user, range(_dataset.num_users))
    pool.join()
    pool.close()

    res = np.array(res)
    hr, ndcg, auc = (res.mean(0)).tolist()

    return hr, ndcg, auc

def _eval_by_user(user):

    map_item_score = {}
    user_input, item_input = _feed_dicts[user]
    feed_dict = {model.user_input: user_input, model.item_input: item_input}
    gtItem = _dataset.testRatings[user][1]
    predictions = _sess.run(_model.output, feed_dict)

    for i in xrange(len(item_input)):
        item = item_input[i]
        map_item_score[item] = predictions[i]

    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)

    hr = gtItem in ranklist
    ndcg = _getNDCG(ranklist, gtItem)
    auc = (predictions < predictions[-1]).sum() / len(user_input)
    return (hr, ndcg, auc)

def _getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def init_logging(args):
    regs = eval(args.regs)
    path = "Log/%s_%s/" % (strftime('%Y-%m-%d_%H', localtime()), args.task)
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(filename=path + "%s_%s_log_embed_size%d_reg1%.7f_reg2%.7f%s" % (
        args.dataset, args.model, args.embed_size, regs[0], regs[1],strftime('%Y_%m_%d_%H_%M_%S', localtime())),
                        level=logging.INFO)
    logging.info("begin training %s model ......" % args.model)
    logging.info("dataset:%s  embedding_size:%d"
                 % (args.dataset, args.embed_size))
    logging.info("regs:%.8f, %.8f  learning_rate:%.4f"
                 % (regs[0], regs[1], args.lr))


if __name__ == '__main__':

    # initialize logging
    args = parse_args()
    init_logging(args)

    #initialize dataset
    dataset = Dataset(args.path + args.dataset)

    #initialize models
    model = GMF(dataset.num_users, dataset.num_items, args)
    model.build_graph()

    #start trainging
    saver = GMFSaver()
    saver.setPrefix("./param")
    training(model, dataset, args)



