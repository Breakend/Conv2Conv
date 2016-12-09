# -*- coding: utf-8 -*-
import sugartensor as tf
from data2 import CornellDataFeeder
import numpy as np

__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 1    # batch size
latent_dim = 100   # hidden layer dimension
gc_latent_dim = 100 # dimension of conditional embedding
num_blocks = 2     # dilated blocks


#
# inputs
#

# ComTrans parallel corpus input tensor ( with QueueRunner )
data = CornellDataFeeder(batch_size=batch_size)

# source, target sentence
x, y, conditionals_x, conditionals_y = data.source, data.target, data.src_cond, data.tgt_cond
voca_size = data.voca_size
conditional_size = data.cond_size # this is the number of cardinal conditionals, for example if 377 is the highest speaker id

# make embedding matrix for source and target
emb_x = tf.sg_emb(name='emb_x', voca_size=voca_size, dim=latent_dim)
emb_y = tf.sg_emb(name='emb_y', voca_size=voca_size, dim=latent_dim)
emb_conditional = tf.sg_emb(name='emb_cond', voca_size=conditional_size, dim=gc_latent_dim)

# shift target for training source
y_src = tf.concat(1, [tf.zeros((batch_size, 1), tf.sg_intx), y[:, :-1]])
# seq_len = tf.not_equal(x.sg_sum(dims=2), 0.).sg_int().sg_sum(dims=1)

conditional_enc_x = None #conditionals_x.sg_lookup(emb=emb_conditional)
conditional_enc_y = None #conditionals_y.sg_lookup(emb=emb_conditional)

# residual block
def res_block(tensor, size, rate, dim=latent_dim, condition=None):

    # filter convolution
    conv_filter = tensor.sg_aconv1d(size=size, rate=rate, bn=True)

    if condition is not None:
        # conv_filter = conv_filter.sg_conv1d(size=1, dim=in_dim)
        # import pdb; pdb.set_trace()
        conv_filter += condition.sg_conv1d(size=1, stride=1, in_dim=gc_latent_dim, dim=latent_dim, pad="SAME", bias=False)
    conv_filter = conv_filter.sg_tanh(name="cond_filter_act")


    # gate convolution
    conv_gate = tensor.sg_aconv1d(size=size, rate=rate, bn=True)
    if condition is not None:
        # conv_gate = conv_gate.sg_conv1d(size=1, dim=in_dim)
        conv_gate += condition.sg_conv1d(size=1, stride=1, in_dim=gc_latent_dim, dim=latent_dim, pad="SAME", bias=False)
    conv_gate = conv_gate.sg_sigmoid(name="cond_gate_act")

    # output by gate multiplying
    out = conv_filter * conv_gate

    # final output
    out = out.sg_conv1d(size=1, dim=dim, act='tanh', bn=True)

    # residual and skip output
    return out + tensor, out

# expand dimension
z = x.sg_lookup(emb=emb_x).sg_conv1d(size=1, dim=latent_dim, act='tanh', bn=True)

# dilated conv block loop
skip = 0  # skip connections
for i in range(num_blocks):
    for r in [1, 2, 4, 8]:
        z, s = res_block(z, size=7, rate=r)
        skip += s

# final logit layers
logit = (skip
         .sg_conv1d(size=1, act='tanh', bn=True)
         .sg_conv1d(size=1, dim=gc_latent_dim))

# expand dimension
enc = y_src.sg_lookup(emb=emb_y)
z = enc.sg_conv1d(size=1, dim=latent_dim, act='tanh', bn=True)

# dilated conv block loop
skip = 0  # skip connections
for i in range(num_blocks):
    for r in [1, 2, 4, 8]:
        z, s = res_block(z, size=7, rate=r, condition=logit)
        skip += s

# final logit layers
label = logit = (skip
         .sg_conv1d(size=1, act='tanh', bn=True)
         .sg_conv1d(size=1, dim=data.voca_size))

label.sg_reshape(shape=[batch_size, -1])
label = logit.sg_argmax()


#
# translate
#

# smaple french sentences for source language
sources = [
    u"Can we make this quick? ."
]

# to batch form
sources = data.to_batch(sources)

# run graph for translating
with tf.Session() as sess:
    # init session vars
    tf.sg_init(sess)

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))

    # initialize character sequence
    pred_prev = np.zeros((batch_size, data.max_len)).astype(np.int32)
    pred = np.zeros((batch_size, data.max_len)).astype(np.int32)

    # generate output sequence
    for i in range(data.max_len):
        # predict character
        # import pdb; pdb.set_trace()
        out = sess.run(label, {x: sources, y_src: pred_prev})
        # update character sequence
        if i < data.max_len - 1:
            pred_prev[:, i + 1] = out[:, i]
        pred[:, i] = out[:, i]

# print result
print '\nsources : --------------'
data.print_index(sources)
print '\ntargets : --------------'
data.print_index(pred)
