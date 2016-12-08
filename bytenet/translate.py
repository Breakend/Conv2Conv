# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
from data import CornellDataFeeder


__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 4
latent_dim = 600   # hidden layer dimension
num_blocks = 3     # dilated blocks

#
# inputs
#

# ComTrans parallel corpus input tensor ( with QueueRunner )
data = CornellDataFeeder(batch_size=batch_size, load=False)

# place holders
x = tf.placeholder(dtype=tf.sg_intx, shape=(batch_size, data.max_len))
y_src = tf.placeholder(dtype=tf.sg_intx, shape=(batch_size, data.max_len))

# make embedding matrix for source and target
emb_x = tf.sg_emb(name='emb_x', voca_size=data.voca_size, dim=latent_dim)
emb_y = tf.sg_emb(name='emb_y', voca_size=data.voca_size, dim=latent_dim)


# residual block
@tf.sg_sugar_func
def sg_res_block(tensor, opt):
    # default rate
    opt += tf.sg_opt(size=3, rate=1, causal=False)

    # input dimension
    in_dim = tensor.get_shape().as_list()[-1]

    # reduce dimension
    input_ = (tensor
              .sg_bypass(act='relu', bn=(not opt.causal), ln=opt.causal)
              .sg_conv1d(size=1, dim=in_dim/2, act='relu', bn=(not opt.causal), ln=opt.causal))

    # 1xk conv dilated
    out = input_.sg_aconv1d(size=opt.size, rate=opt.rate, causal=opt.causal, act='relu', bn=(not opt.causal), ln=opt.causal)

    # dimension recover and residual connection
    out = out.sg_conv1d(size=1, dim=in_dim) + tensor

    return out

# inject residual multiplicative block
tf.sg_inject_func(sg_res_block)


#
# encode graph ( atrous convolution )
#

# embed table lookup
enc = x.sg_lookup(emb=emb_x)
# loop dilated conv block
for i in range(num_blocks):
    enc = (enc
           .sg_res_block(size=5, rate=1)
           .sg_res_block(size=5, rate=2)
           .sg_res_block(size=5, rate=4)
           .sg_res_block(size=5, rate=8)
           .sg_res_block(size=5, rate=16))

# concat merge target source
enc = enc.sg_concat(target=y_src.sg_lookup(emb=emb_y))


#
# decode graph ( causal convolution )
#

# loop dilated causal conv block
dec = enc
for i in range(num_blocks):
    dec = (dec
           .sg_res_block(size=3, rate=1, causal=True)
           .sg_res_block(size=3, rate=2, causal=True)
           .sg_res_block(size=3, rate=4, causal=True)
           .sg_res_block(size=3, rate=8, causal=True)
           .sg_res_block(size=3, rate=16, causal=True))

# final fully convolution layer for softmax
label = dec = dec.sg_conv1d(size=1, dim=data.voca_size)

# greedy search policy

label = dec.sg_argmax()
#label = tf.cast(tf.nn.softmax(tf.cast(label, tf.float64)), tf.float32)


#
# translate
#

# smaple french sentences for source language
sources = [
    u"Who are you?",
    u"What is the meaning of life?",
    u"How are you?",
    u"What is your job?",
    u"What's your name?",
    u"When were you born?",
    u"Where are you from?",
    u"Are you a man or a woman?",
    u"Why are we here?",
    u"okay, bye!",
    u"Did you change your hair?",
    u"Where did he go? He was just here."
]

# to batch form
batches = data.to_batches(sources)

# run graph for translating
with tf.Session() as sess:
    # init session vars
    tf.sg_init(sess)

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))

    for sources in batches:

        # initialize character sequence
        pred_prev = np.zeros((batch_size, data.max_len)).astype(np.int32)
        pred = np.zeros((batch_size, data.max_len)).astype(np.int32)

        # generate output sequence
        for i in range(data.max_len):
            # predict character
            #import pdb; pdb.set_trace()
            out = sess.run(label, {x: sources, y_src: pred_prev})
	    #characters = [np.random.choice(np.arange(139), p=x) for x in out[:,i]]
            #characters = characters
            characters = out[:,i]
            # update character sequence
            if i < data.max_len - 1:
                for j in range(len(characters)):
                    pred_prev[j, i + 1] = int(characters[j])
            pred[:, i] = characters[:]

	# print result
	print '\nsources : --------------'
	data.print_index(sources)
	print '\ntargets : --------------'
	data.print_index(pred)
