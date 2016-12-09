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

batch_size = 8
latent_dim = 400   # hidden layer dimension
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

label = tf.cast(tf.nn.softmax(tf.cast(label, tf.float64)), tf.float32)
label = tf.nn.top_k(label, k=5)
#label = label.sg_argmax()


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
beam_predictions = []
k_beams = 2

# run graph for translating
with tf.Session() as sess:
    # init session vars
    tf.sg_init(sess)

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))

    beam_predictions = []
    for sources in batches:

        # initialize character sequence
        for i in range(k_beams):
            pred_prev = np.zeros((batch_size, data.max_len)).astype(np.int32)
            beam_predictions.append((pred_prev, 0))

        # generate output sequence
        for i in range(data.max_len):
            # predict character
            new_beam_predictions = []
            for beam, value in beam_predictions:
                values, out = sess.run(label, {x: sources, y_src: beam})
                
                if i < data.max_len - 1:
                    num_bms = k_beams if i < 1 else 1
                    for k in range(num_bms):
                        beam_copy = np.copy(beam)
                        beam_copy[:,i+1] = out[:,i,k]
                        beam_value = value + values[:,i,k]
                        new_beam_predictions.append((beam_copy, beam_value))
            if len(new_beam_predictions) > 0:
                beam_predictions = new_beam_predictions
                
        
	    #characters = [np.random.choice(np.arange(139), p=x) for x in out[:,i]]
            #characters = characters
            """characters = out[:,i]
            # update character sequence
            if i < data.max_len - 1:
                for j in range(len(characters)):
                    pred_prev[j, i + 1] = int(characters[j])
            pred[:, i] = characters[:]"""

	# print result
	print '\nsources : --------------'
	data.print_index(sources)
	print '\ntargets : --------------'
        for beam,value in beam_predictions:
	    data.print_index(beam)
            for v in value:
                print("val: %d" % v)
