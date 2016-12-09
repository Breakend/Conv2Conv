# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
from data2 import CornellDataFeeder


__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 16    # batch size
latent_dim = 400   # hidden layer dimension
gc_latent_dim = 400 # dimension of conditional embedding
num_blocks = 3     # dilated blocks

#
# inputs
#

# ComTrans parallel corpus input tensor ( with QueueRunner )
data = CornellDataFeeder(batch_size=batch_size)

# source, target sentence
x, y, conditionals_x, conditionals_y = data.source, data.target, data.src_cond, data.tgt_cond

voca_size = data.voca_size

conditional_size = data.cond_size # this is the number of cardinal conditionals, for example if 377 is the highest speaker id

# TODO: this is for conditioning on speaker also
emb_conditional = tf.sg_emb(name='emb_cond', voca_size=conditional_size, dim=gc_latent_dim)
# make embedding matrix for source and target
emb_x = tf.sg_emb(name='emb_x', voca_size=voca_size, dim=latent_dim)
emb_y = tf.sg_emb(name='emb_y', voca_size=voca_size, dim=latent_dim)

# shift target for training source
y_src = tf.concat(1, [tf.zeros((batch_size, 1), tf.sg_intx), y[:, :-1]])


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

    if opt.conditional is not None:
        out += opt.conditional.sg_conv1d(size=1, stride=1, in_dim=gc_latent_dim, dim=in_dim/2, pad="SAME", act='tanh', bias=False)

    #TODO: add causal gate here

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


in_dim = enc.get_shape().as_list()[-1]
enc = enc.sg_conv1d(size=1, in_dim=in_dim, dim=gc_latent_dim)

# concat merge target source
#enc = enc.sg_concat(target=y_src.sg_lookup(emb=emb_y))


#
# decode graph ( causal convolution )
#

# loop dilated causal conv block
dec = y_src.sg_lookup(emb=emb_y)
for i in range(num_blocks):
    dec = (dec
           .sg_res_block(size=3, rate=1, causal=True, conditional=enc)
           .sg_res_block(size=3, rate=2, causal=True, conditional=enc)
           .sg_res_block(size=3, rate=4, causal=True, conditional=enc)
           .sg_res_block(size=3, rate=8, causal=True, conditional=enc)
           .sg_res_block(size=3, rate=16, causal=True, conditional=enc))

# final fully convolution layer for softmax
dec = dec.sg_conv1d(size=1, dim=data.voca_size)

#import pdb; pdb.set_trace()
#l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if '/b:' not in v.name])
#tf.sg_summary_loss(l2_loss, prefix='l2_loss')
#loss += .00001 * l2_loss
#tf.sg_summary_loss(loss, prefix='total_loss')

# train
label = dec.sg_argmax()


#
# translate
#

# smaple french sentences for source language
sources = [
    u"Can we make this quick?",
    u"Hey, how are you doing?",
    u"What's up?",
    u"My name's Bob? What's yours?",
    u"How are you?",
    u"Where were you born?",
    u"My name is Bob. What's my name?",
    u"Where do you live?",
    u"What's the time?",
    u"What's your favorite color?",
    u"What time is it?",
    u"Hello. How are you?",
    u"It's time for us to go.",
    u"Can you understand me?",
    u"Who?",
    u"What?"
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
