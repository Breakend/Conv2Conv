# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
from data2 import CornellDataFeeder
from cornell_movie_dialogue import CornellMovieData
from tqdm import *


__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 16    # batch size
latent_dim = 600   # hidden layer dimension
gc_latent_dim = 600 # dimension of conditional embedding
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
    out = input_.sg_aconv1d(size=opt.size, rate=opt.rate, causal=opt.causal, bn=(not opt.causal), ln=opt.causal)

    if opt.conditional is not None:
        out += opt.conditional.sg_conv1d(size=1, stride=1, in_dim=gc_latent_dim, dim=in_dim/2, pad="SAME", bias=False)
    out = out.sg_tanh()

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



# concat merge target source
dec = enc.sg_concat(target=y_src.sg_lookup(emb=emb_y))

in_dim = enc.get_shape().as_list()[-1]
enc = enc.sg_conv1d(size=1, in_dim=in_dim, dim=gc_latent_dim)


#
# decode graph ( causal convolution )
#

# loop dilated causal conv block
for i in range(num_blocks):
    dec = (dec
           .sg_res_block(size=3, rate=1, causal=True, conditional=enc)
           .sg_res_block(size=3, rate=2, causal=True, conditional=enc)
           .sg_res_block(size=3, rate=4, causal=True, conditional=enc)
           .sg_res_block(size=3, rate=8, causal=True, conditional=enc)
           .sg_res_block(size=3, rate=16, causal=True, conditional=enc))

# final fully convolution layer for softmax
label = dec = dec.sg_conv1d(size=1, dim=data.voca_size)

#import pdb; pdb.set_trace()
#l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if '/b:' not in v.name])
#tf.sg_summary_loss(l2_loss, prefix='l2_loss')
#loss += .00001 * l2_loss
#tf.sg_summary_loss(loss, prefix='total_loss')

# train
label = tf.log(tf.cast(tf.nn.softmax(tf.cast(label, tf.float64)), tf.float32))

k_beams = 10
label = tf.nn.top_k(label, k=k_beams)

#
# translate
#

# smaple french sentences for source language
orig_sources = [
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


line_pairs = CornellMovieData.get_preprepared_line_pairs('processed_sources_conv2conv.txt', 'processed_targets_val_conv2conv.txt')

orig_sources = [q[0] for q in line_pairs]

print("Found %d source lines" % len(orig_sources))


# to batch form
batches = data.to_batches(orig_sources)
beam_predictions = []
B = 5 


def _match_any(l, arr, i):
    for a2 in l:
        sentence_match = True
        for j in range(a2[0][i].shape[-1]):
            # stop at EOS char
            if a2[0][i][j] != arr[0][i][j]:
                sentence_match = False
            if a2[0][i][j] == 1. and arr[0][i][j] == 1.:
                break
    return False

predictions = []

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
        pred_prev = np.zeros((batch_size, data.max_len)).astype(np.int32)
        beam_predictions.append((pred_prev, 0)) 

        # generate output sequence
        for i in tqdm(range(data.max_len)):
            # predict character
            state_set = []
            if i >= data.max_len - 1:
                break
            for beam, value in beam_predictions:
                values, out = sess.run(label, {x: sources, y_src: beam})
                j = 0
                for k in range(k_beams):
                    beam_copy = np.copy(beam)
                    beam_copy[:,i+1] = out[:,i,k]
                    beam_value = value + values[:,i,k]
                    state_set.append((beam_copy, beam_value))
            beam_predictions = []
            best_batch_states = {}
            for batch_num in range(batch_size):
                best_batch_states[batch_num] = []
                state_set.sort(key=lambda x:x[1][batch_num]) 
                j = len(state_set)-1
                while j >= 0 and B > len(best_batch_states[batch_num]):
                    state = state_set[j]
                    if  0 == len(best_batch_states[batch_num]) or not _match_any(best_batch_states[batch_num], state, batch_num):
                        best_batch_states[batch_num].append(state)
		    j -= 1

            for batch_num, values in best_batch_states.iteritems():                
                for k, v in enumerate(values):
                    if k < len(beam_predictions):
                        if not _match_any(beam_predictions, v, batch_num): 
                            beam_predictions[k][0][batch_num] = v[0][batch_num]
                            beam_predictions[k][1][batch_num] = v[1][batch_num]
                    else:
                        beam_predictions.append(v)

        # print result
        print '\nsources : --------------'
        data.print_index(sources)
        print '\ntargets : --------------'
        for i in range(batch_size):
            beam_predictions.sort(key=lambda x: x[1][i])
            print("%s Val: %s" % (data.print_index2(beam_predictions[-1][0][i], i), beam_predictions[-1][1][i]))
            predictions.append(data.print_index2(beam_predictions[-1][0][i]))

with open('predictions.txt', 'w') as output_file:
    for prediction in predictions:
        output_file.write("%s\n" % prediction)
