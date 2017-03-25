# -*- coding: utf-8 -*-
import argparse
import sugartensor as tf
from twitter_data import TwitterDataFeeder
from helper_ops import *

# Note: modified from https://github.com/buriburisuri/ByteNet


parser = argparse.ArgumentParser()
parser.add_argument("corpus")
parser.add_argument("datapath")
parser.add_argument("--num_steps", default=10000)
parser.add_argument("--learning_rate", default=.0001)
# parser.add_argument("valid")
args = parser.parse_args()

# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 16    # batch size
latent_dim = 500   # hidden layer dimension
gc_latent_dim = 500 # dimension of conditional embedding
num_blocks = 3     # dilated blocks
use_conditional_gate = False
use_l2_norm = False
concat_embedding = True
use_mutual_information = False

#
# inputs
#
sess = tf.Session(config=tf.ConfigProto(
    intra_op_parallelism_threads=4))

# ComTrans parallel corpus input tensor ( with QueueRunner )
if args.corpus == "twitter":
    data = TwitterDataFeeder(batch_size=batch_size, path = args.datapath, sess = sess)
else:
    raise Exception("That corpus isn't permitted.")


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
y_src = tf.concat(axis=1, values=[tf.zeros((batch_size, 1), tf.sg_intx), y[:, :-1]])


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
        out1 = out + opt.conditional.sg_conv1d(size=1, stride=1, in_dim=gc_latent_dim, dim=in_dim/2, pad="SAME", bias=False)
        out2 = out + opt.conditional.sg_conv1d(size=1, stride=1, in_dim=gc_latent_dim, dim=in_dim/2, pad="SAME", bias=False)
        out = out1.sg_tanh() * out2.sg_sigmoid()
    else:
        out = out.sg_relu()

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
if concat_embedding:
    dec = enc.sg_concat(target=y_src.sg_lookup(emb=emb_y))
else:
    dec = y_src.sg_lookup(emb=emb_y)

if use_conditional_gate:
    in_dim = enc.get_shape().as_list()[-1]
    enc = enc.sg_conv1d(size=1, in_dim=in_dim, dim=gc_latent_dim, act='relu')
else:
    enc = None

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
dec = dec.sg_conv1d(size=1, dim=data.voca_size)

# cross entropy loss with logit and mask
loss = dec.sg_ce(target=y, mask=True)

# if use_mutual_information:
#    TODO: get GAN stuff and take enc and dec through the terms and then try to classify them

if use_l2_norm:
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if '/b:' not in v.name])
    tf.sg_summary_loss(l2_loss, prefix='l2_loss')
    loss += .00001 * l2_loss
    tf.sg_summary_loss(loss, prefix='total_loss')

# TODO: properly split validation batch, for now just grab the first batch
# print_validation = tf.sg_print(dec)

optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
# gvs = optimizer.compute_gradients(loss)
# capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]



train_op = tf.sg_optim(sess=sess, optim=optimizer)

# trainable = tf.trainable_variables()
# optim = optimizer.minimize(loss, var_list=trainable)

# Set up logging for TensorBoard.
writer = tf.summary.FileWriter('./assets')
writer.add_graph(tf.get_default_graph())
run_metadata = tf.RunMetadata()
summaries = tf.summary.merge_all()

# Set up session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
init = tf.global_variables_initializer()
sess.run(init)

# train
data.launch_data_threads()

optim = tf.sg_optimize.AdamOptimizer(learning_rate=0.00005)

tf.sg_init(sess)
try:
    custom_train(sess=sess, optim=optim, log_interval=30, lr=0.00005, loss=loss,
                ep_size=data.num_batch, max_ep=100, early_stop=False, lr_reset=True)
except KeyboardInterrupt:
    # Introduce a line break after ^C is displayed so save message
    # is on its own line.
    print()
finally:
    data.cleanup_tensorflow_stuff()
