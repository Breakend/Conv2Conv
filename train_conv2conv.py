"""Training script for the WaveNet network."""

from __future__ import print_function
import argparse
from datetime import datetime
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from cornell_movie_dialogue import CornellMovieData
from model import Conv2Conv
from tqdm import tqdm

BATCH_SIZE = 1
DATA_DIRECTORY = './data'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 500
NUM_STEPS = 4000
LEARNING_RATE = 0.001
WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = None # TODO: this is probably bad
L2_REGULARIZATION_STRENGTH = 0


def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut text samples to this many '
                        'samples.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Disabled by default')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


def main():
    args = get_arguments()

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    # Create coordinator.
    coord = tf.train.Coordinator()

    dataset = CornellMovieData
    dataset.maybe_download_and_extract()
    dialogue_tuples = dataset.get_line_pairs()

    queries = []
    responses = []

    for query, response in dialogue_tuples:
        # TODO: add "STOP" char and then in the generator look for "STOP" char to stop generating
        # TODO: batching
        query = map(lambda x: ord(x), list(query))
        query = np.array(query, dtype='float32')
        query = query.reshape(-1, 1)
        response = map(lambda x: ord(x), list(response))
        response = np.array(response, dtype='float32')
        response = response.reshape(-1, 1)
        queries.append(query)
        responses.append(response)

    # import pdb; pdb.set_trace()
    queries = np.asarray(queries)
    responses = np.asarray(responses)
    # q_pl = tf.placeholder(dtype=tf.float32, shape=None)
    # rq_pl = tf.placeholder(dtype=tf.float32, shape=None)
    # q = tf.PaddingFIFOQueue(len(queries), ['float32'], shapes=[(None, 1)])
    # rq = tf.PaddingFIFOQueue(len(queries), ['float32'], shapes=[(None, 1)])

    # with tf.Session() as sess:
    query_init = tf.placeholder(dtype=tf.float32,
                                    shape=(None, None, 1))
    response_init = tf.placeholder(dtype=tf.float32,
                                    shape=(None, None, 1))
    # input_queries = tf.Variable(query_init, trainable=False)
    # input_responses = tf.Variable(response_init, trainable=False)
    # ...
    # sess.run(input_queries.initializer,
    #        feed_dict={query_init: queries})
    # sess.run(input_responses.initializer,
    #        feed_dict={response_init: responses})
    # query_batch, response_batch = tf.train.batch([query_init, response_init], batch_size=BATCH_SIZE,     dynamic_pad=True)

    # enqueue_op = q.enqueue_many(queries)
    # enqueue_response_op = rq.enqueue_many(responses)
    # sess.run(enqueue_op)
    # sess.run(enqueue_response_op)

    # text_batch, response_batch = tf.train.batch(
    #                                     [queries, responses],
    #                                     batch_size=BATCH_SIZE
    #                                     #,num_threads=1
    #                                     )

    # Create network.
    net = Conv2Conv(
        batch_size=BATCH_SIZE,
        quantization_channels=wavenet_params["quantization_channels"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        dilation_channels=wavenet_params["dilation_channels"],
        skip_channels=wavenet_params["skip_channels"],
        use_biases=wavenet_params["use_biases"],
        dilations=wavenet_params["dilations"])

    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    # import pdb; pdb.set_trace()
    loss = net.loss(query_init, response_init, args.l2_regularization_strength)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, operation_timeout_in_ms=20000))
    init = tf.initialize_all_variables()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(tf.all_variables())

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # reader.start_threads(sess)
    summary_op = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(logdir, graph=tf.get_default_graph())

    try:
        last_saved_step = saved_global_step
        for epoch in range(saved_global_step + 1, args.num_steps):
            step = epoch*len(queries)
            start_time = time.time()
            for i in tqdm(range(len(queries))):
                step = epoch*len(queries)+i
                #TODO: omg batches
                loss_value, _, summary = sess.run([loss, optim, summary_op], feed_dict= {query_init : [queries[i]], response_init: [responses[i]]})
                if step % args.checkpoint_every == 0:
                    save(saver, sess, logdir, step)
                    writer.add_summary(summary, step)
                    last_saved_step = step
            print("fin epoch", step)
            duration = time.time() - start_time
            print('epoch {:d} - loss = {:.3f}, ({:.3f} sec/step)'
                  .format(epoch, loss_value, duration))

            # if step % args.checkpoint_every == 0:
            #     save(saver, sess, logdir, step)
            #     last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
