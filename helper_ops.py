
import tqdm
import sugartensor as tf

def validation_op(label, sess, orig_sources):
    # to batch form
    batches = data.to_batches(orig_sources)
    beam_predictions = []
    B = 5

    def _sent_length(sent_array):
        count = 0
        for num in sent_array:
            if num == 1:
                return count
            else:
                count += 1
        return count


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
    # TODO: this op should run validation on the network like in custom_net_eval with sample outputs if a flag is set
    for sources in batches:
        # initialize character sequence
        pred_prev = np.zeros((batch_size, data.max_len)).astype(np.int32)
        pred = np.zeros((batch_size, data.max_len)).astype(np.int32)
        # generate output sequence
        for i in range(data.max_len):
            # predict character
            out = sess.run(label, {x: sources, y_src: pred_prev})
            # update character sequence
            if i < data.max_len - 1:
                pred_prev[:, i + 1] = out[:, i]
            pred[:, i] = out[:, i]

        # print result
        print '\nsources : --------------'
        data.print_index(sources)
        print '\ntargets : --------------'
        for i in range(batch_size):
            prediction = data.print_index2(pred[i])
            print("[%d] %s" %(i, prediction))
            predictions.append(prediction)


    with open('predictions.txt', 'w') as output_file:
    for prediction in predictions:
        output_file.write("%s\n" % prediction)



def custom_train(**kwargs):
    r"""Trains the model.
    Args:
      **kwargs:
        optim: A name for optimizer. 'MaxProp' (default), 'AdaMax', 'Adam', 'RMSProp' or 'sgd'.
        loss: A 0-D `Tensor` containing the value to minimize.
        lr: A Python Scalar (optional). Learning rate. Default is .001.
        beta1: A Python Scalar (optional). Default is .9.
        beta2: A Python Scalar (optional). Default is .99.
        save_dir: A string. The root path to which checkpoint and log files are saved.
          Default is `asset/train`.
        max_ep: A positive integer. Maximum number of epochs. Default is 1000.
        ep_size: A positive integer. Number of Total batches in an epoch.
          For proper display of log. Default is 1e5.
        save_interval: A Python scalar. The interval of saving checkpoint files.
          By default, for every 600 seconds, a checkpoint file is written.
        log_interval: A Python scalar. The interval of recoding logs.
          By default, for every 60 seconds, logging is executed.
        max_keep: A positive integer. Maximum number of recent checkpoints to keep. Default is 5.
        keep_interval: A Python scalar. How often to keep checkpoints. Default is 1 hour.
        category: Scope name or list to train
        eval_metric: A list of tensors containing the value to evaluate. Default is [].
        tqdm: Boolean. If True (Default), progress bars are shown. If False, a series of loss
            will be shown on the console.
    """
    opt = tf.sg_opt(kwargs)
    assert opt.loss is not None, 'loss is mandatory.'

    # default training options
    opt += tf.sg_opt(optim='MaxProp', lr=0.001, beta1=0.9, beta2=0.99, category='', ep_size=100000)

    # get optimizer
    train_op = sg_optim(opt.loss, optim=opt.optim, lr=0.001,
                        beta1=opt.beta1, beta2=opt.beta2, category=opt.category)

    # for console logging
    loss_ = opt.loss

    # use only first loss when multiple GPU case
    if isinstance(opt.loss, (tuple, list)):
        loss_ = opt.loss[0]

    # define train function
    # noinspection PyUnusedLocal
    @sg_train_func
    def train_func(sess, arg):
        return sess.run([loss_, train_op])[0]

    # run train function
    train_func(**opt)


def custom_train_func(func):
    r""" Decorates a function `func` as sg_train_func.
    Args:
        func: A function to decorate
    """
    @wraps(func)
    def wrapper(**kwargs):
        r""" Manages arguments of `tf.sg_opt`.
        Args:
          **kwargs:
            lr: A Python Scalar (optional). Learning rate. Default is .001.
            save_dir: A string. The root path to which checkpoint and log files are saved.
              Default is `asset/train`.
            max_ep: A positive integer. Maximum number of epochs. Default is 1000.
            ep_size: A positive integer. Number of Total batches in an epoch.
              For proper display of log. Default is 1e5.
            save_interval: A Python scalar. The interval of saving checkpoint files.
              By default, for every 600 seconds, a checkpoint file is written.
            log_interval: A Python scalar. The interval of recoding logs.
              By default, for every 60 seconds, logging is executed.
            max_keep: A positive integer. Maximum number of recent checkpoints to keep. Default is 5.
            keep_interval: A Python scalar. How often to keep checkpoints. Default is 1 hour.
            eval_metric: A list of tensors containing the value to evaluate. Default is [].
            tqdm: Boolean. If True (Default), progress bars are shown. If False, a series of loss
                will be shown on the console.
        """
        opt = tf.sg_opt(kwargs)

        # default training options
        opt += tf.sg_opt(lr=0.001,
                         save_dir='asset/train',
                         max_ep=1000, ep_size=100000,
                         save_interval=600, log_interval=60,
                         eval_metric=[],
                         max_keep=5, keep_interval=1,
                         tqdm=True)

        # training epoch and loss
        epoch, loss = -1, None

        # checkpoint saver
        saver = tf.train.Saver(max_to_keep=opt.max_keep,
                               keep_checkpoint_every_n_hours=opt.keep_interval)

        # add evaluation summary
        for m in opt.eval_metric:
            tf.sg_summary_metric(m)

        # summary writer
        log_dir = opt.save_dir + '/run-%02d%02d-%02d%02d' % tuple(time.localtime(time.time()))[1:5]
        summary_writer = tf.summary.FileWriter(log_dir)

        # console logging function
        def console_log(sess_):
            if epoch >= 0:
                tf.sg_info('\tEpoch[%03d:gs=%d] - loss = %s' %
                           (epoch, sess_.run(tf.sg_global_step()),
                            ('NA' if loss is None else '%8.6f' % loss)))

        # create supervisor
        sv = tf.train.Supervisor(logdir=opt.save_dir,
                                 saver=saver,
                                 save_model_secs=opt.save_interval,
                                 summary_writer=summary_writer,
                                 save_summaries_secs=opt.log_interval,
                                 global_step=tf.sg_global_step(),
                                 local_init_op=tf.sg_phase().assign(True))

        # create session
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            # console logging loop
            if not opt.tqdm:
                sv.loop(opt.log_interval, console_log, args=(sess, ))

            # get start epoch
            _step = sess.run(tf.sg_global_step())
            ep = _step // opt.ep_size

            # check if already finished
            if ep <= opt.max_ep:

                # logging
                tf.sg_info('Training started from epoch[%03d]-step[%d].' % (ep, _step))

                # epoch loop
                for ep in range(ep, opt.max_ep + 1):

                    # update epoch info
                    start_step = sess.run(tf.sg_global_step()) % opt.ep_size
                    epoch = ep

                    # create progressbar iterator
                    if opt.tqdm:
                        iterator = tqdm(range(start_step, opt.ep_size), total=opt.ep_size, initial=start_step,
                                        desc='train', ncols=70, unit='b', leave=False)
                    else:
                        iterator = range(start_step, opt.ep_size)

                    # batch loop
                    for _ in iterator:

                        # exit loop
                        if sv.should_stop():
                            break

                        # TODO: get perplexity and samples

                        # call train function
                        batch_loss = func(sess, opt)

                        # loss history update
                        if batch_loss is not None and \
                                not np.isnan(batch_loss.all()) and not np.isinf(batch_loss.all()):
                            if loss is None:
                                loss = np.mean(batch_loss)
                            else:
                                loss = loss * 0.9 + np.mean(batch_loss) * 0.1

                    # log epoch information
                    console_log(sess)

                # save last version
                saver.save(sess, opt.save_dir + '/model.ckpt', global_step=sess.run(tf.sg_global_step()))

                # logging
                tf.sg_info('Training finished at epoch[%d]-step[%d].' % (ep, sess.run(tf.sg_global_step())))
            else:
                tf.sg_info('Training already finished at epoch[%d]-step[%d].' %
                           (ep - 1, sess.run(tf.sg_global_step())))

    return wrapper
