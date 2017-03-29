
from tqdm import tqdm
import custom_sugartensor as tf
from functools import wraps
import time
import numpy as np


def validation_op(label, loss, sess, batches):
    """batches : sentence batches to be used for validation"""
    # to batch form
    # batches = data.to_batches(orig_sources)
    beam_predictions = []
    B = 5
    # TODO: add beam_search here

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

    label = tf.log(tf.cast(tf.nn.softmax(tf.cast(label, tf.float64)), tf.float32))
    label = label.sg_argmax()

    batch_size = len(batches[0])
    max_len = len(batches[0][0])

    predictions = []
    # TODO: this op should run validation on the network like in custom_net_eval with sample outputs if a flag is set
    losses = []
    for sources in batches:
        # initialize character sequence
        pred_prev = np.zeros((batch_size, max_len)).astype(np.int32)
        pred = np.zeros((batch_size, max_len)).astype(np.int32)
        # generate output sequence
        for i in range(max_len):
            # predict character
            out,loss = sess.run([label, loss], {x: sources, y_src: pred_prev})
            losses.append(loss)
            # update character sequence
            if i < max_len - 1:
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

    return np.mean(losses), predictions
