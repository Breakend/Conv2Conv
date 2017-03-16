# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import threading


def get_twitter_line_pairs(path):
    dialogue_tuples = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            conv_lines = [y.replace('<first_speaker>','').replace('<second_speaker>','').strip() for y in line.split('</d>')[0].split('</s>') if y.strip()]
            for i in range(0, len(conv_lines) - 1):
                #TODO: it would be nice to remove the first speaker and second speaker tags in favour of a unique ID. right now it's None, None
                dialogue_tuples.append((conv_lines[i], conv_lines[i+1], 1, 2))
    return dialogue_tuples



class TwitterDataFeeder(object):

    def __init__(self, sess, batch_size=16, path='data/twitter_train.txt'):

        # load train corpus
        self.corpus_text_path = path
        sources, targets, cond_srcs, cond_tars = self._load_corpus()
        self.batch_size=batch_size

        self.sess = sess
        queue_input_data = tf.placeholder(tf.int32, shape=[20, self.max_len])
        queue_input_target = tf.placeholder(tf.int32, shape=[20, self.max_len])

        queue = tf.FIFOQueue(capacity=50, dtypes=[tf.int32, tf.int32], shapes=[[self.max_len], [self.max_len]])

        enqueue_op = queue.enqueue_many([queue_input_data, queue_input_target])
        dequeue_op = queue.dequeue()

        # tensorflow recommendation:
        # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
        data_batch, target_batch = tf.train.batch(dequeue_op, batch_size=batch_size, capacity=40)
        # use this to shuffle batches:
        # data_batch, target_batch = tf.train.shuffle_batch(dequeue_op, batch_size=15, capacity=40, min_after_dequeue=5)
        def enqueue(sess, raw_data, raw_target):
            """ Iterates over our data puts small junks into our queue."""
            raw_data = np.array(raw_data)
            raw_target = np.array(raw_target)
            under = 0
            max = len(raw_data)
            while True:
              upper = under + 20
              if upper <= max:
                  curr_data = raw_data[under:upper]
                  curr_target = raw_target[under:upper]
                  under = upper
              else:
                  rest = upper - max
                  print(raw_data[under:max].shape)
                  print(raw_data[0:rest].shape)
                  curr_data = np.concatenate((raw_data[under:max], raw_data[0:rest]))
                  curr_target = np.concatenate((raw_target[under:max], raw_target[0:rest]))
                  under = rest
              sess.run(enqueue_op, feed_dict={queue_input_data: curr_data, queue_input_target: curr_target})

        self.enqueue_thread = threading.Thread(target=enqueue, args=[sess, sources, targets])

        # split data
        self.source, self.target, self.src_cond, self.tgt_cond = data_batch, target_batch, None, None
        self.queue = queue

        # calc total batch count
        self.num_batch = len(sources) // batch_size

        # print info
        tf.sg_info('Train data loaded.(total data=%d, total batch=%d)' % (len(sources), self.num_batch))

    def launch_data_threads(self):
        self.enqueue_thread.isDaemon()
        self.enqueue_thread.start()

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

    def _make_vocab(self, line_pairs):
        # use the whole corpus to make a vocabulary
        # make character-level parallel corpus

        all_byte = []
        max_char = 0
        for query, reply, char_1, char_2 in line_pairs:
            src = [ord(ch) for ch in query if ch != '\n']  # source language byte stream
            tgt = [ord(ch) for ch in reply if ch != '\n']  # target language byte stream
            char_1 = int(char_1)
            char_2 = int(char_2)
            if char_1 > max_char:
                max_char = char_1
            if char_2 > max_char:
                max_char = char_2
            all_byte.extend(src + tgt)

        # make vocabulary
        self.cond_size = max_char
        self.index2byte = [0, 1] + list(np.unique(all_byte))  # add <EMP>, <EOS> tokens
        self.byte2index = {}
        for i, b in enumerate(self.index2byte):
            self.byte2index[b] = i
        self.voca_size = len(self.index2byte)
        self.max_len = 140
        self.min_len = 2

    def _load_corpus(self, mode='train'):

        line_pairs = get_twitter_line_pairs(self.corpus_text_path)

        self._make_vocab(line_pairs)

        # make character-level parallel corpus
        all_byte, sources, targets, source_chars, target_chars = [], [], [], [], []
        max_char = 0
        for query, reply, char_1, char_2 in line_pairs:
            src = [ord(ch) for ch in query if ch != '\n']  # source language byte stream
            tgt = [ord(ch) for ch in reply if ch != '\n']  # target language byte stream
            char_1 = int(char_1)
            char_2 = int(char_2)
            sources.append(src)
            targets.append(tgt)
            source_chars.append(char_1)
            target_chars.append(char_2)
            if char_1 > max_char:
                max_char = char_1
            if char_2 > max_char:
                max_char = char_2
            all_byte.extend(src + tgt)

        # remove short and long sentence
        src, tgt = [], []
        for s, t in zip(sources, targets):
            if 2 <= len(s) < self.max_len and 2 <= len(t) < self.max_len:
                src.append(s)
                tgt.append(t)

        # convert to index list and add <EOS> to end of sentence
        for i in range(len(src)):
            src[i] = [self.byte2index[ch] for ch in src[i]] + [1]
            tgt[i] = [self.byte2index[ch] for ch in tgt[i]] + [1]

        # zero-padding
        for i in range(len(tgt)):
            src[i] += [0] * (self.max_len - len(src[i]))
            tgt[i] += [0] * (self.max_len - len(tgt[i]))

        # swap source and target : french -> english
        return src, tgt, source_chars, target_chars

    def to_batch(self, sentences):

        # convert to index list and add <EOS> to end of sentence
        for i in range(len(sentences)):
            sentences[i] = [self.byte2index[ord(ch)] for ch in sentences[i]] + [1]

        # zero-padding
        for i in range(len(sentences)):
            sentences[i] += [0] * (self.max_len - len(sentences[i]))

        return sentences

    def to_batches(self, sentences):
        batches = []

        # convert to index list and add <EOS> to end of sentence
        for i in range(len(sentences)):
            sentences[i] = [self.byte2index[ord(ch)] for ch in sentences[i] if ch != '\n'] + [1]

        # zero-padding
        for i in range(len(sentences)):
            sentences[i] += [0] * (self.max_len - len(sentences[i]))

        batch = []
        num_in_batch = 0
        for sentence in sentences:
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
            batch.append(sentence)

        if batch:
            while len(batch) != self.batch_size:
                batch.append([0]*self.max_len)
            batches.append(batch)

        return batches

    def cleanup_tensorflow_stuff():
        sess.run(self.queue.close(cancel_pending_enqueues=True))
        self.coord.request_stop()
        self.coord.join(threads)
        self.sess.close()

    def print_index2(self, index, i=None):
        str_ = ''
        for ch in index:
            if ch > 1:
                str_ += unichr(self.index2byte[ch])
            elif ch == 1:  # <EOS>
                break
        if i:
            return '[%d]' % i + str_
        else:
            return str_

    def print_index(self, indices):
        for i, index in enumerate(indices):
            str_ = ''
            for ch in index:
                if ch > 1:
                    str_ += unichr(self.index2byte[ch])
                elif ch == 1:  # <EOS>
                    break
            print '[%d]' % i + str_
