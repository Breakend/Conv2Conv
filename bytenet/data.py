# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np

class CornellDataFeeder(object):

    def __init__(self, batch_size=16, name='train', load=True):

        # load train corpus
        sources, targets = self._load_corpus(mode='train')
        self.batch_size = batch_size

        # to constant tensor
        source = tf.convert_to_tensor(sources)
        target = tf.convert_to_tensor(targets)

        # create queue from constant tensor
        source, target = tf.train.slice_input_producer([source, target])

        # create batch queue
        batch_queue = tf.train.shuffle_batch([source, target], batch_size,
                                             num_threads=32, capacity=batch_size*64,
                                             min_after_dequeue=batch_size*32, name=name)

        # split data
        self.source, self.target = batch_queue

        # calc total batch count
        self.num_batch = len(sources) // batch_size

        # print info
        tf.sg_info('Train data loaded.(total data=%d, total batch=%d)' % (len(sources), self.num_batch))

    def _load_corpus(self, mode='train'):

        # load en-fr parallel corpus
        from cornell_movie_dialogue import CornellMovieData
        CornellMovieData.maybe_download_and_extract()
        als = CornellMovieData.get_line_pairs()

        # make character-level parallel corpus
        all_byte, sources, targets = [], [], []
        for query, reply in als:
            src = [ord(ch) for ch in query]  # source language byte stream
            tgt = [ord(ch) for ch in reply]  # target language byte stream
            sources.append(src)
            targets.append(tgt)
            all_byte.extend(src + tgt)

        # make vocabulary
        self.index2byte = [0, 1] + list(np.unique(all_byte))  # add <EMP>, <EOS> tokens
        self.byte2index = {}
        for i, b in enumerate(self.index2byte):
            self.byte2index[b] = i
        self.voca_size = len(self.index2byte)
        print("Vocab size: %d" % self.voca_size)
        self.max_len = 100

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
        return src, tgt

    def to_batches(self, sentences):
        batches = []

        # convert to index list and add <EOS> to end of sentence
        for i in range(len(sentences)):
            sentences[i] = [self.byte2index[ord(ch)] for ch in sentences[i]] + [1]

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

    def print_index(self, indices):
        for i, index in enumerate(indices):
            str_ = ''
            for ch in index:
                if ch > 1:
                    str_ += unichr(self.index2byte[ch])
                elif ch == 1:  # <EOS>
                    break
            print '[%d]' % i + str_
