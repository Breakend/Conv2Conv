# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
from cornell_movie_dialogue import CornellMovieData


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

    def __init__(self, batch_size=16, path='data/twitter_train.txt'):

        # load train corpus
        self.corpus_text_path = path
        sources, targets, cond_srcs, cond_tars = self._load_corpus()
        self.batch_size=batch_size

        # to constant tensor
        source = tf.convert_to_tensor(sources)
        target = tf.convert_to_tensor(targets)
        cond_src = tf.convert_to_tensor(cond_srcs)
        cond_tars = tf.convert_to_tensor(cond_tars)

        #import pdb; pdb.set_trace()
        # create queue from constant tensor
        source, target, cond_src, cond_tars = tf.train.slice_input_producer([source, target, cond_src, cond_tars])

        # create batch queue
        batch_queue = tf.train.shuffle_batch([source, target, cond_src, cond_tars], batch_size,
                                             num_threads=32, capacity=batch_size*64,
                                             min_after_dequeue=batch_size*32, name=name)

        # split data
        self.source, self.target, self.src_cond, self.tgt_cond = batch_queue

        # calc total batch count
        self.num_batch = len(sources) // batch_size

        # print info
        tf.sg_info('Train data loaded.(total data=%d, total batch=%d)' % (len(sources), self.num_batch))

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
        self.max_len = 50
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
