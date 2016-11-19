import fnmatch
import os
import threading

import numpy as np
import tensorflow as tf
from cornell_movie_dialogue import CornellMovieData


def load_generic_text(dataset, directory):
    '''Generator that yields text raw from the directory.'''
    #TODO: yield line pairs from dataset, but turn to list() first
    dataset.maybe_download_and_extract()

    # TODO: make this read it one at a time maybe to conserve RAM? for this dataset should be okay,
    # but for larger ones it definitely won't
    dialogue_tuples = get_line_pairs()

    files = find_files(directory)
    for query, response in dialogue_tuples:
        query = map(lambda x: ord(x), query)
        response = map(lambda x: ord(x), response)
        query = np.array(query, dtype='float32')
        query = query.reshape(-1, 1)
        response = np.array(response, dtype='float32')
        response = response.reshape(-1, 1)
        yield query, response


class ConversationReader(object):
    '''Generic background text reader that preprocesses text files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 text_dir,
                 coord,
                 sample_size=None,
                 queue_size=256):
        self.text_dir = text_dir
        self.coord = coord
        self.sample_size = sample_size
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.response_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 1)])
        self.response_queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 1)])
        self.enqueue_input = self.queue.enqueue([self.sample_placeholder])
        self.enqueue_response = self.response_queue.enqueue([self.response_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        output2 = self.response_queue.dequeue_many(num_elements)
        return output, output2

    def thread_main(self, sess):
        buffer_ = np.array([])
        response_buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_text(CornellMovieData, self.text_dir)
            for text, response in iterator:
                if self.coord.should_stop():
                    self.stop_threads()
                    stop = True
                    break
                if self.sample_size:
                    # Cut samples into fixed size pieces
                    buffer_ = np.append(buffer_, text)
                    while len(buffer_) > self.sample_size:
                        piece = np.reshape(buffer_[:self.sample_size], [-1, 1])
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        buffer_ = buffer_[self.sample_size:]
                    response_buffer_ = np.append(response_buffer_, response)
                    while len(response_buffer_) > self.sample_size:
                        piece = np.reshape(response_buffer_[:self.sample_size], [-1, 1])
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        response_buffer_ = response_buffer_[self.sample_size:]
                    response_buffer_ = np.append(response_buffer_, response)
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: text})
                    sess.run(self.enqueue_response,
                             feed_dict={self.response_placeholder: response})

    def stop_threads():
        for t in self.threads:
            t.stop()

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
