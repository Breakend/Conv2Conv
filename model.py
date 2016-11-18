from utils import *
from layers import *


class Conv2Conv(object):
    """ Using dilated convolutions predict a sequence given a previous sequence
    basically seq2seq with convnets"""

    def __init__(self, batch_size, quantization_channels, filter_width, residual_channels, dilation_channels, skip_channels, use_biases, dilations):
        self.quantization_channels = quantization_channels
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.batch_size = batch_size
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.use_biases = use_biases
        self.dilations = dilations


    def loss(self,
             input_batch,
             l2_regularization_strength=None,
             name='wavenet'):
        '''Creates a WaveNet network and returns the autoencoding loss.
        The variables are all scoped to the given name.
        '''
        with tf.name_scope(name):
            encoded = one_hot(tf.cast(input_batch, tf.int32), self.batch_size, self.quantization_channels)

            network_input = encoded

            raw_output = self._create_network(network_input)

            with tf.name_scope('loss'):
                # Shift original input left by one sample, which means that
                # each output sample has to predict the next input sample.
                shifted = tf.slice(encoded, [0, 1, 0],
                                   [-1, tf.shape(encoded)[1] - 1, -1])
                shifted = tf.pad(shifted, [[0, 0], [0, 1], [0, 0]])

                prediction = tf.reshape(raw_output,
                                        [-1, self.quantization_channels])
                loss = tf.nn.softmax_cross_entropy_with_logits(
                    prediction,
                    tf.reshape(shifted, [-1, self.quantization_channels]))
                reduced_loss = tf.reduce_mean(loss)

                tf.scalar_summary('loss', reduced_loss)

                if l2_regularization_strength is None:
                    return reduced_loss
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if not('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (reduced_loss +
                                  l2_regularization_strength * l2_loss)

                    tf.scalar_summary('l2_loss', l2_loss)
                    tf.scalar_summary('total_loss', total_loss)

                    return total_loss

    def _create_network(self, input_batch):
        with tf.variable_scope('conv2conv'):
            self.causal_layer = causal_layer(input_batch,
                                            self.quantization_channels,
                                            self.filter_width,
                                            self.residual_channels)

            self.dilated_layer = dilated_stack(self.causal_layer,
                                                   self.filter_width,
                                                   self.residual_channels,
                                                   self.dilation_channels,
                                                   self.skip_channels,
                                                   self.use_biases,
                                                   self.dilations)

            self.postprocessing_layer = postprocessing_layer(self.dilated_layer,
                                                            self.skip_channels,
                                                            self.use_biases,
                                                            self.quantization_channels)

            return self.postprocessing_layer
