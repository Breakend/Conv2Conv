from utils import *
import tensorflow as tf


def causal_layer(input_batch, embedding_size, filter_width, residual_channels):
    with tf.name_scope('causal_layer'):
        var_shape = [filter_width, embedding_size, residual_channels]
        filter_var = create_variable('filter', var_shape)
        return causal_conv(input_batch, filter_var, 1)

def postprocessing_layer(inputs, skip_channels, use_biases, quantization_channels):
    with tf.name_scope('postprocessing'):
        # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
        # postprocess the output.

        w1 = create_variable('postprocess1', [1, skip_channels, skip_channels])
        w2 = create_variable('postprocess2', [1, skip_channels, quantization_channels])

        if use_biases:
            b1 = create_bias_variable('postprocess1_bias', [skip_channels])
            b2 = create_bias_variable('postprocess2_bias', [quantization_channels])

        tf.histogram_summary('postprocess1_weights', w1)
        tf.histogram_summary('postprocess2_weights', w2)
        if use_biases:
            tf.histogram_summary('postprocess1_biases', b1)
            tf.histogram_summary('postprocess2_biases', b2)

        # We skip connections from the outputs of each layer, adding them
        # all up here.
        total = sum(inputs)
        transformed1 = tf.nn.relu(total)
        conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
        if use_biases:
            conv1 = tf.add(conv1, b1)
        transformed2 = tf.nn.relu(conv1)
        conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
        if use_biases:
            conv2 = tf.add(conv2, b2)
        return conv2

def dilated_stack(current_layer, filter_width, residual_channels, dilation_channels, skip_channels, use_biases, dilations):
    outputs = list()
    with tf.name_scope('dilated_stack'):
        for layer_index, dilation in enumerate(dilations):
            with tf.name_scope('layer{}'.format(layer_index)):
                output, current_layer = _create_dilation_layer(
                    current_layer, layer_index, dilation,
                    filter_width, dilation_channels, residual_channels,
                    use_biases, skip_channels)
                outputs.append(output)
    return outputs

def _create_dilation_layer(input_batch, layer_index, dilation, filter_width, dilation_channels, residual_channels, use_biases, skip_channels):
    '''Creates a single causal dilated convolution layer.

    The layer contains a gated filter that connects to dense output
    and to a skip connection:

           |-> [gate]   -|        |-> 1x1 conv -> skip output
           |             |-> (*) -|
    input -|-> [filter] -|        |-> 1x1 conv -|
           |                                    |-> (+) -> dense output
           |------------------------------------|

    Where `[gate]` and `[filter]` are causal convolutions with a
    non-linear activation at the output.
    '''
    filter_weight_size = [filter_width, residual_channels, dilation_channels]
    weights_filter = create_variable('filter', filter_weight_size)
    weights_gate = create_variable('gate', filter_weight_size)

    conv_filter = causal_conv(input_batch, weights_filter, dilation)
    conv_gate = causal_conv(input_batch, weights_gate, dilation)

    if use_biases:
        filter_bias = create_bias_variable('filter_bias', [dilation_channels])
        gate_bias = create_bias_variable('gate_bias', [dilation_channels])
        conv_filter = tf.add(conv_filter, filter_bias)
        conv_gate = tf.add(conv_gate, gate_bias)

    out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

    # The 1x1 conv to produce the residual output
    weights_dense = create_variable('dense', [1, dilation_channels, residual_channels])
    transformed = tf.nn.conv1d(out, weights_dense, stride=1, padding="SAME", name="dense")

    # The 1x1 conv to produce the skip output
    weights_skip = create_variable('skip', [1, dilation_channels, skip_channels])
    skip_contribution = tf.nn.conv1d(out, weights_skip, stride=1, padding="SAME", name="skip")

    if use_biases:
        dense_bias = create_bias_variable('dense_bias', [residual_channels])
        skip_bias = create_bias_variable('skip_bias',[skip_channels])
        transformed = transformed + dense_bias
        skip_contribution = skip_contribution + skip_bias

    layer = 'layer{}'.format(layer_index)
    tf.histogram_summary(layer + '_filter', weights_filter)
    tf.histogram_summary(layer + '_gate', weights_gate)
    tf.histogram_summary(layer + '_dense', weights_dense)
    tf.histogram_summary(layer + '_skip', weights_skip)
    if use_biases:
        tf.histogram_summary(layer + '_biases_filter', filter_bias)
        tf.histogram_summary(layer + '_biases_gate', gate_bias)
        tf.histogram_summary(layer + '_biases_dense', dense_bias)
        tf.histogram_summary(layer + '_biases_skip', skip_bias)

    return skip_contribution, input_batch + transformed
