import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops
import numpy as np
from biosppy.signals.tools import get_filter


def upsample_tile(audios, scale):
    _, num_samples, channels = audios.shape.as_list()
    upsampled = tf.tile(audios, [1, 1, scale])
    upsampled = tf.reshape(upsampled, [-1, num_samples * scale, channels])
    return upsampled


def upsample_fir(audios, scale, order=256):
    scale = int(scale)
    if scale == 1:
        return audios
    sample_rate = 16000.  # It is later normalizer, so can be any.
    with tf.variable_scope('upscale_{}'.format(scale)):
        f = get_filter('FIR', 'lowpass', order=order, frequency=sample_rate // scale // 2, sampling_rate=sample_rate)
        filter_b = tf.constant(f['b'][:, np.newaxis, np.newaxis], dtype=tf.float32,
                               name='up_fir_{}_{}'.format(scale, order))

        def _apply_fir(audio):
            exp_audio = tf.transpose(audio)
            exp_audio = tf.expand_dims(exp_audio, axis=-1)
            conv_audio = tf.nn.convolution(exp_audio, filter_b, 'SAME')
            conv_audio = tf.squeeze(conv_audio, axis=-1)
            return tf.transpose(conv_audio)

        # Rework to upsampling with zeros.
        audios_up = upsample_tile(audios, scale)
        up_op = tf.map_fn(_apply_fir, audios_up)
        return up_op


def downsample_fir(audios, scale, order=256):
    scale = int(scale)
    if scale == 1:
        return audios
    sample_rate = 16000.  # It is later normalizer, so can be any.
    with tf.variable_scope('upscale_{}'.format(scale)):
        f = get_filter('FIR', 'lowpass', order=order, frequency=sample_rate // scale // 2, sampling_rate=sample_rate)
        filter_b = tf.constant(f['b'][:, np.newaxis, np.newaxis], dtype=tf.float32,
                               name='up_fir_{}_{}'.format(scale, order))

        def _apply_fir(audio):
            exp_audio = tf.transpose(audio)
            exp_audio = tf.expand_dims(exp_audio, axis=-1)
            conv_audio = tf.nn.convolution(exp_audio, filter_b, 'SAME')
            conv_audio = tf.squeeze(conv_audio, axis=-1)
            return tf.transpose(conv_audio)

        # Rework to upsampling with zeros.
        down_op = tf.map_fn(_apply_fir, audios)
        return down_op[:, ::scale, :]


def prepare_spectrogram(audios, window=400, step=160):
    # TF spectrogram doesn't support multiple channels.
    def _do_spectrogram(audio_tensor):
        return tf.squeeze(audio_ops.audio_spectrogram(audio_tensor, window_size=window, stride=step), axis=0)

    audio_spectrogram = tf.map_fn(_do_spectrogram, audios)
    audio_spectrogram = audio_spectrogram / tf.reduce_max(audio_spectrogram)
    audio_spectrogram = tf.maximum(audio_spectrogram, 1e-4)
    audio_spectrogram = tf.log(audio_spectrogram)
    audio_spectrogram = audio_spectrogram - tf.reduce_min(audio_spectrogram)
    audio_spectrogram = audio_spectrogram / tf.reduce_max(audio_spectrogram)
    return audio_spectrogram


def resnet_block(in_data, kernel_size, num_filters, is_training):
    in_scaled = tf.layers.conv1d(inputs=in_data, kernel_size=1, filters=num_filters,
                                 activation=None, padding='same')
    x = tf.layers.conv1d(inputs=in_data, kernel_size=kernel_size, filters=num_filters,
                         activation=None, padding='same')
    x = tf.layers.BatchNormalization()(x, training=is_training)
    x = tf.nn.leaky_relu(x)
    x = tf.layers.conv1d(inputs=x, kernel_size=kernel_size, filters=num_filters, activation=None,
                         padding='same')
    x = tf.layers.BatchNormalization()(x, training=is_training)
    x = tf.nn.leaky_relu(x + in_scaled)
    return x
