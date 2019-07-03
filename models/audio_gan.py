from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def lrelu(input_, leak=0.2, name="lrelu"):
    return tf.maximum(input_, leak * input_, name=name)


def deconv1d(
        input_, output_shape, k_w, d_w, stddev=0.02, name="deconv1d"):
    """Deconvolution layer."""
    with tf.variable_scope(name):
        w = tf.get_variable(
            "w", [k_w, output_shape[-1], input_.get_shape()[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv1d_transpose(
            input_, w, output_shape=output_shape, strides=[1, d_w, 1])
        biases = tf.get_variable(
            "biases", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        return tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())


def reverse_gradient(x):
    return -x + tf.stop_gradient(2 * x)


class AbstractGAN(t2t_model.T2TModel):
    """Base class for all GANs."""

    def discriminator(self, x, is_training, reuse=False):
        """Discriminator architecture based on InfoGAN.

        Args:
          x: input audios, shape [bs, w, channels]
          is_training: boolean, are we in train or eval model.
          reuse: boolean, should params be re-used.

        Returns:
          out_logit: the output logits (before sigmoid).
        """
        hparams = self.hparams
        num_filters = 32
        kernel_size = 16
        num_blocks = 5
        with tf.variable_scope(
                "discriminator", reuse=reuse,
                initializer=tf.random_normal_initializer(stddev=0.02)):
            x = tf.squeeze(x, axis=3)
            # Mapping x from [bs, s, 1] to [bs, 1]
            net = x
            for b_id in range(num_blocks):
                with tf.variable_scope('block_{}'.format(b_id), reuse=reuse):
                    net = tf.layers.conv1d(net, num_filters, (kernel_size,), strides=(2,),
                                           padding="SAME", name="d_conv1")
                    net = lrelu(net)
                    net = tf.layers.conv1d(net, num_filters, (kernel_size,), strides=(2,),
                                           padding="SAME", name="d_conv2")
                    if hparams.discriminator_batchnorm:
                        net = tf.layers.batch_normalization(net, training=is_training,
                                                            momentum=0.999, name="d_bn2")
                    net = lrelu(net)
            net = tf.layers.flatten(net)  # [bs, s * N]
            net = tf.layers.dense(net, 1024, name="d_fc3")  # [bs, 1024]
            if hparams.discriminator_batchnorm:
                net = tf.layers.batch_normalization(net, training=is_training,
                                                    momentum=0.999, name="d_bn3")
            net = lrelu(net)
            return net

    def generator(self, z, is_training, samples_num):
        """Generator outputting audio in [-1, 1]."""
        hparams = self.hparams
        num_filters = 32
        kernel_size = 16
        num_blocks = 5
        batch_size = hparams.batch_size
        with tf.variable_scope("generator", initializer=tf.random_normal_initializer(stddev=0.02)):
            net = tf.layers.dense(z, 1024, name="g_fc1")
            net = tf.layers.batch_normalization(net, training=is_training,
                                                momentum=0.999, name="g_bn1")
            net = lrelu(net)
            net = tf.layers.dense(net, samples_num // 2**num_blocks * num_filters, name="g_fc2")
            net = tf.layers.batch_normalization(net, training=is_training,
                                                momentum=0.999, name="g_bn2")
            net = lrelu(net)
            net = tf.reshape(net, [batch_size, samples_num // 2**num_blocks, num_filters])
            for b_id in range(num_blocks):
                with tf.variable_scope('block_{}'.format(b_id)):
                    net = deconv1d(net, [batch_size, samples_num // 2**(num_blocks - b_id - 1), num_filters],
                                   kernel_size, 2, name="g_dc{}".format(b_id + 3))
                    net = tf.layers.batch_normalization(net, training=is_training,
                                                        momentum=0.999, name="g_bn{}".format(b_id + 3))
                    net = lrelu(net)
            net = tf.layers.conv1d(net, 1, kernel_size, 1, name="g_dc{}".format(num_blocks + 3),
                                   activation=None, padding='SAME')
            net = net - tf.reduce_mean(net, axis=1, keepdims=True)
            out = tf.nn.tanh(net)
            out = tf.expand_dims(out, axis=-1)
            return out

    def losses(self, inputs, generated):
        """Return the losses dictionary."""
        raise NotImplementedError

    def bottom(self, features):
        return features

    def body(self, features):
        """Body of the model.

        Args:
          features: a dictionary with the tensors.

        Returns:
          A pair (predictions, losses) where predictions is the generated image
          and losses is a dictionary of losses (that get added for the final loss).
        """
        is_training = self.hparams.mode == tf.estimator.ModeKeys.TRAIN

        # Input audios.
        inputs = tf.to_float(features["targets"])
        batch_size, samples_num = common_layers.shape_list(inputs)[0:2]

        # Noise vector.
        z = tf.random_uniform([batch_size, self.hparams.bottleneck_bits],
                              minval=-1, maxval=1, name="z")

        # Generator output: fake images.
        g = self.generator(z, is_training, samples_num)

        losses = self.losses(inputs, g)  # pylint: disable=not-callable

        if is_training:  # Returns an dummy output and the losses dictionary.
            return tf.zeros_like(inputs), losses
        return tf.reshape(g, tf.shape(inputs)), losses

    def top(self, body_output, features):
        """Override the top function to not do anything."""
        return body_output


@registry.register_model
class SlicedAudioGan(AbstractGAN):
    """Sliced GAN for demonstration."""

    def losses(self, inputs, generated):
        """Losses in the sliced case."""
        is_training = self.hparams.mode == tf.estimator.ModeKeys.TRAIN

        def discriminate(x):
            return self.discriminator(x, is_training=is_training, reuse=False)

        generator_loss = common_layers.sliced_gan_loss(
            inputs, reverse_gradient(generated), discriminate,
            self.hparams.num_sliced_vecs)
        return {"training": - generator_loss}

    def infer(self, *args, **kwargs):  # pylint: disable=arguments-differ
        del args, kwargs

        with tf.variable_scope("body/audio_gan", reuse=tf.AUTO_REUSE):
            hparams = self.hparams
            z = tf.random_uniform([hparams.batch_size, hparams.bottleneck_bits],
                                  minval=-1, maxval=1, name="z")
            # TODO: fix num_samples passing
            g_sample = self.generator(z, False, 4096)
            return g_sample


@registry.register_hparams
def sliced_audio_gan():
    """Basic parameters for a vanilla_gan."""
    hparams = common_hparams.basic_params1()
    hparams.optimizer = "adam"
    hparams.learning_rate_constant = 0.0002
    hparams.learning_rate_warmup_steps = 500
    hparams.learning_rate_schedule = "constant * linear_warmup"
    hparams.label_smoothing = 0.0
    hparams.batch_size = 16
    hparams.hidden_size = 128
    hparams.initializer = "uniform_unit_scaling"
    hparams.initializer_gain = 1.0
    hparams.weight_decay = 1e-6
    hparams.kernel_width = 4
    hparams.add_hparam("bottleneck_bits", 128)
    hparams.add_hparam("discriminator_batchnorm", True)
    hparams.add_hparam("num_sliced_vecs", 4096)
    return hparams
