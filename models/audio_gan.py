from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
import tensorflow as tf
import numpy as np

from models.utils import upsample_fir, downsample_fir, resnet_block


def reverse_gradient(x):
    return -x + tf.stop_gradient(2 * x)


def scale_gradient(x, factor):
    return factor * x - tf.stop_gradient((factor - 1.0) * x)


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
        num_filters = 32
        kernel_size = 16
        num_blocks = 5
        shapes = [x.shape]
        with tf.variable_scope(
                "discriminator", reuse=reuse,
                initializer=tf.random_normal_initializer(stddev=0.02)):
            x = tf.squeeze(x, axis=3)
            # Mapping x from [bs, s, 1] to [bs, 1]
            net = resnet_block(x, kernel_size, num_filters, is_training)
            shapes.append(net.shape)
            for b_id in range(num_blocks):
                with tf.variable_scope('block_{}'.format(b_id)):
                    net = downsample_fir(net, 2)
                    net = resnet_block(net, kernel_size, num_filters, is_training)
                    shapes.append(net.shape)
            net = tf.layers.flatten(net)  # [bs, s * N]
            shapes.append(net.shape)
            net = tf.layers.dense(net, 256, activation=None, name="d_fc4")  # [bs, M]
            shapes.append(net.shape)

            dis_params = sum([np.prod([d.value for d in v.shape])
                              for v in tf.trainable_variables() if '/discriminator/discriminator/' in v.name])
            if dis_params > 0:
                shapes = ['[' + ', '.join(map(lambda xx: str(xx.value), x)) + ']' for x in shapes]
                tf.logging.info('Discriminator parameters: {:2.1f}'.format(dis_params / 1000000))
                tf.logging.info('Discriminator shapes flow: {}'.format(' -> '.join(shapes)))

            return net

    def generator(self, z, is_training):
        """Generator outputting audio in [-1, 1]."""
        num_filters = 32
        kernel_size = 16
        num_blocks = 5
        shapes = [z.shape]
        with tf.variable_scope("generator", initializer=tf.random_normal_initializer(stddev=0.02)):
            net = tf.layers.dense(z, self.hparams.samples_num // 2**num_blocks * num_filters, name="g_fc2")
            net = tf.layers.batch_normalization(net, training=is_training,
                                                momentum=0.999, name="g_bn2")
            net = tf.nn.leaky_relu(net)
            net = tf.reshape(net, [-1, self.hparams.samples_num // 2**num_blocks, num_filters])
            shapes.append(net.shape)
            net = resnet_block(net, kernel_size, num_filters, is_training)
            shapes.append(net.shape)
            for b_id in range(num_blocks):
                scale = 2 ** (num_blocks - b_id - 1)
                with tf.variable_scope('block_{}'.format(b_id)):
                    net = upsample_fir(net, 2)
                    net = resnet_block(net, kernel_size, num_filters, is_training)
                    shapes.append(net.shape)
            out = self.to_samples(net, 'samples_scale_{}'.format(scale))
            out = tf.expand_dims(out, axis=-1)
            shapes.append(out.shape)

            gen_params = sum([np.prod([d.value for d in v.shape])
                              for v in tf.trainable_variables('sliced_audio_gan/body/generator')])
            shapes = ['[' + ', '.join(map(lambda xx: str(xx.value), x)) + ']' for x in shapes]
            tf.logging.info('Generator parameters: {:2.1f}'.format(gen_params / 1000000))
            tf.logging.info('Generator shapes flow: {}'.format(' -> '.join(shapes)))

            return out

    @staticmethod
    def to_samples(in_signals, name):
        net = tf.layers.conv1d(in_signals, 1, 1, name=name,
                               activation=tf.tanh, padding='SAME')
        return net

    def losses(self, inputs, generated):
        """Return the losses dictionary."""
        raise NotImplementedError

    def bottom(self, features):
        batch_shape = common_layers.shape_list(features['targets'])
        batch_shape[0] = self.hparams.batch_size
        tf.ensure_shape(features['targets'], batch_shape)
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
        batch_size = common_layers.shape_list(inputs)[0]

        # Noise vector.
        z = tf.random_uniform([batch_size, self.hparams.bottleneck_bits],
                              minval=-1., maxval=1., name="z")

        # Generator output: fake images.
        g = self.generator(z, is_training)

        losses = self.losses(inputs, g)  # pylint: disable=not-callable

        if is_training:  # Returns an dummy output and the losses dictionary.
            return tf.zeros_like(inputs), losses
        return tf.reshape(g, tf.shape(inputs)), losses

    def top(self, body_output, features):
        """Override the top function to not do anything."""
        return body_output

    def infer(self, *args, **kwargs):  # pylint: disable=arguments-differ
        del args, kwargs

        with tf.variable_scope("body/audio_gan", reuse=tf.AUTO_REUSE):
            hparams = self.hparams
            z = tf.random_uniform([hparams.batch_size, hparams.bottleneck_bits],
                                  minval=-1, maxval=1, name="z")
            # TODO: fix num_samples passing
            g_sample = self.generator(z, False)
            return g_sample


@registry.register_model
class SlicedAudioGan(AbstractGAN):
    """Sliced GAN for demonstration."""

    def losses(self, inputs, generated):
        """Losses in the sliced case."""
        is_training = self.hparams.mode == tf.estimator.ModeKeys.TRAIN

        def discriminate(x):
            return self.discriminator(x, is_training=is_training, reuse=False)

        generator_loss = common_layers.sliced_gan_loss(
            inputs, scale_gradient(reverse_gradient(generated), self.hparams.g_grad_factor), discriminate,
            self.hparams.num_sliced_vecs)
        return {"training": - generator_loss}


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
    hparams.initializer = "uniform_unit_scaling"
    hparams.initializer_gain = 1.0
    hparams.weight_decay = 1e-6
    hparams.kernel_width = 4
    hparams.add_hparam("g_grad_factor", 5)
    hparams.add_hparam("bottleneck_bits", 256)
    hparams.add_hparam("num_sliced_vecs", 4096)
    return hparams
