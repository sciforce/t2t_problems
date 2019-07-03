import os
from tensor2tensor.data_generators import audio_encoder
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import modalities
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.utils import registry

import tensorflow as tf


class AudioGanGen(problem.Problem):
    def hparams(self, defaults, model_hparams):
        defaults.modality = {"targets": modalities.ModalityType.REAL}
        defaults.vocab_size = {"targets": None}

    @property
    def max_samples(self):
        raise NotImplementedError()

    @property
    def num_shards(self):
        return 20

    @property
    def target_space_id(self):
        return problem.SpaceID.AUDIO_WAV

    def feature_encoders(self, _):
        return {
            "targets": audio_encoder.AudioEncoder(),
        }

    def example_reading_spec(self):
        data_fields = {
            # "targets": tf.VarLenFeature(tf.float32)
            "targets": tf.FixedLenFeature([self.max_samples], tf.float32)
        }
        data_items_to_decoders = None
        return data_fields, data_items_to_decoders

    def generator(self, data_dir):
        base_dir = os.path.join(data_dir, "wavs")
        encoders = self.feature_encoders(data_dir)
        audio_encoder = encoders["targets"]

        for root, _, files in os.walk(base_dir):
            for file in files:
                filename = os.path.join(root, file)
                wav_data = audio_encoder.encode(filename)
                if len(wav_data) >= self.max_samples:
                    wav_data = wav_data[:self.max_samples]
                else:
                    wav_data = wav_data + [0.] * (self.max_samples - len(wav_data))
                assert len(wav_data) == self.max_samples
                yield {
                    "audio_len": [self.max_samples],
                    "targets": wav_data
                }

    def training_filepaths(self, data_dir, num_shards, shuffled):
        file_basename = self.dataset_filename()
        if not shuffled:
            file_basename += generator_utils.UNSHUFFLED_SUFFIX
        return generator_utils.train_data_filenames(file_basename, data_dir,
                                                    num_shards)

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        train_paths = self.training_filepaths(
            data_dir, self.num_shards, shuffled=False)

        generator_utils.generate_files(
            self.generator(data_dir), train_paths)
        generator_utils.shuffle_dataset(train_paths)

    def eval_metrics(self):
        return ['audio_summary']

    @property
    def all_metrics_fns(self):
        orig_fns = super().all_metrics_fns
        orig_fns['audio_summary'] = audio_summary
        return orig_fns


def audio_summary(predictions, targets, **kwargs):
    num_audios = 6
    summary_g_audio = tf.reshape(predictions[:num_audios, :], [num_audios, -1, 1])
    summary1 = tf.summary.audio("generated", summary_g_audio, sample_rate=16000, max_outputs=num_audios)
    summary_t_audio = tf.reshape(targets[:num_audios, :], [num_audios, -1, 1])
    summary2 = tf.summary.audio("real", summary_t_audio, sample_rate=16000, max_outputs=num_audios)
    summary = tf.summary.merge([summary1, summary2])
    return summary, [0.]


@registry.register_problem
class AudioGan4k(AudioGanGen):
    @property
    def max_samples(self):
        return 4096
