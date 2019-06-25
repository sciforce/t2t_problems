import os
from tqdm import tqdm
import librosa
import tempfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import speech_recognition
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_common_voice_tpu
from tensor2tensor.utils import metrics
from tensor2tensor.layers import modalities
from t2t_problems.utils.ipa_encoder import IPAEncoder

import tensorflow as tf

VOCAB_FILENAME = 'vocab.txt'
TRAIN_DATASET = 'train'
TEST_DATASET = 'test'
DEV_DATASET = 'val'

SAMPLE_RATE = 16000

def _collect_data(directory):
    data_full = []

    def _parse_corpus_csv(dataset):
        data_files = []
        csv_path = os.path.join(directory, dataset + '.csv')
        if os.path.exists(csv_path):
            tf.logging.info('Parsing %s', csv_path)
            with open(csv_path, 'r') as fid:
                lines = fid.read().strip().split('\n')
            for i, transcript_line in tqdm(enumerate(lines)):
                filename, lang, label = transcript_line.split('\t')
                utt_id = '{}-{}'.format(dataset, i)
                data_files.append((utt_id, filename, lang, label, dataset))
        return data_files

    for dataset in (TRAIN_DATASET, TEST_DATASET, DEV_DATASET):
        data_full.extend(_parse_corpus_csv(dataset))
    return data_full


@registry.register_problem()
class AsrIpa(speech_recognition.SpeechRecognitionProblem):
    @property
    def convert_to_ipa(self):
        """If true, we assume that labels contain text, and convert it to IPA."""
        return True

    @property
    def num_shards(self):
        return 20
    
    @property
    def num_dev_shards(self):
        return 1

    @property
    def num_test_shards(self):
        return 1

    @property
    def use_train_shards_for_dev(self):
        """If true, we only generate training data and hold out shards for dev."""
        return False

    def feature_encoders(self, data_dir):
        res = super().feature_encoders(data_dir)
        res["targets"] = IPAEncoder(data_dir)
        return res

    def generator(self,
        data_dir,
        tmp_dir,
        dataset,
        start_from=0,
        how_many=0):
        i = 0
        data_tuples = _collect_data(tmp_dir)
        encoders = self.feature_encoders(data_dir)
        audio_encoder = encoders["waveforms"]
        text_encoder = encoders["targets"]
        try:
            for utt_id, media_file, lang, text_data, utt_dataset in tqdm(
                sorted(data_tuples)[start_from:]):
                if dataset != utt_dataset:
                    continue
                if how_many > 0 and i == how_many:
                    text_encoder.store_vocab()
                    return
                i += 1
                try:
                    wav_data = audio_encoder.encode(media_file)
                except:
                    try:
                        audio, sr = librosa.load(media_file, sr=SAMPLE_RATE)
                        with tempfile.NamedTemporaryFile(suffix='.wav') as fid:
                            librosa.output.write_wav(fid.name, audio, SAMPLE_RATE)
                            wav_data = audio_encoder.encode(fid.name)
                    except Exception as e:
                        tf.logging.error('Error reading file %s. Unhandled exception %s', media_file, str(e))
                try:
                    ipa_data = text_encoder.encode(text_data, lang, is_text=self.convert_to_ipa)
                except Exception as e:
                    tf.logging.warn('Failed transcribing phrase "%s" file: %s Exception: %s', text_data, media_file, str(e))
                    continue
                yield {
                    "waveforms": wav_data,
                    "waveform_lens": [len(wav_data)],
                    "targets": ipa_data,
                    "raw_transcript": [text_data],
                    "utt_id": [utt_id],
                    "lang_id": [lang]
                }
        except GeneratorExit:
            text_encoder.store_vocab()
        text_encoder.store_vocab()

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        train_paths = self.training_filepaths(
            data_dir, self.num_shards, shuffled=False)
        dev_paths = self.dev_filepaths(
            data_dir, self.num_dev_shards, shuffled=False)
        test_paths = self.test_filepaths(
            data_dir, self.num_test_shards, shuffled=True)

        if self.use_train_shards_for_dev:
            all_paths = train_paths + dev_paths + test_paths
            generator_utils.generate_files(
                self.generator(data_dir, tmp_dir, TRAIN_DATASET), all_paths)
            generator_utils.shuffle_dataset(all_paths)
        else:
            generator_utils.generate_files(
                self.generator(data_dir, tmp_dir, TEST_DATASET), test_paths)
            generator_utils.generate_dataset_and_shuffle(
                self.generator(data_dir, tmp_dir, TRAIN_DATASET), train_paths,
                self.generator(data_dir, tmp_dir, DEV_DATASET), dev_paths)

    def hparams(self, defaults, model_hparams):
        super().hparams(defaults, model_hparams)
        vocab_path = os.path.join(model_hparams.data_dir, VOCAB_FILENAME)
        with tf.gfile.Open(vocab_path) as fid:
            vocab = fid.read().strip().split('\n')
        defaults.vocab_size = {"inputs": None,
                               "targets": len(vocab)}
        tf.logging.info('Setting vocabulary size to %d',
                defaults.vocab_size["targets"])

    def example_reading_spec(self):
        data_fields, data_items_to_decoders = super().example_reading_spec()
        data_fields["partial_targets"] = tf.VarLenFeature(tf.int64)

        return data_fields, data_items_to_decoders

@registry.register_problem()
class AsrIpaPrecalc(AsrIpa):
    @property
    def convert_to_ipa(self):
        return False

    @property
    def num_shards(self):
        return 100

    @property
    def num_dev_shards(self):
        return 1

    @property
    def num_test_shards(self):
        return 1

    @property
    def use_train_shards_for_dev(self):
        return True
