import os
from tqdm import tqdm
import tempfile
import librosa
import tgt
import re

import tensorflow as tf
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import speech_recognition
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.models.transformer import transformer_librispeech, update_hparams_for_tpu, transformer_small
from tensor2tensor.utils import metrics

SAMPLE_RATE = 16000

VOCAB_FILENAME = 'vocab.txt'

TEST_SPEAKERS = ['RRBI']
TRAIN_DATASET = 'train'
TEST_DATASET = 'test'

class ArpabetEncoder(text_encoder.TextEncoder):
    def __init__(self, data_dir):
        super(ArpabetEncoder, self).__init__(num_reserved_ids=0)
        self._data_dir = data_dir
        self._vocab_file = os.path.join(self._data_dir, VOCAB_FILENAME)
        self._vocab = text_encoder.RESERVED_TOKENS
        self.load_vocab()

    def encode(self, s):
        res = []
        transcription = s.split(' ')
        for phone in transcription:
            if len(phone) > 0:
                if phone not in self._vocab:
                    self._vocab.append(phone)
                res.append(self._vocab.index(phone))
        return res + [text_encoder.EOS_ID]

    def load_vocab(self):
        tf.logging.info('Loading vocab from %s', self._vocab_file)
        if tf.gfile.Exists(self._vocab_file):
            with tf.gfile.Open(self._vocab_file, 'r') as fid:
                self._vocab = fid.read().strip().split('\n')
        else:
            tf.logging.info('Loading vocab from %s failed', self._vocab_file)

    def store_vocab(self):
        tf.logging.info('Saving vocab to %s', self._vocab_file)
        with tf.gfile.Open(self._vocab_file, 'w') as fid:
            fid.write('\n'.join(self._vocab))

    def decode(self, ids):
        return ' '.join([self._vocab[id] for id in ids])


def _collect_data(directory):
    """Traverses directory collecting input and target files.

    Args:
    directory: base path to extracted audio and transcripts.
    Returns:
    list of (media_base, media_filepath, label, speaker) tuples
    """
    # Returns:
    data_files = []
    speakers = [d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))]
    for speaker in tqdm(speakers, desc='Collecting speakers'):
        speaker_dir = os.path.join(directory, speaker)
        files = os.listdir(os.path.join(speaker_dir, 'wav'))
        for f in tqdm(filter(lambda x: x.endswith('.wav'), files),
            desc='Processing speaker {}'.format(speaker)):
            wav_path = os.path.join(speaker_dir, 'wav', f)
            markup_path = os.path.join(speaker_dir, 'transcript', f.replace('.wav', '.txt'))
            with open(markup_path, 'r') as fid:
                markup_text = fid.read().strip(' \n').replace(',', '')
            dataset = TEST_DATASET if speaker in TEST_SPEAKERS else TRAIN_DATASET
            utt_id = '{}-{}'.format(speaker, f.replace('.wav', ''))
            data_files.append((utt_id, wav_path, markup_text, speaker, dataset))
    return data_files

def _collect_data_textgrids(directory):
    """Traverses directory collecting input and target files.

    Args:
    directory: base path to extracted audio and transcripts.
    Returns:
    list of (media_base, media_filepath, label, speaker) tuples
    """
    # Returns:
    data_files = []
    speakers = [d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))]
    for speaker in tqdm(speakers, desc='Collecting speakers'):
        speaker_dir = os.path.join(directory, speaker)
        files = os.listdir(os.path.join(speaker_dir, 'wav'))
        for f in tqdm(filter(lambda x: x.endswith('.wav'), files),
            desc='Processing speaker {}'.format(speaker)):
            wav_path = os.path.join(speaker_dir, 'wav', f)
            markup_path = os.path.join(speaker_dir, 'annotation', f.replace('.wav', '.txt'))
            try:
                textgrid = tgt.io.read_textgrid(markup_path)
            except Exception:
                markup_path = os.path.join(speaker_dir, 'textgrid', f.replace('.wav', '.TextGrid'))
                textgrid = tgt.io.read_textgrid(markup_path)
            tier = textgrid.get_tier_by_name('phones')
            phones = list()
            parse_error = False
            for phone in tier.annotations:
                phone = phone.text.lower().replace(' ', '')
                if 'spn' in phone:
                    parse_error = True
                    break
                if ',' in phone:
                    _, phone, _ = phone.split(',')
                    if 'err' in phone:
                        continue
                if phone == 'sp':
                    phone = 'sil'
                phone = re.sub(r'[^a-zA-Z1-2]', '', phone)
                phones.append(phone)
            if parse_error:
                continue
            markup_text = ' '.join(phones)
            dataset = TEST_DATASET if speaker in TEST_SPEAKERS else TRAIN_DATASET
            utt_id = '{}-{}'.format(speaker, f.replace('.wav', ''))
            data_files.append((utt_id, wav_path, markup_text, speaker, dataset))
    return data_files

def _is_relative(path, filename):
  """Checks if the filename is relative, not absolute."""
  return os.path.abspath(os.path.join(path, filename)).startswith(path)

@registry.register_problem()
class L2Arctic(speech_recognition.SpeechRecognitionProblem):

    @property
    def num_shards(self):
        return 20
    
    @property
    def num_dev_shards(self):
        return 1

    @property
    def num_test_shards(self):
        return 1

    def generator(self,
        data_dir,
        tmp_dir,
        dataset,
        eos_list=None,
        start_from=0,
        how_many=0):
        del eos_list
        i = 0
        data_tuples = _collect_data(tmp_dir)
        encoders = self.feature_encoders(data_dir)
        audio_encoder = encoders["waveforms"]
        text_encoder = encoders["targets"]
        for utt_id, media_file, text_data, speaker, utt_dataset in tqdm(
            sorted(data_tuples)[start_from:]):
            if dataset != utt_dataset:
                continue
            if how_many > 0 and i == how_many:
                return
            i += 1
            try:
                wav_data = audio_encoder.encode(media_file)
            except AssertionError:
                audio, sr = librosa.load(media_file)
                data_resampled = librosa.resample(audio, sr, SAMPLE_RATE)
                with tempfile.NamedTemporaryFile(suffix='.wav') as fid:
                    librosa.output.write_wav(fid.name, data_resampled, SAMPLE_RATE)
                    wav_data = audio_encoder.encode(fid.name)
            yield {
                "waveforms": wav_data,
                "waveform_lens": [len(wav_data)],
                "targets": text_encoder.encode(text_data),
                "raw_transcript": [text_data],
                "utt_id": [utt_id],
                "spk_id": [speaker],
            }

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        train_paths = self.training_filepaths(
            data_dir, self.num_shards, shuffled=False)
        dev_paths = self.dev_filepaths(
            data_dir, self.num_dev_shards, shuffled=False)
        test_paths = self.test_filepaths(
            data_dir, self.num_test_shards, shuffled=True)

        generator_utils.generate_files(
            self.generator(data_dir, tmp_dir, TEST_DATASET), test_paths)

        all_paths = train_paths + dev_paths
        generator_utils.generate_files(
            self.generator(data_dir, tmp_dir, TRAIN_DATASET), all_paths)
        generator_utils.shuffle_dataset(all_paths)


@registry.register_problem()
class L2ArcticArpabet(L2Arctic):
    def feature_encoders(self, data_dir):
        res = super().feature_encoders(data_dir)
        res["targets"] = ArpabetEncoder(data_dir)
        return res

    def generator(self,
        data_dir,
        tmp_dir,
        dataset,
        eos_list=None,
        start_from=0,
        how_many=0):
        del eos_list
        i = 0
        data_tuples = _collect_data_textgrids(tmp_dir)
        encoders = self.feature_encoders(data_dir)
        audio_encoder = encoders["waveforms"]
        text_encoder = encoders["targets"]
        try:
            for utt_id, media_file, text_data, speaker, utt_dataset in tqdm(
                sorted(data_tuples)[start_from:]):
                if dataset != utt_dataset:
                    continue
                if how_many > 0 and i == how_many:
                    text_encoder.store_vocab()
                    return
                i += 1
                try:
                    wav_data = audio_encoder.encode(media_file)
                except AssertionError:
                    audio, sr = librosa.load(media_file)
                    data_resampled = librosa.resample(audio, sr, SAMPLE_RATE)
                    with tempfile.NamedTemporaryFile(suffix='.wav') as fid:
                        librosa.output.write_wav(fid.name, data_resampled, SAMPLE_RATE)
                        wav_data = audio_encoder.encode(fid.name)
                yield {
                    "waveforms": wav_data,
                    "waveform_lens": [len(wav_data)],
                    "targets": text_encoder.encode(text_data),
                    "raw_transcript": [text_data],
                    "utt_id": [utt_id],
                    "spk_id": [speaker],
                }
        except GeneratorExit:
            text_encoder.store_vocab()
        text_encoder.store_vocab()

    def hparams(self, defaults, model_hparams):
        super().hparams(defaults, model_hparams)
        vocab_path = os.path.join(model_hparams.data_dir, VOCAB_FILENAME)
        with tf.gfile.Open(vocab_path) as fid:
            vocab = fid.read().strip().split('\n')
        model_hparams.vocab_size = {"inputs": None,
                                    "targets": len(vocab)}
        tf.logging.info('Setting vocabulary size to %d',
                model_hparams.vocab_size["targets"])


@registry.register_hparams
def transformer_l2_arctic():
    """HParams for training ASR model on L2 Arctic"""
    hparams = transformer_small()

    hparams.max_length = 1240000
    hparams.max_input_seq_length = 1550
    hparams.max_target_seq_length = 350
    hparams.batch_size = 16
    hparams.learning_rate = 0.15
    hparams.daisy_chain_variables = False
    hparams.filter_size = 1536
    hparams.num_heads = 2
    hparams.ffn_layer = "conv_relu_conv"
    hparams.conv_first_kernel = 9
    hparams.weight_decay = 0
    hparams.layer_prepostprocess_dropout = 0.2
    hparams.relu_dropout = 0.2

    hparams.num_decoder_layers = 1
    hparams.num_encoder_layers = 3
    # hparams.num_hidden_layers = 1
    # hparams.hidden_size = 256

    return hparams

@registry.register_hparams
def transformer_l2_arctic_tpu():
    """HParams for training ASR model on L2 Arctic on TPU"""
    hparams = transformer_l2_arctic()
    update_hparams_for_tpu(hparams)
    hparams.batch_size = 16
    hparams.max_length = 1650 * 80  # this limits inputs[1] * inputs[2]
    hparams.max_input_seq_length = 1650
    hparams.max_target_seq_length = 350
    return hparams
