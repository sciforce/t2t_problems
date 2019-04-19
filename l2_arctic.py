import os
from tqdm import tqdm
import tempfile
import librosa

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import speech_recognition
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.models.transformer import transformer_common_voice_tpu
from tensor2tensor.utils import metrics

SAMPLE_RATE = 16000

def _collect_data(directory):
    """Traverses directory collecting input and target files.

    Args:
    directory: base path to extracted audio and transcripts.
    Returns:
    list of (media_base, media_filepath, label) tuples
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
            utt_id = '{}-{}'.format(speaker, f.replace('.wav', ''))
            data_files.append((utt_id, wav_path, markup_text, speaker))
    return data_files


def _file_exists(path, filename):
  """Checks if the filename exists under the path."""
  return os.path.isfile(os.path.join(path, filename))


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

    def generator(self,
        data_dir,
        tmp_dir,
        eos_list=None,
        start_from=0,
        how_many=0):
        del eos_list
        i = 0
        data_tuples = _collect_data(tmp_dir)
        encoders = self.feature_encoders(data_dir)
        audio_encoder = encoders["waveforms"]
        text_encoder = encoders["targets"]
        for utt_id, media_file, text_data, speaker in tqdm(
            sorted(data_tuples)[start_from:]):
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

        all_paths = train_paths + dev_paths
        generator_utils.generate_files(
            self.generator(data_dir, tmp_dir), all_paths)
        generator_utils.shuffle_dataset(all_paths)