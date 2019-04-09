# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mozilla Common Voice dataset.

Note: Generating the full set of examples can take upwards of 5 hours.
As the Common Voice data are distributed in MP3 format, experimenters will need
to have both SoX (http://sox.sourceforge.net) and on Linux, the libsox-fmt-mp3
package installed. The original samples will be downsampled by the encoder.
"""

import csv
import os
import tarfile
import tqdm  # pylint: disable=g-bad-import-order
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import speech_recognition
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder
from t2t_problems.utils.ipa_utils import get_ipa
from tensor2tensor.models.transformer import transformer_common_voice_tpu
from tensor2tensor.utils import metrics

import tensorflow as tf

# _COMMONVOICE_URL = "https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz"  # pylint: disable=line-too-long
_COMMONVOICE_URLS = ["https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/en.tar.gz",
                     "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/de.tar.gz",
                     "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/fr.tar.gz",
                     "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/cy.tar.gz",
                     "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/tr.tar.gz",
                     "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/tt.tar.gz",
                     "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/ky.tar.gz",
                     "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/ga-IE.tar.gz",
                     "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/ca.tar.gz",
                     "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/zh-TW.tar.gz",
                     "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/sl.tar.gz",
                     "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/it.tar.gz",
                     "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/nl.tar.gz",
                     "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-1/eo.tar.gz"]

_COMMONVOICE_TRAIN_DATASETS = ["validated"]
_COMMONVOICE_DEV_DATASETS = ["dev"]
_COMMONVOICE_TEST_DATASETS = ["test"]

TESTSET_SIZE = 1000

def _collect_data(directory):
  """Traverses directory collecting input and target files.

  Args:
   directory: base path to extracted audio and transcripts.
  Returns:
   list of (media_base, media_filepath, label) tuples
  """
  # Returns:
  data_files = []
  langs = [d for d in os.listdir(directory)
           if os.path.isdir(os.path.join(directory, d))]
  for lang in langs:
    lang_dir = os.path.join(directory, lang)
    transcripts = [
        filename for filename in os.listdir(lang_dir)
        if filename.endswith(".csv") or filename.endswith(".tsv")
    ]
    for transcript in transcripts:
      transcript_path = os.path.join(lang_dir, transcript)
      with open(transcript_path, "r") as transcript_file:
        transcript_reader = csv.reader(transcript_file, dialect="excel-tab")
        # skip header
        _ = next(transcript_reader)
        for transcript_line in transcript_reader:
          utt_id, media_name, label = transcript_line[:3]
          filename = os.path.join(lang_dir, "clips", media_name + ".mp3")
          data_files.append((transcript[:-4] + '_' + utt_id, filename, label, lang))
  return data_files


def _file_exists(path, filename):
  """Checks if the filename exists under the path."""
  return os.path.isfile(os.path.join(path, filename))


def _is_relative(path, filename):
  """Checks if the filename is relative, not absolute."""
  return os.path.abspath(os.path.join(path, filename)).startswith(path)


@registry.register_problem()
class CommonVoice_IPA_UTF(speech_recognition.SpeechRecognitionProblem):
  """Problem spec for Commonvoice using clean and noisy data."""

  # Select only the clean data
  TRAIN_DATASETS = _COMMONVOICE_TRAIN_DATASETS[:1]
  DEV_DATASETS = _COMMONVOICE_DEV_DATASETS[:1]
  TEST_DATASETS = _COMMONVOICE_TEST_DATASETS[:1]

  @property
  def num_shards(self):
    return 100

  @property
  def use_subword_tokenizer(self):
    return False

  @property
  def num_dev_shards(self):
    return 1

  @property
  def num_test_shards(self):
    return 1

  @property
  def use_train_shards_for_dev(self):
    """If true, we only generate training data and hold out shards for dev."""
    return True

  def generator(self,
                data_dir,
                tmp_dir,
                datasets,
                eos_list=None,
                start_from=0,
                how_many=0):
    del eos_list
    i = 0

    for url in _COMMONVOICE_URLS:
      filename = os.path.basename(url)
      compressed_file = generator_utils.maybe_download(tmp_dir, filename,
                                                     url)
      lang = filename[:2]
      target_dir = os.path.join(tmp_dir, lang)
      if os.path.isdir(target_dir):
        continue
      read_type = "r:gz" if filename.endswith(".tgz") else "r"
      with tarfile.open(compressed_file, read_type) as corpus_tar:
        # Create a subset of files that don't already exist.
        #   tarfile.extractall errors when encountering an existing file
        #   and tarfile.extract is extremely slow. For security, check that all
        #   paths are relative.
        members = [
            f for f in corpus_tar if _is_relative(tmp_dir, f.name) and
            not _file_exists(tmp_dir, f.name)
        ]
        tf.logging.info('Extracting %s into directory %s', compressed_file, target_dir)
        corpus_tar.extractall(target_dir, members=members)

    data_tuples = _collect_data(tmp_dir)
    encoders = self.feature_encoders(data_dir)
    audio_encoder = encoders["waveforms"]
    text_encoder = encoders["targets"]
    for dataset in datasets:
      data_tuples = (tup for tup in data_tuples if tup[0].startswith(dataset))
      for utt_id, media_file, text_data, lang in tqdm.tqdm(
          sorted(data_tuples)[start_from:]):
        if how_many > 0 and i == how_many:
          return
        i += 1
        try:
          wav_data = audio_encoder.encode(media_file)
        except Exception:
          tf.logging.warn('Failed encoding file: %s', media_file)
          continue
        try:
          ipa_data = ''.join(get_ipa(text_data, lang))
        except Exception:
          tf.logging.warn('Failed transcribing phrase "%s" file: %s', text_data, media_file)
          continue
        if not wav_data:
          tf.logging.warn('Empty waveform %s', media_file)
          continue
        yield {
            "waveforms": wav_data,
            "waveform_lens": [len(wav_data)],
            "targets": text_encoder.encode(ipa_data),
            "raw_transcript": [text_data],
            "utt_id": [utt_id],
            "spk_id": ["unknown"],
            "lang": lang,
        }

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(
        data_dir, self.num_dev_shards, shuffled=False)
    test_paths = self.test_filepaths(
        data_dir, self.num_test_shards, shuffled=True)

    generator_utils.generate_files(
        self.generator(data_dir, tmp_dir, self.TEST_DATASETS), test_paths, max_cases=TESTSET_SIZE)

    if self.use_train_shards_for_dev:
      all_paths = train_paths + dev_paths
      generator_utils.generate_files(
          self.generator(data_dir, tmp_dir, self.TRAIN_DATASETS), all_paths)
      generator_utils.shuffle_dataset(all_paths)
    else:
      generator_utils.generate_dataset_and_shuffle(
          self.generator(data_dir, tmp_dir, self.TRAIN_DATASETS), train_paths,
          self.generator(data_dir, tmp_dir, self.DEV_DATASETS), dev_paths)
