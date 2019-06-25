import os
import tensorflow as tf
from tensor2tensor.data_generators import text_encoder

VOCAB_FILENAME = 'vocab.txt'

class IPAEncoder(text_encoder.TextEncoder):
  def __init__(self, data_dir, remove_lang_markers=False):
    super(IPAEncoder, self).__init__(num_reserved_ids=0)
    self._data_dir = data_dir
    self._vocab_file = os.path.join(self._data_dir, VOCAB_FILENAME)
    self._vocab = text_encoder.RESERVED_TOKENS
    self._remove_lang_markers = remove_lang_markers
    self.load_vocab()

  def encode(self, s, lang='en', is_text=True):
    from t2t_problems.utils.ipa_utils import get_ipa
    
    res, ipa = [], []
    if is_text:
      ipa = get_ipa(s, lang, remove_semi_stress=False, split_all_diphthongs=True,
                    split_stress_gemination=True,
                    remove_lang_markers=self._remove_lang_markers)
    elif s:
      ipa = s.split(',')
    ipa = ['<{}>'.format(lang)] + ipa
    for phone in ipa:
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
    integers = list(ids)
    if text_encoder.EOS_ID in integers:
      integers = integers[:integers.index(text_encoder.EOS_ID)]
    return ''.join([self._vocab[id] for id in integers])

  @property
  def vocab_size(self):
    return len(self._vocab)