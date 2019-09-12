import tensorflow as tf
import re
from tensorflow.python.util import nest
from tensor2tensor.utils import registry
from tensor2tensor.utils.t2t_model import T2TModel, _del_dict_non_tensors
from tensor2tensor.models.transformer import Transformer, features_to_nonpadding, fast_decode
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils.beam_search import _merge_beam_dim, _unmerge_beam_dim

@registry.register_model
class TransformerSlowDecode(Transformer):
  """Transformer model, but uses slow algorithms for inference.

  Checkpoints between Transformer and TransformerSlowDecode are interchangeable.
  """
  def __init__(self, *args, **kwargs):
    super(TransformerSlowDecode, self).__init__(*args, **kwargs)
    self._name = "transformer"
    self._base_name = "transformer"

  def _greedy_infer(self, features, decode_length, use_tpu=False):
    return T2TModel._greedy_infer(features, decode_length, use_tpu)

  def _beam_decode(self,
                   features,
                   decode_length,
                   beam_size,
                   top_beams,
                   alpha,
                   use_tpu=False):
    return self._beam_decode_slow(features, decode_length, beam_size,
                                  top_beams, alpha, use_tpu)

@registry.register_model
class TransformerWithHistory(Transformer):
  """Transformer model, returns encoder-decoder attention history.

  Checkpoints between Transformer and TransformerWithHistory are interchangeable.
  """
  def __init__(self, *args, **kwargs):
    super(TransformerWithHistory, self).__init__(*args, **kwargs)
    self._name = "transformer"
    self._base_name = "transformer"

  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.modality["targets"]
    target_vocab_size = self._problem_hparams.vocab_size["targets"]
    if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
      target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor
    if "targets_segmentation" in features:
      raise NotImplementedError(
          "Decoding not supported on packed datasets "
          " If you want to decode from a dataset, use the non-packed version"
          " of the dataset when decoding.")
    if self.has_input:
      inputs = features["inputs"]
      if target_modality == modalities.ModalityType.CLASS_LABEL:
        decode_length = 1
      else:
        decode_length = (
            common_layers.shape_list(inputs)[1] + features.get(
                "decode_length", decode_length))

      # TODO(llion): Clean up this reshaping logic.
      inputs = tf.expand_dims(inputs, axis=1)
      if len(inputs.shape) < 5:
        inputs = tf.expand_dims(inputs, axis=4)
      s = common_layers.shape_list(inputs)
      batch_size = s[0]
      inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
      # _shard_features called to ensure that the variable names match
      inputs = self._shard_features({"inputs": inputs})["inputs"]
      input_modality = self._problem_hparams.modality["inputs"]
      input_vocab_size = self._problem_hparams.vocab_size["inputs"]
      if input_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
        input_vocab_size += (-input_vocab_size) % hparams.vocab_divisor
      modality_name = hparams.name.get(
          "inputs",
          modalities.get_name(input_modality))(hparams, input_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get("inputs",
                                    modalities.get_bottom(input_modality))
        inputs = dp(bottom, inputs, hparams, input_vocab_size)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
            self.encode,
            inputs,
            features["target_space_id"],
            hparams,
            features=features)
      encoder_output = encoder_output[0]
      encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
      if 'partial_targets' in features:
        partial_targets = features['partial_targets']
      else:
        partial_targets = None
    else:
      # The problem has no inputs.
      encoder_output = None
      encoder_decoder_attention_bias = None

      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      assert partial_targets is not None

    if partial_targets is not None:
      partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
      partial_targets = tf.to_int64(partial_targets)
      partial_targets_shape = common_layers.shape_list(partial_targets)
      partial_targets_length = partial_targets_shape[1]
      decode_length = (
          partial_targets_length + features.get("decode_length", decode_length))
      batch_size = partial_targets_shape[0]

    if hparams.pos == "timing":
      positional_encoding = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)
    elif hparams.pos == "emb":
      positional_encoding = common_attention.add_positional_embedding(
          tf.zeros([1, decode_length, hparams.hidden_size]), hparams.max_length,
          "body/targets_positional_embedding", None)
    else:
      positional_encoding = None

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get(
            "targets", modalities.get_targets_bottom(target_modality))
        targets = dp(bottom, targets, hparams, target_vocab_size)[0]
      targets = common_layers.flatten4d3d(targets)

      # GO embeddings are all zero, this is because transformer_prepare_decoder
      # Shifts the targets along by one for the input which pads with zeros.
      # If the modality already maps GO to the zero embeddings this is not
      # needed.
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if positional_encoding is not None:
        targets += positional_encoding[:, i:i + 1]
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)

    # Create tensors for encoder-decoder attention history
    att_cache = {"attention_history": {}}
    num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
    att_batch_size, enc_seq_length = common_layers.shape_list(encoder_output)[0:2]
    for layer in range(num_layers):
      att_cache["attention_history"]["layer_%d" % layer] = tf.zeros(
        [att_batch_size, hparams.num_heads, 0, enc_seq_length])
    att_cache["body_outputs"] = tf.zeros([att_batch_size, 1, 0, hparams.hidden_size])

    def update_decoder_attention_history(cache):
      for k in filter(lambda x: "decoder" in x and not "self" in x and not "logits" in x,
        self.attention_weights.keys()):
        m = re.search(r"(layer_\d+)", k)
        if m is None:
          continue
        cache["attention_history"][m[0]] = tf.concat(
            [cache["attention_history"][m[0]], self.attention_weights[k]], axis=2)

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      with tf.variable_scope("body"):
        body_outputs = dp(
            self.decode,
            targets,
            cache.get("encoder_output"),
            cache.get("encoder_decoder_attention_bias"),
            bias,
            hparams,
            cache,
            nonpadding=features_to_nonpadding(features, "targets"))

      update_decoder_attention_history(cache)
      cache["body_outputs"] = tf.concat([cache["body_outputs"], body_outputs[0]], axis=2)

      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        top = hparams.top.get("targets", modalities.get_top(target_modality))
        logits = dp(top, body_outputs, None, hparams, target_vocab_size)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      if partial_targets is not None:
        # If the position is within the given partial targets, we alter the
        # logits to always return those values.
        # A faster approach would be to process the partial targets in one
        # iteration in order to fill the corresponding parts of the cache.
        # This would require broader changes, though.
        vocab_size = tf.shape(ret)[1]

        def forced_logits():
          return tf.one_hot(
              tf.tile(partial_targets[:, i], [beam_size]), vocab_size, 0.0,
              -1e9)

        ret = tf.cond(
            tf.less(i, partial_targets_length), forced_logits, lambda: ret)
      return ret, cache

    ret = fast_decode(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_vocab_size,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha,
        batch_size=batch_size,
        force_decode_length=self._decode_hparams.force_decode_length,
        cache=att_cache)

    if partial_targets is not None:
      if beam_size <= 1 or top_beams <= 1:
        ret["outputs"] = ret["outputs"][:, partial_targets_length:]
      else:
        ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
    return ret

  def estimator_spec_predict(self, features, use_tpu=False):
    """Constructs `tf.estimator.EstimatorSpec` for PREDICT (inference) mode."""
    decode_hparams = self._decode_hparams
    top_beams = decode_hparams.beam_size if decode_hparams.return_beams else 1
    infer_out = self.infer(
        features,
        beam_size=decode_hparams.beam_size,
        top_beams=top_beams,
        alpha=decode_hparams.alpha,
        decode_length=decode_hparams.extra_length,
        use_tpu=use_tpu)
    if isinstance(infer_out, dict):
      outputs = infer_out["outputs"]
      scores = infer_out["scores"]
      attention_history = infer_out["cache"]["attention_history"]
      body_outputs = {"body_outputs": infer_out["cache"]["body_outputs"]}
    else:
      outputs = infer_out
      scores = None
      attention_history = {}
      body_outputs = {}

    inputs = features.get("inputs")
    if inputs is None:
      inputs = features["targets"]

    predictions = {
        "outputs": outputs,
        "scores": scores,
        "inputs": inputs,
        "targets": features.get("infer_targets")
    }
    predictions.update(attention_history)
    predictions.update(body_outputs)

    # Pass through remaining features
    for name, feature in features.items():
      if name not in list(predictions.keys()) + ["infer_targets"]:
        if name == "decode_loop_step":
          continue
        if not feature.shape.as_list():
          # All features must have a batch dimension
          batch_size = common_layers.shape_list(outputs)[0]
          feature = tf.tile(tf.expand_dims(feature, 0), [batch_size])
        predictions[name] = feature

    _del_dict_non_tensors(predictions)

    export_out = {"outputs": predictions["outputs"]}
    if "scores" in predictions:
      export_out["scores"] = predictions["scores"]

    export_out.update(attention_history)
    export_out.update(body_outputs)

    # Necessary to rejoin examples in the correct order with the Cloud ML Engine
    # batch prediction API.
    if "batch_prediction_key" in predictions:
      export_out["batch_prediction_key"] = predictions["batch_prediction_key"]

    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.estimator.export.PredictOutput(export_out)
    }
    if use_tpu:
      # Note: important to call this before remove_summaries()
      if self.hparams.tpu_enable_host_call:
        host_call = self.create_eval_host_call()
      else:
        host_call = None

      remove_summaries()

      return tf.contrib.tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          host_call=host_call,
          export_outputs=export_outputs)
    else:
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs=export_outputs)
