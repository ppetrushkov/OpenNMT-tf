"""Sequence classifier."""

import tensorflow as tf

from opennmt import inputters
from opennmt.models.model import Model
from opennmt.utils.cell import last_encoding_from_state
from opennmt.utils.misc import print_bytes
from opennmt.utils.losses import cross_entropy_loss


class EmbeddingClassifier(Model):
  """A sequence classifier."""

  def __init__(self,
               inputter,
               labels_vocabulary_file_key,
               daisy_chain_variables=False,
               hidden_sizes=None,
               dropout=0.0,
               name="seqclassifier"):
    """Initializes a sequence classifier.

    Args:
      inputter: A :class:`opennmt.inputters.inputter.Inputter` to process the
        input data.
      encoder: A :class:`opennmt.encoders.encoder.Encoder` to encode the input.
      labels_vocabulary_file_key: The data configuration key of the labels
        vocabulary file containing one label per line.
      encoding: "average" or "last" (case insensitive), the encoding vector to
        extract from the encoder outputs.
      daisy_chain_variables: If ``True``, copy variables in a daisy chain
        between devices for this model. Not compatible with RNN based models.
      name: The name of this model.

    Raises:
      ValueError: if :obj:`encoding` is invalid.
    """
    super(EmbeddingClassifier, self).__init__(
        name,
        features_inputter=inputter,
        labels_inputter=ClassInputter(labels_vocabulary_file_key),
        daisy_chain_variables=daisy_chain_variables)
    self.hidden_sizes = hidden_sizes
    self.dropout = dropout

  #def _build(self, features, labels, params, mode, config=None):
  def _call(self, features, labels, params, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope("encoder"):
      inputs = self.features_inputter.transform_data(
          features,
          mode=mode) 
    input_size = inputs.get_shape().as_list()[-1]

    if self.dropout > 0.0:
      inputs = tf.layers.dropout(inputs, self.dropout, training=training)

    # Build hidden layers
    if self.hidden_sizes:
      with tf.variable_scope("network", initializer=self._initializer(params)):
        for hidden_size in self.hidden_sizes:
          layer = tf.layers.Dense(
              hidden_size,
              activation=tf.tanh,
              use_bias=True, dtype=self.dtype,
              kernel_initializer=self._initializer(params))
          layer.build([None, input_size])
          input_size = hidden_size
          inputs = layer(inputs)
          if self.dropout > 0.0:
            inputs = tf.layers.dropout(inputs, self.dropout, training=training)

    with tf.variable_scope("generator", initializer=self._initializer(params)):
      output_layer = tf.layers.Dense(
              self.labels_inputter.vocabulary_size,
              use_bias=True, dtype=self.dtype,
              kernel_initializer=self._initializer(params))
      output_layer.build([None, input_size])
      logits = output_layer(inputs)

    if mode != tf.estimator.ModeKeys.TRAIN:
      labels_vocab_rev = self.labels_inputter.vocabulary_lookup_reverse()
      classes_prob = tf.nn.softmax(logits)
      classes_id = tf.argmax(classes_prob, axis=1)
      predictions = {
          "classes": labels_vocab_rev.lookup(classes_id)
      }
    else:
      predictions = None

    return logits, predictions

  def _compute_loss(self, features, labels, outputs, params, mode):
    return cross_entropy_loss(
        outputs,
        labels["classes_id"],
        label_smoothing=params.get("label_smoothing", 0.0),
        mode=mode)

  def compute_loss(self, outputs, labels, training=True, params=None):
    return cross_entropy_loss(
        outputs,
        labels["classes_id"],
        label_smoothing=params.get("label_smoothing", 0.0),
        training=training)

  def _compute_metrics(self, features, labels, predictions):
    return {
        "accuracy": tf.metrics.accuracy(labels["classes"], predictions["classes"])
    }

  def auto_config(self, num_devices=1):
      return {"params": {"minimum_learning_rate": "0.0001"} }

  def print_prediction(self, prediction, params=None, stream=None):
    print_bytes(prediction["classes"], stream=stream)


class ClassInputter(inputters.TextInputter):
  """Reading class from a text file."""

  def __init__(self, vocabulary_file_key):
    super(ClassInputter, self).__init__(
        vocabulary_file_key=vocabulary_file_key, num_oov_buckets=0)

  def make_features(self, element=None, features=None, training=None):
    return {
        "classes": element,
        "classes_id": self.vocabulary.lookup(element)
    }
