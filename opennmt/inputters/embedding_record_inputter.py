import tensorflow as tf

from opennmt.inputters.record_inputter import SequenceRecordInputter

class EmbeddingRecordInputter(SequenceRecordInputter):

  def make_features(self, element=None, features=None, training=None):
    if features is None:
      features = {}
    if "tensor" in features:
      return features
    example = tf.parse_single_example(element, features={
        "shape": tf.VarLenFeature(tf.int64),
        "values": tf.VarLenFeature(tf.float32)
    })
    values = example["values"].values
    shape = tf.cast(example["shape"].values, tf.int32)
    tensor = tf.reshape(values, shape)
    tensor.set_shape([self.input_depth])
    features["length"] = tf.constant(1)
    features["tensor"] = tf.cast(tensor, self.dtype)
    return features


