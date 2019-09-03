"""Define inputters reading from raw embedding output files."""

import tensorflow as tf
import numpy as np

from opennmt.inputters.inputter import Inputter


class EmbeddingTextInputter(Inputter):
  """Inputter that reads variable-length tensors.

  Each record contains the following fields:

   * ``shape``: the shape of the tensor as a ``int64`` list.
   * ``values``: the flattened tensor values as a :obj:`dtype` list.

  Tensors are expected to be of shape ``[time, depth]``.
  """

  def __init__(self, dtype=tf.float32):
    """Initializes the parameters of the record inputter.

    Args:
      dtype: The values type.
    """
    super(EmbeddingTextInputter, self).__init__(dtype=dtype)
    self._cached = {}

  def read_data(self, data_file):
    def gen():
      if data_file not in self._cached:
        dataset = []
        with open(data_file, 'r') as f:
          for line in f:
            dataset.append(np.array([float(x) for x in line.strip().split()], dtype=np.float32))
        self._cached[data_file] = dataset
      for elem in self._cached[data_file]:
        yield elem
    return gen

  def make_dataset(self, data_file, training=None):
    data = np.array([i for i in self.read_data(data_file)()])
    self.input_depth = data.shape[-1]
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(data))
    return dataset

  def get_dataset_size(self, data_file):
    return sum(1 for _ in self.read_data(data_file)())

  def get_receiver_tensors(self):
    return {
        "tensor": tf.placeholder(self.dtype, shape=(None, None, self.input_depth)),
        "length": tf.placeholder(tf.int32, shape=(None,))
    }

  def make_features(self, element=None, features=None, training=None):
    if features is None:
      features = {}
    if "tensor" in features:
      return features
    features["tensor"] = element
    features["length"] = 1
    return features

  def make_inputs(self, features, training=None):
    return features["tensor"]

