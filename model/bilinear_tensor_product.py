from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer
import numpy as np

class BilinearTensorProduct(Layer):

  def __init__(self, output_dim, **kwargs):
    """
    @param output_dim : size of the output vector
    @param input_dim : size of the word embedding input vector
    """
    self.output_dim = output_dim # k
    super(BilinearTensorLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    """
    @param input_shape : Should be (2, d, d)
    where d is the length of word embedding vectors
    """
    self.input_dim = input_shape[1] # d
    assert input_shape[0] == 2, "Error: input should have two vectors."
    assert input_shape[1] == input_shape[2], "Error: input vectors should be of equal length."

    k = self.output_dim
    d = self.input_dim
    initalizer = initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
    self.T = self.add_weight(name='tensor_weights', 
                            shape=(k, d, d),
                            initializer=initializer,
                            trainable=True)

    super(BilinearTensorLayer, self).build(input_shape)

  def call(self, inputs):
    """
    @param inputs : [OBJECT (length d vector), ACTION (length d vector)]
    """
    k = self.output_dim
    d = self.input_dim
    V1 = inputs[0] # OBJECT
    V2 = inputs[1] # ACTION

    if type(inputs) is not np.ndarray or len(inputs) != 2:
      raise Exception("Expected list of two vectors as input to BilinearTensorLayer.")
    
    # output is a length k vector
    output = [np.dot(np.dot(V1, self.T[slice_idx]), V2.transpose) for slice_idx in range(len(k))]
    return output

  def compute_output_shape(self, input_shape):
    batch_size = input_shape[0]
    return (batch_size, self.output_dim)