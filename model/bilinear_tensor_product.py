from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer
import numpy as np

class BilinearTensorProduct(Layer):

  def __init__(self, output_dim, input_dim=300, **kwargs):
    """
    @param output_dim : size of the output vector
    @param input_dim : size of the word embedding input vector
    """
    self.output_dim = output_dim # k
    self.input_dim = input_dim # d
    super(BilinearTensorProduct, self).__init__(**kwargs)

  def build(self, input_shape):
    """
    @param input_shape : Should be (batch_size, 2, d)
    where d is the length of word embedding vectors
    """
    # assert self.input_dim == input_shape[2], "Error: input_dim does not match input_shape[2]"
    k = self.output_dim
    d = self.input_dim
    # print "Input shape:", input_shape
    # self.batch_size = input_shape[0]
    initializer = initializers.RandomNormal(mean=0.0, stddev=1, seed=None)

    # Tensor weights
    self.T = self.add_weight(name='tensor_weights', 
                            shape=(k, d, d),
                            initializer=initializer,
                            trainable=True)

    # Standard feedforward weights
    self.W = self.add_weight(name='feedforward_weights',
                            shape=(2*d, k),
                            initializer=initializer,
                            trainable=True)

    # Bias weights
    self.b = self.add_weight(name='bias_weights',
                            shape=(d,),
                            initializer=initializer,
                            trainable=True)

    self.built = True

  def call(self, inputs):
    """
    @param inputs : [vector 1 (d), vector 2(d)]
    """
    k = self.output_dim
    d = self.input_dim

    if type(inputs) is not list or len(inputs) != 2:
      raise Exception("Expected list of two vectors as input to BilinearTensorLayer.")

    V1 = inputs[0] # length d
    V2 = inputs[1] # length d

    self.batch_size = K.shape(V1)[0]
    feedforward_product = K.dot(K.concatenate([V1, V2]), self.W)

    # must infer the shape here, cannot set explicitly
    bilinear_tensor_product = [ K.sum((V2 * K.dot(V1, self.T[0])) + self.b, axis=1) ]
    for i in range(k)[1:]:
      bilinear_tensor_product.append(K.sum((V2 * K.dot(V1, self.T[i])) + self.b, axis=1))

    result = K.tanh(feedforward_product + bilinear_tensor_product)
    return result

  def compute_output_shape(self, input_shape):
    batch_size = input_shape[0][0]
    return (batch_size, self.output_dim)
