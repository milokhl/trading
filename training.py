from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.optimizers import SGD

from model.bilinear_tensor_product import BilinearTensorProduct

from dataset import loadInputTensor
import numpy as np

import tensorflow as tf

def contrastive_max_margin_loss(y_true, y_pred):
	# why is loss always 1?
	loss = tf.maximum(0.0, 1 + tf.subtract(y_pred[:,1:], y_pred[:,:1])) # right - left
	return loss

def avg_cmm_loss(y_true, y_pred):
	loss = tf.maximum(0.0, 1 + tf.subtract(y_pred[:,1:], y_pred[:,:1])) # right - left
	return K.mean(loss)

def loadData(num_batches):
	"""
	Creates a giant tensor of shape (6, n, d)
	where n is the number of training examples
	and d is the dimension of word embedding vectors
	"""
	data = loadInputTensor(1)
	for i in range(num_batches)[1:]:
		data = np.concatenate((data, loadInputTensor(i)), axis=1)
	return data

# DEFINE CONSTANTS
data = loadData(100)
n = data.shape[1]
d = 300
k = 50
l = 40
batch_size = 32
num_epochs = 50

def printHyperParams():
	print "n: %d d: %d k: %d l: %d batch_size: %d num_epochs: %d" % (n, d, k, l, batch_size, num_epochs)
printHyperParams()

subj_input = Input(shape=(d,), dtype='float32', name='subj_input')
act_input = Input(shape=(d,), dtype='float32', name='act_input')
pred_input = Input(shape=(d,), dtype='float32', name='pred_input')

subj_corr_input = Input(shape=(d,), dtype='float32')
act_corr_input = Input(shape=(d,), dtype='float32')
pred_corr_input = Input(shape=(d,), dtype='float32')

# embed object/action pairs into relations
shared_btp_1 = BilinearTensorProduct(k, input_dim=d, name='shared_tensor1')
shared_btp_2 = BilinearTensorProduct(k, input_dim=d, name='shared_tensor2')

out1 = shared_btp_1([subj_input, act_input])
out2 = shared_btp_2([pred_input, act_input])

out3 = shared_btp_1([subj_corr_input, act_corr_input])
out4 = shared_btp_2([pred_corr_input, act_corr_input])

# embed relation/relation pairs into events
shared_btp_3 = BilinearTensorProduct(l, input_dim=k, name='shared_tensor3')
real_embedding = shared_btp_3([out1, out2])
corr_embedding = shared_btp_3([out3, out4])

# calculate embedding scores
shared_linear = Dense(1, input_shape=(l,), name='linear_add_layer')
real_score = shared_linear(real_embedding)
corr_score = shared_linear(corr_embedding)

# concatenate the 2 outputs into a single output
output = Concatenate(axis=1, name='concat_output')([real_score, corr_score])

model = Model(inputs=[subj_input, act_input, pred_input, subj_corr_input, act_corr_input, pred_corr_input],
							outputs=[output])

optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=optimizer,
              loss=contrastive_max_margin_loss,
              metrics=[avg_cmm_loss])

model.fit(x=[data[0], data[1], data[2], data[3], data[4], data[5]],
					y=[np.zeros((n, 2))], epochs=num_epochs, batch_size=batch_size)
