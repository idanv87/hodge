import os
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

from utils import MultiHeadAttention, train_step

input_seq_length = 5  # Maximum length of the input sequence
h = 1  # Number of self-attention heads
d_k = 2  # Dimensionality of the linearly projected queries and keys
d_v = 2  # Dimensionality of the linearly projected values
d_model = 2  # Dimensionality of the model sub-layers' outputs
batch_size = 64  # Batch size from the training process


queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))
y=queries
x=keys


class my_model(tf.keras.Model):
  def __init__(self, h, d_k, d_v, d_model):
    super(my_model, self).__init__()
    self.layer1= MultiHeadAttention(h, d_k, d_v, d_model)
  def call(self,x):
    return self.layer1(x,x,x)  

L=my_model(h, d_k, d_v, d_model)
L.optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)

epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * batch_size))

    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)



# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
# with tf.GradientTape() as tape:
#   loss=tf.keras.losses.MeanSquaredError()(L(x),y)
#   print(loss)
# gradients = tape.gradient(loss, L.trainable_variables)
# # print(loss)
# optimizer.apply_gradients(zip(gradients, L.trainable_variables))
# loss=tf.keras.losses.MeanSquaredError()(L(x),y)
# print(loss)
# # print(x1[5]-x2[5])

# print(np.array(x1))


# train_loss = tf.keras.metrics.Mean(name='train_loss')

# def create_checkpoint(model_name, epochs):
#   checkpoint_dir = os.path.join('/content/drive/My Drive/nlp_checkpoints', model_name)
#   checkpoint_prefix = os.path.join(checkpoint_dir, str(epochs) + "_epochs", str(epochs) + "_epochs_cp.ckpt")

#   checkpoint_callback = ModelCheckpoint(
#       filepath=checkpoint_prefix,
#       save_weights_only=True,
#       save_best_only=True,
#       monitor='val_loss',
#       mode='min',
#       verbose=1)
#   return checkpoint_callback, checkpoint_prefix

# def train_step(X,Y,model):
#   with tf.GradientTape() as tape:
#     loss = model.loss_function(model(X), Y)
#   gradients = tape.gradient(loss, model.trainable_variables)
#   model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))





# class layer(tf.keras.layers.Layer):
#     def __init__(self,n):
#      super(layer,self).__init__()
#      self.layer1=tf.keras.layers.Dense(n, input_shape=(n,), activation='relu',use_bias=False)
#     def call(self, x):
#       return self.layer1(x)

# class my_model(tf.keras.Model):
#   def __init__(self, n):
#     super(my_model, self).__init__()
#     self.layer1=layer(n)
#   def call(self,x):
#     return self.layer1(x)  
 
# L=my_model(3)
# x=np.array([[1,2,4], [1,2,4]])
# y=np.array([[1,2,3], [1,2,4]])

# x=x.reshape((2,3,1))
# y=y.reshape((2,3,1))

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# with tf.GradientTape() as tape:

#  loss=tf.keras.losses.MeanSquaredError()(y[0,:,:],L((x[0,:,:])))

# gradients = tape.gradient(loss, L.trainable_variables)
# # print(loss)
# print(L.trainable_weights)
# optimizer.apply_gradients(zip(gradients, L.trainable_variables))
# print(L.trainable_weights)

  
  



# # # print(tf.keras.layers.Dense(10, activation='relu')(x))
# # # print(L.call(x))
# # print(L(x))
# # epochs = 2
# # for epoch in range(epochs):
# #     print("\nStart of epoch %d" % (epoch,))
# #     start_time = time.time()

# #     # Iterate over the batches of the dataset.
# #     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
# #         loss_value = train_step(x_batch_train, y_batch_train)

# #         # Log every 200 batches.
# #         if step % 200 == 0:
# #             print(
# #                 "Training loss (for one batch) at step %d: %.4f"
# #                 % (step, float(loss_value))
# #             )
# #             print("Seen so far: %d samples" % ((step + 1) * batch_size))

# #     # Run a validation loop at the end of each epoch.
# #     for x_batch_val, y_batch_val in val_dataset:
# #         test_step(x_batch_val, y_batch_val)

# #     val_acc = val_acc_metric.result()
# #     val_acc_metric.reset_states()
# #     print("Validation acc: %.4f" % (float(val_acc),))
# #     print("Time taken: %.2fs" % (time.time() - start_time))
  