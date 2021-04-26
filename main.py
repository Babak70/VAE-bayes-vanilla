import tensorflow as tf
import datetime
import os
import time
from matplotlib import pyplot as plt
from IPython import display
from read_file import load
from my_network import compute_loss,CVAE
import numpy as np



WIDTH = 60
HEIGHT = 36
WIDTH_label = 20
HEIGHT_label = WIDTH_label


PATH = []
BATCH_SIZE = 128
BATCH_SIZE_test = 1000
EPOCHS = 1000
num_train_D=1
num_train_G=1
BUFFER_SIZE = 512
log_dir="logs/"
record_bytes_images=WIDTH *HEIGHT 
record_bytes_labels=WIDTH_label *HEIGHT_label 



latent_dim = 2
num_examples_to_generate = 16


#Train
data_files=['./train_labelsF.bin','./train_dataF.bin']
record_bytes=[record_bytes_labels,record_bytes_images]
first_dim=[HEIGHT_label,HEIGHT]

#Test
data_files_test=['./eval_labelsF.bin','./eval_dataF.bin']
record_bytes_test=[record_bytes_labels,record_bytes_images]
first_dim_test=[HEIGHT_label,HEIGHT]




# #Train dataset pipeline
# NUM_files=len(data_files)
# DATASETS=[None]*NUM_files
# for i, data_file in enumerate(data_files):
#   train_dataset_portion = tf.data.FixedLengthRecordDataset(data_file, record_bytes=record_bytes[i])
#   train_dataset_portion = train_dataset_portion.map( lambda x: load(x,record_bytes[i],first_dim[i]),
#                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)

#   DATASETS[i]=train_dataset_portion
# train_dataset = tf.data.Dataset.zip(tuple(DATASETS)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# #Test dataset pipeline
# NUM_files_test=len(data_files_test)
# DATASETS_test=[None]*NUM_files_test

# for i, data_file in enumerate(data_files_test):
#   test_dataset_portion = tf.data.FixedLengthRecordDataset(data_file, record_bytes=record_bytes_test[i])
#   test_dataset_portion = test_dataset_portion.map( lambda x: load(x,record_bytes_test[i],first_dim_test[i]),
#                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)

#   DATASETS_test[i]=test_dataset_portion
# test_dataset = tf.data.Dataset.zip(tuple(DATASETS_test)).batch(BATCH_SIZE_test )




# __________________________________________

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)
train_size = 60000
batch_size = 32
test_size = 10000

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

# ___________________________________________




random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)


optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer)






summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(model, x, optimizer):

  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))




def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start_time = time.time()

    display.clear_output(wait=True)


    print("Epoch: ", epoch)

    # Train
    for _ in range(num_train_G):
       for train_x in train_dataset:
         train_step(model, train_x, optimizer)
       end_time = time.time()
       loss = tf.keras.metrics.Mean()
       for test_x in test_dataset:
         loss(compute_loss(model, test_x))
       elbo = -loss.result()
       display.clear_output(wait=False)
       print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
       generate_and_save_images(model, epoch, test_sample)

    
    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        end_time-start_time))
  checkpoint.save(file_prefix = checkpoint_prefix)
  
  
def generate_and_save_images(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


assert BATCH_SIZE_test >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]

generate_and_save_images(model, 0, test_sample)
fit(train_dataset, EPOCHS, test_dataset)

