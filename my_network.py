import tensorflow as tf
import numpy as np
# from main import WIDTH, WIDTH_label,HEIGHT,HEIGHT_label


WIDTH = 60
HEIGHT = 36
WIDTH_label = 20
HEIGHT_label = WIDTH_label


def Fully_connected(HEIGHT_in,WIDTH_in,HEIGHT_out,WIDTH_out, apply_batchnorm=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Reshape((HEIGHT_in*WIDTH_in,)))

  result.add(
      tf.keras.layers.Dense(HEIGHT_out*WIDTH_out,input_shape=(WIDTH_in*HEIGHT_in,),activation='sigmoid',kernel_initializer=initializer))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
    
  result.add(tf.keras.layers.Reshape((HEIGHT_out,WIDTH_out,1)))

  return result



# --------------------------------VAE----------------------------
class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits



# ----------------------------------losses----------------------------




def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)



def generator_loss(disc_gen_generated_output, target_gen):
    
  loss=tf.math.reduce_mean(tf.keras.losses.MSE(disc_gen_generated_output, target_gen))
  return loss
  


def discriminator_loss( disc_generated_output, target_disc):
    
  loss=tf.math.reduce_mean(tf.keras.losses.MSE(disc_generated_output, target_disc))

  return loss

def generator_corr(logits,labels):
    
  m1=tf.reduce_mean(logits)
  m2=tf.reduce_mean(labels)
  n1=tf.reduce_mean(tf.multiply(tf.subtract(logits,m1),tf.subtract(labels,m2)))
  dn1=tf.sqrt((tf.reduce_mean(tf.math.squared_difference(logits, m1))))
  dn2=tf.sqrt(tf.reduce_mean(tf.math.squared_difference(labels, m2)))
  return tf.divide(n1,dn1*dn2,name=None)
