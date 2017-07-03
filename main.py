#Import the libraries we will need.
import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy
import random

from os import walk

TRAIN = True

image_directory = os.path.dirname(os.path.realpath(__file__)) + '/1_ad' #Directory to save sample images from generator in.
sample_directory = os.path.dirname(os.path.realpath(__file__)) + '/images' #Directory to save sample images from generator in.
model_directory = os.path.dirname(os.path.realpath(__file__)) + '/models' #Directory to save trained model to.

#This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)
    
#The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
#They allow for saving sample images from the generator to follow progress
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2. * 255.0

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img

def read_image(file_path):
    image_array = scipy.misc.imread(file_path)
    return image_array

def rescale_image(image_data):
     return image_data.astype(dtype=np.uint8) / 255.

### Generator network
def generator(z):
    zP = slim.fully_connected(z,4*4*256,normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_project',weights_initializer=initializer)
    zCon = tf.reshape(zP,[-1,4,4,256])
    
    gen1 = slim.convolution2d_transpose(\
        zCon,num_outputs=64,kernel_size=[5,5],stride=[2,2],\
        padding="SAME",normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_conv1', weights_initializer=initializer)
    
    gen2 = slim.convolution2d_transpose(\
        gen1,num_outputs=32,kernel_size=[5,5],stride=[2,2],\
        padding="SAME",normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_conv2', weights_initializer=initializer)
    
    gen3 = slim.convolution2d_transpose(\
        gen2,num_outputs=16,kernel_size=[5,5],stride=[2,2],\
        padding="SAME",normalizer_fn=slim.batch_norm,\
        activation_fn=tf.nn.relu,scope='g_conv3', weights_initializer=initializer)
    
    g_out = slim.convolution2d_transpose(\
        gen3,num_outputs=3,kernel_size=[32,32],padding="SAME",\
        biases_initializer=None,activation_fn=tf.nn.tanh,\
        scope='g_out', weights_initializer=initializer)
    
    return g_out

### Descriminator network
def discriminator(bottom, reuse=False):
    
    dis1 = slim.convolution2d(bottom,16,[4,4],stride=[2,2],padding="SAME",\
        biases_initializer=None,activation_fn=lrelu,\
        reuse=reuse,scope='d_conv1',weights_initializer=initializer)
    
    dis2 = slim.convolution2d(dis1,32,[4,4],stride=[2,2],padding="SAME",\
        normalizer_fn=slim.batch_norm,activation_fn=lrelu,\
        reuse=reuse,scope='d_conv2', weights_initializer=initializer)
    
    dis3 = slim.convolution2d(dis2,64,[4,4],stride=[2,2],padding="SAME",\
        normalizer_fn=slim.batch_norm,activation_fn=lrelu,\
        reuse=reuse,scope='d_conv3',weights_initializer=initializer)
    
    d_out = slim.fully_connected(slim.flatten(dis3),1,activation_fn=tf.nn.sigmoid,\
        reuse=reuse,scope='d_out', weights_initializer=initializer)
    
    return d_out

### GAN
tf.reset_default_graph()

z_size = 100 #Size of z vector used for generator.

#This initializaer is used to initialize all the weights of the network.
initializer = tf.truncated_normal_initializer(stddev=0.02)

#These two placeholders are used for input into the generator and discriminator, respectively.
z_in = tf.placeholder(shape=[None,z_size],dtype=tf.float32) #Random vector
real_in = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32) #Real images

Gz = generator(z_in) #Generates images from random z vectors
Dx = discriminator(real_in) #Produces probabilities for real images
Dg = discriminator(Gz,reuse=True) #Produces probabilities for generator images

#These functions together define the optimization objective of the GAN.
d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) #This optimizes the discriminator.
g_loss = -tf.reduce_mean(tf.log(Dg)) #This optimizes the generator.

tvars = tf.trainable_variables()

#The below code is responsible for applying gradient descent to update the GAN.
trainerD = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
trainerG = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
d_grads = trainerD.compute_gradients(d_loss,tvars[9:]) #Only update the weights for the discriminator network.
g_grads = trainerG.compute_gradients(g_loss,tvars[0:9]) #Only update the weights for the generator network.

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerG.apply_gradients(g_grads)

image_data_pool = []

### Training
if TRAIN:
    batch_size = 128 #Size of image batch to apply at each iteration.
    iterations = 100001 #Total number of iterations to use.

    f = []
    for (dirpath, dirnames, filenames) in walk(image_directory):
        f.extend(filenames)
        break

    for filename in f:
        image_data = read_image(image_directory +'/' + filename)
        if image_data.shape[2] == 4:
            r = image_data[:,:,0]
            g = image_data[:,:,1]
            b = image_data[:,:,2]
            image_data = np.stack((r, g, b), axis=2)

        image_data = scipy.misc.imresize(image_data, [28, 28, 3])
        image_data = rescale_image(image_data)
        image_data_pool.append(image_data)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:  
        sess.run(init)
        for i in range(iterations):
            zs = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) #Generate a random z batch

            xs = random.sample(image_data_pool, batch_size)
            xs = np.array(xs)

            #xs,_ = mnist.train.next_batch(batch_size) #Draw a sample batch from MNIST dataset.
            xs = (np.reshape(xs, [-1, 28, 28, 3]) - 0.5) * 2.0 #Transform it to be between -1 and 1
            xs = np.lib.pad(xs, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32
            _,dLoss = sess.run([update_D,d_loss],feed_dict={z_in:zs,real_in:xs}) #Update the discriminator
            _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs}) #Update the generator, twice for good measure.
            _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs})
            if i % 500 == 0:
                print "Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss)
                z2 = np.random.uniform(-1.0,1.0,size=[1,z_size]).astype(np.float32) #Generate another z batch
                newZ = sess.run(Gz,feed_dict={z_in:z2}) #Use new z to get sample images from generator.
                if not os.path.exists(sample_directory):
                    os.makedirs(sample_directory)
                #Save sample generator images for viewing training progress.
                scipy.misc.imsave(sample_directory+'/fig'+str(i)+'.png', inverse_transform(newZ[0]))
            if i % 500 == 0 and i != 0:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess,model_directory+'/model-'+str(i)+'.cptk')
                print "Saved Model"

###
batch_size_sample = 1
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:  
    sess.run(init)
    #Reload the model.
    print 'Loading Model...'
    ckpt = tf.train.get_checkpoint_state(model_directory)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Successfully loaded network weights')
    else:
        print('Could not find old network weights')
    
    zs = np.random.uniform(-1.0,1.0,size=[batch_size_sample,z_size]).astype(np.float32) #Generate a random z batch
    newZ = sess.run(Gz,feed_dict={z_in:zs}) #Use new z to get sample images from generator.

    if not os.path.exists(sample_directory):
        os.makedirs(sample_directory)

    scipy.misc.imsave(sample_directory + '/result.png', inverse_transform(newZ[0]))

    #save_images(np.reshape(newZ[0:batch_size_sample],[1,32,32]),[1,1],sample_directory+'/fig.png')