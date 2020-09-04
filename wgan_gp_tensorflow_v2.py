import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import cv2
from numpy import newaxis
import pickle
import input_data
from tensorflow.python.framework import dtypes
#import tensorflow.compat.v2 as tf
#tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


data = []

crop_img_size = 64

# Load in the images
if not os.path.isfile('slike_np.p'):
    for filepath in os.listdir('./Slike/'):
        img = cv2.imread('Slike/{0}'.format(filepath), 0)
        height = img.shape[0]
        width = img.shape[1]
        crop_img=img

        #crop_img = img[int(0.1 * height):int(0.9 * height), int(0.1 * width):int(0.9 * width)]
        #resized = cv2.resize(crop_img, (crop_img.shape[0], crop_img.shape[1]), interpolation=cv2.INTER_AREA)
        resized = cv2.resize(crop_img, (crop_img_size, crop_img_size), interpolation=cv2.INTER_AREA)
        height_resized = resized.shape[0]
        width_resized = resized.shape[1]
        #resized = cv2.line(resized, (0, int(0.2 * height)), (int(0.5 * width), 0), (0, 255, 0) , 9)
        pts = np.array([[0, int(0.4 * height_resized)], [int(0.59 * width), 0], [0,0]], np.int32)
        cv2.fillPoly(resized, [pts], color=[0, 0, 0])

        pts = np.array([[width_resized, int(0.2 * height_resized)], [int(0.4 * width_resized), 0], [width_resized, 0]], np.int32)
        cv2.fillPoly(resized, [pts], color=[0, 0, 0])

        pts = np.array([[0, int(0.6 * height_resized)], [int(0.6 * width_resized), height_resized], [0, height_resized]], np.int32)
        cv2.fillPoly(resized, [pts], color=[0, 0, 0])

        pts = np.array([[int(0.4 * width_resized), height_resized], [width_resized, int(0.6 * height_resized)], [width_resized, height_resized]], np.int32)
        cv2.fillPoly(resized, [pts], color=[0, 0, 0])

        #plt.imshow(resized, cmap='gray')
        #plt.show()

        data.append(resized)
    data = np.array(data)
    data = data[:, :, :, newaxis]
    print(data.shape)
    pickle.dump(data, open('slike_np.p', 'wb'), protocol=2 )
    print("Sacuvan numpy slika!")
else:
    data = pickle.load( open('slike_np.p', "rb" ) )
    print("Ucitan numpy red slika!")

print(data.shape)

options = dict(dtype=dtypes.float32, reshape=True, seed=None)
train = input_data._DataSet(data, np.ones(data.shape[0]), **options)
input = input_data._Datasets(train=train, validation=None, test=None)



img_size = data.shape[1]
print(img_size)

mb_size = 32
X_dim = img_size*img_size
z_dim = 10
h_dim = img_size
lam = 10
n_disc = 5
lr = 1e-4


def plot(samples):
    fig = plt.figure(figsize=(1, 1))
    #gs = gridspec.GridSpec(4, 4)
    #gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        #ax = plt.subplot(gs[i])
        plt.axis('off')
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #ax.set_aspect('equal')
        plt.imshow(sample.reshape(img_size, img_size), cmap='Greys_r')
        fig.set_size_inches(2.1, 2.1)

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random.normal(shape=size, stddev=xavier_stddev)


X = tf.compat.v1.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W3 = tf.Variable(xavier_init([h_dim, 1]))
D_b3 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2, D_W3, D_b3]

z = tf.compat.v1.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W3 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b3 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2, G_W3, G_b3]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def G(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, 2*G_W2) + G_b2)
    G_log_prob = tf.matmul(G_h2, G_W3) + G_b3
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def D(X):
    D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, 2*D_W2) + D_b2)
    out = tf.matmul(D_h2, D_W3) + D_b3
    return out


G_sample = G(z)
D_real = D(X)
D_fake = D(G_sample)

eps = tf.random.uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*X + (1. - eps)*G_sample
grad = tf.gradients(ys=D(X_inter), xs=[X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum(input_tensor=(grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean(input_tensor=(grad_norm - 1)**2)

D_loss = tf.reduce_mean(input_tensor=D_fake) - tf.reduce_mean(input_tensor=D_real) + grad_pen
G_loss = -tf.reduce_mean(input_tensor=D_fake)

global_step = tf.Variable(0, name='global_step', trainable=False)

D_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(D_loss, var_list=theta_D, global_step=global_step))
G_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(G_loss, var_list=theta_G, global_step=global_step))


sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

saver = tf.compat.v1.train.Saver()

if not os.path.exists('out/'):
    os.makedirs('out/')

try:
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint(os.path.curdir + "/checkpoints/."))
    print("Model restored.")
except:
    print("No model saved, starting from scratch...")


if not os.path.exists('checkpoints/'):
    os.makedirs('checkpoints/')

i = global_step.eval(session=sess)

for it in range(i, 1000000):

    for _ in range(n_disc):
        X_mb, _ = input.train.next_batch(mb_size)
        _, D_loss_curr = sess.run(
            [D_solver, D_loss],
            feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
        )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: sample_z(mb_size, z_dim)}
    )

    #saving checkpoint
    if it % 100 == 0:
        save_path = saver.save(sess, os.path.curdir + "/checkpoints/model.ckpt", global_step=global_step, write_meta_graph=False)
        print("Model saved in path: %s" % save_path)

    if it % 100 == 0:
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))

        if it % 100 == 0:
            samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'
                        .format(str(it).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)