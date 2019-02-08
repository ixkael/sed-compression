
import numpy as np
import tensorflow as tf

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0)

def next_batch(params, models, i_batch, batch_size):
    batch_params = params[i_batch*batch_size: (i_batch + 1)*batch_size, :]
    batch_models = models[i_batch*batch_size: (i_batch + 1)*batch_size, :]
    return batch_params, batch_models

def leaky(input):
    return tf.nn.leaky_relu(input)

def drop_out(input, keep_prob):
    return tf.nn.dropout(input, keep_prob)

def dense(inputs, units, name):
    return tf.layers.dense(inputs=inputs, units=units, reuse=tf.AUTO_REUSE,
                           name=name, kernel_initializer=initializer)

def neural_network(input, n_out, num_hiddens, keep_prob, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        net = input
        for i, n_hidden in enumerate(num_hiddens):
            net = drop_out(leaky(dense(net, n_hidden, name=name+"_dense_"+str(i+1))), keep_prob)
    return dense(net, n_out, name =name+"_dense_end")


def train_encoder(train_params, train_models, valid_params, valid_models,
    num_latent, num_hiddens, num_epoch, learning_rate, batch_size, keep_prob):

    num_total_batches = len(train_params) // batch_size

    Params_in = tf.placeholder(dtype=tf.float32, shape=[None, train_params.shape[1]], name ="params_in")
    Model_in = tf.placeholder(dtype=tf.float32, shape=[None, train_models.shape[1]], name ="model_in")
    Keep_prob = tf.placeholder(dtype=tf.float32, name = "drop_rate")

    Latent = neural_network(Params_in, num_latent, num_hiddens, Keep_prob, name="encoder")
    Model_out = tf.nn.softplus(neural_network(Latent, train_models.shape[1], num_hiddens, Keep_prob, name="decoder"))
    Likes = -0.5*tf.square(Model_out - Model_in)
    Likelihood = tf.reduce_mean(tf.reduce_sum(Likes, axis=1))

    Recon_error = - Likelihood
    Regul_error = 0
    ELBO_loss = Recon_error + Regul_error

    Optim_op = tf.train.AdamOptimizer(learning_rate=learning_rate)
    Train_op = Optim_op.minimize(ELBO_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_epoch):
            loss_total = 0
            for j in range(num_total_batches):
                train_params_batch, train_models_batch = next_batch(train_params, train_models, j, batch_size)
                loss_batch, mod, op = sess.run([ELBO_loss, Model_out, Train_op],
                    feed_dict={Params_in: train_params_batch, Model_in: train_models_batch, Keep_prob: keep_prob})
                loss_total += loss_batch / num_total_batches
                if not np.isfinite(loss_batch):
                    print(i, j, 'NAN loss!', loss_total)
                    stop
            if i % 100 == 0:
                print(i, '%.3e' % loss_total, end=" ; ")

            train_latent, train_models_out = sess.run([Latent, Model_out],
                feed_dict={Params_in: train_params, Model_in: train_models, Keep_prob: keep_prob})
            valid_latent, valid_models_out = sess.run([Latent, Model_out],
                feed_dict={Params_in: valid_params, Model_in: valid_models, Keep_prob: keep_prob})

    return train_latent, train_models_out, valid_latent, valid_models_out
