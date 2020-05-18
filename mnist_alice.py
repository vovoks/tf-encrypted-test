import tf_encrypted as tfe
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import random as ran


def provide_data(features):
    dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.repeat()
    dataset = dataset.batch(10)
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    batch = tf.reshape(batch, [10,784])
    return batch


remote_config = tfe.RemoteConfig.load("config.json")
tfe.set_config(remote_config)

tfe.set_protocol(tfe.protocol.Pond())
players = remote_config.players
server0 = remote_config.server(players[0].name)

tfe.set_protocol(tfe.protocol.Pond(
    tfe.get_config().get_player("alice"),
    tfe.get_config().get_player("bob")
))

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_data = mnist.train.images[:100, :]
train_labels = mnist.train.labels[:100]

x_train_0 = tfe.define_private_input(
    "alice",
    lambda: provide_data(train_data)
)

