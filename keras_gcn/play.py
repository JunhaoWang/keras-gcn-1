import tensorflow as tf

tf.enable_eager_execution()

from gcn_layers import GraphConv

from tensorflow.keras.layers import Lambda

import tensorflow.keras.backend as K

import numpy as np

import networkx as nx

from tqdm import tqdm

from sklearn.decomposition import PCA

import scipy.sparse as sp

import tensorflow_probability as tfp

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized.toarray()

def make_gaussian_mixture_prior(latent_size, mixture_components):
  """Creates the mixture of Gaussians prior distribution.
  Args:
    latent_size: The dimensionality of the latent representation.
    mixture_components: Number of elements of the mixture.
  Returns:
    random_prior: A `tfd.Distribution` instance representing the distribution
      over encodings in the absence of any evidence.
  """
  if mixture_components == 1:
    # See the module docstring for why we don't learn the parameters here.
    return tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros([latent_size]),
        scale_identity_multiplier=1.0)

  loc = tf.compat.v1.get_variable(
      name="loc", shape=[mixture_components, latent_size])
  raw_scale_diag = tf.compat.v1.get_variable(
      name="raw_scale_diag", shape=[mixture_components, latent_size])
  mixture_logits = tf.compat.v1.get_variable(
      name="mixture_logits", shape=[mixture_components])

  return tfp.distributions.MixtureSameFamily(
      components_distribution=tfp.distributions.MultivariateNormalDiag(
          loc=loc,
          scale_diag=tf.nn.softplus(raw_scale_diag)),
      mixture_distribution=tfp.distributions.Categorical(logits=mixture_logits),
      name="prior")

################### toy

# input_data = [
#         [
#             [1, 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 1, 0],
#             [0, 0, 0, 1],
#         ]
#     ]
#
# input_edge = [
#     [
#         [1, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 1],
#     ]
# ]
#
# labels = [0,1,1,2]
#
# X = np.array(input_data)
# A = np.array(input_edge)
# labels = np.array(labels)
# num_node = 4


################### cora

# def load_cora():
#     num_nodes = 2708
#     num_feats = 1433
#     feat_data = np.zeros((num_nodes, num_feats))
#     labels = np.empty((num_nodes,1), dtype=np.int64)
#     node_map = {}
#     label_map = {}
#     with open("cora/cora.content") as fp:
#         for i,line in enumerate(fp):
#             info = line.strip().split()
#             feat_data[i,:] = list(map(float, info[1:-1]))
#             node_map[info[0]] = i
#             if not info[-1] in label_map:
#                 label_map[info[-1]] = len(label_map)
#             labels[i] = label_map[info[-1]]
#
#     adj_lists = defaultdict(set)
#     with open("cora/cora.cites") as fp:
#         for i,line in enumerate(fp):
#             info = line.strip().split()
#             paper1 = node_map[info[0]]
#             paper2 = node_map[info[1]]
#             adj_lists[paper1].add(paper2)
#             adj_lists[paper2].add(paper1)
#     return feat_data, labels, adj_lists
#
# feat_data, labels, adj_lists = load_cora()
# num_node = feat_data.shape[0]
# A = np.zeros((num_node, num_node))
# for i in adj_lists:
#     for j in adj_lists[i]:
#         A[i][j] = 1
#
# norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)
#
# A = preprocess_graph(A)
#
# A = np.expand_dims(A, 0)
# X = np.expand_dims(feat_data, 0)
# labels = labels.reshape(-1)

################### sbm

## diff 3
sizes = [200, 400]
self_probs = [0.5, 0.1]

scale_factor = .5
probs = np.diag(self_probs)
probs = np.array(self_probs).reshape(-1, 1) * scale_factor + probs
g = nx.stochastic_block_model(sizes, probs, directed = True, seed=0)
A = nx.to_numpy_array(g)
A = preprocess_graph(A)
A = np.expand_dims(A, 0)
X = np.eye(600)
X = np.expand_dims(X, 0)
num_node = 600
labels = np.array([0] * 200 + [1] * 400)



class GAE(tf.keras.Model):
    def __init__(self, num_node):
        super(GAE, self).__init__()
        self.num_node = num_node
        self.conv1 = GraphConv(
            units=32,
            name='GraphConv',
        )
        self.conv2 = GraphConv(
            units=16,
            name='GraphConv',
        )

        self.dot = Lambda(lambda x: tf.reshape(K.dot(x, K.transpose(x)), [-1]))

    def call(self, inputs):
        """Run the model."""
        X = inputs[0]
        A = inputs[1]
        inputs_tensor = [tf.cast(tf.convert_to_tensor(X), tf.double),
                         tf.cast(tf.convert_to_tensor(A), tf.double)]
        result = self.conv1(inputs_tensor)
        result = self.conv2([tf.cast(result, tf.double),
                             tf.cast(tf.convert_to_tensor(A), tf.double)])
        result = tf.squeeze(result, [0])
        result = tf.reshape(self.dot(result), [-1])
        return result

    def encode(self, inputs):
        X = inputs[0]
        A = inputs[1]
        inputs_tensor = [tf.cast(tf.convert_to_tensor(X), tf.double),
                         tf.cast(tf.convert_to_tensor(A), tf.double)]
        return tf.squeeze(self.conv1(inputs_tensor))

class VGAE(tf.keras.Model):
    def __init__(self, num_node):
        super(VGAE, self).__init__()
        self.num_node = num_node
        self.prior = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=tf.zeros(16), scale=1),
                        reinterpreted_batch_ndims=1)
        self.conv1 = GraphConv(
            units=32,
            name='GraphConv',
        )

        self.dense1 = tf.keras.layers.Dense(
                   tfp.layers.MultivariateNormalTriL.params_size(16),
                   activation=None
        )
        self.dense2 = tfp.layers.MultivariateNormalTriL(
            16,
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                self.prior, weight=1.0))

        self.dot = Lambda(lambda x: tf.reshape(K.dot(x, K.transpose(x)), [-1]))

    def call(self, inputs):
        """Run the model."""
        X = inputs[0]
        A = inputs[1]
        inputs_tensor = [tf.cast(tf.convert_to_tensor(X), tf.double),
                         tf.cast(tf.convert_to_tensor(A), tf.double)]
        latent = self.conv1(inputs_tensor)
        dist_params = self.dense1(latent)
        dist_sample = self.dense2(dist_params)

        result = tf.squeeze(dist_sample, [0])
        result = tf.reshape(self.dot(result), [-1])
        return result

    def encode(self, inputs):
        X = inputs[0]
        A = inputs[1]
        inputs_tensor = [tf.cast(tf.convert_to_tensor(X), tf.double),
                         tf.cast(tf.convert_to_tensor(A), tf.double)]
        latent = self.conv1(inputs_tensor)
        dist_params = self.dense1(latent)
        dist_mean = self.dense2(dist_params).mean()
        return tf.squeeze(dist_mean)


class MDGAE(tf.keras.Model):
    def __init__(self, num_node, num_cluster = 7):
        super(MDGAE, self).__init__()
        self.num_node = num_node
        self.num_cluster = num_cluster
        self.prior = make_gaussian_mixture_prior(16, self.num_cluster)
        self.prior = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=tf.zeros(16), scale=1),
                        reinterpreted_batch_ndims=1)
        self.conv1 = GraphConv(
            units=32,
            name='GraphConv',
        )

        self.dense1 = tf.keras.layers.Dense(
               tfp.layers.MixtureNormal.params_size(self.num_cluster, [16]),
               activation=None
        )
        self.dense2 = tfp.layers.MixtureNormal(self.num_cluster, [16],
                activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                    self.prior, weight=10.0)
        )

        self.dot = Lambda(lambda x: tf.reshape(K.dot(x, K.transpose(x)), [-1]))


    def call(self, inputs):
        """Run the model."""
        X = inputs[0]
        A = inputs[1]
        inputs_tensor = [tf.cast(tf.convert_to_tensor(X), tf.double),
                         tf.cast(tf.convert_to_tensor(A), tf.double)]
        latent = self.conv1(inputs_tensor)
        dist_params = self.dense1(latent)
        dist_sample = self.dense2(dist_params)

        result = tf.squeeze(dist_sample, [0])
        result = tf.reshape(self.dot(result), [-1])
        return result

    def encode(self, inputs):
        X = inputs[0]
        A = inputs[1]
        inputs_tensor = [tf.cast(tf.convert_to_tensor(X), tf.double),
                         tf.cast(tf.convert_to_tensor(A), tf.double)]
        latent = self.conv1(inputs_tensor)
        dist_params = self.dense1(latent)
        dist_mean = self.dense2(dist_params).mean()
        return tf.squeeze(dist_mean)


pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()

gae = GAE(num_node)

# print(gae([X,A]))
# print(tf.reshape(gae([X,A]), [num_node, num_node]))

optimizer = tf.train.AdamOptimizer()

loss_history = []

for epoch in tqdm(range(2000)):
    with tf.GradientTape() as tape:
        logits = gae([X, A])
        loss_value = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.cast(tf.convert_to_tensor(A.reshape(-1)), tf.double),
            logits=tf.cast(tf.convert_to_tensor(logits), tf.double),
            pos_weight=pos_weight
        ))

    loss_history.append(loss_value.numpy())
    grads = tape.gradient(loss_value, gae.trainable_variables)
    optimizer.apply_gradients(zip(grads, gae.trainable_variables),
                            global_step=tf.train.get_or_create_global_step())
print(gae([X,A]))
print(tf.reshape(gae([X,A]), [num_node, num_node]))

import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()

embeddings = gae.encode([X, A]).numpy()

X_embedded = PCA().fit_transform(embeddings)

for l in set(labels):
    plt.scatter(X_embedded[np.argwhere(labels == l), 0], X_embedded[np.argwhere(labels == l), 1])
plt.show()


