import tensorflow as tf

tf.enable_eager_execution()

from gcn_layers import GraphConv

from tensorflow.keras.layers import Lambda

import tensorflow.keras.backend as K

import numpy as np

from collections import defaultdict

from tqdm import tqdm

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


################### cora

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

feat_data, labels, adj_lists = load_cora()
num_node = feat_data.shape[0]
A = np.zeros((num_node, num_node))
for i in adj_lists:
    for j in adj_lists[i]:
        A[i][j] = 1
A = np.expand_dims(A, 0)
X = np.expand_dims(feat_data, 0)
labels = labels.reshape(-1)

class GAE(tf.keras.Model):
    def __init__(self, num_node):
        super(GAE, self).__init__()
        self.num_node = num_node
        self.conv1 = GraphConv(
            units=16,
            name='GraphConv',
        )
        self.dot = Lambda(lambda x: tf.reshape(K.dot(x, K.transpose(x)), [-1]))

    def call(self, inputs):
        """Run the model."""
        inputs_tensor = [tf.cast(tf.convert_to_tensor(inputs[0]), tf.double),
                         tf.cast(tf.convert_to_tensor(inputs[1]), tf.double)]
        result = self.conv1(inputs_tensor)
        result = tf.squeeze(result, [0])
        result = tf.reshape(self.dot(result), [-1])
        return result

    def encode(self, inputs):
        inputs_tensor = [tf.cast(tf.convert_to_tensor(inputs[0]), tf.double),
                         tf.cast(tf.convert_to_tensor(inputs[1]), tf.double)]
        return tf.squeeze(self.conv1(inputs_tensor))

print(A)

pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()

gae = GAE(num_node)

print(gae([X,A]))
print(tf.reshape(gae([X,A]), [num_node, num_node]))

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
for l in set(labels):
    plt.scatter(embeddings[np.argwhere(labels == l), 0], embeddings[np.argwhere(labels == l), 1])
plt.show()
