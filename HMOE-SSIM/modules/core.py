import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, Concatenate, Flatten
from tensorflow.python.keras.regularizers import l2

tf.random.set_seed(2022)

class FeatureEmbedding(keras.layers.Layer):
    def __init__(self, field_size, feature_size, embedding_size, regularizer, kernel_initializer="glorot_uniform"):
        super(FeatureEmbedding, self).__init__()
        self.field_size = field_size
        self.regularizer = keras.regularizers.get(regularizer)
        self.embedding = tf.keras.layers.Embedding(feature_size, embedding_size,
                                                   embeddings_regularizer=self.regularizer,
                                                   embeddings_initializer=kernel_initializer)

    @tf.function
    def call(self, inputs):
        feat_ids = inputs
        embeddings = self.embedding(feat_ids)  # None * F * K
        return embeddings


class MLP(keras.layers.Layer):
    def __init__(self, hidden_units, activation, dnn_dropout, l2_reg, w_initializer='glorot_uniform',
                 b_initializer="zeros", use_batch_norm=False):
        super(MLP, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation, kernel_regularizer=l2(l2_reg),
                                  kernel_initializer=w_initializer, bias_regularizer=l2(l2_reg),
                                  bias_initializer=b_initializer) for unit in hidden_units]
        self.dropout = [Dropout(do) for do in dnn_dropout]
        self.use_batch_norm = use_batch_norm
        self.bn = BatchNormalization()

    @tf.function
    def call(self, inputs, training=None):
        x = inputs
        for dnn, dropout in zip(self.dnn_network, self.dropout):
            x = dnn(x)
            if self.use_batch_norm:
                x = self.bn(x, training=training)
            x = dropout(x, training=training)
        return x


class SingleMLP(keras.layers.Layer):
    def __init__(self, hidden_units, activation, dnn_dropout, l2_reg, w_initializer='glorot_uniform',
                 b_initializer="zeros"):
        super(SingleMLP, self).__init__()
        self.dnn_network = Dense(units=hidden_units, activation=activation, kernel_regularizer=l2(l2_reg),
                                 kernel_initializer=w_initializer, bias_regularizer=l2(l2_reg),
                                 bias_initializer=b_initializer)
        self.dropout = Dropout(dnn_dropout)

    @tf.function
    def call(self, inputs, training=None):
        return self.dnn_network(inputs)


# todo：mtl prediction
class PredictLayer(keras.layers.Layer):
    def __init__(self, l2_reg, use_bias=True, w_initializer='glorot_uniform', b_initializer="zeros"):
        super(PredictLayer, self).__init__()
        self.dense = Dense(1, activation=None, kernel_regularizer=l2(l2_reg), kernel_initializer=w_initializer,
                           use_bias=use_bias, bias_regularizer=l2(l2_reg), bias_initializer=b_initializer)

    @tf.function
    def call(self, inputs):
        outputs = tf.reshape(self.dense(inputs), [-1, 1])  # (batch_size, 1)
        outputs = tf.nn.sigmoid(outputs)
        return outputs


class PredictLayerWithLogit(keras.layers.Layer):
    def __init__(self, l2_reg, use_bias=True, w_initializer='glorot_uniform', b_initializer="zeros"):
        super(PredictLayerWithLogit, self).__init__()
        self.dense = Dense(1, activation=None, kernel_regularizer=l2(l2_reg), kernel_initializer=w_initializer,
                           use_bias=use_bias, bias_regularizer=l2(l2_reg), bias_initializer=b_initializer)

    @tf.function
    def call(self, inputs):
        logits = tf.reshape(self.dense(inputs), [-1, 1])  # (batch_size, 1)
        prob = tf.nn.sigmoid(logits)
        return prob, logits


class SEBlockLayer(keras.layers.Layer):
    def __init__(self, hidden_units, field_size, activation, l2_reg, w_initializer='glorot_uniform',
                 b_initializer="zeros"):
        super(SEBlockLayer, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation, kernel_regularizer=l2(l2_reg),
                                  kernel_initializer=w_initializer, bias_regularizer=l2(l2_reg),
                                  bias_initializer=b_initializer) for unit in hidden_units]

        self.out_network = Dense(units=field_size, activation=tf.sigmoid, kernel_regularizer=l2(l2_reg),
                                 kernel_initializer=w_initializer, bias_regularizer=l2(l2_reg),
                                 bias_initializer=b_initializer)
        self.field_size = field_size

    @tf.function
    def call(self, inputs):
        x = tf.stop_gradient(inputs)  # None * F * D
        x_pool = tf.reduce_mean(x, axis=-1)  # None * F
        for dnn in self.dnn_network:
            x_pool = dnn(x_pool)
        emb_gate = 2 * self.out_network(x_pool)  # None * F
        emb_gate = tf.expand_dims(emb_gate, 2)  # None * F * 1
        gated_inputs = inputs * emb_gate  # None * F * K
        return gated_inputs


class MMoE(keras.layers.Layer):
    def __init__(self, task_num, expert_num, expert_dim, l2_reg, w_initializer='VarianceScaling',
                 b_initializer="zeros"):
        super(MMoE, self).__init__()
        self.task_num = task_num
        self.expert_layers = [Dense(expert_dim, activation='relu', kernel_regularizer=l2(l2_reg),
                                    kernel_initializer=w_initializer, bias_regularizer=l2(l2_reg),
                                    bias_initializer=b_initializer) for _ in range(expert_num)]
        self.gate_layers = [Dense(expert_num, use_bias=False, activation='softmax', kernel_regularizer=l2(l2_reg),
                                  kernel_initializer=w_initializer) for _ in range(task_num)]

    @tf.function
    def call(self, inputs):
        expert_outs = [expert_net(inputs) for expert_net in self.expert_layers]  # exp_num * b * exp_dim
        towers = []
        for i in range(self.task_num):
            gate = self.gate_layers[i](inputs)  # b, exp_num
            gate = tf.expand_dims(gate, axis=-1)  # b, exp_num, 1
            exp_out = Concatenate(axis=1)([expert[:, tf.newaxis, :] for expert in expert_outs])  # b, exp_num, exp_dim
            _tower = tf.matmul(exp_out, gate, transpose_a=True)  # b, exp_dim, 1
            towers.append(Flatten()(_tower))  # (bs,expert_dim)
        return towers


class MMoE_MULTI_LAYER(keras.layers.Layer):
    def __init__(self, task_num, expert_num, expert_units_list, expert_dropout_list, l2_reg, w_initializer='VarianceScaling'):
        super(MMoE_MULTI_LAYER, self).__init__()
        self.task_num = task_num
        self.expert_layers = [MLP(expert_units_list, 'relu', expert_dropout_list, l2_reg) for _ in range(expert_num)]
        self.gate_layers = [Dense(expert_num, use_bias=False, activation='softmax', kernel_regularizer=l2(l2_reg),
                                  kernel_initializer=w_initializer) for _ in range(task_num)]

    @tf.function
    def call(self, inputs):
        expert_outs = [expert_net(inputs) for expert_net in self.expert_layers]  # exp_num * b * exp_dim
        towers = []
        for i in range(self.task_num):
            gate = self.gate_layers[i](inputs)  # b, exp_num
            gate = tf.expand_dims(gate, axis=-1)  # b, exp_num, 1
            exp_out = Concatenate(axis=1)([expert[:, tf.newaxis, :] for expert in expert_outs])  # b, exp_num, exp_dim
            _tower = tf.matmul(exp_out, gate, transpose_a=True)  # b, exp_dim, 1
            towers.append(Flatten()(_tower))  # (bs,expert_dim)
        return towers


class CGC(keras.layers.Layer):
    def __init__(self, task_num, expert_nums, share_expert_num, expert_dim, l2_reg,
                 w_initializer='VarianceScaling', b_initializer="zeros", is_last=False):
        super(CGC, self).__init__()
        self.task_num = task_num
        self.is_last = is_last
        self.share_layer = [Dense(expert_dim, activation="relu", kernel_regularizer=l2(l2_reg),
                                  kernel_initializer=w_initializer, bias_regularizer=l2(l2_reg),
                                  bias_initializer=b_initializer) for _ in range(share_expert_num)]
        self.E_layers = []
        for i in range(task_num):
            sub_exp = [Dense(expert_dim, activation='relu', kernel_regularizer=l2(l2_reg),
                             kernel_initializer=w_initializer, bias_regularizer=l2(l2_reg),
                             bias_initializer=b_initializer)
                       for _ in range(expert_nums[i])]
            self.E_layers.append(sub_exp)
        self.gate_layers = [Dense(share_expert_num + expert_nums[i], use_bias=False, activation='softmax',
                                  kernel_regularizer=l2(l2_reg), kernel_initializer=w_initializer)
                            for i in range(task_num)]
        if not self.is_last:
            self.share_gate_layer = Dense(share_expert_num + sum(expert_nums), use_bias=False, activation='softmax',
                                          kernel_regularizer=l2(l2_reg), kernel_initializer=w_initializer)

    @tf.function
    def call(self, inputs):
        E_net = [[expert(inputs[i]) for expert in sub_expert] for i, sub_expert in enumerate(self.E_layers)]
        share_net = [expert(inputs[-1]) for expert in self.share_layer]
        towers = []
        for i in range(self.task_num):
            gate = self.gate_layers[i](inputs[i])
            gate = tf.expand_dims(gate, axis=-1)  # b, exp_num, 1
            experts = share_net + E_net[i]
            exp_out = Concatenate(axis=1)([expert[:, tf.newaxis, :] for expert in experts])  # b, exp_num, exp_dim
            _tower = tf.matmul(exp_out, gate, transpose_a=True)  # b, exp_dim, 1
            towers.append(Flatten()(_tower))  # (bs,expert_dim)
        if self.is_last:
            towers.append(None)
        else:
            share_gate = self.share_gate_layer(inputs[-1])
            share_gate = tf.expand_dims(share_gate, axis=-1)  # b, exp_num, 1
            share_experts = share_net + sum(E_net, [])
            share_exp_out = Concatenate(axis=1)(
                [expert[:, tf.newaxis, :] for expert in share_experts])  # b, exp_num, exp_dim
            _share = tf.matmul(share_exp_out, share_gate, transpose_a=True)  # b, exp_dim, 1
            towers.append(Flatten()(_share))
        return towers


class CGC_MULTI_LAYER(keras.layers.Layer):
    def __init__(self, task_num, expert_nums, share_expert_num, expert_units_list, expert_dropout_list, l2_reg,
                 w_initializer='VarianceScaling', b_initializer="zeros", is_last=False):
        super(CGC_MULTI_LAYER, self).__init__()
        self.task_num = task_num
        self.is_last = is_last
        self.share_layer = [MLP(expert_units_list, 'relu', expert_dropout_list, l2_reg) for _ in range(share_expert_num)]
        self.E_layers = []
        for i in range(task_num):
            sub_exp = [MLP(expert_units_list, 'relu', expert_dropout_list, l2_reg) for _ in range(expert_nums[i])]
            self.E_layers.append(sub_exp)
        self.gate_layers = [Dense(share_expert_num + expert_nums[i], use_bias=False, activation='softmax',
                                  kernel_regularizer=l2(l2_reg), kernel_initializer=w_initializer)
                            for i in range(task_num)]
        if not self.is_last:
            self.share_gate_layer = Dense(share_expert_num + sum(expert_nums), use_bias=False, activation='softmax',
                                          kernel_regularizer=l2(l2_reg), kernel_initializer=w_initializer)

    @tf.function
    def call(self, inputs):
        E_net = [[expert(inputs[i]) for expert in sub_expert] for i, sub_expert in enumerate(self.E_layers)]
        share_net = [expert(inputs[-1]) for expert in self.share_layer]
        towers = []
        for i in range(self.task_num):
            gate = self.gate_layers[i](inputs[i])
            gate = tf.expand_dims(gate, axis=-1)  # b, exp_num, 1
            experts = share_net + E_net[i]
            exp_out = Concatenate(axis=1)([expert[:, tf.newaxis, :] for expert in experts])  # b, exp_num, exp_dim
            _tower = tf.matmul(exp_out, gate, transpose_a=True)  # b, exp_dim, 1
            towers.append(Flatten()(_tower))  # (bs,expert_dim)
        if self.is_last:
            towers.append(None)
        else:
            share_gate = self.share_gate_layer(inputs[-1])
            share_gate = tf.expand_dims(share_gate, axis=-1)  # b, exp_num, 1
            share_experts = share_net + sum(E_net, [])
            share_exp_out = Concatenate(axis=1)(
                [expert[:, tf.newaxis, :] for expert in share_experts])  # b, exp_num, exp_dim
            _share = tf.matmul(share_exp_out, share_gate, transpose_a=True)  # b, exp_dim, 1
            towers.append(Flatten()(_share))
        return towers


class SingleLayerPerceptron(keras.layers.Layer):
    def __init__(self, units, activation='relu', use_bn=False, dropout=0.0,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer='l2', bias_regularizer='l2', use_bias=True):
        super(SingleLayerPerceptron, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bn = use_bn
        self.dropout_rate = dropout
        self.kernel_initialzier = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularzier = keras.regularizers.get(bias_regularizer)
        self.use_bias = use_bias
        self.kernel = None
        self.bias = None
        self.bn = None
        self.dropout = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=[last_dim, self.units],
            regularizer=self.kernel_regularizer,
            initializer=self.kernel_initialzier,
            dtype=tf.float32,
            trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.units, ],
                regularizer=self.bias_regularzier,
                initializer=self.bias_initializer,
                dtype=tf.float32,
                trainable=True
            )
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization()

        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

    @tf.function
    def call(self, inputs, training=None, *args, **kwargs):
        outputs = tf.matmul(a=inputs, b=self.kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.use_bn:
            outputs = self.bn(outputs, training=training)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs, training=training)
        return outputs
