import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dropout, Activation, BatchNormalization, Dense
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import numpy as np
import random
import os
random.seed(2022)
tf.random.set_seed(2022)
np.random.seed(2022)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def multi_scene_loss(predicts, sid, labels, scenes):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    loss_value = 0
    for s in scenes:
        mask_scene = tf.cast(tf.equal(sid, s), tf.float32)
        mask_scene = tf.squeeze(mask_scene, axis=-1)
        scene_sample_num = tf.reduce_sum(mask_scene)
        s_loss = bce(labels, predicts[str(s)])
        m_loss = tf.reduce_sum(mask_scene * s_loss) / (scene_sample_num + 1e-10)
        loss_value += m_loss
    return loss_value

def multi_scene_loss_s(predicts, labels, scenes):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    loss_value = 0
    for s in scenes:
        s_loss = bce(labels, predicts[str(s)])
        m_loss = tf.reduce_sum(s_loss) / (len(labels) + 1e-10)
        loss_value += m_loss
    return loss_value

def multi_scene_loss_is_search(predicts, sid, labels, scenes, smask, lambda1=0.5):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    kl_loss_fn = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    loss_value = 0
    smask = tf.squeeze(smask, axis=-1)
    ones = tf.ones_like(smask, dtype=tf.float32)
    smask_eq_0 = tf.cast(tf.equal(smask, 0), tf.float32)
    for s in scenes:
        mask_scene = tf.cast(tf.equal(sid, s), tf.float32)
        mask_scene = tf.squeeze(mask_scene, axis=-1)
        scene_num = tf.reduce_sum(mask_scene * smask_eq_0)
        scene_s_num = tf.reduce_sum(smask)
        s_loss = bce(labels, predicts[str(s)])
        m_loss = tf.reduce_sum(mask_scene * smask_eq_0 * s_loss + smask * s_loss) / (scene_num + scene_s_num + 1e-10)
        loss_value += m_loss
    kl_loss = kl_loss_fn(tf.expand_dims(ones, axis=1), tf.expand_dims(smask, axis=1))
    loss_value += lambda1 * kl_loss
    return loss_value

def multi_scene_loss_is_retrain(predicts, sid, labels, scenes, smask):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    loss_value = 0
    smask = tf.squeeze(smask, axis=-1)
    smask_eq_0 = tf.cast(tf.equal(smask, 0), tf.float32)
    for s in scenes:
        mask_scene = tf.cast(tf.equal(sid, s), tf.float32)
        mask_scene = tf.squeeze(mask_scene, axis=-1)
        scene_num = tf.reduce_sum(mask_scene * smask_eq_0)
        scene_s_num = tf.reduce_sum(smask)
        s_loss = bce(labels, predicts[str(s)])
        m_loss = tf.reduce_sum(mask_scene * smask_eq_0 * s_loss + smask * s_loss) / (scene_num + scene_s_num + 1e-10)
        loss_value += m_loss
    return loss_value
    

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


class MultiLayerPerceptron(keras.layers.Layer):
    def __init__(self, input_dim, mlp_dims, dropout, output_layer=True, use_bn=False, use_ln=False):
        super(MultiLayerPerceptron, self).__init__()
        self.mlps = []
        for mlp_dim in mlp_dims:
            self.mlps.append(tf.keras.layers.Dense(units=mlp_dim, input_dim=(input_dim,)))
            if use_bn:
                self.mlps.append(tf.keras.layers.BatchNormalization())
            if use_ln:
                self.mlps.append(tf.keras.layers.LayerNormalization())
            self.mlps.append(tf.keras.layers.ReLU())
            self.mlps.append(tf.keras.layers.Dropout(dropout))
            input_dim = mlp_dim
        if output_layer:
            self.mlps.append(tf.keras.layers.Dense(1))

    def call(self, inputs):
        """
        :param inputs: Float tensor of size ``(batch_size, embed_dim)``
        :return : tensor of size (batch_size, mlp_dims[-1])
        """
        x = inputs
        for layer in self.mlps:
            x = layer(x)
        return x

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

class Par_Norm(Layer):
    def __init__(self, scenes, momentum=0.99, epsilon=0.001):
        super(Par_Norm, self).__init__()
        self.scenes = scenes
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma_initializer='ones'
        self.beta_initializer='zeros'
        self.moving_mean_initializer='zeros'
        self.moving_variance_initializer='ones'
        self.scale_com = None
        self.offset_com = None
        self.moving_mean = None
        self.moving_var = None
        self.scale_scenes = None
        self.offset_scenes = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.scale_com = self.add_weight(name="Gamma_com", shape=[last_dim], initializer=self.gamma_initializer,
                                         dtype=tf.float32, trainable=True)
        self.offset_com = self.add_weight(name="Beta_com", shape=[last_dim], initializer=self.beta_initializer,
                                          dtype=tf.float32, trainable=True)
        self.moving_mean = [self.add_weight(name="mu_"+str(s), shape=[last_dim], initializer=self.moving_mean_initializer,
                                            dtype=tf.float32, trainable=False) for s in self.scenes]
        self.moving_var = [self.add_weight(name="sigma_"+str(s), shape=[last_dim], initializer=self.moving_variance_initializer,
                                           dtype=tf.float32, trainable=False) for s in self.scenes]
        self.scale_scenes = [self.add_weight(name="Gamma_"+str(s), shape=[last_dim], initializer=self.gamma_initializer,
                                             dtype=tf.float32, trainable=True) for s in self.scenes]
        self.offset_scenes = [self.add_weight(name="Beta_" + str(s), shape=[last_dim], initializer=self.beta_initializer,
                                              dtype=tf.float32, trainable=True) for s in self.scenes]

    @tf.function
    def call(self, inputs, sid, training=None):
        outputs = []
        if training:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            for i, s in enumerate(self.scenes):
                mask_scene = tf.cast(tf.equal(sid, s), tf.float32)
                scene_input = mask_scene * inputs
                scene_mean, scene_var = tf.nn.moments(scene_input, [0])
                update_mv_mean = tf.compat.v1.assign(self.moving_mean[i], self.moving_mean[i] * self.momentum + scene_mean * (1 - self.momentum))
                update_mv_var = tf.compat.v1.assign(self.moving_var[i], self.moving_var[i] * self.momentum + scene_var * (1 - self.momentum))
                with tf.control_dependencies([update_mv_mean, update_mv_var]):
                    cur_scale = self.scale_com * self.scale_scenes[i]
                    cur_offset = self.offset_com + self.offset_scenes[i]
                    output_tensor = tf.nn.batch_normalization(inputs, batch_mean, batch_var, cur_offset, cur_scale, self.epsilon)
                    outputs.append(output_tensor)
        else:
            for i in range(len(self.scenes)):
                cur_scale = self.scale_com * self.scale_scenes[i]
                cur_offset = self.offset_com + self.offset_scenes[i]
                output_tensor = tf.nn.batch_normalization(inputs, self.moving_mean[i], self.moving_var[i], cur_offset, cur_scale, self.epsilon)
                outputs.append(output_tensor)
        return outputs


class STAR_FCN(Layer):
    def __init__(self, unit, activation, dropout, l2_reg, scenes, use_bias=True, w_initializer='glorot_uniform',
                 b_initializer="zeros", use_batch_norm=False):
        super(STAR_FCN, self).__init__()
        self.unit = unit
        self.scenes = scenes
        self.l2_regularizer = l2(l2_reg)
        self.kernel_initializer = w_initializer
        self.bias_initializer = b_initializer
        self.activation = Activation(activation)
        self.dropout = Dropout(dropout) if dropout else None
        self.use_bias = use_bias
        self.use_batch_norm = use_batch_norm
        self.bn = BatchNormalization()
        self.kernel_com = None
        self.bias_com = None
        self.kernel_scenes = None
        self.bias_scenes = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape[0])
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.kernel_com = self.add_weight(name="kernel_com", shape=[last_dim, self.unit],
                                          regularizer=self.l2_regularizer, initializer=self.kernel_initializer,
                                          dtype=tf.float32, trainable=True)
        self.bias_com = self.add_weight(name="bias_com", shape=[self.unit, ],
                                        regularizer=self.l2_regularizer, initializer=self.bias_initializer,
                                        dtype=tf.float32, trainable=True)
        self.kernel_scenes = [self.add_weight(name="kernel_"+str(s), shape=[last_dim, self.unit],
                                              regularizer=self.l2_regularizer, initializer=self.kernel_initializer,
                                              dtype=tf.float32, trainable=True) for s in self.scenes]
        self.bias_scenes = [self.add_weight(name="bias_"+str(s), shape=[self.unit, ],
                                            regularizer=self.l2_regularizer, initializer=self.bias_initializer,
                                            dtype=tf.float32, trainable=True) for s in self.scenes]

    @tf.function
    def call(self, inputs, training=None):
        scenes_outputs = []
        for i, s in enumerate(self.scenes):
            cur_kernel = self.kernel_com * self.kernel_scenes[i]
            cur_bias = self.bias_com + self.bias_scenes[i]
            outputs = tf.matmul(a=inputs[i], b=cur_kernel)
            if self.use_bias:
                outputs = tf.nn.bias_add(outputs, cur_bias)
            if self.use_batch_norm:
                outputs = self.bn(outputs, training=training)
            if self.activation:
                outputs = self.activation(outputs)
            if self.dropout:
                outputs = self.dropout(outputs, training=training)
            scenes_outputs.append(outputs)
        return scenes_outputs


class STAR(Model):
    def __init__(self, param_dict):
        super(STAR, self).__init__()
        self.field_size = param_dict['field_size']
        self.feature_size = param_dict['feature_size']
        self.embedding_size = param_dict['embedding_size']
        self.l2_reg = param_dict['l2_reg']
        self.share_units = param_dict['share_units']
        self.share_dropout = param_dict['share_dropout']
        self.fcn_units = param_dict['fcn_units']
        self.fcn_dropout = param_dict['fcn_dropout']
        self.activation = param_dict['activation']
        self.scenes = param_dict['scenes']
        self.user_pn = param_dict['user_pn']
        self.aux_net = param_dict['aux_net']

        self.embedding = FeatureEmbedding(self.field_size, self.feature_size, self.embedding_size, l2(self.l2_reg))
        self.share_mlp = MLP(self.share_units, self.activation, self.share_dropout, self.l2_reg) \
            if len(self.share_units) > 0 else None
        self.pn = Par_Norm(self.scenes)
        self.aux_mlp = MLP([16], self.activation, [0.0], self.l2_reg)
        self.aux_out = Dense(1, activation=None, kernel_regularizer=l2(self.l2_reg), bias_regularizer=l2(self.l2_reg))
        self.fcns = [STAR_FCN(units, self.activation, dropout, self.l2_reg, self.scenes)
                     for (units, dropout) in zip(self.fcn_units, self.fcn_dropout)]
        self.fcn_out = STAR_FCN(1, None, None, self.l2_reg, self.scenes)
        
        # ssim
        self.embed_dims = param_dict["mlp_dims"]
        self.dropout = param_dict["mlp_dropout"]
        self.use_bn = param_dict["use_bn"]
        self.input_dim = self.field_size * self.embedding_size
        self.ticket = False
        self.topk = 0.1
        self.temp = 1
        self.thre = 0.5
        self.lambda1 = 0.5
        self.domain_embedding = FeatureEmbedding(self.field_size, self.feature_size, self.embedding_size, l2(self.l2_reg))
        self.domain_hypernet = MultiLayerPerceptron(self.input_dim, self.embed_dims, output_layer=False, dropout=self.dropout,
                                                    use_bn=self.use_bn)
                                                    
        self.domain1_mask = tf.keras.layers.Dense(units=1, input_shape=(self.embed_dims[-1],))
        self.domain2_mask = tf.keras.layers.Dense(units=1, input_shape=(self.embed_dims[-1],))
        self.domain3_mask = tf.keras.layers.Dense(units=1, input_shape=(self.embed_dims[-1],))

    @tf.function
    def call(self, inputs, label, sid, training=None):
        
        # ssim
        smask = self.compute(inputs, sid)

        feat_emb = self.embedding(inputs)
        scene_emb = feat_emb[:, -1, :]  # None * K
        deep_input = tf.reshape(feat_emb, shape=[-1, self.field_size * self.embedding_size])  # None * (F*K)
        share_output = self.share_mlp(deep_input, training=training) if self.share_mlp else deep_input

        fcn_outputs = self.pn(share_output, sid, training=training) if self.user_pn else [share_output for _ in self.scenes]
        for fcn in self.fcns:
            fcn_outputs = fcn(fcn_outputs, training=training)
        fcn_outputs = self.fcn_out(fcn_outputs, training=training)

        if self.aux_net:
            aux_output = self.aux_mlp(scene_emb)
            aux_output = self.aux_out(aux_output)
            aux_output = tf.reshape(aux_output, [-1, 1])

        scene_outputs = {}
        for i, s in enumerate(self.scenes):
            outputs = tf.reshape(fcn_outputs[i], [-1, 1])
            if self.aux_net:
                outputs = outputs + aux_output
            outputs = tf.nn.sigmoid(outputs)
            scene_outputs[str(s)] = outputs
        return scene_outputs, sid, label, smask

    def compute(self, x, d):
        se_embedding = self.embedding(x)
        re_embedding = self.domain_embedding(x)
        if self.ticket:
            d_dnn = tf.reshape(re_embedding, [-1, self.field_size * self.embedding_size])
            hyper_output = self.domain_hypernet(d_dnn)

            m1 = self.domain1_mask(hyper_output)
            m2 = self.domain2_mask(hyper_output)
            m3 = self.domain3_mask(hyper_output)

            if self.topk is None:
                smask1 = tf.cast(tf.greater(m1, 0), tf.float32)
                smask2 = tf.cast(tf.greater(m2, 0), tf.float32)
                smask3 = tf.cast(tf.greater(m3, 0), tf.float32)

                d_eq_0 = tf.cast(tf.equal(d, 0), tf.float32)
                d_eq_1 = tf.cast(tf.equal(d, 1), tf.float32)
                d_eq_2 = tf.cast(tf.equal(d, 2), tf.float32)

                smask = smask1 * d_eq_0 + smask2 * d_eq_1 + smask3 * d_eq_2
                smask = binary_ste(tf.nn.relu(smask - self.thre))
            else:
                m = m1 * tf.cast(tf.equal(d, 0), tf.float32) + \
                    m2 * tf.cast(tf.equal(d, 1), tf.float32) + \
                    m3 * tf.cast(tf.equal(d, 2), tf.float32)
                m = tf.squeeze(m, axis=1)
                smask = tf.zeros(tf.shape(d)[0], dtype=tf.float32)
                k = tf.cast(float(tf.shape(d)[0]) * self.topk, tf.int32)
                _, topk_indices = tf.math.top_k(m, k=k, sorted=True)
                smask = tf.tensor_scatter_nd_update(smask, tf.expand_dims(topk_indices, axis=1), tf.ones_like(topk_indices, dtype=tf.float32))
                smask = tf.expand_dims(smask, axis=1)

        else:
            d_dnn = tf.reshape(se_embedding, [-1, self.field_size * self.embedding_size])
            hyper_output = self.domain_hypernet(d_dnn)
            
            m1 = self.domain1_mask(hyper_output)
            m2 = self.domain2_mask(hyper_output)
            m3 = self.domain3_mask(hyper_output)

            smask1 = tf.sigmoid(self.temp * m1)
            smask2 = tf.sigmoid(self.temp * m2)
            smask3 = tf.sigmoid(self.temp * m3)

            d_eq_0 = tf.cast(tf.equal(d, 0), tf.float32)
            d_eq_1 = tf.cast(tf.equal(d, 1), tf.float32)
            d_eq_2 = tf.cast(tf.equal(d, 2), tf.float32)

            smask = smask1 * d_eq_0 + smask2 * d_eq_1 + smask3 * d_eq_2
            smask = binary_ste(tf.nn.relu(smask - self.thre))
            

        return smask


class STAR_OPT(object):
    def __init__(self):
        super(STAR_OPT, self).__init__()
    @tf.function
    def call(self, model, optimizer, x_batch_train, y_batch_train, d_batch_train, retrain=False):
        with tf.GradientTape() as tape:
            predicts, sid, _, smask = model(x_batch_train, y_batch_train, d_batch_train, training=True)
            l2_loss = tf.add_n(model.losses)
            if not retrain:
                ms_loss = multi_scene_loss_is_search(predicts, sid, y_batch_train, model.scenes, smask, model.lambda1)
            else:
                ms_loss = multi_scene_loss_is_retrain(predicts, sid, y_batch_train, model.scenes, smask)
            loss_value = ms_loss + l2_loss
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        share_num = tf.reduce_sum(smask)
        return loss_value, int(share_num)

class STAR_Eval(object):
    def __init__(self):
        super(STAR_Eval, self).__init__()

    def call(self, model, x_batch_val, y_batch_val, d_batch_val):
        predicts, sid, _, _ = model.call(x_batch_val, y_batch_val, d_batch_val, training=False)
        pred = 0
        for s in model.scenes:
            mask_scene = tf.cast(tf.equal(sid, s), tf.float32)
            pred = pred + mask_scene * predicts[str(s)]
        pred = pred.numpy()
        return pred

@tf.custom_gradient
def binary_ste(x):
    # Forward pass: Apply sign function
    y = tf.sign(x)

    # Define the gradient function
    def grad(dy):
        return dy

    return y, grad

      
def kmax_pooling(x, k):
    _, topk_indices = tf.math.top_k(x, k=k, sorted=True)
    
    return topk_indices



