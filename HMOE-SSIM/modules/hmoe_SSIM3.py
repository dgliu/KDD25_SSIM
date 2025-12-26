import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.regularizers import l2
from core import FeatureEmbedding, MLP, MMoE, PredictLayer
from tensorflow import keras
import numpy as np
import random
import os
random.seed(2022)
tf.random.set_seed(2022)
np.random.seed(2022)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

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
        s_loss = bce(labels, tf.expand_dims(predicts[str(s)], axis=-1))
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
        s_loss = bce(labels, tf.expand_dims(predicts[str(s)], axis=-1))
        m_loss = tf.reduce_sum(mask_scene * smask_eq_0 * s_loss + smask * s_loss) / (scene_num + scene_s_num + 1e-10)
        loss_value += m_loss
    return loss_value

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

class HMoE(Model):
    def __init__(self, param_dict):
        super(HMoE, self).__init__()
        self.field_size = param_dict['field_size']
        self.feature_size = param_dict['feature_size']
        self.embedding_size = param_dict['embedding_size']
        self.l2_reg = param_dict['l2_reg']
        self.share_units = param_dict['share_units']
        self.share_dropout = param_dict['share_dropout']
        self.expert_dim = param_dict['expert_dim']
        self.expert_num = param_dict['expert_num']
        self.tower_units = param_dict['tower_units']
        self.tower_dropout = param_dict['tower_dropout']
        self.activation = param_dict['activation']
        self.scenes = param_dict['scenes']
        self.scenes_num = len(self.scenes)

        self.embedding = FeatureEmbedding(self.field_size, self.feature_size, self.embedding_size, l2(self.l2_reg))
        self.share_mlp = MLP(self.share_units, self.activation, self.share_dropout, self.l2_reg) \
            if len(self.share_units) > 0 else None
        self.mmoe = MMoE(self.scenes_num, self.expert_num, self.expert_dim, self.l2_reg)
        self.tower_mlps = [MLP(self.tower_units, self.activation, self.tower_dropout, self.l2_reg) for _ in self.scenes]
        self.tower_outs = [PredictLayer(self.l2_reg) for _ in self.scenes]
        self.score_gates = [Dense(self.scenes_num, use_bias=False, activation='softmax', kernel_regularizer=l2(self.l2_reg))
                            for _ in self.scenes]
        
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
        deep_input = tf.reshape(feat_emb, shape=[-1, self.field_size * self.embedding_size])  # None * (F*K)
        share_output = self.share_mlp(deep_input, training=training) if self.share_mlp else deep_input

        mmoe_outputs = self.mmoe(share_output)

        scene_scores = []
        for i, s in enumerate(self.scenes):
            deep_outputs = self.tower_mlps[i](mmoe_outputs[i], training=training)
            tower_outputs = self.tower_outs[i](deep_outputs)
            scene_scores.append(tower_outputs)
        scene_outputs = {}
        for i, s in enumerate(self.scenes):
            score_list = [score if i == j else tf.stop_gradient(score) for j, score in enumerate(scene_scores)]
            score_concat = Concatenate(axis=1)(score_list)  # (batch_size, scene_num)
            weights = self.score_gates[i](share_output)  # (batch_size, scene_num)
            outputs = weights * score_concat
            outputs = tf.reduce_sum(outputs, -1)
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


class HMoE_OPT(object):
    def __init__(self):
        super(HMoE_OPT, self).__init__()

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

class HMoE_Eval(object):
    def __init__(self):
        super(HMoE_Eval, self).__init__()

    def call(self, model, x_batch_val, y_batch_val, d_batch_val):
        predicts, sid, _, _ = model.call(x_batch_val, y_batch_val, d_batch_val, training=False)
        pred = 0
        for s in model.scenes:
            mask_scene = tf.cast(tf.equal(sid, s), tf.float32)
            mask_scene = tf.squeeze(mask_scene, axis=-1)
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