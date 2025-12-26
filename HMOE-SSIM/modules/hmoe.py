import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.regularizers import l2
from core import FeatureEmbedding, MLP, MMoE, PredictLayer
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
        s_loss = bce(labels, tf.expand_dims(predicts[str(s)], axis=-1))
        m_loss = tf.reduce_sum(mask_scene * s_loss) / (scene_sample_num + 1e-10)
        loss_value += m_loss
    return loss_value

def multi_scene_loss_s(predicts, labels, scenes):   # shared all sample
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    loss_value = 0
    for s in scenes:
        s_loss = bce(labels, tf.expand_dims(predicts[str(s)], axis=-1))
        m_loss = tf.reduce_sum(s_loss) / (len(labels) + 1e-10)
        loss_value += m_loss
    return loss_value


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

    @tf.function
    def call(self, inputs, label, sid, training=None):

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
        return scene_outputs, sid, label


class HMoE_OPT(object):
    def __init__(self):
        super(HMoE_OPT, self).__init__()

    @tf.function
    def call(self, model, optimizer, x_batch_train, y_batch_train, d_batch_train):
        with tf.GradientTape() as tape:
            predicts, sid, _ = model(x_batch_train, y_batch_train, d_batch_train, training=True)
            l2_loss = tf.add_n(model.losses)
            ms_loss = multi_scene_loss(predicts, sid, y_batch_train, model.scenes)
            #ms_loss = multi_scene_loss_s(predicts, y_batch_train, model.scenes)  # shared all sample
            loss_value = ms_loss + l2_loss
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value

class HMoE_Eval(object):
    def __init__(self):
        super(HMoE_Eval, self).__init__()

    def call(self, model, x_batch_val, y_batch_val, d_batch_val):
        predicts, sid, _ = model.call(x_batch_val, y_batch_val, d_batch_val, training=False)
        pred = 0
        for s in model.scenes:
            mask_scene = tf.cast(tf.equal(sid, s), tf.float32)
            mask_scene = tf.squeeze(mask_scene, axis=-1)
            pred = pred + mask_scene * predicts[str(s)]
        pred = pred.numpy()
        return pred
