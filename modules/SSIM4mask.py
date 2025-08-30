import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.layers import MultiLayerPerceptron, FactorizationMachine, FeaturesLinear, FeatureEmbedding
import modules.layers as layer


class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)

class BasicModel(torch.nn.Module):
    def __init__(self, opt):
        super(BasicModel, self).__init__()
        self.latent_dim = opt["latent_dim"]
        self.feature_num = opt["feat_num"]
        self.field_num = opt["field_num"]
        self.embed_dims = opt["mlp_dims"]
        self.dropout = opt["mlp_dropout"]
        self.use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.ticket = False
        self.topk = None
        self.temp = 1
        self.thre = 0.5
        print(self.field_num)
        print(self.feature_num)
        self.embedding = FeatureEmbedding(self.feature_num, self.latent_dim)
        self.domain_embedding = FeatureEmbedding(self.feature_num, self.latent_dim)
        self.sign = LBSign.apply

        self.domain_hypernet = MultiLayerPerceptron(self.dnn_dim, self.embed_dims, output_layer=False, dropout=self.dropout,
                                                    use_bn=self.use_bn)
        self.domain1_mask = torch.nn.Linear(self.embed_dims[-1], 1)
        self.domain2_mask = torch.nn.Linear(self.embed_dims[-1], 1)
        self.domain3_mask = torch.nn.Linear(self.embed_dims[-1], 1)
        self.domain4_mask = torch.nn.Linear(self.embed_dims[-1], 1)


    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, field_num)``

        """
        pass

    def reg(self):
        return 0.0
    
    def compute(self, x, d):
        if self.ticket:
            d_embedding = self.domain_embedding(x)
            d_dnn = d_embedding.view(-1, self.dnn_dim)
            hyper_outpot = self.domain_hypernet(d_dnn)

            m1 = self.domain1_mask(hyper_outpot)
            m2 = self.domain2_mask(hyper_outpot)
            m3 = self.domain3_mask(hyper_outpot)
            m4 = self.domain4_mask(hyper_outpot)

            if self.topk is None:
                smask1 = (m1 > 0).float()
                smask2 = (m2 > 0).float()
                smask3 = (m3 > 0).float()
                smask4 = (m4 > 0).float()

                smask = smask1 * torch.eq(d, 0).type(torch.long) + smask2 * torch.eq(d, 1).type(torch.long) \
                        + smask3 * torch.eq(d, 2).type(torch.long) + smask4 * torch.eq(d, 3).type(torch.long)
                smask = self.sign(torch.relu(smask - self.thre))
            else:
                m = m1 * torch.eq(d, 0).type(torch.long) + m2 * torch.eq(d, 1).type(torch.long) \
                        + m3 * torch.eq(d, 2).type(torch.long) + m4 * torch.eq(d, 3).type(torch.long)
                smask = torch.zeros(len(d)).to('cuda:0')
                prob_idx = kmax_pooling(m, 0, int(len(d) * self.topk)).squeeze()
                smask[prob_idx] = 1
                smask = smask.unsqueeze(1)

        else:
            d_embedding = self.embedding(x)
            d_dnn = d_embedding.view(-1, self.dnn_dim)
            hyper_outpot = self.domain_hypernet(d_dnn)

            m1 = self.domain1_mask(hyper_outpot)
            m2 = self.domain2_mask(hyper_outpot)
            m3 = self.domain3_mask(hyper_outpot)
            m4 = self.domain4_mask(hyper_outpot)

            smask1 = torch.sigmoid(self.temp * m1)
            smask2 = torch.sigmoid(self.temp * m2)
            smask3 = torch.sigmoid(self.temp * m3)
            smask4 = torch.sigmoid(self.temp * m4)

            smask = smask1 * torch.eq(d, 0).type(torch.long) + smask2 * torch.eq(d, 1).type(torch.long) \
                    + smask3 * torch.eq(d, 2).type(torch.long) + smask4 * torch.eq(d, 3).type(torch.long)
            smask = self.sign(torch.relu(smask - self.thre))

        return smask


class FM(BasicModel):
    def __init__(self, opt):
        super(FM, self).__init__(opt)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x_embedding = self.embedding(x)
        output_fm = self.fm(x_embedding)
        logit = output_fm
        return logit


class MaskDNN(BasicModel):
    def __init__(self, opt):
        super(MaskDNN, self).__init__(opt)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.dnn1 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)
        self.dnn2 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)
        self.dnn3 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)
        self.dnn4 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)

    def forward(self, x, d):
        smask = self.compute(x, d)

        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_dnn1 = self.dnn1(x_dnn)
        output_dnn2 = self.dnn2(x_dnn)
        output_dnn3 = self.dnn3(x_dnn)
        output_dnn4 = self.dnn4(x_dnn)
        logit1 = output_dnn1
        logit2 = output_dnn2
        logit3 = output_dnn3
        logit4 = output_dnn4
        return logit1, logit2, logit3, logit4, smask


class MaskDeepFM(FM):
    def __init__(self, opt):
        super(MaskDeepFM, self).__init__(opt)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.dnn1 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)
        self.dnn2 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)
        self.dnn3 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)
        self.dnn4 = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)

    def forward(self, x, d):
        smask = self.compute(x, d)
        
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_fm1 = self.fm(x_embedding)
        output_fm2 = self.fm(x_embedding)
        output_fm3 = self.fm(x_embedding)
        output_fm4 = self.fm(x_embedding)
        output_dnn1 = self.dnn1(x_dnn)
        output_dnn2 = self.dnn2(x_dnn)
        output_dnn3 = self.dnn3(x_dnn)
        output_dnn4 = self.dnn4(x_dnn)
        logit1 = output_dnn1 + output_fm1
        logit2 = output_dnn2 + output_fm2
        logit3 = output_dnn3 + output_fm3
        logit4 = output_dnn4 + output_fm4
        return logit1, logit2, logit3, logit4, smask


class MaskDeepCross(BasicModel):
    def __init__(self, opt):
        super(MaskDeepCross, self).__init__(opt)
        cross_num = opt["cross"]
        mlp_dims = opt["mlp_dims"]
        use_bn = opt["use_bn"]
        dropout = opt["mlp_dropout"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.cross1 = layer.CrossNetwork(self.dnn_dim, cross_num)
        self.dnn1 = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination1 = torch.nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)
        self.cross2 = layer.CrossNetwork(self.dnn_dim, cross_num)
        self.dnn2 = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination2 = torch.nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)
        self.cross3 = layer.CrossNetwork(self.dnn_dim, cross_num)
        self.dnn3 = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination3 = torch.nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)
        self.cross4 = layer.CrossNetwork(self.dnn_dim, cross_num)
        self.dnn4 = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination4 = torch.nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)

    def forward(self, x, d):
        smask = self.compute(x, d)
        
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_cross1 = self.cross1(x_dnn)
        output_dnn1 = self.dnn1(x_dnn)
        comb_tensor1 = torch.cat((output_cross1, output_dnn1), dim=1)
        output_cross2 = self.cross2(x_dnn)
        output_dnn2 = self.dnn2(x_dnn)
        comb_tensor2 = torch.cat((output_cross2, output_dnn2), dim=1)
        output_cross3 = self.cross3(x_dnn)
        output_dnn3 = self.dnn3(x_dnn)
        comb_tensor3 = torch.cat((output_cross3, output_dnn3), dim=1)
        output_cross4 = self.cross4(x_dnn)
        output_dnn4 = self.dnn4(x_dnn)
        comb_tensor4 = torch.cat((output_cross4, output_dnn4), dim=1)
        logit1 = self.combination1(comb_tensor1)
        logit2 = self.combination2(comb_tensor2)
        logit3 = self.combination3(comb_tensor3)
        logit4 = self.combination4(comb_tensor4)
        return logit1, logit2, logit3, logit4, smask


def getOptim(network, optim, lr, m_lr,  l2):
    weight_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'domain' not in p[0], network.named_parameters()))
    mask_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'domain' in p[0], network.named_parameters()))

    optim = optim.lower()
    if optim == "sgd":
        return [torch.optim.SGD(weight_params, lr=lr, weight_decay=l2), torch.optim.SGD(mask_params, lr=m_lr)]
    elif optim == "adam":
        return [torch.optim.Adam(weight_params, lr=lr, weight_decay=l2), torch.optim.Adam(mask_params, lr=m_lr)]
    else:
        raise ValueError("Invalid optimizer type: {}".format(optim))

def getModel(model: str, opt):
    model = model.lower()
    if model == "deepfm":
        return MaskDeepFM(opt)
    elif model == "dcn":
        return MaskDeepCross(opt)
    elif model == "dnn":
        return MaskDNN(opt)
    else:
        raise ValueError("Invalid model type: {}".format(model))

def sigmoid(x):
    return float(1. / (1. + np.exp(-x)))
    
def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return index


