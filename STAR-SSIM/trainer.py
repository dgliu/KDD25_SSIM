import tensorflow as tf
import argparse
import logging
import os, sys
from pathlib import Path
import numpy as np
from modules import star
from sklearn import metrics
from utils import trainUtils
import random

parser = argparse.ArgumentParser(description="star trainer")
parser.add_argument("--dataset", type=str, help="specify dataset", default="ali-ccp")

# training hyperparameters
parser.add_argument("--lr", type=float, help="learning rate", default=3e-4)
parser.add_argument("--l2", type=float, help="L2 regularization", default=1e-6)
parser.add_argument("--bsize", type=int, help="batchsize", default=4096)
parser.add_argument("--epoch", type=int, default=20, help="maxmium epochs")
parser.add_argument("--save_dir", type=Path, default="save/", help="model save directory")

# neural network hyperparameters
parser.add_argument("--embedding_size", type=int, help="embedding dimension", default=16)
parser.add_argument("--scenes", type=int, nargs='+', default=[0, 1, 2], help="scenes")
parser.add_argument("--activation", type=str, default='relu', help="activation")
parser.add_argument("--share_units", type=int, nargs='+', default=[], help="share mlp layer size")
parser.add_argument("--share_dropout", type=float, default=[], help="share mlp dropout rate")
parser.add_argument("--fcn_units", type=int, nargs='+', default=[128, 64, 32], help="fcn_units layer size")
parser.add_argument("--fcn_dropout", type=float, default=[0.1, 0.3, 0.3], help="fcn_dropout rate")
parser.add_argument("--aux_net", action="store_true", default=True, help="auxiliary network")
parser.add_argument("--user_pn", action="store_true", default=True, help="mlp batch normalization")

# device information
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=4, help="device info")

args = parser.parse_args()

my_seed = 2022
random.seed(my_seed)
np.random.seed(my_seed)
tf.random.set_seed(my_seed)
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.cuda], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.cuda], True)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['TF_DETERMINISTIC_OPS'] = '1'


class Trainer(object):
    def __init__(self, opt):
        self.lr = opt['lr']
        self.l2 = opt['l2']
        self.bs = opt['bsize']
        self.model_dir = opt["save_dir"]
        self.dataloader = trainUtils.getDataLoader(opt["dataset"], opt["data_dir"])
        self.model = star.STAR(opt['model_opt'])
        self.model_opt = star.STAR_OPT()
        self.model_eval = star.STAR_Eval()
        self.optim = tf.optimizers.Adam(learning_rate=opt['lr'])
        self.logger = trainUtils.get_log(opt['dataset'])

    def train(self, epochs):
        cur_auc = 0.0
        ds = self.dataloader.get_all_data("train", batch_size=self.bs)
        for epoch_idx in range(int(epochs)):
            train_loss = .0
            step = 0
            for feature, label, domain in ds: # self.dataloader.get_data("train", batch_size=self.bs):
                loss = self.model_opt.call(self.model, self.optim, feature, label, domain)
                train_loss += loss
                step += 1
                if step % 1000 == 0:
                    self.logger.info("[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".
                                     format(epoch=epoch_idx, setp=step, loss=loss))
            train_loss /= step
            val_auc, val_loss = self.evaluate_val("val")
            self.logger.info(
                "[Epoch {epoch:d} | Train Loss:{loss:.6f} | Val AUC:{val_auc:.6f}, Val Loss:{val_loss:.6f}".
                format(epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss))
            early_stop = False
            if val_auc > cur_auc:
                cur_auc = val_auc
                self.model.save_weights("save4.h5")
            else:
                self.model.load_weights("save4.h5")
                early_stop = True
                te_auc, te_loss = self.evaluate_test("test")
                for s in self.model.scenes:
                    self.logger.info("Early stop at epoch {epoch:d}|Test AUC{d:}: {te_auc:.6f}, Test Loss{d:}:{te_loss:.6f}".
                                 format(epoch=epoch_idx, d=s, te_auc=te_auc[s], te_loss=te_loss[s]))
                break
        if not early_stop:
            te_auc, te_loss = self.evaluate_test("test")
            for s in self.model.scenes:
                self.logger.info(
                    "Final Test AUC{d:}:{te_auc:.6f}, Test Loss{d:}:{te_loss:.6f}".format(d=s, te_auc=te_auc, te_loss=te_loss))

    def evaluate_val(self, on: str):
        preds, trues = [], []
        for feature, label, domain in self.dataloader.get_data(on, batch_size=self.bs * 10):
            pred = self.model_eval.call(self.model, feature, label, domain)
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss

    def evaluate_test(self, on: str):
        preds_dist = {}
        trues_dist = {}
        for s in self.model.scenes:
            preds_dist[s] = []
            trues_dist[s] = []
        for feature, label, domain in self.dataloader.get_data(on, batch_size=self.bs * 10):
            pred = self.model_eval.call(self.model, feature, label, domain)
            for s in self.model.scenes:
                mask_scene = tf.equal(domain, s)
                preds_dist[s].append(pred[mask_scene])
                trues_dist[s].append(label[mask_scene])
        auc = {}
        loss = {}
        for s in self.model.scenes:
            y_pred = np.concatenate(preds_dist[s]).astype("float64")
            y_true = np.concatenate(trues_dist[s]).astype("float64")
            auc[s] = metrics.roc_auc_score(y_true, y_pred)
            loss[s] = metrics.log_loss(y_true, y_pred)
        return auc, loss


def main():
    sys.path.extend(["./modules", "./dataloader", "./utils"])
    if args.dataset.lower() == "ali-ccp":
        field_dim = trainUtils.get_stats("../data/ali-ccp/stats")
        data_dir = "../data/ali-ccp/tfrecord"
        field = len(field_dim)
        feature = sum(field_dim)
    elif args.dataset.lower() == "ali-mama":
        field_dim = trainUtils.get_stats("../data/ali-mama/stats_2")
        data_dir = "../data/ali-mama/threshold_2"
        field = len(field_dim)
        feature = sum(field_dim)
    model_opt = {
        "embedding_size": args.embedding_size, "feature_size": feature, "field_size": field, "scenes": args.scenes, "activation": args.activation,
        "share_units": args.share_units, "share_dropout": args.share_dropout, "fcn_units": args.fcn_units, "fcn_dropout": args.fcn_dropout,
        "aux_net": args.aux_net, "user_pn": args.user_pn, "l2_reg": args.l2
    }

    opt = {"model_opt": model_opt, "dataset": args.dataset, "lr": args.lr, "l2": args.l2,
           "bsize": args.bsize, "epoch": args.epoch, "data_dir": data_dir,
           "save_dir": args.save_dir, "cuda": args.cuda
           }
    print(opt)
    print(gpus)
    trainer = Trainer(opt)
    trainer.train(args.epoch)


if __name__ == "__main__":
    """
    python trainer.py  --dataset ali-ccp
    """
    main()
