import torch
import torch.nn.functional as F
import argparse
import logging
import os, sys
import time, statistics
from pathlib import Path
import numpy as np
from sklearn import metrics
from utils import trainUtils
from modules import SSIM3mask
import pickle

parser = argparse.ArgumentParser(description="ssim trainer")
parser.add_argument("--dataset", type=str, help="specify dataset", default="ali-ccp")
parser.add_argument("--model", type=str, help="specify model", default="dnn")


# training hyperparameters
parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
parser.add_argument("--l2", type=float, help="L2 regularization", default=1e-7)
parser.add_argument("--bsize", type=int, help="batchsize", default=4096)
parser.add_argument("--optim", type=str, default="Adam", help="optimizer type")
parser.add_argument("--max_epoch", type=int, default=20, help="maxmium epochs")
parser.add_argument("--save_dir", type=Path, default="save/", help="model save directory")

# neural network hyperparameters
parser.add_argument("--dim", type=int, help="embedding dimension", default=16)
parser.add_argument("--mlp_dims", type=int, nargs='+', default=[1024, 512, 256], help="mlp layer size")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="mlp dropout rate (default:0.0)")
parser.add_argument("--mlp_bn", action="store_true", default=False, help="mlp batch normalization")
parser.add_argument("--cross", type=int, help="cross layer", default=3)

# device information
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=0, help="device info")

# mask information
parser.add_argument("--m_lr", type=float, help="mask network learning rate", default=1e-5)
parser.add_argument("--r_lr", type=float, help="retrain learning rate", default=1e-3)
parser.add_argument("--r_l2", type=float, help="retrain L2 regularization", default=1e-7)
parser.add_argument("--final_temp", type=float, default=1000, help="final temperature")
parser.add_argument("--search_epoch", type=int, default=1, help="search epochs")
parser.add_argument("--thre", type=float, default=0.55, help="thre")
parser.add_argument("--lambda1", type=float, default=0.4, help="lambda")
parser.add_argument("--topk", type=float, default=[0.95], help="sclect_rate")
args = parser.parse_args()

my_seed = 2022
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'


class Trainer(object):
    def __init__(self, opt):
        self.lr = opt['lr']
        self.l2 = opt['l2']
        self.bs = opt['bsize']
        self.model_dir = opt["save_dir"]
        self.m_lr = opt["m_lr"]
        self.r_lr = opt['r_lr']
        self.r_l2 = opt['r_l2']
        self.epochs = opt["search_epoch"]
        self.lambda1 = opt["lambda1"]
        self.thre = opt["thre"]
        self.topk = opt["topk"]
        self.dataloader = trainUtils.getDataLoader(opt["dataset"], opt["data_dir"])
        self.device = trainUtils.getDevice(opt["cuda"])
        self.network = SSIM3mask.getModel(opt["model"], opt["model_opt"]).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.optim = SSIM3mask.getOptim(self.network, opt["optimizer"], self.lr, self.m_lr, self.l2)
        self.logger = trainUtils.get_log(opt['model'])
        self.domain_list = [0, 1, 2]
        # self.batch_num = (38070670 // self.bs) + 1
        self.model_opt = opt["model_opt"]
        self.opt = opt
        self.model_init_dir = 'model_init_s3'
        torch.save(self.network.state_dict(), self.model_init_dir)
        
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.ds = self.dataloader.get_all_data("train", batch_size=self.bs)
        self.batch_num = len(self.ds)

    def train_on_batch(self, label, data, domain, retrain=False):
        self.network.train()
        self.network.zero_grad()
        data, label, domain = data.to(self.device), label.to(self.device), domain.to(self.device)
        logit1, logit2, logit3, smask = self.network(data, domain)
        logloss1 = self.criterion(logit1, label)
        logloss2 = self.criterion(logit2, label)
        logloss3 = self.criterion(logit3, label)
 
        if not retrain:
            main_loss = torch.mean(logloss1 * torch.eq(domain, 0).type(torch.long) * torch.eq(smask, 0).type(torch.long) +
                                   logloss2 * torch.eq(domain, 1).type(torch.long) * torch.eq(smask, 0).type(torch.long)+
                                   logloss3 * torch.eq(domain, 2).type(torch.long) * torch.eq(smask, 0).type(torch.long)+
                                   (logloss1 * smask + logloss2 * smask + logloss3 * smask) / 3)
            ones = torch.ones(smask.shape).to(self.device)
            kl_loss = self.kl_loss(smask, ones)
            loss = main_loss + self.lambda1 * kl_loss
        else:
            loss = torch.mean(logloss1 * torch.eq(domain, 0).type(torch.long) * torch.eq(smask, 0).type(torch.long) +
                                   logloss2 * torch.eq(domain, 1).type(torch.long) * torch.eq(smask, 0).type(torch.long)+
                                   logloss3 * torch.eq(domain, 2).type(torch.long) * torch.eq(smask, 0).type(torch.long)+
                                   (logloss1 * smask + logloss2 * smask + logloss3 * smask) / 3)

        loss.backward()
        for optim in self.optim:
            optim.step()
        return loss.item(), torch.sum(smask).item()

    def eval_on_batch(self, data, domain):
        self.network.eval()
        with torch.no_grad():
            data, domain = data.to(self.device), domain.to(self.device)
            logit1, logit2, logit3, _ = self.network(data, domain)
            logit = logit1 * torch.eq(domain, 0).type(torch.long) + logit2 * torch.eq(domain, 1).type(torch.long) \
                    + logit3 * torch.eq(domain, 2).type(torch.long)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob

    def search(self):
        self.logger.info("ticket:{t}".format(t=self.network.ticket))
        self.logger.info("-----------------Begin Search-----------------")
        self.network.thre = self.thre
        temp_increase = self.opt["final_temp"] ** (1. / (self.batch_num - 1))
        for epoch_idx in range(int(self.epochs)):
            train_loss = .0
            train_share_num = 0
            step = 0
            for feature, label, domain in self.ds:
                if step > 0:
                    self.network.temp *= temp_increase
                loss, share_num = self.train_on_batch(label, feature, domain)
                train_loss += loss
                train_share_num += share_num
                step += 1
                if step % 1000 == 0:
                    print('share_num:', share_num, 'train_share_num:', train_share_num)
                    self.logger.info("[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".
                                     format(epoch=epoch_idx, setp=step, loss=loss))
            train_loss /= step
            self.logger.info("Temp:{temp:.6f}".format(temp=self.network.temp))
            val_auc, val_loss = self.evaluate_val("val")
            print('share_num:', share_num, 'train_share_num:', train_share_num)
            self.logger.info(
                "[Epoch {epoch:d} | Train Loss: {loss:.6f} | Val AUC: {val_auc:.6f}, Val Loss: {val_loss:.6f}]".format(
                    epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss))

        te_auc, te_loss = self.evaluate_test("test")
        for d in self.domain_list:
            self.logger.info(
                "Early stop at epoch {epoch:d}|Test AUC{d:}: {te_auc:.6f}, Test Loss{d:}:{te_loss:.6f}".
                    format(epoch=epoch_idx, d=d, te_auc=te_auc[d], te_loss=te_loss[d]))

        torch.save({
            'domain_hypernet_state_dict': self.network.domain_hypernet.state_dict(),
            'domain_embedding_state_dict': self.network.embedding.state_dict(),
            'domain1_mask_state_dict': self.network.domain1_mask.state_dict(),
            'domain2_mask_state_dict': self.network.domain2_mask.state_dict(),
            'domain3_mask_state_dict': self.network.domain3_mask.state_dict()
        }, 'mask_params_s3.pth')

    def evaluate_val(self, on: str):
        preds, trues = [], []
        for feature, label, domain in self.dataloader.get_data(on, batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature, domain)
            label = label.detach().cpu().numpy()
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
        for d in self.domain_list:
            preds_dist[d] = []
            trues_dist[d] = []
        for feature, label, domain in self.dataloader.get_data(on, batch_size=self.bs * 10):
            pred = self.eval_on_batch(feature, domain)
            label = label.detach().cpu().numpy()
            domain = domain.detach().cpu().numpy()
            for d in self.domain_list:
                ind = np.nonzero(domain == d)[0]
                preds_dist[d].append(pred[ind])
                trues_dist[d].append(label[ind])
        auc = {}
        loss = {}
        for d in self.domain_list:
            y_pred = np.concatenate(preds_dist[d]).astype("float64")
            y_true = np.concatenate(trues_dist[d]).astype("float64")
            auc[d] = metrics.roc_auc_score(y_true, y_pred)
            loss[d] = metrics.log_loss(y_true, y_pred)
        return auc, loss

    def train(self, epochs, retrain_id=0):
        print('#'*150)
        print("retrain id:", retrain_id)
        #self.opt["model_opt"]["mlp_dropout"] = 0.0
        print("mlp_dropout:", self.opt["model_opt"]["mlp_dropout"])
        self.network = SSIM3mask.getModel(self.opt["model"], self.opt["model_opt"]).to(self.device)
        self.network.load_state_dict(torch.load(self.model_init_dir))

        checkpoint = torch.load('mask_params_s3.pth')
        self.network.domain_hypernet.load_state_dict(checkpoint['domain_hypernet_state_dict'])
        self.network.domain_embedding.load_state_dict(checkpoint['domain_embedding_state_dict'])
        self.network.domain1_mask.load_state_dict(checkpoint['domain1_mask_state_dict'])
        self.network.domain2_mask.load_state_dict(checkpoint['domain2_mask_state_dict'])
        self.network.domain3_mask.load_state_dict(checkpoint['domain3_mask_state_dict'])
        self.network.ticket = True
        self.network.thre = self.thre
        self.network.topk = self.topk[retrain_id]

        cur_auc = 0.0
        early_stop = False
        self.optim = SSIM3mask.getOptim(self.network, "adam", self.r_lr, self.m_lr, self.r_l2)[:1]

        self.logger.info("-----------------Begin Train-----------------")
        self.logger.info("Ticket:{t}".format(t=self.network.ticket))
        self.logger.info("Topk:{k}".format(k=self.network.topk))

        for epoch_idx in range(int(epochs)):
            train_loss = .0
            train_share_num = 0
            step = 0
            for feature, label, domain in self.ds:
                loss, share_num = self.train_on_batch(label, feature, domain, retrain=True)
                train_loss += loss
                train_share_num += share_num
                step += 1
                if step % 1000 == 0:
                    print('share_num:', share_num, 'train_share_num:', train_share_num)
                    self.logger.info("[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".
                                     format(epoch=epoch_idx, setp=step, loss=loss))
            train_loss /= step
            val_auc, val_loss = self.evaluate_val("val")
            print('share_num:', share_num, 'train_share_num:', train_share_num)
            self.logger.info(
                "[Epoch {epoch:d} | Train Loss: {loss:.6f} | Val AUC: {val_auc:.6f}, Val Loss: {val_loss:.6f}]".format(
                    epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss))
            if val_auc > cur_auc:
                cur_auc = val_auc
                torch.save(self.network.state_dict(), self.model_dir)
            else:
                self.network.load_state_dict(torch.load(self.model_dir))
                self.network.to(self.device)
                early_stop = True
                te_auc, te_loss = self.evaluate_test("test")
                for d in self.domain_list:
                    self.logger.info(
                        "Early stop at epoch {epoch:d}|Test AUC{d:}: {te_auc:.6f}, Test Loss{d:}:{te_loss:.6f}".
                            format(epoch=epoch_idx, d=d, te_auc=te_auc[d], te_loss=te_loss[d]))
                break
        if not early_stop:
            te_auc, te_loss = self.evaluate_test("test")
            for d in self.domain_list:
                self.logger.info(
                    "Final Test AUC{d:}:{te_auc:.6f}, Test Loss{d:}:{te_loss:.6f}".format(d=d, te_auc=te_auc[d],
                                                                                          te_loss=te_loss[d]))


def main():
    sys.path.extend(["./modules", "./dataloader", "./utils"])
    if args.dataset.lower() == "ali-ccp":
        field_dim = trainUtils.get_stats("./data/ali-ccp/stats")
        data_dir = "./data/ali-ccp/tfrecord"
        field = len(field_dim)
        feature = sum(field_dim)
    else:
        print("dataset error")
    model_opt = {
        "latent_dim": args.dim, "feat_num": feature, "field_num": field,
        "mlp_dropout": args.mlp_dropout, "use_bn": args.mlp_bn, "mlp_dims": args.mlp_dims, "cross": args.cross
    }

    opt = {
        "model_opt": model_opt, "dataset": args.dataset, "model": args.model, "lr": args.lr, "l2": args.l2,
        "bsize": args.bsize, "optimizer": args.optim, "data_dir": data_dir, "save_dir": args.save_dir,
        "cuda": args.cuda, "search_epoch": args.search_epoch, "lambda1": args.lambda1, "r_lr": args.r_lr, "r_l2": args.r_l2,
        "final_temp": args.final_temp, "thre": args.thre, "m_lr": args.m_lr, "topk": args.topk
    }
    print(opt)
    trainer = Trainer(opt)
    if args.search_epoch > 0:
        trainer.search()
        
    trainer.train(args.max_epoch)


if __name__ == "__main__":
    """
    python SSIM3trainer.py --dataset 'ali-ccp' --model 'dnn'   
    """
    main()
