import torch
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin

import os
import time
import numpy as np
from collections import OrderedDict, defaultdict

from utils.eval_t2m import evaluation_vae, test_vae
from utils.utils import print_current_loss

def def_value():
    return 0.0

class VAETrainer:
    def __init__(self, opt, vae):
        self.opt = opt
        self.vae = vae

        if opt.is_train:
            self.logger = SummaryWriter(opt.log_dir)
            if opt.recon_loss == "l1":
                self.recon_criterion = torch.nn.L1Loss()
            elif opt.recon_loss == "l1_smooth":
                self.recon_criterion = torch.nn.SmoothL1Loss()
        

    def train_forward(self, MB_rep_data, name_list):
        MB_motion = MB_rep_data.to(self.opt.device, dtype=torch.float32)

        pred_MB_motion, loss_dict = self.vae.forward(MB_motion)

        self.MB_motion = MB_motion
        self.pred_MB_motion = pred_MB_motion
        self.name_list=name_list

        # loss
        loss_rec = self.recon_criterion(pred_MB_motion, MB_motion)
        loss_kl = loss_dict["loss_kl"]

        loss = loss_rec + loss_kl * self.opt.lambda_kl

        loss_dict["loss_recon"] = loss_rec

        return loss, loss_dict


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optim.param_groups:
            param_group["lr"] = current_lr


    def save(self, file_name, epoch, total_iter):
        state = {
            "vae": self.vae.state_dict(),
            "optim": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "total_iter": total_iter,
        }
        torch.save(state, file_name)


    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.opt.device)
        self.vae.load_state_dict(checkpoint["vae"])
        self.optim.load_state_dict(checkpoint["optim"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        return checkpoint["epoch"], checkpoint["total_iter"]


    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        self.vae.to(self.opt.device)

        # optimizer
        self.optim = torch.optim.AdamW(self.vae.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)    ## lr=2e-4, weight_decay = 0
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.opt.milestones, gamma=self.opt.gamma) ## milestones=[150000, 250000], gamma = 0.05

        epoch = 0
        it = 0
        if self.opt.is_continue:    ## False
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)    ## 返回的是cumsum的结果，实际的训练个数应该是20942。此处结果大约是7217
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(eval_val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        # eval
        best_mse, writer = evaluation_vae(
            self.opt.model_dir, eval_val_loader, self.vae, self.logger, epoch, best_mse=100000, 
            eval_wrapper=eval_wrapper)

        # training loop
        while epoch < self.opt.max_epoch:
            self.vae.train()
            for i, data in enumerate(train_loader):
                batch_data = data[0]
                MB_rep_data = data[1]
                name_list = data[2]
                it += 1 ## 这里的it就是steps的意思
                if it < self.opt.warm_up_iter:
                    curr_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                # forward
                self.optim.zero_grad()
                loss, loss_dict = self.train_forward(MB_rep_data, name_list)
                loss.backward()
                self.optim.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                
                # log
                logs["loss"] += loss.item()
                logs["lr"] += self.optim.param_groups[0]['lr']
                for tag, value in loss_dict.items():
                    logs[tag] += value.item()

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, f'{str(epoch)}_{str(it)}.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            print('Validation time:')
            self.vae.eval()
            val_log = defaultdict(def_value, OrderedDict())
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    batch_data = data[0]
                    MB_rep_data = data[1]
                    name_list = data[2]
                    loss, loss_dict = self.train_forward(MB_rep_data, name_list)

                    val_log["loss"] += loss.item()
                    for tag, value in loss_dict.items():
                        val_log[tag] += value.item()
            
            msg = "Validation loss: "
            for tag, value in val_log.items():
                self.logger.add_scalar('Val/%s'%tag, value / len(val_loader), epoch)
                msg += "%s: %.3f, " % (tag, value / len(val_loader))
            print(msg)
            
            # evaluation
            if epoch % self.opt.eval_every_e == 0:
                best_mse, writer = evaluation_vae(
                    self.opt.model_dir, eval_val_loader, self.vae, self.logger, epoch, best_mse=best_mse, 
                    eval_wrapper=eval_wrapper)

                # data = torch.cat([self.motion[:4], self.pred_motion[:4]], dim=0).detach().cpu().numpy()
                data = torch.cat([self.MB_motion[:4], self.pred_MB_motion[:4]], dim=0).detach().cpu().numpy() ## [256,64,17,512]->两个cat成[512,64,17,512]
                   
                save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(MB_motion_data = data, name_list=self.name_list[:4], save_dir = save_dir)
    
    
    @torch.no_grad()
    def test(self, eval_wrapper, eval_val_loader, repeat_time, save_dir, cal_mm=True):
        os.makedirs(save_dir, exist_ok=True)
        f = open(pjoin(save_dir, 'eval.log'), 'w')

        self.vae.eval()
        metrics = {
            "fid": [],
            "div": [],
            "top1": [],
            "top2": [],
            "top3": [],
            "matching": [],
            "mpjpe": [],
            "mm": []
        }
        for i in range(repeat_time):
            fid, diversity, R_precision, matching_score, mpjpe, multimodality = test_vae(
                eval_val_loader, self.vae, i, eval_wrapper, self.opt.joints_num, cal_mm=cal_mm
            )
            metrics["fid"].append(fid)
            metrics["div"].append(diversity)
            metrics["top1"].append(R_precision[0])
            metrics["top2"].append(R_precision[1])
            metrics["top3"].append(R_precision[2])
            metrics["matching"].append(matching_score)
            metrics["mpjpe"].append(mpjpe)
            metrics["mm"].append(multimodality)

        fid = np.array(metrics["fid"])
        div = np.array(metrics["div"])
        top1 = np.array(metrics["top1"])
        top2 = np.array(metrics["top2"])
        top3 = np.array(metrics["top3"])
        matching = np.array(metrics["matching"])
        mpjpe = np.array(metrics["mpjpe"])
        mm = np.array(metrics["mm"])

        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMPJPE: {np.mean(mpjpe):.3f}, conf. {np.std(mpjpe)*1.96/np.sqrt(repeat_time):.3f}\n" \
                    f"\tMultimodality: {np.mean(mm):.3f}, conf. {np.std(mm)*1.96/np.sqrt(repeat_time):.3f}\n\n"
        print(msg_final)
        print(msg_final, file=f, flush=True)

        f.close()