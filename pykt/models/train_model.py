import os
import torch
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
import numpy as np
from .evaluate_model import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cal_loss(model, ys, r, rshft, sm, preloss=[], epoch=0, flag=False):
    model_name = model.model_name

    if model_name in ["atdkt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # print(f"loss1: {y.shape}")
        loss1 = binary_cross_entropy(y.double(), t.double())

        if model.emb_type.find("predcurc") != -1:
            if model.emb_type.find("his") != -1:
                loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
            else:
                loss = model.l1*loss1+model.l2*ys[1]
        elif model.emb_type.find("predhis") != -1:
            loss = model.l1*loss1+model.l2*ys[1]
        else:
            loss = loss1
        if flag:
            loss = loss1

    return loss

def model_forward(model, data, epoch):
    model_name = model.model_name
    dcur = data
    q, c, r, t = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["tseqs"]
    qshft, cshft, rshft, tshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_tseqs"]
    m, sm = dcur["masks"], dcur["smasks"]

    ys, preloss = [], []

    if model_name in ["atdkt"]:
        # is_repeat = dcur["is_repeat"]
        y, y2, y3 = model(dcur, train=True)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys = [y, y2, y3] # first: yshft
    loss = cal_loss(model, ys, r, rshft, sm, preloss, epoch)
    return loss
    

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False):
    max_auc, best_epoch = 0, -1
    train_step = 0
    
    for i in range(1, num_epochs + 1):
        loss_mean = []
        for data in train_loader:
            train_step+=1
            model.train()
            loss = model_forward(model, data, i)
            opt.zero_grad()
            loss.backward()#compute gradients 
            opt.step()#update model’s parameters
                
            loss_mean.append(loss.detach().cpu().numpy())
       
        loss_mean = np.mean(loss_mean)
        auc, acc = evaluate(model, valid_loader, model.model_name)

        if auc > max_auc:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type+"_model.ckpt"))
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
            # window_testauc, window_testacc = -1, -1
            validauc, validacc = round(auc, 4), round(acc, 4)#model.evaluate(valid_loader, emb_type)
            # trainauc, trainacc = model.evaluate(train_loader, emb_type)
            testauc, testacc, window_testauc, window_testacc = round(testauc, 4), round(testacc, 4), round(window_testauc, 4), round(window_testacc, 4)
            max_auc = round(max_auc, 4)
        print(f"Epoch: {i}, validauc: {validauc}, validacc: {validacc}, best epoch: {best_epoch}, best auc: {max_auc}, loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(f"            testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")

        if i - best_epoch >= 10:
            break
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch
