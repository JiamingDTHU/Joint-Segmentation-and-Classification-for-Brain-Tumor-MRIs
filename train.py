# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 22:29:41 2022

@author: 19813
"""
import torch
import torch.nn as nn
import copy
import time
import pandas as pd
def train(model,traindataloader,train_rate,criterion,optimizer,num_epochs=25):
    batch_num=len(traindataloader)
    train_batch_num=round(batch_num*train_rate)
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=0.0
    train_loss_all=[]
    train_acc_all=[]
    val_loss_all=[]
    val_acc_all=[]
    since=time.time
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_loss=0.0
        train_correct=0
        train_num=0
        
        val_loss=0.0
        val_corrects=0
        val_num=0
        for step,(b_x,b_y) in enumerate(traindataloader):
           if step < train_batch_num:
               model.train()#改为训练模式
               output=model(b_x)
               pre_lab=torch.argmax(output,1)
               loss=criterion(output,b_y)#损失函数
               optimizer.zero_grad()
               loss.backward
               optimizer.step()
               train_loss +=loss.item()*b_x.size(0)
               train_num +=b_x.size(0)
               
           else:
               model.eval()#改为评估模式
               output=model(b_x)
               pre_lab=torch.argmax(output,1)
               loss=criterion(output,b_y)
               val_loss +=loss.item()*b_x.size(0)
        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        train_acc_all=()#这里的预测准确率函数用dice，还没写入，其他朋友可以填入
        val_acc_all=()
        
        print('{}Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{}val Loss:{:.4f} val Acc:{:.4f}'.format(epoch,val_loss_all[-1],val_acc_all[-1]))
        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())
        time_use=time.time()-since
        print('train and val complete in {:.0f} m {:.0f}s'.format(time_use//60, time_use%60))
    model.load_state_dict(best_model_wts)
    train_process=pd.DataFrame(
        data={'epoch':range(num_epochs),
              'train_loss_all':train_loss_all,
              'val_loss_all':val_loss_all,
              'train_acc_all':train_acc_all,
              'val_acc_all':val_acc_all})
    return model,train_process
def cUNet():
    It=[]
optimizer=torch.optim.Adam(cUNet.parameters(),lr=0.0003)
cUNet,train_process=train(cUNet,train_loader,0.8,criterion,optimizer,num_epochs=25)