#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from ase.io import read, write
import sys
import numpy as np
# sys.path.append('/scratch/work/wun2/github')
sys.path.append('/scratch/work/wun2/github/SpookyNet/')
from data.AtomicData import neighbor_list_and_relative_vec
import torch
import torch.utils.data as Data
import torch
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import time
import re
import ase_utils 
from data import AtomicData
from utils.torch_geometric import Batch, Dataset

from torch.utils.data import DataLoader
from spookynet import SpookyNet
from tqdm import tqdm


# In[2]:



def preprocess(batch_data):
        Z=batch_data['atomic_numbers'].squeeze()
        R=batch_data['pos']
#        R=torch.tensor(batch_data['pos'], requires_grad=True)
        idx_i=batch_data['edge_index'][0]
        idx_j=batch_data['edge_index'][1]
        batch_seg=batch_data['batch'].squeeze()
        Eref=batch_data['total_energy'].squeeze()
        Fref=batch_data['forces']
        Qaref=batch_data['atomic_charges'].squeeze()
        Qref=batch_data['total_charge'].squeeze()
        Sref=batch_data['total_charge'].squeeze()
        cell=batch_data['cell']
        cell_offsets=batch_data['edge_cell_shift']
        num_batch=(batch_seg[-1]+1).tolist()
        return Z, R, idx_i, idx_j, batch_seg, Eref, Fref, Qaref, Qref, Sref, cell, cell_offsets, num_batch


def save_loss_metrics(id_n=None, atomic_n=None, ref_e=None,ref_f=None,ref_qa=None,pred_e=None, pred_f=None, pred_qa=None, components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }, outfile='outfile'):
    loss_fn=torch.nn.MSELoss(reduction='mean')
    loss_total_e=loss_fn(pred_e,ref_e).cpu().detach().numpy().item()
    loss_total_f=loss_fn(pred_f,ref_f).cpu().detach().numpy().item()
    loss_total_qa=loss_fn(pred_qa, ref_qa).cpu().detach().numpy().item()
    loss_f_specie={}
    loss_qa_specie={}
    atomic_n_repeat=torch.repeat_interleave(atomic_n.unsqueeze(1), 3, dim=1)
    for i, j in components.items():       
        loss_f_specie[i]=loss_fn(pred_f[atomic_n_repeat==j],ref_f[atomic_n_repeat==j]).cpu().detach().numpy().item()
        loss_qa_specie[i]=loss_fn(pred_qa[atomic_n==j],ref_qa[atomic_n==j]).cpu().detach().numpy().item()
        
    with open(outfile, 'a') as f:
            f.write('%s, %s, %s, %s, %s, %s \n' %  (id_n, loss_total_e, loss_total_f, loss_total_qa, ', '.join([str(loss_f_specie[i]) for i in components.keys()]), ', '.join([str(loss_qa_specie[i]) for i in components.keys()])))

    return loss_f_specie, loss_qa_specie


if __name__=="__main__":  


    data_xyz=read('/scratch/phys/sin/Eric_summer_project_2022/Nian_calculations/process_OUTCAT_nian/nian_889_charge.extxyz',format='extxyz',index=':')
    dataset=[AtomicData.from_ase(data_xyz[i],3.5) for i in range(len(data_xyz))]
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [679, 210])
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=Batch.from_data_list)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, collate_fn=Batch.from_data_list)
    
    model=SpookyNet().float().cuda()
    model.load_state_dict(torch.load('results_spooky_2/spooky_charge_13461_param.pkl'))
    loss_fn=torch.nn.MSELoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001,amsgrad=True)
    components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }
    
    result_path='results_spooky_species'
    try:
        os.mkdir(result_path)
    except:
        pass
    train_output=os.path.join(result_path,'output_spooky_charge_train.txt')
    val_output=os.path.join(result_path,'output_spooky_charge_val.txt')
    val_output_total=os.path.join(result_path,'output_spooky_charge_val_total.txt')
    model_params=os.path.join(result_path,'spooky_charge_%s_param.pkl')
    with open(train_output, 'a') as f:
        f.write('epoch, loss_e, loss_f, loss_qa, '+ ', '.join([i+'_f_mae' for i in components.keys()])+ ', '+', '.join([i+'_qa_mae' for i in components.keys()])+'\n')
    with open(val_output, 'a') as f:
        f.write('epoch, loss_e, loss_f, loss_qa, '+ ', '.join([i+'_f_mae' for i in components.keys()])+ ', '+', '.join([i+'_qa_mae' for i in components.keys()])+'\n')
    with open(val_output_total, 'a') as f:
        f.write('epoch, loss_e, loss_f, loss_qa, '+ ', '.join([i+'_f_mae' for i in components.keys()])+ ', '+', '.join([i+'_qa_mae' for i in components.keys()])+'\n')


    epochs=20000
    
    
    # In[5]:
    
    
    
    for epoch in tqdm(range(epochs)): 
        for train_i, train_d in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            Z, R, idx_i, idx_j, batch_seg, Eref, Fref, Qaref, Qref, Sref, cell, cell_offsets, num_batch=preprocess(train_d)
            Z, R, idx_i, idx_j, batch_seg, Eref, Fref, Qaref, Qref, Sref, cell, cell_offsets=[i.cuda() for i in [Z, R, idx_i, idx_j, batch_seg, Eref, Fref, Qaref, Qref, Sref, cell, cell_offsets]]
        
            R.requires_grad=True
           
            energy, forces, dipole, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6, qa_2 = model(Z, Qref, Sref, R, idx_i, idx_j, num_batch=num_batch, batch_seg=batch_seg, cell=cell, cell_offsets=cell_offsets)
            loss_t_e=loss_fn(energy,Eref)
            loss_t_f=loss_fn(forces,Fref)
            loss_t_q=loss_fn(qa_2, Qaref)
            loss_t=loss_t_e+loss_t_f+loss_t_q    
            loss_t.backward()
            optimizer.step()
            save_loss_metrics(id_n=train_i, atomic_n=Z, ref_e=Eref,ref_f=Fref,ref_qa=Qaref,pred_e=energy, pred_f=forces, pred_qa=qa_2, components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }, outfile=train_output)

            
            
        if epoch%20==1:
            torch.save(model.state_dict(),model_params % epoch)
            model.eval()
            loss_val_sum=[]
            loss_val_e_sum=[]
            loss_val_f_sum=[]
            loss_val_q_sum=[]

        #    with torch.no_grad():
            for val_i, val_d in enumerate(val_dataloader):
                Z_v, R_v, idx_i_v, idx_j_v, batch_seg_v, Eref_v, Fref_v, Qaref_v, Qref_v, Sref_v, cell_v, cell_offsets_v, num_batch_v=preprocess(val_d)
                Z_v, R_v, idx_i_v, idx_j_v, batch_seg_v, Eref_v, Fref_v, Qaref_v, Qref_v, Sref_v, cell_v, cell_offsets_v=[i.cuda() for i in [Z_v, R_v, idx_i_v, idx_j_v, batch_seg_v, Eref_v, Fref_v, Qaref_v, Qref_v, Sref_v, cell_v, cell_offsets_v]]
                with torch.enable_grad():
                    R_v.requires_grad=True
                    energy_v, forces_v, dipole_v, f_v, ea_v, qa_v, ea_rep_v, ea_ele_v, ea_vdw_v, pa_v, c6_v, qa_2_v = model(Z_v, Qref_v, Sref_v, R_v, idx_i_v, idx_j_v, num_batch=num_batch_v, batch_seg=batch_seg_v, cell=cell_v, cell_offsets=cell_offsets_v)
                loss_v_e=loss_fn(energy_v,Eref_v)
                loss_v_f=loss_fn(forces_v,Fref_v)
                loss_v_q=loss_fn(qa_2_v, Qaref_v)
                loss_v=loss_v_e+loss_v_f+loss_v_q
                loss_val_sum.append(loss_t.cpu().detach().numpy().item())
                loss_val_e_sum.append(loss_t_e.cpu().detach().numpy().item())
                loss_val_f_sum.append(loss_t_f.cpu().detach().numpy().item())
                loss_val_q_sum.append(loss_t_q.cpu().detach().numpy().item())
                save_loss_metrics(id_n=val_i, atomic_n=Z_v, ref_e=Eref_v,ref_f=Fref_v,ref_qa=Qaref_v,pred_e=energy_v, pred_f=forces_v, pred_qa=qa_2_v, components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }, outfile=val_output)

#               print("####epoch %s    loss_v:  %s   loss_v_e:  %s   loss_v_f:  %s  loss_q: %s #####" % (val_i,loss_v.cpu().detach().numpy().item(), loss_v_e.cpu().detach().numpy().item(), loss_v_f.cpu().detach().numpy().item(), loss_v_q.cpu().detach().numpy().item()))
            loss_val_avg=np.array(loss_val_sum).mean()
            loss_val_e_avg=np.array(loss_val_e_sum).mean()
            loss_val_f_avg=np.array(loss_val_f_sum).mean()
            loss_val_q_avg=np.array(loss_val_q_sum).mean()              
            with open(val_output_total, 'a') as f:
                f.write('Epoch:%s, %s, %s, %s, %s \n' %  (epoch,  loss_val_avg.item(), loss_val_e_avg.item(), loss_val_f_avg.item(), loss_val_q_avg.item())) 
            print("####epoch %s    loss_t:  %s   loss_e:  %s   loss_f:  %s  loss_q: %s #####" % (epoch,loss_val_avg.item(), loss_val_e_avg.item(), loss_val_f_avg.item(), loss_val_q_avg.item()))               
        
        

