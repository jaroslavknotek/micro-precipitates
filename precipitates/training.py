from tqdm.auto import tqdm
import precipitates.nnet as nnet
from functools import partial
import itertools

import numpy as np
import torch

#don't import wandb

import logging
logging.basicConfig()
logger = logging.getLogger('pred')
logger.setLevel(logging.DEBUG)



def _calculate_loss(
    y,
    pred,
    mask,
    has_label,
    wm,
    loss_denoise,
    loss,
    denoise_loss_weight = 2):
    
    ls_segmentation = loss(pred[:,1:,...]*has_label,y[:,1:,...]*has_label)
    ls_seg_masked = (ls_segmentation*wm).mean()
    
    if denoise_loss_weight == 0:
        return ls_seg_masked
    else:   
        ls_denoise = loss_denoise(pred[:,0] * mask, y[:,0] * mask)
        return ls_seg_masked + ls_denoise*denoise_loss_weight

def train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_denoise,
    loss,
    model_save_dir = None,
    denoise_loss_weight = 2,
    wandb = None,
    device = 'cpu',
    patience = 5,
    epochs_n = None
):
    early_stopper = nnet.EarlyStopper(patience=patience, min_delta=0)
    
    loss_dict = {}
    if epochs_n is None:
        epochs = itertools.count()
    else:
        epochs = range(epochs_n)
        
    for e in tqdm(epochs,desc='Training epochs'):
        model.train()
        train_losses = []
        val_losses = []

        for rec in train_dataloader:
            x, y,mask,has_label,wm = [r.to(device) for r in rec]
            optimizer.zero_grad()
            pred = model(x)
            ls = _calculate_loss(
                y,
                pred,
                mask,
                has_label,
                wm,
                loss_denoise,
                loss,
                denoise_loss_weight
            )

            ls.backward()
            optimizer.step()
            train_losses.append(ls.item())

        with torch.no_grad():
            model.eval()
            for rec in val_dataloader:
                x, y,mask,has_label,wm = [r.to(device) for r in rec]            
                pred = model(x)
                ls = _calculate_loss(
                    y,
                    pred,
                    mask,
                    has_label,
                    wm,
                    loss_denoise,
                    loss,
                    denoise_loss_weight
                )
                val_losses.append(ls.item())

        loss_dict.setdefault("train_loss",[]).append(train_losses)
        loss_dict.setdefault("val_loss",[]).append(val_losses)

        validation_loss = np.mean(val_losses)
        if wandb is not None:
            wandb.log({"val_loss": validation_loss})
        
        logger.info(f"epoch {e} val:{validation_loss:.5f} stopper counter:{early_stopper.counter}")
        if early_stopper.early_stop(validation_loss):             
            logger.info("early stopping")
            break
        
        if model_save_dir!=None and early_stopper.counter == 0:
            logger.info("Saving best model")
            best_model_path = model_save_dir/'model-best.torch'
            torch.save(model,best_model_path)
            
    return model, loss_dict