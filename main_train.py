import cv2
import os, glob, shutil, copy
import numpy as np
import random
import torch
import json
from model_convlstm_ae import ST_AutoEncoder
from joblib import Parallel, delayed
from time import time
from copy import deepcopy
from utils import get_seq_img



f = open("config.json")
config =  json.load(f)
f.close()

device = torch.device("cuda")
model = ST_AutoEncoder(config["n_channels"], config["contrastive_loss"]["n_outliers"]).to(device)

################  get data: #####################
################### end get data ###############

loss_min, loss_val = 9999, 0

height, width = config["height"], config["width"]
scratch = config["path_scratch"]
shutil.rmtree(scratch, ignore_errors=True)
os.mkdir(scratch)

epoch = 0
opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-8)

outdir = config["path_save_model"]
if os.path.exists(outdir + "/best.pth"):
    """
    if os.path.exists(outdir + "/best.txt"):        
        f = open(outdir + "/best.txt", 'r')
        losses = f.read()
        losses = [float(l) for l in losses.splitlines()]
        f.close()
        loss_min = losses[-1]
    """        
    print("\n...load best train state\n")
    checkpoint = torch.load(outdir + '/best.pth')
    model.load_state_dict(checkpoint["model"])
    opt.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint['epoch']    
else:
    shutil.rmtree(outdir, ignore_errors=True)
    os.mkdir(outdir)


#model.to(device)
    
#x_worst=None
#index_worst=0
y = None
#restart = True
while epoch < config["epochs"]:

    batch_size, batch_size_val, n_outliers, n_outliers_val = config["batch_size_train"], config["batch_size_val"], config["contrastive_loss"]["n_outliers"], config["contrastive_loss"]["n_outliers_val"]
    time_s = time()    
    x = Parallel(n_jobs=max(1, batch_size // 2)) (delayed(get_seq_img)(config["seqlen"], config["dtheta"], angle_start=0, random_start=True) for _ in range(batch_size))

    #if epoch == 0 or restart:
    y = Parallel(n_jobs=max(1, n_outliers // 2)) (delayed(get_seq_img)(config["seqlen"], config["contrastive_loss"]["dtheta"], angle_start=0, random_start=True) for _ in range(n_outliers))
    #restart = False

        
    x = x + y    
    x = (np.array(x).reshape(batch_size+n_outliers, config["n_channels"], config["seqlen"], height, width) / 255).astype(np.float32)
    x = torch.tensor(x).to(device)
    model.train()    
    opt.zero_grad()

    #if epoch > 0:
    #    x[index_worst] = x_worst
        
    x_out, loss_contrastive  = model(x)
    #if loss_contrastive.item() > 0:    
    #    loss_contrastive  = loss_contrastive / batch_size /n_outliers
    
    if config["loss_type"] == "mse":
        loss =   ((x[:batch_size, :, :, :, :] - x_out[:batch_size, :, :, :, :]) ** 2).sum() / config["seqlen"] / config["n_channels"] / batch_size
    elif config["loss_type"] == "max":
        loss =   torch.max(((x[:batch_size, :, :, :, :] - x_out[:batch_size, :, :, :, :]) ** 2).sum(dim=4).sum(dim=3).sum(dim=2) / config["seqlen"] / config["n_channels"])
        ###index_worst = int(torch.argmax(loss))
        ###x_worst = deepcopy(x[index_worst]) # keep the worst sequence for repeatition
    else:
        print("error: loss type is undefined")
        quit()

    loss_tot = loss + loss_contrastive
        
    loss_tot.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        x_val = Parallel(n_jobs=max(1, batch_size // 2))( delayed(get_seq_img)(config["seqlen"], config["dtheta"], angle_start=0, random_start=True) for _ in range(batch_size_val))
        y_val = Parallel(n_jobs=max(1, n_outliers_val // 2)) (delayed(get_seq_img)(config["seqlen"], config["contrastive_loss"]["dtheta"], angle_start=0, random_start=True) for _ in range(n_outliers_val))       
        x_val = x_val + y_val        
        x_val = (np.array(x_val).reshape(batch_size_val + n_outliers_val, config["n_channels"], config["seqlen"], height, width) / 255).astype(np.float32)
        x_val = torch.tensor(x_val).to(device)
        x_val_out, loss_contrastive_val  = model(x_val)

        #if loss_contrastive_val.item() > 0:
        #    loss_contrastive_val  = loss_contrastive_val  / batch_size_val / n_outliers_val
        
    if config["loss_type"] == "mse":
        loss_val = ((x_val[:batch_size_val, :, :, :, :] - x_val_out[:batch_size_val, :, :, :, :]) ** 2).sum().item()/ config["seqlen"] / config["n_channels"] / batch_size_val
    elif config["loss_type"] == "max":        
        loss_val = torch.max(((x_val[:batch_size_val, :, :, :, :] - x_val_out[:batch_size_val, :, :, :, :]) ** 2).sum(dim=4).sum(dim=3).sum(dim=2)).item()/ config["seqlen"]  / config["n_channels"]
    else:
        print("error: loss type is undefined")
        quit()

    loss_val_tot = loss_val + loss_contrastive_val
        
    if loss_val_tot < loss_min and epoch > 10:
        loss_min = copy.deepcopy(loss_val_tot)
        checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': opt.state_dict() }        
        torch.save(checkpoint, outdir + '/best.pth')
        #torch.save(model, outdir + '/best.pth')
        
        with open(outdir + "/best.txt", "a") as f:
            f.write(str(loss_min.item()) + '\n')
            f.close()

    if epoch%10==0:
        torch.save(model, outdir + '/last.pth')

    if epoch % 10 == 0:
        cv2.imwrite(scratch + '/' + str(epoch).zfill(4) + '_grt-rec_' + str(loss_val_tot.item())[:7] + '.png',
                    cv2.hconcat([cv2.hconcat([cv2.vconcat(
                        x_val.detach().cpu().numpy().reshape(batch_size_val+n_outliers_val, config["seqlen"] , height, width, config["n_channels"])[i][:][
                        :][:] * 255), cv2.vconcat(x_val_out.detach().cpu().numpy().reshape(batch_size_val+n_outliers_val, config["seqlen"] , height, width, config["n_channels"])[i][:][:][:] * 255)]) for i in range(batch_size_val+n_outliers_val)]))
        
    print('epoch %10d  loss_t  %8.5f  loss_ct  %8.5f  loss_v  %8.5f  loss_cv  %8.5f  loss_min  %8.5f' % (epoch, loss, loss_contrastive, loss_val, loss_contrastive_val, loss_min))
    epoch += 1
    
    """
    Read config at new iteration again. You can adjust in config.json 
    some reasonable parameters affecting the training during run:

    config["epochs"]
    config["learning_rate"]
    config["loss_type"]
    config["batch_size_train"], config["batch_size_val"]
    config['preprocess']['gauss'], config['preprocess']['median']    
   
    you might also edit the others, but this might kill your run
    """
    
    f = open("config.json")
    config =  json.load(f)
    f.close()

    
