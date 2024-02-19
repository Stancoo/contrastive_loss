import cv2
import os, glob, shutil
import numpy as np
import random
import torch
import json
from warnings import warn
from model_convlstm_ae import ST_AutoEncoder
from utils import get_seq_img



f = open("config.json")
config =  json.load(f)
f.close()

device = torch.device("cuda")
model = ST_AutoEncoder(config["n_channels"],0).to(device)

################  get data: #####################
################### end get data ###############

weights = config["inference"]["weights"]
if os.path.exists(weights):
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint["model"])    
else:
    warn("model is not found")
    quit()

height, width = config["height"], config["width"]

model.eval()
losses = []

npixels = config["n_channels"] * config["seqlen"] * height * width


resdir = config["inference"]["results"]
shutil.rmtree(resdir, ignore_errors=True)
os.mkdir(resdir)

with torch.no_grad():
    for i in range(config["inference"]["n_samples"]):
        x_val = get_seq_img(config["seqlen"], config["dtheta"], angle_start=0, random_start=True, random_seq=eval(config["inference"]["random_seq"]))
        #x_val = get_seq_img_reverse(config["seqlen"], config["dtheta"], angle_start=0, random_start=True)        
        x_val = (np.array(x_val).reshape(1, config["n_channels"], config["seqlen"], height, width) / 255).astype(np.float32)            
        x_val = torch.tensor(x_val).to(device)
        x_val_out, _  = model(x_val)
        if config["inference"]["loss_type"] == "mse":
            loss_val = (((x_val[0, :, :, :] - x_val_out[0, :, :, :]) ** 2).sum() / config["seqlen"] / config["n_channels"]).item()  # euclidian distance
        elif config["inference"]["loss_type"] == "max":
            loss_val =   torch.max(((x_val[0, :, :, :] - x_val_out[0, :, :, :]) ** 2).sum(dim=3).sum(dim=2).sum(dim=1)).item()/ config["seqlen"] / config["n_channels"]
        else:
            print("error: loss type is undefined")
            quit()

        cv2.imwrite(resdir + '/' + str(i).zfill(6) + '_' + "%.5f" % loss_val + '.png',
                    cv2.hconcat([cv2.vconcat(
                        x_val.detach().cpu().numpy().reshape(config["seqlen"], height, width, config["n_channels"])[:][:][:] * 255),
                                 cv2.vconcat(
                                     x_val_out.detach().cpu().numpy().reshape(config["seqlen"], height, width, config["n_channels"])[:][:][:] * 255)]))

        print('%10i  loss_v  %8.5f' % (i, loss_val))
        losses.append(loss_val)

losses = np.array(losses)

if len(losses) > 0:
    bins = max(int(len(losses) // 10), 100)
    hist = np.histogram(losses, bins=bins, density=True)
    with open(resdir + "/hist.dat", "w+") as f:
        np.savetxt(f, np.column_stack([hist[1][:len(hist[0])], hist[0]]), fmt='%.10f')
        f.close()
