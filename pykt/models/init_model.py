import torch
import numpy as np
import os

from .atdkt import ATDKT

device = "cpu" if not torch.cuda.is_available() else "cuda"

import pandas as pd
def save_qcemb(model, emb_save, ckpt_path):
    fs = []
    for k in ckpt_path.split("/"):
        if k.strip() != "":
            fs.append(k)
    fname = "_".join(fs)
    for n, p in model.question_emb.named_parameters():
        pd.to_pickle(p, os.path.join(emb_save, fname+"qemb_from_atdkt.pkl"))
    for n, p in model.concept_emb.named_parameters():
        pd.to_pickle(p, os.path.join(emb_save, fname+"cemb_from_atdkt.pkl"))
    for n, p in model.interaction_emb.named_parameters():
        pd.to_pickle(p, os.path.join(emb_save, fname+"xemb_from_atdkt.pkl"))

def init_model(model_name, model_config, data_config, emb_type):
    print(f"in init_model, model_name: {model_name}")
    if model_name == "atdkt":
        model = ATDKT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    else:
        print(f"The wrong model name: {model_name} was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    infs = model_name.split("_")
    save = False
    if len(infs) == 2:
        model_name, save = infs[0], True
    print(f"in load model! model name: {model_name}, save: {save}")
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
    model.load_state_dict(net)
    if model_name == "atdkt" and save:
        save_qcemb(model, data_config["emb_save"], ckpt_path)
    return model
