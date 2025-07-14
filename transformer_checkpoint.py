import torch


def convert(path, out_path):
    # ckpt = torch.load(path, map_location="cpu")
    ckpt = torch.load(path)
    state = ckpt["state_dict"]
    keys = list(state.keys())

    PREFIX = "up."
    out = {}
    for k in keys:
        index = k.find(PREFIX)
        if index != -1:
            layer = int(k[index + len(PREFIX)])
            out_k = k.replace(PREFIX + str(layer), PREFIX + str(5 - layer))
            out[out_k] = state[k]
        else:
            out[k] = state[k]

    ckpt["state_dict"] = out
    torch.save(out, out_path)

import os

if __name__ == "__main__":
    files = os.listdir("./logs/deg-unet/original")
    
    for file in files:
        path = os.path.join("./logs/deg-unet/original", file)
        out_path = os.path.join("./logs/deg-unet/ckpts3", file)
        convert(path, out_path)