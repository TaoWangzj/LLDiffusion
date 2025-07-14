from argparse import ArgumentParser

import numpy as np
import torch


def convert(src_keys):
    PREFIX = "up."
    tgt_keys = []
    for k in src_keys:
        index = k.find(PREFIX)
        if index != -1:
            layer = int(k[index + len(PREFIX)])
            k = k.replace(PREFIX + str(layer), PREFIX + str(5 - layer))
        tgt_keys.append(k)
    return tgt_keys


def main():
    # path, out_path = args.input, args.output

    path = "./logs/deg-unet/original/best-10899-18.39.pth"
    out_path = "./logs/deg-unet/original_revise/best-10899-18.39.pth"
    ckpt = torch.load(path, map_location="cpu")

    state = ckpt["state_dict"]
    src_keys = list(state.keys())
    tgt_keys = convert(src_keys)
    state = {k: state[src_keys[i]] for i, k in enumerate(tgt_keys)}
    ckpt["state_dict"] = state

    ema_state = ckpt["ema_helper"]
    ema_src_keys = list(ema_state.keys())
    ema_tgt_keys = convert(ema_src_keys)
    ema_state = {k: ema_state[ema_src_keys[i]]
                 for i, k in enumerate(ema_tgt_keys)}
    ckpt["ema_helper"] = ema_state

    tgt_up_keys = [key for key in tgt_keys if key.find(
        "module.deg_unet.up.") != -1]
    src_up_keys = [key for key in src_keys if key.find(
        "module.deg_unet.up.") != -1]

    up_indices = [int(key[len("module.deg_unet.up.")]) for key in tgt_up_keys]
    up_indices = np.array(up_indices)
    up_indices = np.argsort(up_indices)
    tgt_up_keys = [tgt_up_keys[i] for i in up_indices]
    src_up_keys = [src_up_keys[i] for i in up_indices]

    start = src_keys.index(src_up_keys[0])
    src_pos_keys = src_keys.copy()
    src_pos_keys[start: start + len(tgt_up_keys)] = src_up_keys
    
    tgt_indices = [src_keys.index(k) for k in src_pos_keys]

    optimizer_state = ckpt["optimizer"]["state"]
    optimizer_state = {i: optimizer_state[i] for i in range(
        len(tgt_indices)) if i in optimizer_state}

    ckpt["optimizer"]["state"] = optimizer_state

    torch.save(ckpt, out_path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input",default="logs/deg-unet/original/best-10899-18.39.pth")
    parser.add_argument("-o", "--output",default="logs/deg-unet/original_revise/")
    return parser.parse_args()


if __name__ == "__main__":
    # args = parse_args()
    # print(args.input)
    # print(args.output)
    main()