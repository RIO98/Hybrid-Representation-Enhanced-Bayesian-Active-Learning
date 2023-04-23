import argparse
import os
import random
from collections import OrderedDict
from glob import glob
from itertools import chain

import numpy as np
import pandas as pd

from bal.evaluator.volume import ActiveEvaluator, ACTIVE_SELECTORS
from bal.utils.utils import multiprocess_agent, get_init_parameter_names, read_lines, write_txt


def active_iterator(*args):
    return ActiveEvaluator(*args).run()


def get_args():
    parser = argparse.ArgumentParser(description="Bayesian Active Learning Iterator")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--n_layers", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", type=str, choices=["databank", "testing"], required=True)
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("-t", "--threshold", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--coef", type=float, default=0.25)
    parser.add_argument("--loss", type=str, default="focal")
    parser.add_argument("--structure", type=str, default="Quad")
    parser.add_argument("--metric", type=str, default="dice")
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--increment", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=6)
    parser.add_argument("--clip", action="store_false")
    parser.add_argument("--img_vmin", type=int, default=0)
    parser.add_argument("--img_vmax", type=int, default=255)
    parser.add_argument("--iteration", type=int, default=100000)
    parser.add_argument("--eval_save_root", type=str, default=r"./")
    parser.add_argument("--image_root", type=str, default=r"./")
    parser.add_argument("--txt_save_root", type=str, default=r"./")
    parser.add_argument("--pred_save_root", type=str, default=r"./")
    return parser.parse_args()


def class_initializer(cls, args):
    parameter_names = get_init_parameter_names(cls)
    return cls(**{k: v for k, v in args.items() if k in parameter_names})


def testing_mode(args, save_root, exp_n):
    out = multiprocess_agent(active_iterator, args)
    df = pd.concat(out)
    df.to_excel(os.path.join(save_root, f"{exp_n.split('_')[1]}_testing.xlsx"))


def databank_mode(args, in_args, exp_n, prev_bank_cases, prev_train_cases):
    out = multiprocess_agent(active_iterator, args)
    df = pd.concat(out)
    df.to_excel(os.path.join(in_args.eval_save_root, f"{exp_n.split('_')[1]}_databank.xlsx"))

    df = df.sort_values(by=["all"], ascending=False)
    buffer_cases = list(df.head(in_args.buffer_size).index.values)
    uncertainties = list(df.head(in_args.buffer_size).values)
    print(f"Pick {in_args.increment} cases from the buffer cases: {len(buffer_cases)}")
    parameters = OrderedDict([
        ("file_root", in_args.image_root),
        ("buffer_cases", buffer_cases),
        ("n_outputs", in_args.increment),
        ("bank_paths", prev_bank_cases),
        ("train_paths", prev_train_cases),
        ("coef", in_args.coef),
        ("clip", in_args.clip),
        ("img_vmin", in_args.img_vmin),
        ("img_vmax", in_args.img_vmax),
        ("uncertainty", uncertainties),
    ])

    ic = class_initializer(ACTIVE_SELECTORS[in_args.method], parameters)
    adding_cases = ic.run()
    adding_slices = [i for i in prev_bank_cases if i.split(os.sep)[-2] in adding_cases]
    train_slices = list(chain(prev_train_cases, adding_slices))
    bank_slices = [i for i in prev_bank_cases if i not in adding_slices]
    new_train_txt_pth = os.path.join(in_args.txt_save_root, f"id-list_trial-{in_args.stage + 1}_training-0.txt")
    new_bank_txt_pth = os.path.join(in_args.txt_save_root, f"id-list_trial-{in_args.stage + 1}_databank-0.txt")
    write_txt(train_slices, new_train_txt_pth)
    write_txt(bank_slices, new_bank_txt_pth)


def main():
    in_args = get_args()
    random.seed(in_args.seed)
    np.random.seed(in_args.seed)
    exp_n = f"stage{in_args.stage}_iter{in_args.iteration}"
    test_root = os.path.join(in_args.pred_save_root, "testing")
    bank_root = os.path.join(in_args.pred_save_root, "databank")
    prev_train_txt_pth = os.path.join(in_args.txt_save_root, f"id-list_trial-{in_args.stage}_training-0.txt")
    prev_bank_txt_pth = os.path.join(in_args.txt_save_root, f"id-list_trial-{in_args.stage}_databank-0.txt")
    os.makedirs(in_args.eval_save_root, exist_ok=True)

    prev_train_cases = read_lines(prev_train_txt_pth)
    prev_bank_cases = read_lines(prev_bank_txt_pth)

    test_cases = [os.path.basename(r) for r in glob(os.path.join(test_root, "*"))]
    bank_cases = [os.path.basename(r) for r in glob(os.path.join(bank_root, "*"))]

    if in_args.mode == "testing":
        print(f"test cases: {len(test_cases)}")
        args = [[pt, in_args.threshold, test_root, bank_root, in_args.image_root, in_args.mode, in_args.n_classes] for
                pt in test_cases]
        testing_mode(args, in_args.eval_save_root, exp_n)

    elif in_args.mode == "databank":
        print(f"bank cases: {len(bank_cases)}")
        print(f"clip option: {in_args.clip}, img_vmin: {in_args.img_vmin}, img_vmax: {in_args.img_vmax}")
        args = [[pt, in_args.threshold, test_root, bank_root, in_args.image_root, in_args.mode, in_args.n_classes] for
                pt in bank_cases]
        databank_mode(args, in_args, exp_n, prev_bank_cases, prev_train_cases)
    else:
        raise NotImplementedError("mode should be either testing or databank")


if __name__ == '__main__':
    main()
