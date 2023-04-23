import argparse
import os
import random
from collections import OrderedDict
from glob import glob
from itertools import chain

import numpy as np
import pandas as pd

from bal.evaluator.slice import ActiveEvaluator, ACTIVE_SELECTORS
from bal.utils.utils import multiprocess_agent, get_init_parameter_names, read_lines, write_txt


def active_iterator(pt, threshold, test_root, bank_root, ref_root, mode, n_classes):
    active_iter = ActiveEvaluator(pt,
                                  threshold,
                                  test_root,
                                  bank_root,
                                  ref_root,
                                  mode,
                                  n_classes)

    return active_iter.run()


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
    return cls(**{k: v for k, v in vars(args).items() if k in parameter_names})


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
    print(len(buffer_cases))
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
    random.seed(int(in_args.seed))
    np.random.seed(int(in_args.seed))
    _stage = in_args.stage
    struct = in_args.structure
    mode = in_args.mode
    threshold = in_args.threshold
    method = in_args.method
    exp_n = f'{struct}_all_stage{_stage}_iter{in_args.iteration}'
    size = in_args.size
    root = in_args.root
    txt_root = r'/win/salmon/user/li/Project/MRI2CT/Bayesian_unet/scripts/exp/{}/seed_{}/{}layer/{}'.format(in_args.exp,
                                                                                                            in_args.seed,
                                                                                                            in_args.n_layers,
                                                                                                            method)
    image_root = in_args.image_root
    test_root = os.path.join(root, str(size), in_args.loss, str(in_args.exp), 'seed_{}'.format(in_args.seed),
                             '{}layer'.format(in_args.n_layers), method, exp_n, 'testing')
    bank_root = os.path.join(root, str(size), in_args.loss, str(in_args.exp), 'seed_{}'.format(in_args.seed),
                             '{}layer'.format(in_args.n_layers), method, exp_n, 'databank')
    ref_root = in_args.reference_root
    prev_train_txt_pth = os.path.join(txt_root, 'id-list_trial-{}_training-0.txt'.format(_stage))
    prev_bank_txt_pth = os.path.join(txt_root, 'id-list_trial-{}_databank-0.txt'.format(_stage))
    save_root = os.path.join(in_args.save_root, struct, 'dice', str(size), in_args.loss, str(in_args.exp),
                             'seed_{}'.format(in_args.seed), '{}layer'.format(in_args.n_layers), method, exp_n)
    os.makedirs(save_root, exist_ok=True)

    prev_train_cases = read_lines(prev_train_txt_pth)
    prev_bank_cases = read_lines(prev_bank_txt_pth)

    test_cases = [os.path.basename(r) for r in glob(os.path.join(test_root, '*'))]
    bank_cases = [os.path.join(*(r.split(os.sep)[-2:])) for r in
                  glob(os.path.join(bank_root, '*', 'image_*_label.mha'))]

    print('test cases: {}'.format(len(test_cases)), 'bank cases: {}'.format(len(bank_cases)))

    args = []
    if mode == 'testing':
        for pt in test_cases:
            args.append([pt, threshold, test_root, bank_root, ref_root, mode, in_args.n_classes])

        with mp.Pool(mp.cpu_count() - 1) as p:
            out_dfs = p.starmap(active_iterator, args)

        df = pd.concat(out_dfs)
        df.to_excel(os.path.join(save_root, exp_n.split('_')[1] + '_' + mode + '.xlsx'))

    elif mode == 'databank':
        if method == 'random':
            adding_cases = random.sample(prev_bank_cases, in_args.increment)
        else:
            for pt in bank_cases:
                args.append([pt, threshold, test_root, bank_root, ref_root, mode, in_args.n_classes])

            with mp.Pool(mp.cpu_count() - 1) as p:
                out_dfs = p.starmap(active_iterator, args)

            df = pd.concat(out_dfs)
            df.to_excel(os.path.join(save_root, exp_n.split('_')[1] + '_' + mode + '.xlsx'))

            df = df.sort_values(by=['all'], ascending=False)
            uncertainties = list(df.head(in_args.buffer_size)['all'].values)
            if _stage == 1 and method == 'simi_mi':
                case_names = np.array([r.split('/')[0] for r in df.index.values])
                _, idx = np.unique(case_names, return_index=True)
                unique_case_names = case_names[np.sort(idx)]
                case_dfs = []

                for case_name in unique_case_names:
                    num_slice = in_args.buffer_size // len(unique_case_names)
                    if len(case_dfs) == 0:
                        num_slice = in_args.buffer_size - num_slice * (len(unique_case_names) - 1)
                    tmp_indices = [i for i, r in enumerate(df.index.values) if case_name in r]
                    print('Case: {}, number of slices: {}'.format(case_name, len(tmp_indices)))
                    tmp_df = df.iloc[tmp_indices]
                    case_dfs.append(tmp_df.head(num_slice))
                buffer_cases = list(pd.concat(case_dfs).index.values)
            else:
                buffer_cases = list(df.head(in_args.buffer_size).index.values)
            print(len(buffer_cases))

            ic = SliceCluster(image_root, buffer_cases, in_args.increment,
                              method=method,
                              bank_paths=prev_bank_cases,
                              train_paths=prev_train_cases,
                              coef=in_args.coef,
                              clip=in_args.clip,
                              img_vmin=in_args.img_vmin,
                              img_vmax=in_args.img_vmax,
                              uncertainty=uncertainties)
            adding_indices = ic.run()
            adding_cases = [buffer_cases[i] for i in adding_indices]
        train_cases = list(chain(prev_train_cases, adding_cases))
        bank_cases = [i for i in prev_bank_cases if i not in adding_cases]
        new_train_txt_pth = os.path.join(txt_root, 'id-list_trial-{}_training-0.txt'.format(_stage + 1))
        new_bank_txt_pth = os.path.join(txt_root, 'id-list_trial-{}_databank-0.txt'.format(_stage + 1))
        write_txt(train_cases, new_train_txt_pth)
        write_txt(bank_cases, new_bank_txt_pth)


if __name__ == '__main__':
    main()
