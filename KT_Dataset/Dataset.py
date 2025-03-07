import csv
import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
import os
import numpy as np

def cat(seq, max_len, step_len=None):
    step_len = max_len if step_len is None else step_len
    seq_arr = []
    idx = 0
    while idx + max_len <= len(seq):
        seq_arr.append(seq[idx:idx + max_len])
        idx += step_len
    if idx < len(seq):
        seq_arr.append(seq[max(len(seq) - max_len, 0): len(seq)])
    return seq_arr


def pad(seq, max_len):
    seq.extend([-1 for i in range(max_len - len(seq))])
    return seq


def pad_c_seq(seq, max_len):
    seq.extend([torch.zeros(seq[0].shape[0]) for i in range(max_len - len(seq))])
    return seq


def get_iter(x, tqdm_info):
    if tqdm_info:
        return tqdm(x)
    else:
        return x


class KtDataset(Dataset):
    def __init__(self, max_len=200, step_len=200, pth="Junyi", tqdm_info=True):
        with open(os.path.join(pth, "P2C.json"), encoding='utf8') as i_f:
            self.problem_to_concept = json.load(i_f)
        with open(os.path.join(pth, "data.json"), encoding='utf8') as i_f:
            log_dict = json.load(i_f)
            self.data = log_dict

        self.problem_seq_list = []
        self.concepts_seq_list = []
        self.response_seq_list = []

        self.max_pid = 0
        self.max_cid = 0
        for stu_log in get_iter(self.data.values(), tqdm_info):
            for p_id, response in stu_log:
                self.max_pid = max(self.max_pid, p_id)
                for c_id in self.problem_to_concept[str(p_id)]:
                    self.max_cid = max(self.max_cid, c_id)

        for stu_log in get_iter(self.data.values(), tqdm_info):
            p_seq = []
            c_seq = []
            r_seq = []
            for p_id, response in stu_log:
                p_seq.append(p_id)
                r_seq.append(response)
                c_onehot = torch.zeros(self.max_cid+1)
                for c_id in self.problem_to_concept[str(p_id)]:
                    c_onehot[c_id] = 1
                c_seq.append(c_onehot)

            if len(p_seq) > max_len:
                p_seq_arr = cat(seq=p_seq, max_len=max_len, step_len=step_len)
                c_seq_arr = cat(seq=c_seq, max_len=max_len, step_len=step_len)
                r_seq_arr = cat(seq=r_seq, max_len=max_len, step_len=step_len)

                self.problem_seq_list.extend(p_seq_arr)
                self.concepts_seq_list.extend(c_seq_arr)
                self.response_seq_list.extend(r_seq_arr)

            else:
                pad(p_seq, max_len)
                pad_c_seq(c_seq, max_len)
                pad(r_seq, max_len)

                self.problem_seq_list.append(p_seq)
                self.concepts_seq_list.append(c_seq)
                self.response_seq_list.append(r_seq)
        assert (len(self.problem_seq_list) == len(self.concepts_seq_list) and
                len(self.problem_seq_list) == len(self.response_seq_list))

    def __len__(self):
        return len(self.problem_seq_list)

    def __getitem__(self, item):
        return (torch.tensor(self.problem_seq_list[item]),
                torch.stack(self.concepts_seq_list[item]),
                torch.tensor(self.response_seq_list[item]))


def generate_dataloader(full_dataset, fold, batch_size=32):
    assert 1 <= fold <= 10, "Fold number must be between 1 and 10"

    # 计算数据集的大小
    dataset_size = len(full_dataset)

    # 生成数据集的所有索引
    indices = list(range(dataset_size))

    # 打乱索引
    np.random.seed(42)  # 固定随机种子以保证可重复性
    np.random.shuffle(indices)

    # 定义每个fold的大小
    fold_size = dataset_size // 10

    # 划分测试集的索引
    test_start = (fold - 1) * fold_size
    test_end = fold * fold_size if fold < 10 else dataset_size
    test_indices = indices[test_start:test_end]

    # 剩余的数据用来划分训练集和验证集
    remaining_indices = indices[:test_start] + indices[test_end:]

    # 划分训练集和验证集的大小
    train_size = int(0.8 * len(remaining_indices))

    # 划分训练集和验证集的索引
    train_indices = remaining_indices[:train_size]
    val_indices = remaining_indices[train_size:]

    # 定义采样器
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # 定义DataLoader
    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


def load_data(dataset_list, max_len=200, step_len=200, verbose=True, fold=1):
    train_arr = []
    val_arr = []
    test_arr = []
    p_constant_arr = []
    c_constant_arr = []
    for dataset_name in dataset_list:
        if verbose:
            print(f"Loading {dataset_name} ...")
        dataset_i = KtDataset(max_len=max_len, step_len=step_len, pth=dataset_name, tqdm_info=verbose)
        train_loader, val_loader, test_loader = generate_dataloader(dataset_i, fold=fold)

        train_arr.append(train_loader)
        val_arr.append(val_loader)
        test_arr.append(test_loader)
        p_constant_arr.append(dataset_i.max_pid)
        c_constant_arr.append(dataset_i.max_cid)

    return train_arr, val_arr, test_arr, p_constant_arr, c_constant_arr


def get_dataset_names():
    current_directory = os.path.dirname(__file__)
    files_and_directories = os.listdir(current_directory)
    directories = [d for d in files_and_directories if os.path.isdir(os.path.join(current_directory, d))]
    print(directories)



if __name__ == "__main__":
    get_dataset_names()
    exit()
    load_data(["ASSIST09", "ASSIST17", "ALGEBRA05", "Junyi"], max_len=200, step_len=200, verbose=True, fold=1)