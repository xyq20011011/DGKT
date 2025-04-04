import csv
import json

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
min_log = 20


def divide_data():
    problem_code = {}
    concept_code = {}
    stu_code = {}

    problem_to_concept = {}
    stu_log = {}
    # 打开CSV文件
    with open('skill_builder_data.csv', newline='', encoding="ISO-8859-1") as csvfile:
        # 创建一个CSV阅读器对象
        csv_reader = csv.reader(csvfile)
        next(csv_reader)

        # 逐行读取CSV文件内容并打印
        for row in tqdm(csv_reader):
            order = row[0]
            user_id = row[2]
            problem_name = row[4]
            concept_name = row[16]

            if problem_name not in problem_code.keys():
                p_id = len(problem_code.keys())
                problem_code[problem_name] = p_id
            else:
                p_id = problem_code[problem_name]

            if concept_name not in concept_code.keys():
                c_id = len(concept_code.keys())
                concept_code[concept_name] = c_id
            else:
                c_id = concept_code[concept_name]

            if p_id not in problem_to_concept.keys():
                problem_to_concept[p_id] = [c_id]

            elif c_id not in problem_to_concept[p_id]:
                problem_to_concept[p_id].append(c_id)

        with open('P2C.json', 'w', encoding='utf8') as output_file:
            json.dump(problem_to_concept, output_file, indent=4, ensure_ascii=False)

    order_set = set()
    with open('skill_builder_data.csv', newline='', encoding="ISO-8859-1") as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)

        for row in tqdm(csv_reader):
            order_id = row[0]
            stu_id = row[2]
            p_id = problem_code[row[4]]
            response = 0.0 if row[6] == '0' else 1.0

            if order_id in order_set:
                continue
            order_set.add(order_id)

            if stu_id not in stu_log.keys():
                stu_log[stu_id] = []
            stu_log[stu_id].append((p_id, response))

        filtered_stu_log = {key: value for key, value in stu_log.items() if len(value) >= min_log}
        filtered_stu_log = {index: value for index, (_, value) in enumerate(filtered_stu_log.items())}
        print("学生总数:")
        print(len(filtered_stu_log))
        log_len = [len(log) for log in filtered_stu_log.values()]
        print("平均学生序列长度：")
        print(sum(log_len)/len(log_len))

        with open('data.json', 'w', encoding='utf8') as output_file:
            json.dump(filtered_stu_log, output_file, indent=4, ensure_ascii=False)
        with open('data_sample.json', 'w', encoding='utf8') as output_file:
            json.dump(dict(list(filtered_stu_log.items())[:100]), output_file, indent=4, ensure_ascii=False)
        print(dict(list(filtered_stu_log.items())[:100]))

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


class KtDataset(Dataset):
    def __init__(self, max_len=200, step_len=200):
        with open("P2C.json", encoding='utf8') as i_f:
            self.problem_to_concept = json.load(i_f)
        with open("data.json", encoding='utf8') as i_f:
            log_dict = json.load(i_f)
            self.data = log_dict

        self.problem_seq_list = []
        self.concepts_seq_list = []
        self.response_seq_list = []

        self.max_pid = 0
        self.max_cid = 0
        for stu_log in self.data.values():
            for p_id, response in stu_log:
                self.max_pid = max(self.max_pid, p_id)
                for c_id in self.problem_to_concept[str(p_id)]:
                    self.max_cid = max(self.max_cid, c_id)

        for stu_log in tqdm(self.data.values()):
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
        return torch.tensor(self.problem_seq_list[item]), torch.stack(self.concepts_seq_list[item]), torch.tensor(self.response_seq_list[item])


if __name__ == "__main__":
    divide_data()
    dataset = KtDataset()
    # a, b, c = dataset[1]
    # print(a.shape)
    # print(b.shape)
    # print(c.shape)
