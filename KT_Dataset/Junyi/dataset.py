import csv
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import random
min_log = 200


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
    seq.extend([[-1] for i in range(max_len - len(seq))])
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

        for stu_log in tqdm(self.data.values()):
            p_seq = []
            c_seq = []
            r_seq = []
            for p_id, response in stu_log:
                p_seq.append(p_id)
                c_seq.append(self.problem_to_concept[str(p_id)])
                r_seq.append(response)

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
        return self.problem_seq_list[item], self.concepts_seq_list[item], self.response_seq_list[item]


def divide_data():
    problem_code = {}
    concept_code = {}
    stu_code = {}

    problem_to_concept = {}
    stu_log = {}
    # 打开CSV文件
    with open('/data/xyq2/CD/Data/Junyi/junyi_Exercise_table.csv', newline='') as csvfile:
        # 创建一个CSV阅读器对象
        csv_reader = csv.reader(csvfile)
        next(csv_reader)

        # 逐行读取CSV文件内容并打印
        for row in tqdm(csv_reader):
            problem_name = row[0]
            concept_name = row[9]
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
            problem_to_concept[p_id] = [c_id]


        with open('P2C.json', 'w', encoding='utf8') as output_file:
            json.dump(problem_to_concept, output_file, indent=4, ensure_ascii=False)

    with open('/data/xyq2/CD/Data/Junyi/junyi_ProblemLog_original.csv', newline='') as csvfile:
        # 创建一个CSV阅读器对象
        csv_reader = csv.reader(csvfile)
        next(csv_reader)

        for row in tqdm(csv_reader):
            stu_name = row[0]
            p_id = problem_code[row[1]]
            response = 0.0 if row[10] == "false" else 1.0

            if stu_name not in stu_code.keys():
                stu_id = len(stu_code.keys())
                stu_code[stu_name] = stu_id
                stu_log[stu_id] = []
            else:
                stu_id = stu_code[stu_name]

            stu_log[stu_id].append((p_id, response))

        filtered_stu_log = {key: value for key, value in stu_log.items() if len(value) >= min_log and len(value) <= 1000}
        filtered_stu_log = {index: value for index, (_, value) in enumerate(filtered_stu_log.items())}
        random.seed(42)
        filtered_stu_log = dict(random.sample(filtered_stu_log.items(), 10000))
        print("学生总数:")
        print(len(filtered_stu_log))
        log_len = [len(log) for log in filtered_stu_log.values()]
        print("平均学生序列长度：")
        print(sum(log_len)/len(log_len))
        print("最长学生序列长度：")
        print(max(log_len))


        with open('data.json', 'w', encoding='utf8') as output_file:
            json.dump(filtered_stu_log, output_file, indent=4, ensure_ascii=False)
        with open('data_sample.json', 'w', encoding='utf8') as output_file:
            json.dump(dict(list(filtered_stu_log.items())[:100]), output_file, indent=4, ensure_ascii=False)
        # print(dict(list(filtered_stu_log.items())[:100]))


if __name__ == "__main__":
    divide_data()
    # dataset = KtDataset()
    # print(dataset[1])


