import os
import torch.nn.functional
import torch.nn as nn
import Dataset
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import random
import matplotlib.pyplot as plt

seed = 114514
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def save_dict(my_dict, path):
    with open(path, 'wb') as f:
        pickle.dump(my_dict, f)


def load_dict(path):
    with open(path, 'rb') as f:
        my_dict = pickle.load(f)
    return my_dict


class ModelData:
    def __init__(self, source_domains, target_domain):
        self.data = {"current_epoch": 0,
                     "train_loss_arr": [], "train_auc_arr": [], "train_acc_arr": [],
                     "val_loss_arr": [], "val_auc_arr": [], "val_acc_arr": [],
                     "source_domains": source_domains, "target_domain": target_domain}

    def __getitem__(self, item):
        return self.data[item]

    def print_info(self):
        num_domains = len(self.data["source_domains"])
        print("Model Info :")
        print("current_epoch : %d" % self.data["current_epoch"])
        print("Train loss : %f" % (sum(self.data["train_loss_arr"][-1]) / num_domains), end="   ")
        print("Train average auc : %f" % (sum(self.data["train_auc_arr"][-1]) / num_domains), end="   ")
        print("Train average acc : %f" % (sum(self.data["train_acc_arr"][-1]) / num_domains))
        for i, source_domain in enumerate(self.data["source_domains"]):
            print("%s :" % source_domain)
            print("Valid loss : %f" % self.data["val_loss_arr"][-1][i], end="   ")
            print("Valid auc : %f" % self.data["val_auc_arr"][-1][i], end="   ")
            print("Valid acc : %f" % self.data["val_acc_arr"][-1][i])

    def add(self, train_losses, train_aucs, train_accs, val_losses, val_aucs, val_accs):
        self.data["train_loss_arr"].append(train_losses)
        self.data["train_auc_arr"].append(train_aucs)
        self.data["train_acc_arr"].append(train_accs)

        self.data["val_loss_arr"].append(val_losses)
        self.data["val_auc_arr"].append(val_aucs)
        self.data["val_acc_arr"].append(val_accs)

        self.data["current_epoch"] += 1

    def is_best(self):
        now_auc = sum(self.data["val_auc_arr"][-1])
        for aucs in self.data["val_auc_arr"][:-1]:
            if sum(aucs) > now_auc:
                return False
        return True

    def get_best(self):
        best_auc = sum(self.data["val_auc_arr"][-1])
        for aucs in self.data["val_auc_arr"][:-1]:
            if sum(aucs) > best_auc:
                best_auc = sum(aucs)
        return best_auc / len(self.data["val_auc_arr"][-1])

    def plot(self):
        epochs = range(1, self.data["current_epoch"] + 1)

        # Plotting train and validation loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(epochs, [sum(losses) / len(losses) for losses in self.data["train_loss_arr"]], label='Train Loss')
        plt.plot(epochs, [sum(losses) / len(losses) for losses in self.data["val_loss_arr"]], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plotting train and validation AUC
        plt.subplot(1, 3, 2)
        plt.plot(epochs, [sum(aucs) / len(aucs) for aucs in self.data["train_auc_arr"]], label='Train AUC')
        plt.plot(epochs, [sum(aucs) / len(aucs) for aucs in self.data["val_auc_arr"]], label='Validation AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.title('Training and Validation AUC')
        plt.legend()

        # Plotting train and validation accuracy
        plt.subplot(1, 3, 3)
        plt.plot(epochs, [sum(accs) / len(accs) for accs in self.data["train_acc_arr"]], label='Train Accuracy')
        plt.plot(epochs, [sum(accs) / len(accs) for accs in self.data["val_acc_arr"]], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()


class Trainer:
    def __init__(self, trainer_name):
        self.train_loader_arr = None
        self.dataset_list = None
        self.model_info = None
        self.model = None
        self.train_iters = None
        self.val_loader_arr = None
        self.test_loader_arr = None
        self.c_constant_arr = None
        self.trainer_name = trainer_name
        self.verbose = True
        self.lr = 1e-3
        self.weight_decay = 0
        self.sample_num = 60
        self.criterion = nn.BCEWithLogitsLoss().cuda()

    def load_data(self, target_domain=None, max_len=200, step_len=200, fold=1, dataset_list=None):
        assert target_domain is not None or dataset_list is not None
        if dataset_list is None:
            dataset_list = Dataset.get_dataset_names()
            assert target_domain in dataset_list, f"{target_domain} not included!"
            dataset_list.remove(target_domain)
            dataset_list.append(target_domain)

        self.dataset_list = dataset_list
        self.train_loader_arr, self.train_iters, self.val_loader_arr, self.test_loader_arr, _, self.c_constant_arr = \
            Dataset.load_data(dataset_list=dataset_list, max_len=max_len, step_len=step_len,
                              verbose=self.verbose, fold=fold)

    def init_model(self, model_class):
        # assert not os.path.exists(os.path.join("saved_model", self.trainer_name, "model")), \
        #     f"saved_model/{self.trainer_name}/model already exists!"

        assert self.train_iters is not None, "Data not loaded!"

        self.model = model_class(c_list=self.c_constant_arr).cuda()
        self.model_info = ModelData(self.dataset_list[:-1], self.dataset_list[-1])
        if not os.path.exists(os.path.join("saved_model", self.trainer_name)):
            os.makedirs(os.path.join("saved_model", self.trainer_name))

    def save_model(self, name: str = None):
        self.print("Model saved to %s" % self.trainer_name)
        if name is None:
            data_pth = os.path.join("saved_model", self.trainer_name, "model_data.pickle")
            torch_model_pth = os.path.join("saved_model", self.trainer_name, "model")
        else:
            data_pth = os.path.join("saved_model", self.trainer_name, f"model_data_{name}.pickle")
            torch_model_pth = os.path.join("saved_model", self.trainer_name, f"model_{name}")
        save_dict(self.model_info, data_pth)
        torch.save(self.model, torch_model_pth)

    def load_model(self, name: str = None):
        if name is None:
            data_pth = os.path.join("saved_model", self.trainer_name, "model_data.pickle")
            torch_model_pth = os.path.join("saved_model", self.trainer_name, "model")
        else:
            data_pth = os.path.join("saved_model", self.trainer_name, f"model_data_{name}.pickle")
            torch_model_pth = os.path.join("saved_model", self.trainer_name, f"model_{name}")
        assert os.path.exists(torch_model_pth) and os.path.exists(data_pth), f"{torch_model_pth} does not exist!"
        self.model_info = load_dict(data_pth)
        self.model = torch.load(torch_model_pth)
        self.model.cuda()

    def train(self, num_epoch, centroid=None, best_name=None, train_emb=False):
        assert self.model is not None, "Model not loaded!"
        assert self.train_iters is not None, "Data not loaded!"

        if train_emb:
            optimizer = torch.optim.Adam(self.model.concept_emb.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay)
        best_auc = 0
        for epoch in self._get_epoch_iterator(
                range(self.model_info["current_epoch"] + 1, self.model_info["current_epoch"] + 1 + num_epoch)):

            """
            train on each source domain
            """
            train_losses = [0 for i in self.model_info["source_domains"]]
            train_aucs = [0 for i in self.model_info["source_domains"]]
            train_accs = [0 for i in self.model_info["source_domains"]]
            self.print("Training on source domains...")
            for _ in self._get_iterator(range(self.sample_num)):
                sum_loss = 0
                for domain_idx, train_it in enumerate(self.train_iters[:-1]):
                    q_seq, concept_seq, response_seq = next(train_it)
                    q_seq = q_seq.cuda()
                    concept_seq = concept_seq.cuda()
                    response_seq = response_seq.cuda()

                    label_mask = (response_seq != -1).cuda()
                    label_mask[:, 0] = False
                    label = torch.masked_select(response_seq, label_mask).to(torch.float32)

                    out = self.model(q_seq, concept_seq, response_seq, domain=domain_idx, use_centroid=centroid)
                    out_mask = label_mask.unsqueeze(-1)
                    selected_out = torch.masked_select(out, out_mask).to(torch.float32)

                    loss = self.criterion(selected_out, label)
                    selected_out = selected_out.cpu()
                    label = label.cpu()
                    try:
                        auc = roc_auc_score(label.cpu(), selected_out.detach().cpu())
                        acc = accuracy_score(label.cpu(), selected_out.detach().cpu() > 0)
                    except ValueError:
                        auc = 0.5
                        acc = 0.5
                    sum_loss += loss
                    train_losses[domain_idx] += loss
                    train_aucs[domain_idx] += auc
                    train_accs[domain_idx] += acc

                optimizer.zero_grad()
                sum_loss.backward()
                optimizer.step()
            # if epoch % 10 == 0:
            #     self.plot()

            train_losses = [i.item() / self.sample_num for i in train_losses]
            train_aucs = [i / self.sample_num for i in train_aucs]
            train_accs = [i / self.sample_num for i in train_accs]

            """
            valid on each source domain
            """
            valid_losses, valid_aucs, valid_accs = self.valid(centroid=centroid)

            """
            save model
            """
            self.model_info.add(train_losses, train_aucs, train_accs, valid_losses, valid_aucs, valid_accs)
            if self.verbose:
                self.model_info.print_info()
            mean_auc = sum(train_aucs) / len(train_aucs)
            if mean_auc > best_auc and best_name is not None:
                self.save_model(best_name)
        print(self.model_info.get_best())

    def train_target(self, num_epoch=150, target_batch=1, centroid="target"):
        assert self.model is not None, "Model not loaded!"
        assert self.train_iters is not None, "Data not loaded!"

        optimizer = torch.optim.Adam(self.model.concept_emb.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        best_val_auc = 0
        for epoch in self._get_epoch_iterator(
                range(self.model_info["current_epoch"] + 1, self.model_info["current_epoch"] + 1 + num_epoch)):

            """
            train on each source domain
            """

            self.print("Training on target domain...")

            train_loader = self.train_loader_arr[-1]
            batch_cnt = 0
            for batch in train_loader:
                if batch_cnt >= target_batch:
                    break
                batch_cnt += 1
                q_seq, concept_seq, response_seq = batch
                q_seq = q_seq.cuda()
                concept_seq = concept_seq.cuda()
                response_seq = response_seq.cuda()

                label_mask = (response_seq != -1).cuda()
                label_mask[:, 0] = False
                label = torch.masked_select(response_seq, label_mask).to(torch.float32)

                out = self.model(q_seq, concept_seq, response_seq, domain=-1, use_centroid=centroid)
                out_mask = label_mask.unsqueeze(-1)
                selected_out = torch.masked_select(out, out_mask).to(torch.float32)

                loss = self.criterion(selected_out, label)
                selected_out = selected_out.cpu()
                label = label.cpu()
                try:
                    auc = roc_auc_score(label.cpu(), selected_out.detach().cpu())
                    acc = accuracy_score(label.cpu(), selected_out.detach().cpu() > 0)
                except ValueError:
                    auc = 0.5
                    acc = 0.5

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        """
        valid on each source domain
        """
        valid_loss, valid_auc, valid_acc = self.valid_target(centorid=centroid)
        self.print(f"Val loss : {valid_loss}    Val auc : {valid_auc}   Val acc : {valid_acc}")

        print("target result : ", valid_auc)
        return valid_auc

    def valid(self, centroid=None):
        """
        valid on each source domain
        """
        valid_losses = [0 for i in self.model_info["source_domains"]]
        valid_aucs = [0 for i in self.model_info["source_domains"]]
        valid_accs = [0 for i in self.model_info["source_domains"]]

        for domain_idx, valid_loader in enumerate(self.val_loader_arr[:-1]):
            pred_arr = []
            label_arr = []
            sum_loss = 0
            self.print("Validating on %s..." % self.model_info["source_domains"][domain_idx])
            for batch in self._get_iterator(valid_loader):
                q_seq, concept_seq, response_seq = batch
                q_seq = q_seq.cuda()
                concept_seq = concept_seq.cuda()
                response_seq = response_seq.cuda()

                label_mask = (response_seq != -1).cuda()
                label_mask[:, 0] = False
                label = torch.masked_select(response_seq, label_mask).to(torch.float32)

                out = self.model(q_seq, concept_seq, response_seq, domain=domain_idx, use_centroid=centroid)
                out_mask = label_mask.unsqueeze(-1)
                selected_out = torch.masked_select(out, out_mask).to(torch.float32)

                loss = self.criterion(selected_out, label)
                selected_out = selected_out.cpu()
                label = label.cpu()

                pred_arr.append(selected_out.detach().cpu())
                label_arr.append(label.detach().cpu())
                sum_loss += loss.item()

            cat_pred = torch.concatenate(pred_arr)
            cat_label = torch.concatenate(label_arr)
            try:
                auc = roc_auc_score(cat_label.cpu(), cat_pred.detach().cpu())
                acc = accuracy_score(cat_label.cpu(), cat_pred.detach().cpu() > 0)
            except ValueError:
                auc = 0.5
                acc = 0.5

            valid_losses[domain_idx] = sum_loss / len(valid_loader)
            valid_aucs[domain_idx] = auc
            valid_accs[domain_idx] = acc
        return valid_losses, valid_aucs, valid_accs

    def valid_target(self, centorid="target"):
        valid_loader = self.val_loader_arr[-1]
        pred_arr = []
        label_arr = []
        sum_loss = 0
        self.print("Validating on %s..." % self.model_info["source_domains"][-1])
        for batch in self._get_iterator(valid_loader):
            q_seq, concept_seq, response_seq = batch
            q_seq = q_seq.cuda()
            concept_seq = concept_seq.cuda()
            response_seq = response_seq.cuda()

            label_mask = (response_seq != -1).cuda()
            label_mask[:, 0] = False
            label = torch.masked_select(response_seq, label_mask).to(torch.float32)

            out = self.model(q_seq, concept_seq, response_seq, domain=-1, use_centroid=centorid)
            out_mask = label_mask.unsqueeze(-1)
            selected_out = torch.masked_select(out, out_mask).to(torch.float32)

            loss = self.criterion(selected_out, label)
            selected_out = selected_out.cpu()
            label = label.cpu()

            pred_arr.append(selected_out.detach().cpu())
            label_arr.append(label.detach().cpu())
            sum_loss += loss.item()

        cat_pred = torch.concatenate(pred_arr)
        cat_label = torch.concatenate(label_arr)
        try:
            auc = roc_auc_score(cat_label.cpu(), cat_pred.detach().cpu())
            acc = accuracy_score(cat_label.cpu(), cat_pred.detach().cpu() > 0)
        except ValueError:
            auc = 0.5
            acc = 0.5

        return sum_loss / len(valid_loader), auc, acc

    def concept_aggregation(self, k=5):
        self.model.concept_emb.cluster_emb(k=k)

    def init_target_embedding(self):
        self.model.concept_emb.init_target_embedding()

    def plot(self):
        self.model_info.plot()

    def print(self, x):
        if self.verbose:
            print(x)

    def _get_iterator(self, data, total=None):
        if self.verbose:
            return tqdm(data, total=total)
        else:
            return data

    def _get_epoch_iterator(self, data):
        if self.verbose:
            return data
        else:
            return tqdm(data, desc="Epoch")
