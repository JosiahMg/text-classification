import pandas as pd
from collections import Counter
import numpy as np
import logging
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import os
import tqdm
import torch.nn.functional as F
import time
from sklearn.metrics import f1_score
from transformers import BertModel, BertConfig, AdamW

class Vocab:
    def __init__(self, df, vocab_file, max_len=512, segment_num=2):
        self.max_len = max_len
        self.segment_num = segment_num
        self._id2word, self._word2id = self.load_vocab(df, vocab_file)
        self.unk = self._word2id.get('[UNK]')
        self.num_label = len(np.unique(df.label.values))

    def load_vocab(self, df, vocab_file):
        with open(vocab_file, 'r') as f:
            lines = f.readlines()
            id2word = list(map(lambda x: x.strip(), lines))
            word2id = dict(zip(id2word, range(len(id2word))))
            return id2word, word2id

    def tokenize(self, text):
        text = text.strip().split()[:self.max_len-2]
        actual_len = len(text) + 2
        tokens = ['[PAD]']*self.max_len
        tokens[:actual_len] = ["[CLS]"] + text + ["[SEP]"]
        output_tokens = self.token2id(tokens)
        return output_tokens

    # def tokenize(self, text):
    #     total_seq_len = (self.max_len - 2) * self.segment_num
    #     segment_seq_len = self.max_len - 2
    #     tokens = [0] * (self.max_len * self.segment_num)
    #     text = text.strip().split()[:total_seq_len]
    #     for i in range(self.segment_num):
    #
    #
    #
    #     actual_len = len(text) + 2
    #
    #     tokens[:actual_len] = ["[CLS]"] + text + ["[SEP]"]
    #     output_tokens = self.token2id(tokens)
    #     return output_tokens

    def token2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def num_classes(self):
        return self.num_label


class NewsDataset(Dataset):
    def __init__(self, df, vocab):
        inputs = df.text.apply(vocab.tokenize).to_list()
        lens = [(np.sum(np.array(text)>0)).tolist() for text in inputs]
        self.input_ids = torch.LongTensor(inputs)
        self.targets = torch.LongTensor(df['label'].tolist())
        self.lens = torch.LongTensor(lens)
        self.token_type_ids = torch.zeros_like(self.input_ids)
        self.attention_mask = (self.input_ids > 0).long()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return {'input_ids': self.input_ids[item],
                'token_type_ids': self.token_type_ids[item],
                'attention_mask': self.attention_mask[item],
                'len': self.lens[item],
                'target': self.targets[item]
                }


class BertClassifier(nn.Module):
    def __init__(self, vocab, bert_model_path, drop=0.3):
        super(BertClassifier, self).__init__()
        config = BertConfig.from_pretrained(bert_model_path+'config.json')
        self.bert = BertModel.from_pretrained(bert_model_path+'pytorch_model.bin', config=config)
        self.fc = nn.Linear(config.hidden_size, vocab.num_classes)
        self.dropout = nn.Dropout(drop)

    def forward(self, input_ids, attention_mask, token_type_ids):  # input_ids.shape: (batch_size, seq_len)
        sequence_output, pooled_output = self.bert(input_ids=input_ids,
                                                   attention_mask=attention_mask,
                                                   token_type_ids=token_type_ids)
        reps = sequence_output[:, 0, :]
        reps = self.dropout(reps)
        return self.fc(reps)


class Optimizer:
    def __init__(self, model_parameters, lr, weight_decay=0, lr_scheduler=False):
        # self.optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, correct_bias=False)
        # 每decay_step个epoch降低LR一次
        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler:
            lr_func = lambda step: 0.75 ** (step // 100)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_func)

    def step(self):
        self.optimizer.step()
        if self.lr_scheduler:
            self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    # 手动更新学习率, 每次更新后变成原来的decay倍
    def update_lr(self, decay=0.8):
        for p in self.optimizer.param_groups:
            p['lr'] *= decay
        print(f'Update LR to {self.get_lr()}')

    # 用于模型的保存和加载
    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, chkpoint):
        return self.optimizer.load_state_dict(chkpoint)


class Trainer:
    """
    state_dict_path: 模型保存路径
    prefix_model_name: 模型名称前缀
    """
    def __init__(self, model, optimizer, loss_fn, state_dict_path='./state_dict', prefix_model_name='bert.model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
          print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
          print('No GPU available, using the CPU instead.')

        self.state_dict_path = state_dict_path
        self.prefix_model_name = prefix_model_name
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(self, train_dataloader, val_dataloader, epoch_nums=999, patient=5, resume=False):
        if resume and self.check_checkpoint():
            start_epoch = self.load_model()
            start_epoch += 1
            print(f'Load checkpoint to continue training: start_epoch: {start_epoch}, lr: {self.optimizer.get_lr()}')

        else:
            start_epoch = 1

        all_loss = []
        all_acc = []
        all_score = []
        threshold = 0
        for epoch in range(start_epoch, epoch_nums + 1):
            self.train(epoch, train_dataloader)
            loss, acc, score = self.test(epoch, val_dataloader)

            all_loss.append(loss)
            all_acc.append(acc)
            all_score.append(score)

            best_acc = max(all_acc)
            best_score = max(all_score)
            best_loss = min(all_loss)

            if all_score[-1] == best_score or all_loss[-1] == best_loss or all_acc[-1] == best_acc:
                self.save_state_dict(epoch, all_loss, all_acc, all_score)
                threshold = 0
            else:
                threshold += 1
                self.optimizer.update_lr(decay=0.8)

            if threshold >= patient:
                print("epoch {} has the lowest loss: {:4f}".format(start_epoch + np.argmin(np.array(all_loss)), min(all_loss)))
                print("epoch {} has the highest acc: {:4f}".format(start_epoch + np.argmax(np.array(all_acc)), max(all_acc)))
                print("epoch {} has the highest f1 score: {:4f}".format(start_epoch + np.argmax(np.array(all_score)), max(all_score)))
                print("early stop!")
                break

    def train(self, epoch, train_dataloader):
        self.model.train()
        self._iteration(epoch, train_dataloader)

    def test(self, epoch, val_dataloader):
        self.model.eval()
        with torch.no_grad():
            return self._iteration(epoch, val_dataloader, train=False)

    def _iteration(self, epoch, dataloader, train=True):
        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(dataloader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(dataloader),
                              bar_format="{l_bar}{r_bar}")

        total_loss = 0.
        corrects = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        for i, data in data_iter:
            inputs = data['input_ids'].to(self.device)  # (batch, seq_len)
            token_type_ids = data['token_type_ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device)
            targets = data['target'].to(self.device)  # (batch)


            outputs = self.model(inputs, attention_mask, token_type_ids)  # (batch, num_classes)
            loss = self.loss_fn(outputs, targets)

            # 训练模式进行反向传播
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 多分类任务
            result = torch.max(outputs, 1)[1].view(targets.size())  # (batch)

            if not train:
                all_predictions.extend(result.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

            # 计算准确率
            corrects += (result == targets).sum().item()
            total_samples += targets.size(0)

            total_loss += loss.item()

            if train:
                log_dic = {
                    "epoch": epoch,
                    "train_loss": total_loss / (i + 1), "train_acc": corrects / total_samples,
                    "test_loss": 0, "test_acc": 0
                }

            else:
                log_dic = {
                    "epoch": epoch,
                    "train_loss": 0, "train_acc": 0,
                    "test_loss": total_loss / (i + 1), "test_acc": corrects / total_samples
                }

            # 打印日志信息
            if (i + 1) % 500 == 0 or (i + 1) == len(data_iter):
                data_iter.write(str({k: v for k, v in log_dic.items() if v != 0}))

        if not train:
            score = f1_score(y_true=all_targets, y_pred=all_predictions, average='macro')
            return total_loss / len(data_iter), corrects / total_samples, score

    def predict(self, test_dataloader, save_file=False, model_name=None, k=0):
        if model_name == None:
            model_name = self.find_most_recent_state_dict()
        ckpoint = torch.load(model_name)
        self.model.load_state_dict(ckpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        all_predictions = []

        data_iter = tqdm.tqdm(enumerate(test_dataloader),
                              desc="Predicting",
                              total=len(test_dataloader),
                              bar_format="{l_bar}{r_bar}")

        with torch.no_grad():
            for i, data in data_iter:
                inputs = data['input_ids'].to(self.device)  # (batch, seq_len)
                token_type_ids = data['token_type_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)

                outputs = self.model(inputs, attention_mask, token_type_ids)

                result = torch.max(outputs, 1)[1].cpu().numpy()
                all_predictions.extend(result)

        if save_file:
            df = pd.DataFrame()
            df['label'] = all_predictions
            if k != 0:
                file_name = self.state_dict_path + "/" + self.prefix_model_name + '_{:.4f}_{}.csv'.format(ckpoint['acc'][-1], k)
            else:
                file_name = self.state_dict_path + "/" + self.prefix_model_name + '_{:.4f}.csv'.format(ckpoint['acc'][-1])
            df.to_csv(file_name, index=None)

        return all_predictions

    def save_state_dict(self, epoch, loss, acc, f1):
        if not os.path.exists(self.state_dict_path):
            os.mkdir(self.state_dict_path)

        save_path = self.state_dict_path + '/' + self.prefix_model_name + '.epoch.{}'.format(str(epoch))
        self.model.to('cpu')

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'acc': acc,
            'f1_score': f1
        }

        torch.save(checkpoint, save_path)

        print("{} saved in epoch:{}, loss:{:.4f}, acc:{:.4f}, f1_score:{:.4f}".format(save_path, epoch, loss[-1], acc[-1], f1[-1]))

        self.model.to(self.device)

    def load_model(self):
        checkpoint_dir = self.find_most_recent_state_dict()

        checkpoint = torch.load(checkpoint_dir, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

        torch.cuda.empty_cache()
        self.model.to(self.device)

        print(f"{checkpoint_dir} loaded, epoch={start_epoch}")
        return start_epoch

    def find_most_recent_state_dict(self):
        """
        :param dir_path: 存储所有模型文件的目录
        :return: 返回最新的模型文件路径, 按模型名称最后一位数进行排序
        """
        dic_lis = [i for i in os.listdir(self.state_dict_path)]
        dic_lis = [i for i in dic_lis if r"model.epoch." in i]
        if len(dic_lis) == 0:
            raise FileNotFoundError("can not find any state dict in {}!".format(self.state_dict_path))
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return self.state_dict_path + "/" + dic_lis[-1]

    # 判断是否有保存的模型
    def check_checkpoint(self):
        if os.path.exists(self.state_dict_path):
            dic_lis = [i for i in os.listdir(self.state_dict_path)]
            dic_lis = [i for i in dic_lis if r"model.epoch." in i]
            if len(dic_lis) != 0:
                return True
        return False

def delete_spec_file_subname(path, str):
    dic_lis = [i for i in os.listdir(path)]
    print(dic_lis)
    dic_lis = [i for i in dic_lis if str in i]
    print(dic_lis)
    for file in dic_lis:
        os.remove(path+file)
        print(f'Delete file {file}')


if __name__ == '__main__':
    train_file = '../NLPFramework/corpus/tianchi_news/train_set.csv'
    test_file = '../NLPFramework/corpus/tianchi_news/test_a.csv'
    vocab_file = '../NLPFramework/pre_trained_models/bert-mini/vocab.txt'
    bert_model_path = '../NLPFramework/pre_trained_models/bert-mini/'


    batch_size = 32
    seq_len = 256
    lr = 1e-4
    drop = 0.3
    l2_regu = 1e-3

    # 读取文件中的数据
    df_all = pd.read_csv(train_file, sep='\t')

    # 构建词典
    vocab = Vocab(df_all, vocab_file, max_len=seq_len)

    # 训练集和验证集数据
    # df_train, df_val = train_test_split(df_all, test_size=0.2, random_state=10)
    gkf = GroupKFold(n_splits=5).split(X=df_all, groups=df_all.index)

    for k, (train_idx, val_idx) in enumerate(gkf):
        print(f'-----Train k={k+1}-----')
        df_train = df_all.iloc[train_idx]
        df_val = df_all.iloc[val_idx]
        print(f'Train data shape: {df_train.shape}')
        print(f'Validation data shape: {df_val.shape}')


        train_dataset = NewsDataset(df_train, vocab)
        val_dataset = NewsDataset(df_val, vocab)

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

        vocab_size = vocab.vocab_size
        num_classes = vocab.num_classes

        print(f'vocabulary size is {vocab_size}, num_classes is {num_classes}')
        # 创建模型 优化器  损失函数
        model = BertClassifier(vocab, bert_model_path, drop=drop)
        optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=l2_regu)
        loss_fn = torch.nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, loss_fn, state_dict_path='./state_dict')

        trainer.fit(train_dataloader, val_dataloader, resume=False)



        df_test = pd.read_csv(test_file, sep='\t')
        df_test['label'] = np.zeros(len(df_test), dtype=np.int)
        test_dataset = NewsDataset(df_test, vocab)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

        trainer.predict(test_dataloader, save_file=True, k=k+1)
        delete_spec_file_subname(r'./state_dict/', r'epoch.')

