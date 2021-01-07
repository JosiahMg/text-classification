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


"""
任务: 多分类
模型: TextCNN
"""


class NewsVocab():
    """
    构建字典:
    train_data.keys(): ['text', 'label']
    train_data['text']: iterable of iterables
    train_data['label']: iterable
    """
    def __init__(self, train_data, embfile, min_count=5):
        self.min_count = min_count
        self.pad = 0
        self.unk = 1
        self.embfile = embfile

        self._id2word = ['[PAD]', '[UNK]']

        self._id2label = []
        self.target_names = []

        self.build_vocab(train_data)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._id2label)

        logging.info("Build vocab: words %d, labels %d." % (self.word_size, self.label_size))

    def build_vocab(self, data):
        word_counter = Counter()

        for words in data['text']:
            words = words.strip().split()
            for word in words:
                word_counter[word] += 1

        for word, count in word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}

        label_counter = Counter(data['label'])

        for label in range(len(label_counter)):
            count = label_counter[label]
            self._id2label.append(label)
            self.target_names.append(label2name[label])

    def load_pretrained_embs(self):
        with open(self.embfile, encoding='utf-8') as f:
            lines = f.readlines()
            items = lines[0].split()
            word_count, embedding_dim = int(items[0]), int(items[1])

        vocab_size = self.word_size
        embeddings = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
        # 词典中所有的索引值
        all_index = np.array(range(vocab_size))
        # 用于存放word2vec中的索引值
        word2vec_index = np.array([0, 1])
        count = 0

        for i, line in enumerate(lines[1:], 1):
            values = line.strip().split()
            index = self._word2id.get(values[0], self.unk)
            vector = np.array(values[1:], dtype=np.float)

            if index not in [self.pad, self.unk]:
                embeddings[index] = vector
                word2vec_index = np.append(word2vec_index, index)
            embeddings[self.unk] += vector
            count = i

        # 获取word2vec中没有的word的index
        diff_index = np.setdiff1d(all_index, word2vec_index, assume_unique=True)
        embeddings[self.unk] = embeddings[self.unk] / count
        embeddings[diff_index] = embeddings[self.unk]
        # embeddings = embeddings / np.std(embeddings)

        return embeddings


    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)


    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    @property
    def word_size(self):
        return len(self._id2word)


    @property
    def label_size(self):
        return len(self._id2label)


def preprocess(df, vocab):
    texts = df['text'].to_list()
    labels = df['label'].to_list()

    texts = [vocab.word2id(text.strip().split()) for text in texts]
    labels = vocab.label2id(labels)
    return {'text': texts, 'label': labels}


class NewsDataset(Dataset):
    def __init__(self, df, seq_len=256, front=True):
        inputs = df['text']
        self.labels = torch.LongTensor(df['label'])
        self.lens = torch.LongTensor([len(text) if len(text)<seq_len else seq_len for text in inputs])
        # 统一相同的长度并变成LongTensor类型
        self.inputs = torch.zeros(len(inputs), seq_len, dtype=torch.int64)
        for i, text in enumerate(self.inputs):
            if front:
                self.inputs[i, :self.lens[i]] = torch.LongTensor(inputs[i][:seq_len])
            else:
                self.inputs[i, :self.lens[i]] = torch.LongTensor(inputs[i][-seq_len:])


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return {'text': self.inputs[item], 'len': self.lens[item], 'label': self.labels[item]}


class TextCNN(nn.Module):
    def __init__(self, vocab, emb_size, out_channels, filter_sizes=[2, 3, 4], drop=0.3):
        super(TextCNN, self).__init__()
        vocab_size = vocab.word_size
        num_classes = vocab.label_size
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)

        # 使用预训练的词向量
        extword_embed = vocab.load_pretrained_embs()
        self.extword_embed = nn.Embedding.from_pretrained(torch.from_numpy(extword_embed), padding_idx=0)

        self.convs = nn.ModuleList([nn.Conv2d(1, out_channels, (k, emb_size)) for k in filter_sizes])
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(len(filter_sizes) * out_channels, num_classes)

    def forward(self, x):  # x.shape: (batch_size, seq_len)
        emb1 = self.emb(x)  # x.shape: (batch_size, seq_len, emb_size)
        emb2 = self.extword_embed(x)  # x.shape: (batch_size, seq_len, emb_size)
        x = emb1 + emb2      # x.shape: (batch_size, seq_len, emb_size)
        # x = torch.cat((emb1, emb2), dim=2)  # x.shape: (batch_size, seq_len, 2*emb_size)

        x = x.unsqueeze(1)  # x.shape: (batch_size, 1, seq_len, emb_size)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(self.convs)*(batch_size, out_channels, w)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(self.convs)*(batch_size, out_channels)
        x = torch.cat(x, dim=1)  # (batch, len(self.convs)*out_channels)
        x = self.dropout(x)
        return self.fc(x)


class Optimizer:
    def __init__(self, model_parameters, lr, weight_decay=0, lr_scheduler=False):
        self.optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
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
    def __init__(self, model, optimizer, loss_fn, state_dict_path='./state_dict', prefix_model_name='textcnn.model'):
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
                self.save_state_dict(epoch, all_loss[-1], all_acc[-1], all_score[-1])
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
            inputs = data['text'].to(self.device)  # (batch, seq_len)
            targets = data['label'].to(self.device)  # (batch)

            outputs = self.model(inputs)  # (batch, num_classes)
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

    def predict(self, test_dataloader, save_file=False, model_name=None):
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
                input_ids = data['text'].to(self.device)

                outputs = self.model(input_ids)

                result = torch.max(outputs, 1)[1].cpu().numpy()
                all_predictions.extend(result)

        if save_file:
            df = pd.DataFrame()
            df['label'] = all_predictions
            df.to_csv(self.state_dict_path + "/" + self.prefix_model_name + '_{:.4f}.csv'.format(ckpoint['acc']), index=None)

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

        print("{} saved in epoch:{}, loss:{:.4f}, acc:{:.4f}, f1_score:{:.4f}".format(save_path, epoch, loss, acc, f1))

        self.model.to(self.device)

    def load_model(self):
        checkpoint_dir = self.find_most_recent_state_dict()

        checkpoint = torch.load(checkpoint_dir)

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



if __name__ == '__main__':
    # train_file = './corpus/tianchi_news/train_set.csv'
    # test_file = './corpus/tianchi_news/test_a.csv'
    train_file = '../NLPFramework/corpus/tianchi_news/train_set.csv'
    test_file = '../NLPFramework/corpus/tianchi_news/test_a.csv'

    embfile = '../NLPFramework/corpus/tianchi_news/word2vec_100.txt'

    batch_size = 64
    seq_len = 512
    emb_size = 100
    out_channels = 200
    lr = 1e-2

    # 读取文件中的数据
    df_all = pd.read_csv(train_file, sep='\t')

    # 构建词典
    vocab = NewsVocab(df_all, embfile)

    # 训练集和验证集数据
    df_train, df_val = train_test_split(df_all, test_size=0.1, random_state=10)


    #预处理数据
    train_data = preprocess(df_train, vocab)
    val_data = preprocess(df_val, vocab)

    train_dataset = NewsDataset(train_data, seq_len=seq_len, front=True)
    val_dataset = NewsDataset(val_data, seq_len=seq_len, front=True)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    vocab_size = vocab.word_size
    num_classes = vocab.label_size

    print(f'vocabulary size is {vocab_size}, num_classes is {num_classes}')
    # 创建模型 优化器  损失函数
    model = TextCNN(vocab, emb_size, out_channels)
    optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=2e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, state_dict_path='./state_dict')

    trainer.fit(train_dataloader, val_dataloader, resume=False)



    df_test = pd.read_csv(test_file, sep='\t')
    df_test['label'] = np.zeros(len(df_test), dtype=np.int)
    test_data = preprocess(df_test, vocab)
    test_dataset = NewsDataset(test_data, seq_len=seq_len, front=True)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    trainer.predict(test_dataloader, save_file=True)
    # trainer.predict(test_dataloader, save_file=True, model_name='./state_dict/textcnn.model.epoch.62')
