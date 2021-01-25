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
模型: BERT
"""


class ScaledDotProductAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):  # (batch, n_head, seq_len, dim)
        scores = torch.matmul(query, key.transpose(-2, -1))/np.sqrt(query.size(-1))  # (batch, *, seq_len_q, seq_len_v)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        # (batch, *, seq_len_q, dim)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    # h is n_head
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model//h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # (batch, n_head, seq_len, self.d_k)
        query, key, value = [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for linear, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x: (batch, n_head, seq_len, d_k)
        # attn: (batch, n_head, seq_len_q, seq_len_k)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # x.shape: (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN = max(0, xw_1 + b_1)w_2 + b_2 equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerEncoderBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


# equeal to : emb = nn.Embedding(vocab_size, emb_size, padding_idx)
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, emb_size=512):
        super().__init__(vocab_size, emb_size, padding_idx=0)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, emb_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    # sequence.shape: (batch, seq_len)
    # segment_label: (batch, seq_len)
    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_encoder_blocks = nn.ModuleList(
            [TransformerEncoderBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])


    def forward(self, x, segment_info):  # x.shape: (batch, seq)  segment_info.shape: (batch, seq_len)
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_encoder_blocks:
            x = transformer.forward(x, mask)

        return x


class BertClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.bert = BERT(vocab_size, hidden, n_layers, attn_heads, dropout)
        self.fc = nn.Linear(2*hidden, n_classes)

    def forward(self, x, segment_info):
        x = self.bert(x, segment_info)
        pooled_avg = torch.mean(x, dim=1)  # (batch, hidden_size)
        pooled_max, _ = torch.max(x, dim=1)  # (batch, hidden_size)
        outputs = torch.cat((pooled_avg, pooled_max), dim=1)   # （batch, 2*hidden_size)
        # outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        return outputs



class NewsVocab():
    """
    构建字典:
    train_data.keys(): ['text', 'label']
    train_data['text']: iterable of iterables
    train_data['label']: iterable
    """
    def __init__(self, train_data, min_count=5):
        self.min_count = min_count
        self.pad = 0
        self.unk = 1

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

    def load_pretrained_embs(self, embfile):
        with open(embfile, encoding='utf-8') as f:
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
    def __init__(self, model, optimizer, loss_fn, state_dict_path='./state_dict', prefix_model_name='textrnn.model'):
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
            inputs = data['text'].to(self.device)  # (batch, seq_len)
            lens = data['len'].to(self.device)  # (batch)
            targets = data['label'].to(self.device)  # (batch)
            segments = torch.ones_like(inputs).long()

            outputs = self.model(inputs, segments)  # (batch, num_classes)
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
                input_ids = data['text'].to(self.device)
                segments = torch.ones_like(input_ids).long()

                outputs = self.model(input_ids, segments)

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


#  将path目录下的.csv文件统计并生成新的csv

def merge_submit(path, match='.csv'):
    files = []
    for file in os.listdir(path):
        if match in file:
            files.append(file)

    df_tmp = pd.read_csv(files[0])
    n_classes = len(set(df_tmp.label.values))

    all_labels = np.zeros((len(df_tmp), n_classes))
    print(f"The shape of all_labels {all_labels.shape}")

    for file in files:
        df = pd.read_csv(file)
        for i, label in enumerate(df.label.values):
            all_labels[i, label] += 1

    new_df = pd.DataFrame(columns=['label'])
    new_df['label'] = all_labels.argmax(axis=1).astype(np.int)
    new_df.to_csv("submit.csv", index=None)
    return new_df['label']



if __name__ == '__main__':
    # train_file = './corpus/tianchi_news/train_set.csv'
    # test_file = './corpus/tianchi_news/test_a.csv'
    train_file = '../NLPFramework/corpus/tianchi_news/train_set.csv'
    test_file = '../NLPFramework/corpus/tianchi_news/test_a.csv'

    word2vec_file = '../NLPFramework/corpus/tianchi_news/word2vec_100.txt'
    glove_file = '../NLPFramework/corpus/tianchi_news/glove_200.txt'

    batch_size = 64
    seq_len = 512
    emb_size = 100
    hidden_size = 100
    lr = 1e-2
    drop = 0.3
    l2_regu = 1e-4
    num_layers = 1
    n_heads = 5

    # 读取文件中的数据
    df_all = pd.read_csv(train_file, sep='\t')

    # 构建词典
    vocab = NewsVocab(df_all)

    # 训练集和验证集数据
    # df_train, df_val = train_test_split(df_all, test_size=0.1, random_state=10)

    gkf = GroupKFold(n_splits=10).split(X=df_all, groups=df_all.index)

    for k, (train_idx, val_idx) in enumerate(gkf):
        print(f'-----Train k={k+1}-----')
        df_train = df_all.iloc[train_idx]
        df_val = df_all.iloc[val_idx]
        print(f'Train data shape: {df_train.shape}')
        print(f'Validation data shape: {df_val.shape}')

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
        model = BertClassifier(vocab_size, num_classes, hidden=hidden_size, n_layers=num_layers, attn_heads=n_heads, dropout=drop)
        optimizer = Optimizer(model.parameters(), lr=lr, weight_decay=l2_regu)
        loss_fn = torch.nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, loss_fn, state_dict_path='./state_dict')

        # trainer.fit(train_dataloader, val_dataloader, resume=False)



        df_test = pd.read_csv(test_file, sep='\t')
        df_test['label'] = np.zeros(len(df_test), dtype=np.int)
        test_data = preprocess(df_test, vocab)
        test_dataset = NewsDataset(test_data, seq_len=seq_len, front=True)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

        trainer.predict(test_dataloader, save_file=True, k=k+1)

        delete_spec_file_subname(r'./state_dict/', r'epoch.')

