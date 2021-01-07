import fasttext
import pandas as pd


train_file = '../NLPFramework/corpus/tianchi_news/train_set.csv'
test_file = '../NLPFramework/corpus/tianchi_news/test_a.csv'

df_all = pd.read_csv(train_file, sep='\t')

# 构造fasttext所有需要的格式的文件
filename = '../NLPFramework/corpus/tianchi_news/train_fasttext.csv'

df_all['label_ft'] = '__label__' + df_all['label'].astype(str)
df_all[['text', 'label_ft']].to_csv(filename, index=None, header=None, sep='\t')

# wordNgrams: 设置ngram
# verbose: 2表示显示所有epoch  1表示显示最后一个epcoh
# loss: 优化方法 hs表示层次softmax
model = fasttext.train_supervised(filename, lr=0.2, wordNgrams=2, verbose=2,
                                             minCount=1, epoch=25, loss='hs')


df_test = pd.read_csv(test_file, sep='\t')
preds = [model.predict(text)[0][0].split("__")[-1] for text in df_test['text']]

df = pd.DataFrame()
df['label'] = preds
df.to_csv('submit_{}.csv'.format('fasttext'), index=None)

# score:0.9176