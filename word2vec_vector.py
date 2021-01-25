from gensim.models import word2vec
import pandas as pd


train_file = '../NLPFramework/corpus/tianchi_news/train_set.csv'
test_file = '../NLPFramework/corpus/tianchi_news/test_a.csv'
word2vec_file = '../NLPFramework/corpus/tianchi_news/word2vec_100.txt'

num_features = 100     # Word vector dimensionality
num_workers = 8       # Number of threads to run in parallel


df_train = pd.read_csv(train_file, sep='\t')
df_test = pd.read_csv(test_file, sep='\t')
df_all = pd.concat([df_train, df_test], axis=0)

df_all['train_text'] = df_all.text.apply(lambda text: text.split())


model = word2vec.Word2Vec(df_all['train_text'], window=5, min_count=5, size=num_features, workers=num_workers)

model.wv.save_word2vec_format(word2vec_file, binary=False)





