import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier


train_file = '../NLPFramework/corpus/tianchi_news/train_set.csv'
test_file = '../NLPFramework/corpus/tianchi_news/test_a.csv'

df_all = pd.read_csv(train_file, sep='\t', nrows=100)

# 构造tfidf模型
tfidf_model = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)

# 提取tfidf特征
train_X = tfidf_model.fit_transform(df_all['text']).toarray()
train_y = df_all['label'].values

# 分类器模型
model = RidgeClassifier()
# 训练模型
model.fit(train_X, train_y)

# 预测test数据集
df_test = pd.read_csv(test_file, sep='\t')
test_data = tfidf_model.transform(df_test['text']).toarray()
val_pred = model.predict(test_data)

# 保存预测结果到csv文件
df = pd.DataFrame()
df['label'] = val_pred
df.to_csv('submit_{}.csv'.format('Tfidf'), index=None)

# 0.8625
