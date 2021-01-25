# text-classification

## 文本分类的算法选择
算法选型的出发点就是权衡各种约束，考虑模型的天花板能力，选择合适的模型。一个比较使用框架模型如下：  
contcat of embedding -> spartial droput(0.2) -> LSTM -> LSTM -> concat(max_pool, meanpool) -> FC

结合前面的任务难度定义，推荐的算法选型行为:
- Fasttext（垃圾邮件/主题分类） 简单的任务，速度快
- TextCNN（主题分类/领域识别） 简单的任务，类别多，速度快
- LSTM（情感分类/意图识别） 稍微复杂的任务
- Bert（细粒度情感/阴阳怪气/小样本识别）难任务

