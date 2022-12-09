# supervised_classifier
* 收錄常用的個人常用的 Supervised learning，並建立成方法便於後續應用，若需要使用更多的演算法可參考網站：
> ref: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

## 使用方式：
```python=
# 實例化
classifer=supervisedClassifier()

# 選擇演算法
classifer.select_algo('kNN')

# 開始訓練
classifer.train(train_feature)

# 讀取pre_train model
classifer.load_model('xxx.sav')

# 開始預測
prediction=classifer.predict(test_feature)

# 預測機率
predict_prob=classifer.predict_prob(test_feature)

# 印出 accuracy
classifer.accuracy_score(gt,test_feature)