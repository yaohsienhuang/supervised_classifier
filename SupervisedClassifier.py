import os
import time
import pickle
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import linear_model
from sklearn.metrics import accuracy_score


class supervisedClassifier:
    outliers_fraction=0.2
    algorithms = {
            "kNN":neighbors.KNeighborsClassifier(),
            "decision_tree":tree.DecisionTreeClassifier(),
            "SVM":svm.SVC(),
            "SGDClassifier":linear_model.SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
    }
    
    def __init__(self):
        print('algorithms:')
        n=1
        for name,algo in self.algorithms.items():
            print(f'({n}) {name} : {algo}')
            n+=1
    
    def select_algo(self,algo):
        self.algo=algo
        self.model=self.algorithms[algo]
        print('%s has been selected.'%self.algo)
    
    def train(self,dataset,gt):
        print('%s start trainig...'%self.algo)
        self.trained_model=self.model.fit(dataset,gt)
        self.save_model(self.trained_model)
    
    def predict(self,dataset):
        prediction = self.trained_model.predict(dataset)
        return prediction
    
    def predict_prob(self,dataset):
        predict_prob = self.trained_model.predict_proba(dataset)
        return predict_prob
    
    def accuracy_score(self,gt,prediction):
        accuracy=round(accuracy_score(gt, prediction)*100, 2)
        print('accuracy=',accuracy)
    
    def save_model(self,model):
        time_str = time.strftime("%Y%m%d%H%M")
        folder=f'model/{self.algo}'
        if not os.path.isdir(f'model/{self.algo}'):
            os.makedirs(f'model/{self.algo}')
        filename = folder+f'{time_str}-{self.algo}_nu{self.outliers_fraction}.sav'
        pickle.dump(model, open(filename, 'wb'))
        print('model_save:',filename)
        
    def load_model(self,path):
        self.trained_model = pickle.load(open(path, 'rb'))

if __name__ == '__main__':
    classifer=supervisedClassifier()
    classifer.select_algo('kNN')
    classifer.train(train_feature)
    classifer.load_model('xxx.sav')
    prediction=classifer.predict(test_feature)
    predict_prob=classifer.predict_prob(test_feature)
    classifer.accuracy_score(gt,test_feature)

   
    