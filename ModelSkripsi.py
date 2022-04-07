# -*- coding: utf-8 -*-
"""
Created on Fri May 28 02:25:35 2021

@author: erwin
"""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier



class nb:
    
    def __init__(self,features, Label, k):
        self.features = features
        self.Label = Label
        self.k = k
    
    def klasifikasi_nb(self):
        rata =0
        kf = KFold(n_splits=self.k)
        NB_classifier = GaussianNB()
  
        for train_index, test_index in kf.split(self.features):
            X_train, X_test, Y_train, Y_test = self.features[train_index], self.features[test_index], self.Label[train_index], self.Label[test_index]
  
            model = NB_classifier.fit(X_train, Y_train)
            pred = model.predict(X_test)
            akurasi = accuracy_score(Y_test, pred)*100
            rata += akurasi
        return str('{:.2f}'.format(rata/self.k))


class rf:
    
    def __init__(self,features, Label, k, tree):
        self.features = features
        self.Label = Label
        self.k = k
        self.tree = tree
    
    def klasifikasi_rf(self):
        rata = 0
        kf = KFold(n_splits=self.k)
        RF_classifier = RandomForestClassifier(criterion='entropy',n_estimators=self.tree, random_state=0)

        for train_index, test_index in kf.split(self.features):
            X_train, X_test, Y_train, Y_test = self.features[train_index], self.features[test_index], self.Label[train_index], self.Label[test_index]
  
            model = RF_classifier.fit(X_train, Y_train)
            pred = model.predict(X_test)
            akurasi = accuracy_score(Y_test, pred)*100
            rata += akurasi
        return str('{:.2f}'.format(rata/self.k))
    
    
class UjiDataTunggal:
    
    def __init__(self, features, Label, dataTunggal, k, tree):
        self.features = features
        self.Label = Label
        self.dataTunggal = dataTunggal
        self.k = k
        self.tree = tree
        
    def KlasifikasiDataTunggal(self):
        kf = KFold(n_splits=self.k)
        RF_classifier = RandomForestClassifier(criterion='entropy',n_estimators=self.tree, random_state=0)

        for train_index, test_index in kf.split(self.features):
            X_train, Y_train = self.features[train_index], self.Label[train_index]
            model = RF_classifier.fit(X_train, Y_train)
            pred = model.predict(self.dataTunggal)
            if (pred == 1):
                hasil_string = "Lulus Tepat Waktu"
            else:
                hasil_string = "Tidak Lulus Tepat Waktu"
            return hasil_string.upper() 
       