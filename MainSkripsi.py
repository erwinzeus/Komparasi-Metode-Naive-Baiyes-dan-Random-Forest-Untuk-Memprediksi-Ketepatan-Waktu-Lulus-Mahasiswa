# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FinalSkripsiUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from ModelSkripsi import nb, rf, UjiDataTunggal
from PandasModel import PandasModel

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.setFixedSize(1166, 693)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(0, 50, 1161, 71))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 100, 1181, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 10, 1161, 71))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 10, 91, 81))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("../Images/Webp.net-resizeimage.png"))
        self.label_2.setObjectName("label_2")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(1060, 10, 81, 81))
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("../Images/Webp.net-resizeimage (1).png"))
        self.label_4.setObjectName("label_4")
        self.browse = QtWidgets.QPushButton(self.centralwidget)
        self.browse.setGeometry(QtCore.QRect(280, 160, 71, 23))
        self.browse.setObjectName("browse")
        self.addressBar = QtWidgets.QLineEdit(self.centralwidget)
        self.addressBar.setGeometry(QtCore.QRect(20, 160, 251, 21))
        self.addressBar.setReadOnly(True)
        self.addressBar.setObjectName("addressBar")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 130, 301, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.dataView = QtWidgets.QTableView(self.centralwidget)
        self.dataView.setGeometry(QtCore.QRect(20, 190, 331, 481))
        self.dataView.setObjectName("dataView")
        self.listFeatures = QtWidgets.QListWidget(self.centralwidget)
        self.listFeatures.setGeometry(QtCore.QRect(390, 160, 331, 271))
        self.listFeatures.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.listFeatures.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.listFeatures.setTextElideMode(QtCore.Qt.ElideLeft)
        self.listFeatures.setObjectName("listFeatures")
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listFeatures.addItem(item)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(390, 130, 341, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.modellingButton = QtWidgets.QPushButton(self.centralwidget)
        self.modellingButton.setGeometry(QtCore.QRect(640, 540, 75, 23))
        self.modellingButton.setObjectName("modellingButton")
        self.clearModel = QtWidgets.QPushButton(self.centralwidget)
        self.clearModel.setGeometry(QtCore.QRect(560, 540, 71, 23))
        self.clearModel.setObjectName("clearModel")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(760, 150, 391, 371))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.label_12 = QtWidgets.QLabel(self.groupBox)
        self.label_12.setGeometry(QtCore.QRect(10, 10, 131, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.prodi = QtWidgets.QLineEdit(self.groupBox)
        self.prodi.setEnabled(True)
        self.prodi.setGeometry(QtCore.QRect(10, 30, 100, 20))
        self.angkatan = QtWidgets.QLineEdit(self.groupBox)
        self.angkatan.setGeometry(QtCore.QRect(10, 80, 100, 20))
        self.angkatan.setObjectName("angkatan")
        self.label_13 = QtWidgets.QLabel(self.groupBox)
        self.label_13.setGeometry(QtCore.QRect(10, 60, 141, 16))
        self.label_13.setObjectName("label_13")
        self.nilaimasuk = QtWidgets.QLineEdit(self.groupBox)
        self.nilaimasuk.setGeometry(QtCore.QRect(10, 130, 100, 20))
        self.nilaimasuk.setObjectName("nilaimasuk")
        self.label_14 = QtWidgets.QLabel(self.groupBox)
        self.label_14.setGeometry(QtCore.QRect(10, 110, 141, 16))
        self.label_14.setObjectName("label_14")
        self.ips1 = QtWidgets.QLineEdit(self.groupBox)
        self.ips1.setGeometry(QtCore.QRect(10, 180, 100, 20))
        self.ips1.setObjectName("ips1")
        self.label_15 = QtWidgets.QLabel(self.groupBox)
        self.label_15.setGeometry(QtCore.QRect(10, 160, 141, 16))
        self.label_15.setObjectName("label_15")
        self.ips2 = QtWidgets.QLineEdit(self.groupBox)
        self.ips2.setGeometry(QtCore.QRect(10, 230, 100, 20))
        self.ips2.setObjectName("ips2")
        self.label_16 = QtWidgets.QLabel(self.groupBox)
        self.label_16.setGeometry(QtCore.QRect(10, 210, 141, 16))
        self.label_16.setObjectName("label_16")
        self.ips3 = QtWidgets.QLineEdit(self.groupBox)
        self.ips3.setGeometry(QtCore.QRect(10, 280, 100, 20))
        self.ips3.setText("")
        self.ips3.setObjectName("ips3")
        self.label_17 = QtWidgets.QLabel(self.groupBox)
        self.label_17.setGeometry(QtCore.QRect(10, 260, 141, 16))
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.groupBox)
        self.label_18.setGeometry(QtCore.QRect(150, 10, 141, 16))
        self.label_18.setObjectName("label_18")
        self.ips5 = QtWidgets.QLineEdit(self.groupBox)
        self.ips5.setGeometry(QtCore.QRect(150, 30, 100, 20))
        self.ips5.setText("")
        self.ips5.setObjectName("ips5")
        self.ips6 = QtWidgets.QLineEdit(self.groupBox)
        self.ips6.setGeometry(QtCore.QRect(150, 80, 100, 20))
        self.ips6.setText("")
        self.ips6.setObjectName("ips6")
        self.label_19 = QtWidgets.QLabel(self.groupBox)
        self.label_19.setGeometry(QtCore.QRect(150, 60, 141, 16))
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.groupBox)
        self.label_20.setGeometry(QtCore.QRect(150, 110, 141, 16))
        self.label_20.setObjectName("label_20")
        self.ips7 = QtWidgets.QLineEdit(self.groupBox)
        self.ips7.setGeometry(QtCore.QRect(150, 130, 100, 20))
        self.ips7.setText("")
        self.ips7.setObjectName("ips7")
        self.ips8 = QtWidgets.QLineEdit(self.groupBox)
        self.ips8.setGeometry(QtCore.QRect(150, 180, 100, 20))
        self.ips8.setText("")
        self.ips8.setObjectName("ips8")
        self.label_25 = QtWidgets.QLabel(self.groupBox)
        self.label_25.setGeometry(QtCore.QRect(150, 160, 141, 16))
        self.label_25.setObjectName("label_25")
        self.label_27 = QtWidgets.QLabel(self.groupBox)
        self.label_27.setGeometry(QtCore.QRect(150, 210, 141, 16))
        self.label_27.setObjectName("label_27")
        self.sks1 = QtWidgets.QLineEdit(self.groupBox)
        self.sks1.setGeometry(QtCore.QRect(150, 230, 100, 20))
        self.sks1.setText("")
        self.sks1.setObjectName("sks1")
        self.sks3 = QtWidgets.QLineEdit(self.groupBox)
        self.sks3.setGeometry(QtCore.QRect(150, 330, 100, 20))
        self.sks3.setText("")
        self.sks3.setObjectName("sks3")
        self.label_29 = QtWidgets.QLabel(self.groupBox)
        self.label_29.setGeometry(QtCore.QRect(150, 310, 111, 16))
        self.label_29.setObjectName("label_29")
        self.sks4 = QtWidgets.QLineEdit(self.groupBox)
        self.sks4.setGeometry(QtCore.QRect(280, 30, 100, 20))
        self.sks4.setText("")
        self.sks4.setObjectName("sks4")
        self.label_30 = QtWidgets.QLabel(self.groupBox)
        self.label_30.setGeometry(QtCore.QRect(280, 10, 111, 16))
        self.label_30.setObjectName("label_30")
        self.label_31 = QtWidgets.QLabel(self.groupBox)
        self.label_31.setGeometry(QtCore.QRect(280, 60, 111, 16))
        self.label_31.setObjectName("label_31")
        self.sks5 = QtWidgets.QLineEdit(self.groupBox)
        self.sks5.setGeometry(QtCore.QRect(280, 80, 100, 20))
        self.sks5.setText("")
        self.sks5.setObjectName("sks5")
        self.label_32 = QtWidgets.QLabel(self.groupBox)
        self.label_32.setGeometry(QtCore.QRect(280, 110, 111, 16))
        self.label_32.setObjectName("label_32")
        self.sks6 = QtWidgets.QLineEdit(self.groupBox)
        self.sks6.setGeometry(QtCore.QRect(280, 130, 100, 20))
        self.sks6.setText("")
        self.sks6.setObjectName("sks6")
        self.label_33 = QtWidgets.QLabel(self.groupBox)
        self.label_33.setGeometry(QtCore.QRect(280, 160, 111, 16))
        self.label_33.setObjectName("label_33")
        self.sks7 = QtWidgets.QLineEdit(self.groupBox)
        self.sks7.setGeometry(QtCore.QRect(280, 180, 100, 20))
        self.sks7.setText("")
        self.sks7.setObjectName("sks7")
        self.label_34 = QtWidgets.QLabel(self.groupBox)
        self.label_34.setGeometry(QtCore.QRect(150, 260, 141, 16))
        self.label_34.setObjectName("label_34")
        self.label_21 = QtWidgets.QLabel(self.groupBox)
        self.label_21.setGeometry(QtCore.QRect(10, 310, 141, 16))
        self.label_21.setObjectName("label_21")
        self.label_35 = QtWidgets.QLabel(self.groupBox)
        self.label_35.setGeometry(QtCore.QRect(280, 210, 111, 16))
        self.label_35.setObjectName("label_35")
        self.sks2 = QtWidgets.QLineEdit(self.groupBox)
        self.sks2.setGeometry(QtCore.QRect(150, 280, 100, 20))
        self.sks2.setText("")
        self.sks2.setObjectName("sks2")
        self.sks8 = QtWidgets.QLineEdit(self.groupBox)
        self.sks8.setGeometry(QtCore.QRect(280, 230, 100, 20))
        self.sks8.setText("")
        self.sks8.setObjectName("sks8")
        self.ips4 = QtWidgets.QLineEdit(self.groupBox)
        self.ips4.setGeometry(QtCore.QRect(10, 330, 100, 20))
        self.ips4.setText("")
        self.ips4.setObjectName("ips4")
        self.poin = QtWidgets.QLineEdit(self.groupBox)
        self.poin.setGeometry(QtCore.QRect(280, 280, 100, 20))
        self.poin.setText("")
        self.poin.setObjectName("poin")
        self.label_28 = QtWidgets.QLabel(self.groupBox)
        self.label_28.setGeometry(QtCore.QRect(280, 260, 111, 16))
        self.label_28.setObjectName("label_28")
        self.lamaTA = QtWidgets.QLineEdit(self.groupBox)
        self.lamaTA.setGeometry(QtCore.QRect(280, 330, 100, 20))
        self.lamaTA.setText("")
        self.lamaTA.setObjectName("lamaTA")
        self.label_26 = QtWidgets.QLabel(self.groupBox)
        self.label_26.setGeometry(QtCore.QRect(280, 310, 141, 16))
        self.label_26.setObjectName("label_26")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(760, 130, 331, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.klasifikasi = QtWidgets.QPushButton(self.centralwidget)
        self.klasifikasi.setGeometry(QtCore.QRect(1060, 530, 75, 23))
        self.klasifikasi.setObjectName("klasifikasi")
        self.clearUji = QtWidgets.QPushButton(self.centralwidget)
        self.clearUji.setGeometry(QtCore.QRect(980, 530, 75, 23))
        self.clearUji.setObjectName("clearUji")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(400, 470, 111, 21))
        self.label_22.setObjectName("label_22")
        self.masukanK = QtWidgets.QLineEdit(self.centralwidget)
        self.masukanK.setGeometry(QtCore.QRect(450, 500, 61, 21))
        self.masukanK.setObjectName("masukanK")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(410, 500, 31, 21))
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        self.label_24.setGeometry(QtCore.QRect(530, 500, 111, 21))
        self.label_24.setObjectName("label_24")
        self.masukanTree = QtWidgets.QLineEdit(self.centralwidget)
        self.masukanTree.setGeometry(QtCore.QRect(560, 500, 61, 21))
        self.masukanTree.setObjectName("masukanTree")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(410, 580, 261, 91))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setGeometry(QtCore.QRect(20, 20, 73, 13))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.groupBox_2)
        self.label_9.setGeometry(QtCore.QRect(150, 20, 59, 13))
        self.label_9.setObjectName("label_9")
        self.label_11 = QtWidgets.QLabel(self.groupBox_2)
        self.label_11.setGeometry(QtCore.QRect(110, 40, 16, 31))
        self.label_11.setObjectName("label_11")
        self.readRF = QtWidgets.QLineEdit(self.groupBox_2)
        self.readRF.setGeometry(QtCore.QRect(40, 40, 61, 31))
        self.readRF.setReadOnly(True)
        self.readRF.setObjectName("readRF")
        self.readNB = QtWidgets.QLineEdit(self.groupBox_2)
        self.readNB.setGeometry(QtCore.QRect(160, 40, 61, 31))
        self.readNB.setReadOnly(True)
        self.readNB.setObjectName("readNB")
        self.label_10 = QtWidgets.QLabel(self.groupBox_2)
        self.label_10.setGeometry(QtCore.QRect(230, 40, 20, 31))
        self.label_10.setObjectName("label_10")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(760, 570, 371, 101))
        self.groupBox_3.setObjectName("groupBox_3")
        self.readHasil = QtWidgets.QLineEdit(self.groupBox_3)
        self.readHasil.setGeometry(QtCore.QRect(30, 30, 321, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.readHasil.setFont(font)
        self.readHasil.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.readHasil.setReadOnly(True)
        self.readHasil.setObjectName("readHasil")
        self.seleksiButton = QtWidgets.QPushButton(self.centralwidget)
        self.seleksiButton.setGeometry(QtCore.QRect(640, 440, 75, 23))
        self.seleksiButton.setObjectName("seleksiButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.prodi.setEnabled(False)
        self.angkatan.setEnabled(False)
        self.nilaimasuk.setEnabled(False)
        self.ips1.setEnabled(False)
        self.ips2.setEnabled(False)
        self.ips3.setEnabled(False)
        self.ips4.setEnabled(False)
        self.ips5.setEnabled(False)
        self.ips6.setEnabled(False)
        self.ips7.setEnabled(False)
        self.ips8.setEnabled(False)
        self.sks1.setEnabled(False)
        self.sks2.setEnabled(False)
        self.sks3.setEnabled(False)
        self.sks4.setEnabled(False)
        self.sks5.setEnabled(False)
        self.sks6.setEnabled(False)
        self.sks7.setEnabled(False)
        self.sks8.setEnabled(False)
        self.poin.setEnabled(False)
        self.lamaTA.setEnabled(False)
        
        self.browse.clicked.connect(self.OpenPath)
        self.seleksiButton.clicked.connect(self.showDataTunggal)
        self.seleksiButton.clicked.connect(self.readFile)
        self.modellingButton.clicked.connect(self.modelling)
        self.klasifikasi.clicked.connect(self.ujiTunggal)      
        self.clearModel.clicked.connect(self.ClearModel)
        self.clearUji.clicked.connect(self.ClearKlasifikasi)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    def OpenPath(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Single File', QtCore.QDir.rootPath())
        data = pd.read_csv(fileName)
        df = PandasModel(data)
        
        self.dataView.setModel(df)
        self.addressBar.setText(fileName)
        
    def DataPreprocessing(self):
        path = self.addressBar.text()
        data = pd.read_csv(path)
        
        # data cleaning
        dc = data.dropna()
                
        # menyamakan skala pada atribut Nilai Masuk
        dataUnscaled = dc.loc[(dc['Jalur'] == "Gelombang  ") |          
                              (dc['Jalur'] == "Jalur Kerjasama ") | 
                              (dc['Jalur'] == "Jalur Tes ")] 
        for mult in dataUnscaled['Nilai Masuk']:
            dataUnscaled[mult]=dataUnscaled['Nilai Masuk']*10
        dc['Nilai Masuk'].loc[dc['Nilai Masuk']<=10] = dataUnscaled[mult]
        
        # normalisasi 
        scaler = MinMaxScaler() 
        dc['Nilai Masuk']=scaler.fit_transform(dc[['Nilai Masuk']])
        
        # data transformasi
        dc['Lama Studi'] = np.where(dc['Lama Studi'] <= 8, 1, dc['Lama Studi'])
        dc['Lama Studi'] = np.where(dc['Lama Studi'] >= 9, 0, dc['Lama Studi'])
        
        # konversi data kategori menjadi int
        categoricalColumns = ['Prodi']
        for cat in categoricalColumns:
            labelencoder = LabelEncoder()
            dc[cat] = labelencoder.fit_transform(dc[cat].astype(str))
         
        
        dataProcessed = dc.drop(columns=['No','Kabupaten Sekolah',
                                         'Propinsi Sekolah','Jalur',
                                         'Tgl Masuk USD','Tgl Lulus',
                                         'Lama Studi Bulan'])
        
        #mengambil dan mengubah list item menjadi list String
        feature_list = self.listFeatures.selectedItems()
        selected_features = [i.text() for i in list(feature_list)]   

        features = dataProcessed[selected_features]
        Label = dataProcessed['Lama Studi']
        
        return features, Label
    
    def readFile(self):
        features, Label = self.DataPreprocessing()
        data = pd.concat([features, Label], axis = 1, join='inner')
        df = PandasModel(data)
        self.dataView.setModel(df)
        
    def modelling(self):
        try:
            features, Label = self.DataPreprocessing()
            
            features = np.array(features)
            Label = np.array(Label)
            
            k = int(self.masukanK.text())
            tree = int(self.masukanTree.text())
        
            modellingNB = nb(features, Label, k)
            modellingRF = rf(features, Label, k, tree)
            
            self.readNB.setText(str(modellingNB.klasifikasi_nb()))
            self.readRF.setText(str(modellingRF.klasifikasi_rf()))
        except ValueError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('Salah Memasukkan Data.')
            msg.setInformativeText('Harap masukkan nilai tree dan k dengan tipe data numerik atau nilai k dengan data numerik lebih dari 1.')
            msg.setWindowTitle("Error")
            msg.exec_()
        
    def ClearModel(self):
        self.readNB.clear()
        self.readRF.clear()
        
    def ujiTunggal(self):
        try:
            if self.prodi.text() == 'TI' or self.prodi.text() == 'TE' or self.prodi.text() == 'MAT' or self.prodi.text() == 'TM':
                if self.prodi.text() == 'TI':
                    prodi = 2
                elif self.prodi.text() == 'MAT':
                    prodi = 0
                elif self.prodi.text() == 'TE':
                    prodi = 1
                elif self.prodi.text() == 'TM':
                    prodi = 4
                    
                DataTunggal = [prodi, self.angkatan.text(), self.nilaimasuk.text()
                                , self.ips1.text(), self.ips2.text(), self.ips3.text()
                                , self.ips4.text(), self.ips5.text(), self.ips6.text()
                                , self.ips7.text(), self.ips8.text(), self.sks1.text()
                                ,self.sks2.text(), self.sks3.text(), self.sks4.text()
                                , self.sks5.text(), self.sks6.text(), self.sks7.text()
                                , self.sks8.text(), self.poin.text(), self.lamaTA.text()]
                tunggal = [[x for x in DataTunggal if x]]
                features, Label = self.DataPreprocessing()
                features = np.array(features)
                Label = np.array(Label)
                    
                k = int(self.masukanK.text())
                tree = int(self.masukanTree.text())
                
                UDT = UjiDataTunggal(features, Label, tunggal, k, tree)
                self.readHasil.setText(str(UDT.KlasifikasiDataTunggal()))  
            else:
                DataTunggal = [self.prodi.text(), self.angkatan.text(), self.nilaimasuk.text()
                                , self.ips1.text(), self.ips2.text(), self.ips3.text()
                                , self.ips4.text(), self.ips5.text(), self.ips6.text()
                                , self.ips7.text(), self.ips8.text(), self.sks1.text()
                                ,self.sks2.text(), self.sks3.text(), self.sks4.text()
                                , self.sks5.text(), self.sks6.text(), self.sks7.text()
                                , self.sks8.text(), self.poin.text(), self.lamaTA.text()]
              
                tunggal = [[x for x in DataTunggal if x]]
                
                features, Label = self.DataPreprocessing()
                features = np.array(features)
                Label = np.array(Label)
                    
                k = int(self.masukanK.text())
                tree = int(self.masukanTree.text())
                
                UDT = UjiDataTunggal(features, Label, tunggal, k, tree)
                self.readHasil.setText(str(UDT.KlasifikasiDataTunggal()))
        
        except ValueError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('Salah Memasukkan Tipe Data.')
            msg.setInformativeText('Harap masukkan data uji tunggal sesuai dengan pedoman di setiap atribut.')
            msg.setWindowTitle("Error")
            msg.exec_()
                
    def showDataTunggal(self):
        self.readNB.clear()
        self.readRF.clear()
        self.prodi.clear()
        self.angkatan.clear()
        self.nilaimasuk.clear()
        self.ips1.clear()
        self.ips2.clear()
        self.ips3.clear()
        self.ips4.clear()
        self.ips5.clear()
        self.ips6.clear()
        self.ips7.clear()
        self.ips8.clear()
        self.sks1.clear()
        self.sks2.clear()
        self.sks3.clear()
        self.sks4.clear()
        self.sks5.clear()
        self.sks6.clear()
        self.sks7.clear()
        self.sks8.clear()
        self.poin.clear()
        self.lamaTA.clear()
        self.readHasil.clear()
        
        featureList = self.listFeatures.selectedItems()
        selected_features = [i.text() for i in list(featureList)]
       
        self.prodi.setEnabled(False)
        self.angkatan.setEnabled(False)
        self.nilaimasuk.setEnabled(False)
        self.ips1.setEnabled(False)
        self.ips2.setEnabled(False)
        self.ips3.setEnabled(False)
        self.ips4.setEnabled(False)
        self.ips5.setEnabled(False)
        self.ips6.setEnabled(False)
        self.ips7.setEnabled(False)
        self.ips8.setEnabled(False)
        self.sks1.setEnabled(False)
        self.sks2.setEnabled(False)
        self.sks3.setEnabled(False)
        self.sks4.setEnabled(False)
        self.sks5.setEnabled(False)
        self.sks6.setEnabled(False)
        self.sks7.setEnabled(False)
        self.sks8.setEnabled(False)
        self.poin.setEnabled(False)
        self.lamaTA.setEnabled(False)
    
        for i in selected_features:
            if i == "Prodi":
                self.prodi.setEnabled(True)
            elif i == "Angkatan":
                self.angkatan.setEnabled(True)
            elif i == "Nilai Masuk":
                self.nilaimasuk.setEnabled(True)
            elif i == "IPS 1":
                self.ips1.setEnabled(True)
            elif i == "IPS 2":
                self.ips2.setEnabled(True)
            elif i == "IPS 3":
                self.ips3.setEnabled(True)
            elif i == "IPS 4":
                self.ips4.setEnabled(True)
            elif i == "IPS 5":
                self.ips5.setEnabled(True)
            elif i == "IPS 6":
                self.ips6.setEnabled(True)
            elif i == "IPS 7":
                self.ips7.setEnabled(True)
            elif i == "IPS 8":
                self.ips8.setEnabled(True)
            elif i == "SKS 1":
                self.sks1.setEnabled(True)
            elif i == "SKS 2":
                self.sks2.setEnabled(True)
            elif i == "SKS 3":
                self.sks3.setEnabled(True)
            elif i == "SKS 4":
                self.sks4.setEnabled(True)
            elif i == "SKS 5":
                self.sks5.setEnabled(True)
            elif i == "SKS 6":
                self.sks6.setEnabled(True)
            elif i == "SKS 7":
                self.sks7.setEnabled(True)
            elif i == "SKS 8":
                self.sks8.setEnabled(True)
            elif i == "Poin":
                self.poin.setEnabled(True)
            elif i == "Lama TA":
                self.lamaTA.setEnabled(True)
                
    def ClearKlasifikasi(self):
        self.prodi.clear()
        self.angkatan.clear()
        self.nilaimasuk.clear()
        self.ips1.clear()
        self.ips2.clear()
        self.ips3.clear()
        self.ips4.clear()
        self.ips5.clear()
        self.ips6.clear()
        self.ips7.clear()
        self.ips8.clear()
        self.sks1.clear()
        self.sks2.clear()
        self.sks3.clear()
        self.sks4.clear()
        self.sks5.clear()
        self.sks6.clear()
        self.sks7.clear()
        self.sks8.clear()
        self.poin.clear()
        self.lamaTA.clear()
        self.readHasil.clear()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Prediksi Ketepatan Waktu Lulus Mahasiswa"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt;\">oleh Erwinsyah Rico Agusta / 175314101</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">Prediksi Ketepatan Waktu Lulus Mahasiswa<br/>Fakultas Sains dan Teknologi Universitas Sanata Dharma</span></p></body></html>"))
        self.browse.setText(_translate("MainWindow", "Browse"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:11pt;\">1. Masukkan Data :</span></p><p><br/></p></body></html>"))
        __sortingEnabled = self.listFeatures.isSortingEnabled()
        self.listFeatures.setSortingEnabled(False)
        item = self.listFeatures.item(0)
        item.setText(_translate("MainWindow", "IPS 8"))
        item = self.listFeatures.item(1)
        item.setText(_translate("MainWindow", "Lama TA"))
        item = self.listFeatures.item(2)
        item.setText(_translate("MainWindow", "SKS 8"))
        item = self.listFeatures.item(3)
        item.setText(_translate("MainWindow", "IPS 7"))
        item = self.listFeatures.item(4)
        item.setText(_translate("MainWindow", "IPS 2"))
        item = self.listFeatures.item(5)
        item.setText(_translate("MainWindow", "SKS 3"))
        item = self.listFeatures.item(6)
        item.setText(_translate("MainWindow", "IPS 1"))
        item = self.listFeatures.item(7)
        item.setText(_translate("MainWindow", "SKS 6"))
        item = self.listFeatures.item(8)
        item.setText(_translate("MainWindow", "SKS 5"))
        item = self.listFeatures.item(9)
        item.setText(_translate("MainWindow", "IPS 6"))
        item = self.listFeatures.item(10)
        item.setText(_translate("MainWindow", "IPS 3"))
        item = self.listFeatures.item(11)
        item.setText(_translate("MainWindow", "Prodi"))
        item = self.listFeatures.item(12)
        item.setText(_translate("MainWindow", "SKS 7"))
        item = self.listFeatures.item(13)
        item.setText(_translate("MainWindow", "IPS 5"))
        item = self.listFeatures.item(14)
        item.setText(_translate("MainWindow", "SKS 1"))
        item = self.listFeatures.item(15)
        item.setText(_translate("MainWindow", "SKS 2"))
        item = self.listFeatures.item(16)
        item.setText(_translate("MainWindow", "IPS 4"))
        item = self.listFeatures.item(17)
        item.setText(_translate("MainWindow", "Nilai Masuk"))
        item = self.listFeatures.item(18)
        item.setText(_translate("MainWindow", "SKS 4"))
        item = self.listFeatures.item(19)
        item.setText(_translate("MainWindow", "Angkatan"))
        item = self.listFeatures.item(20)
        item.setText(_translate("MainWindow", "Poin"))
        self.listFeatures.setSortingEnabled(__sortingEnabled)
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:11pt;\">2. Atribut yang ingin digunakan (urut dari yang terbaik) :</span></p></body></html>"))
        self.modellingButton.setText(_translate("MainWindow", "Modelling"))
        self.clearModel.setText(_translate("MainWindow", "Bersihkan"))
        self.label_12.setText(_translate("MainWindow", "Prodi (TI,TM,TE atau MAT)"))
        self.label_13.setText(_translate("MainWindow", "Angkatan (Format: TTTT)"))
        self.label_14.setText(_translate("MainWindow", "Nilai Masuk (0-100)"))
        self.label_15.setText(_translate("MainWindow", "IPS 1 (0-4.00)"))
        self.label_16.setText(_translate("MainWindow", "IPS 2 (0-4.00)"))
        self.label_17.setText(_translate("MainWindow", "IPS 3 (0-4.00)"))
        self.label_18.setText(_translate("MainWindow", "IPS 5 (0-4.00)"))
        self.label_19.setText(_translate("MainWindow", "IPS 6 (0-4.00)"))
        self.label_20.setText(_translate("MainWindow", "IPS 7 (0-4.00)"))
        self.label_25.setText(_translate("MainWindow", "IPS 8 (0-4.00)"))
        self.label_27.setText(_translate("MainWindow", "SKS 1 (0-24)"))
        self.label_29.setText(_translate("MainWindow", "SKS 3 (0-24)"))
        self.label_30.setText(_translate("MainWindow", "SKS 4 (0-24)"))
        self.label_31.setText(_translate("MainWindow", "SKS 5 (0-24)"))
        self.label_32.setText(_translate("MainWindow", "SKS 6 (0-24)"))
        self.label_33.setText(_translate("MainWindow", "SKS 7 (0-24)"))
        self.label_34.setText(_translate("MainWindow", "SKS 2 (0-24)"))
        self.label_21.setText(_translate("MainWindow", "IPS 4 (0-4.00)"))
        self.label_35.setText(_translate("MainWindow", "SKS 8 (0-24)"))
        self.label_28.setText(_translate("MainWindow", "Poin Kegiatan"))
        self.label_26.setText(_translate("MainWindow", "Lama TA (Semester)"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:11pt;\">3. Uji data tunggal</span></p><p><br/></p></body></html>"))
        self.klasifikasi.setText(_translate("MainWindow", "Klasifikasi"))
        self.clearUji.setText(_translate("MainWindow", "Bersihkan"))
        self.label_22.setText(_translate("MainWindow", "Masukan :"))
        self.label_23.setText(_translate("MainWindow", "K-fold"))
        self.label_24.setText(_translate("MainWindow", "Tree"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Akurasi"))
        self.label_8.setText(_translate("MainWindow", "Random Forest"))
        self.label_9.setText(_translate("MainWindow", "Naive Bayes"))
        self.label_11.setText(_translate("MainWindow", "%"))
        self.label_10.setText(_translate("MainWindow", "%"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Hasil Klasifikasi"))
        self.seleksiButton.setText(_translate("MainWindow", "Seleksi"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

