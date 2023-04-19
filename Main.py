
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
from sklearn.metrics import accuracy_score 
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

main = tkinter.Tk()
main.title("LiSA-G: Authenticate and Identify Users On Widely Available Commercial Smartwatches") #designing main screen
main.geometry("1300x1200")

global filename
global random_acc,knn_acc,mlp_acc
global X, Y, X_train, X_test, y_train, y_test
global data
global features
global classifier

def upload():
    global filename
    global data
    global features
    filename = filedialog.askopenfilename(initialdir="dataset")
    data = pd.read_csv(filename)
    rows = data.shape[0]  # gives number of row count
    cols = data.shape[1]  # gives number of col count
    features = cols - 1
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    text.insert(END,"Number of features found in dataset : "+str(cols)+"\n");
    
    
def splitdataset(balance_data):
    global X, Y, X_train, X_test, y_train, y_test
    X = balance_data.values[:, 0:features] 
    Y = balance_data.values[:, features]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test

def generateModel():
    X, Y, X_train, y_train, X_test, y_test = splitdataset(data)
    text.insert(END,"Dataset Length : "+str(len(X))+"\n");
    text.insert(END,"Splitted Training Length : "+str(len(X_train))+"\n");
    text.insert(END,"Splitted Test Length : "+str(len(y_train))+"\n\n");
        
            

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    return accuracy    


def knnAlgorithm():
    global knn_acc
    text.delete('1.0', END)
    cls = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
    cls.fit(X_train, y_train) 
    prediction_data = prediction(X_test, cls)
    knn_acc = cal_accuracy(y_test, prediction_data,'KNearest Neighbor Accuracy')
    

def randomAlgorithm():
    global random_acc
    global classifier
    text.delete('1.0', END)
    cls = RandomForestClassifier(max_depth=50, random_state=0)
    cls.fit(X_train, y_train) 
    prediction_data = prediction(X_test, cls)
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Accuracy')
    classifier = cls
    

def MLPAlgorithm():
    global mlp_acc
    text.delete('1.0', END)
    srhl_tanh = MLPRandomLayer(n_hidden=30, activation_func='tanh')
    cls = GenELMClassifier(hidden_layer=srhl_tanh)
    cls.fit(X_train, y_train)
    prediction_data = prediction(X_test, cls) 
    mlp_acc = cal_accuracy(y_test, prediction_data,'Multilayer Perceptron Algorithm Accuracy') 


    
def graph():
    height = [knn_acc,random_acc,mlp_acc]
    bars = ('KNN Accuracy','Random Forest Accuracy','Multilayer Perceptron Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def Authentication():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(filename)
    test = test.values[:, 0:features]
    text.insert(END,filename+" test file loaded\n");
    y_pred = classifier.predict(test)
    print(y_pred)
    for i in range(len(test)):
        text.insert(END,str(test[i])+" Authenticated as user "+str(y_pred[i])+" from Accelerometer & Gyroscope Sensor Data\n\n");

font = ('times', 16, 'bold')
title = Label(main, text='LiSA-G: Authenticate and Identify Users On Widely Available Commercial Smartwatches')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=75)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=640,y=100)
text.config(font=font1)


font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Accelerometer & Gyroscope Sensor Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

generateButton = Button(main, text="Generate Train Test Model", command=generateModel)
generateButton.place(x=50,y=150)
generateButton.config(font=font1) 

knnButton = Button(main, text="Run KNN Algorithm", command=knnAlgorithm)
knnButton.place(x=50,y=200)
knnButton.config(font=font1) 

randomButton = Button(main, text="Run Random Forest Algorithm", command=randomAlgorithm)
randomButton.place(x=50,y=250)
randomButton.config(font=font1) 

mlpButton = Button(main, text="Run Multilayer Perceptron Algorithm", command=MLPAlgorithm)
mlpButton.place(x=50,y=300)
mlpButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)

authButton = Button(main, text="Authenticate User using Accelerometer & Gyroscope Sensor Data ", command=Authentication)
authButton.place(x=50,y=400)
authButton.config(font=font1) 

main.config(bg='OliveDrab2')
main.mainloop()
