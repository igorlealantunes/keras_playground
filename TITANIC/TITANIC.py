#IMPORTANTE
# - Comparar resultados entre validaçaão cruzada e holdout.

import pandas as pd

def one_hot_encode(X):
    #FOR SEX
    cleanup_nums = {"Sex" : {"male": 1, "female": 0}}
    X = X.replace(cleanup_nums)
    
    #FOR EMBARKED
    dummies = pd.get_dummies(X['Embarked'], drop_first=True)
    X = X.drop(columns=['Embarked'])
    X = X.join(dummies)
    
    return X

def normalize(X, X_temp):
    #z-score normalizing
    X['Fare'] = (X['Fare'] - X_temp['Fare'].mean())/X_temp['Fare'].std()
    X['Age'] = (X['Age'] - X_temp['Age'].mean())/X_temp['Age'].std()
    X['Family'] = (X['Family'] -  X_temp['Family'].mean())/X_temp['Family'].std()
    
    #min-max normalizing
    X['Pclass'] = (X['Pclass'] - X_temp['Pclass'].min())/(X_temp['Pclass'].max() - X_temp['Pclass'].min())
    return X

def read_data():
    #READ TRAIN
    df = pd.read_csv('titanic_data.csv', sep=';')
    df['Family'] = df['SibSp'] + df['Parch']
    df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'])
    #df.isna().sum()
    df = df.dropna()
    
    #df.describe()
    
    #Split Train and Test 80-20
    X_train = df.sample(frac=0.8)
    y_train = X_train['Survived']
    
    X_test = df.drop(X_train.index)
    y_test = X_test['Survived']
    
    #Encode Sex and Embarked
    X_train = one_hot_encode(X_train).drop(columns=['Survived'])
    X_test  = one_hot_encode(X_test).drop(columns=['Survived'])
    
    #NORMALIZE
    X_train = normalize(X_train, df)
    X_test  = normalize(X_test, df)
    
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test= read_data()

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initializing the ANN
classifier = Sequential()

#HIDDEN LAYERS
classifier.add(Dense(units=20, kernel_initializer='uniform', activation = 'relu', input_dim=X_train.shape[1]))
classifier.add(Dense(units=20, kernel_initializer='uniform', activation = 'relu'))
#ADD OUTPUT LAYER
classifier.add(Dense(units=1, kernel_initializer='uniform', activation = 'sigmoid'))

#compiling ANN
classifier.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False), metrics=['accuracy'])
    
#Fitting the ANN
history = classifier.fit(x = X_train, y = y_train, validation_data = (X_test, y_test), batch_size = 5, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5).astype('int')

import matplotlib.pyplot as plt


#PLOT ACC
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Accuracy Train', 'Accuracy Validation'], loc='upper left')
plt.show()

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
confusion_matrix(y_test, y_pred)
print("Baseline Error: %.2f%%" % (100-accuracy_score(y_test, y_pred)*100))