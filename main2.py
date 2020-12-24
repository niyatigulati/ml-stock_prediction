import pandas as pd
import numpy as np
from IPython.display import display
import LinearRegressionModel
import randomForestModel
import lstm, time 
import decessionTreeModel
import visualize as vs
import stock_data as sd
import preprocess as prepr
tickers = ['null','IBM', 'MSFT', 'NYSE:WIT']
def linearreg():
    display(stocks.head())
    X_train, X_test, y_train, y_test, label_range= sd.train_test_split_linear_regression(stocks)
    model = LinearRegressionModel.build_model(X_train,y_train)
    predictions = LinearRegressionModel.predict_prices(model,X_test, label_range)
    vs.plot_prediction(y_test,predictions)
def randomForest():
    display(stocks.head())
    X_train, X_test, y_train, y_test, label_range= sd.train_test_split_randomForest(stocks)
    model = randomForestModel.build_model(X_train,y_train)
    predictions = LinearRegressionModel.predict_prices(model,X_test, label_range)
    vs.plot_prediction(y_test,predictions)    
def dectre():
    X = stocks.iloc[:, 1:2].values
    y = stocks.iloc[:, 2].values
    model= decessionTreeModel.build_model(X, y)
    y_pred = model.predict(X[:360])
    print(y_pred)
    import matplotlib.pyplot as plt
    plt.plot(stocks.item[:360],y[:360], color = 'red', label='Adjusted Close')
    plt.plot(stocks.item[:360],y_pred, color = 'blue', label='Predicted Close')
    plt.title('Trading vs Prediction')
    plt.xlabel('Price USD')
    plt.ylabel('Trading Days')
    plt.legend(loc='upper left')
    plt.show()
def lstmm():
    stocks = pd.read_csv('preprocessed.csv')
    stocks_data = stocks.drop(['item'], axis =1)

    display(stocks_data.head())
    
    X_train, X_test,y_train, y_test = sd.train_test_split_lstm(stocks_data, 5)
    
    unroll_length = 50
    
    X_train = sd.unroll(X_train, unroll_length)
    X_test = sd.unroll(X_test, unroll_length)
    y_train = y_train[-X_train.shape[0]:]
    y_test = y_test[-X_test.shape[0]:]
    
    print("x_train", X_train.shape)
    print("y_train", y_train.shape)
    print("x_test", X_test.shape)
    print("y_test", y_test.shape)
    model = lstm.build_basic_model(input_dim = X_train.shape[-1],output_dim = unroll_length, return_sequences=True)
    start = time.time()
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    print('compilation time : ', time.time() - start)
    history = model.fit(
        X_train,
        y_train,
        epochs=1,
        validation_split=0.05)
    print("Accuracy:-")
    print(history.history.keys())
    print(history.history['acc'],'%')    
    predictions = model.predict(X_test)
    vs.plot_lstm_prediction(y_test,predictions)
print("choose Stock Index")
print('1-Bitcoin')
print('2-MSFT')
print('3-Forex')
index = int(input('Enter Value 1 to 3  and press enter:-'))
stocks = prepr.get_alpha_vantage_data(tickers[index])
# stocks = pd.read_csv('preprocessed.csv')
print(stocks)
print ("Select Classifier:")
print ("1- Linear Regression ")
print ("2- Random Forest")
print ("3- recurrent neural network (RNN)  ")
print ("4- Decision Tree")
classifr =input('Enter Value from 1 to 4 and press enter:-')
if  (classifr == "1"):
    linearreg()
elif(classifr == "2"):
    randomForest()
elif(classifr == "3"):
    lstmm()
elif(classifr == "4"):
    dectre()
       