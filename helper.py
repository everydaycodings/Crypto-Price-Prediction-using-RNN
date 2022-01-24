from cProfile import label
import pickle
import time, datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os


epochs = 2
plt.figure(figsize=(5,6))

checkpoint_filepath = "temp/model.hdf5"
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode="min",
    save_best_only=True)

callbacks = [
             EarlyStopping(patience=10, monitor="val_loss", mode="min"),
             model_checkpoint_callback,
]

def fetch_options():

    asset_file = pickle.load(open("files/asset.pickle", 'rb'))

    return asset_file


def fetch_data(ticker):
    period1 = int(time.mktime(datetime.datetime(2010, 12, 1, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2025, 12, 20, 23, 59).timetuple()))
    interval = '1d' # 1d, 1m

    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'

    df = pd.read_csv(query_string)

    return df



def display_loss_plot(model):

    loss = model.history["loss"]
    val_loss = model.history["val_loss"]

    graph = {"Loss": loss,
            "Val_Loss": val_loss}

    graph = pd.DataFrame(graph)

    return graph



def convert_dataset_matrix(dataset, time_step=7):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)



def train_model(df):

    global X_train, X_test, scaler, df1, test_data
    df1=df.reset_index()['Close']
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

    training_size=int(len(df1)*0.65)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

    time_step = 100
    X_train, y_train = convert_dataset_matrix(train_data, time_step)
    X_test, ytest = convert_dataset_matrix(test_data, time_step)


    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    model = model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=epochs,batch_size=64,verbose=1, callbacks = callbacks)

    return model

def display_accuracy_graph_plot():
    model = load_model("temp/model.hdf5")
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

    look_back=100
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(df1), label="complete Dataset")
    plt.plot(trainPredictPlot, label="trained Dataset")
    plt.plot(testPredictPlot, label="Predicted data")
    plt.legend()
    pic_path = "temp/pic.png"
    plt.savefig(pic_path)
    plot = pic_path
    return plot



def predict():

    model = load_model("temp/model.hdf5")
    lst_output=[]
    n_steps=100
    i=0

    x_input=test_data[len(test_data) - n_steps:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    while(i<30):
        
        if(len(temp_input)>100):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)

    plt.close()
    plt.plot(day_new,scaler.inverse_transform(df1[len(df1) - n_steps:]), label="Current Graph")
    plt.plot(day_pred,scaler.inverse_transform(lst_output), label="Future Prediction")
    plt.legend()
    pic_path = "temp/pic1.png"
    plt.savefig(pic_path)
    plot = pic_path

    plt.close()
    df3=df1.tolist()
    df3.extend(lst_output)
    plt.plot(df3[1200:])
    pic_path = "temp/pic2.png"
    plt.savefig(pic_path)
    plot1 = pic_path

    return plot, plot1



