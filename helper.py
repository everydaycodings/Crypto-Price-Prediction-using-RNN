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


epochs = 2

checkpoint_filepath = "model/model.hdf5"
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

    model = pickle.load(open("files/asset.pickle", 'rb'))

    return model


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

    global X_train, X_test, scaler
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

def display_accuracy_graph_plot(df1):
    model = load_model("model/model.hdf5")
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
    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plot = plt.show()
    return plot

def predict():
    pass



