import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"#限于各驱动程序差异，此处禁止GPU运算处理
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, LeakyReLU, Dropout
from sklearn.preprocessing import MinMaxScaler


def univariate_data(close_dataset,open_dataset,high_dataset,low_dataset, start_index, end_index, history_size, target_size):#数据预处理
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(close_dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        close_data = close_dataset[indices].flatten()
        open_data = open_dataset[indices].flatten()
        high_data = high_dataset[indices].flatten()
        low_data = low_dataset[indices].flatten()
        res_data = [close_data.tolist(),open_data.tolist(),high_data.tolist(),low_data.tolist()]
        data_val = np.asarray(res_data)
        data.append(data_val)
        labels.append(close_dataset[i + target_size])
    return np.array(data), np.array(labels)

stock=['601318','601988','601939','600519','002230','000858','002594','601099','600019','600036']

for i in stock:

    filepath="D:/Documents/finance/data/"+str(i)
    ds = pd.read_csv(filepath+".csv",index_col="ts_code")[["pct_chg","close","open","high","low"]]#读取表格中的各项数据
    min_max_scaler = MinMaxScaler()#对数据进行均一化处理

    close_price = ds[['close']]
    close_norm_data = min_max_scaler.fit_transform(close_price.values)
    open_price = ds[['open']]
    open_norm_data = min_max_scaler.fit_transform(open_price.values)
    high_price = ds[['high']]
    high_norm_data = min_max_scaler.fit_transform(high_price.values)
    low_price = ds[['low']]
    low_norm_data = min_max_scaler.fit_transform(low_price.values)

    past_history = 5
    future_target = 0

    # 取 80%数据进行训练 20%进行测试
    TRAIN_SPLIT = int(len(close_norm_data) * 0.8)

    # 训练数据集
    x_train, y_train = univariate_data(close_norm_data,
                                       open_norm_data,
                                       high_norm_data,
                                       low_norm_data,
                                       0,
                                       TRAIN_SPLIT,
                                       past_history,
                                       future_target)
    print('x:'+str(x_train))
    print('y:'+str(y_train))
    # 测试数据集
    x_test, y_test = univariate_data(close_norm_data,
                                     open_norm_data,
                                     high_norm_data,
                                     low_norm_data,
                                     TRAIN_SPLIT,
                                     len(close_norm_data),
                                     past_history,
                                     future_target)

    #配置神经网络相关参数
    num_units = 64
    learning_rate = 0.0001
    activation_function = 'linear'
    adam = Adam(lr=learning_rate)
    loss_function = 'mse'
    batch_size = 65
    num_epochs = 50

    # Initialize the RNN
    model = Sequential()
    # input_shape 时间步,每个时间步的特征长度(dim)
    model.add(LSTM(units=num_units, activation=activation_function, input_shape=(4, 5)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compiling the RNN
    model.compile(optimizer=adam, loss=loss_function)

    # 训练模型
    # Using the training set to train the model
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=num_epochs,
        shuffle=False
    )

    #绘图并加入标签
    original_value = min_max_scaler.inverse_transform(y_test).flatten()
    predictions_value = min_max_scaler.inverse_transform(model.predict(x_test)).flatten()
    df_result = pd.DataFrame(
        {"key": pd.Series(range(0, len(original_value))), "original": original_value, "predictions": predictions_value})
    df_result["original"].plot(label="original")
    df_result["predictions"].plot(label="predictions")
    plt.xlabel('time')
    plt.ylabel('index')
    plt.title('Stock:'+i+'\'s chart')
    plt.legend()
    plt.savefig(filepath+"lstm.png")
    plt.close()
