from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv('psspredict_new02.csv', header=0, index_col=0)
values = dataset.values
print("1 values")
print("0",values[0])
print("1",values[1])
print("2",values[2])
print("3",values[3])
print("4",values[4])
print("5",values[5])
print("6",values[6])
print("7",values[7])
print("8",values[8])
print("9",values[9])
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
print("2 values")
print("0",values[0])
print("1",values[1])
print("2",values[2])
print("3",values[3])
print("4",values[4])
print("5",values[5])
print("6",values[6])
print("7",values[7])
print("8",values[8])
print("9",values[9])

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print("3 values")
print("0",values[0])
print("1",values[1])
print("2",values[2])
print("3",values[3])
print("4",values[4])
print("5",values[5])
print("6",values[6])
print("7",values[7])
print("8",values[8])
print("9",values[9])

print("4 scaled")
print("0",scaled[0])
print("1",scaled[1])
print("2",scaled[2])
print("3",scaled[3])
print("4",scaled[4])
print("5",scaled[5])
print("6",scaled[6])
print("7",scaled[7])
print("8",scaled[8])
print("9",scaled[9])

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
print("5 scaled")
print("0",scaled[0])
print("1",scaled[1])
print("2",scaled[2])
print("3",scaled[3])
print("4",scaled[4])
print("5",scaled[5])
print("6",scaled[6])
print("7",scaled[7])
print("8",scaled[8])
print("9",scaled[9])
print("6 reframed")
print("0",reframed.head(10))

# drop columns we don't want to predict
reframed.drop(reframed.columns[[1,2,3,4,5,6,7]], axis=1, inplace=True)
print("7 reframed")
print("0",reframed.head(10))


print(reframed.head(10))
 
# split into train and test sets
values = reframed.values
print("8 values")
print("0",values[0])
print("1",values[1])
print("2",values[2])
print("3",values[3])
print("4",values[4])
print("5",values[5])
print("6",values[6])
print("7",values[7])
print("8",values[8])
print("9",values[9])

n_train_hours = 365 * 24 * 15 
train = values[:n_train_hours, :]
print("9 train")
print("0",train[0])
print("1",train[1])
print("2",train[2])
print("3",train[3])
print("4",train[4])
print("5",train[5])
print("6",train[6])
print("7",train[7])
print("8",train[8])
print("9",train[9])

test = values[n_train_hours:, :]
print("10 test")
print("0",test[0])
print("1",test[1])
print("2",test[2])
print("3",test[3])
print("4",test[4])
print("5",test[5])
print("6",test[6])
print("7",test[7])
print("8",test[8])
print("9",test[9])

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
print("11 train_X")
print("0",train_X[0])
print("1",train_X[1])
print("2",train_X[2])
print("3",train_X[3])
print("4",train_X[4])
print("5",train_X[5])
print("6",train_X[6])
print("7",train_X[7])
print("8",train_X[8])
print("9",train_X[9])
print("12 train_y")
print("0",train_y[0])
print("1",train_y[1])
print("2",train_y[2])
print("3",train_y[3])
print("4",train_y[4])
print("5",train_y[5])
print("6",train_y[6])
print("7",train_y[7])
print("8",train_y[8])
print("9",train_y[9])

test_X, test_y = test[:, :-1], test[:, -1]
print("13 test_X")
print("0",test_X[0])
print("1",test_X[1])
print("2",test_X[2])
print("3",test_X[3])
print("4",test_X[4])
print("5",test_X[5])
print("6",test_X[6])
print("7",test_X[7])
print("8",test_X[8])
print("9",test_X[9])
print("14 test_y")
print("0",test_y[0])
print("1",test_y[1])
print("2",test_y[2])
print("3",test_y[3])
print("4",test_y[4])
print("5",test_y[5])
print("6",test_y[6])
print("7",test_y[7])
print("8",test_y[8])
print("9",test_y[9])

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
print("15 train_X")
print("0",train_X[0])
print("1",train_X[1])
print("2",train_X[2])
print("3",train_X[3])
print("4",train_X[4])
print("5",train_X[5])
print("6",train_X[6])
print("7",train_X[7])
print("8",train_X[8])
print("9",train_X[9])

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print("16 test_X")
print("0",test_X[0])
print("1",test_X[1])
print("2",test_X[2])
print("3",test_X[3])
print("4",test_X[4])
print("5",test_X[5])
print("6",test_X[6])
print("7",test_X[7])
print("8",test_X[8])
print("9",test_X[9])

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#print("17 history")
#print("0",history.head(10))


# plot history
 
# make a prediction
print("17 test_X")
print("0",test_X[0])
print("1",test_X[1])
print("2",test_X[2])
print("3",test_X[3])
print("4",test_X[4])
print("5",test_X[5])
print("6",test_X[6])
print("7",test_X[7])
print("8",test_X[8])
print("9",test_X[9])

yhat = model.predict(test_X)
print("18 yhat")
print("0",yhat[0])
print("1",yhat[1])
print("2",yhat[2])
print("3",yhat[3])
print("4",yhat[4])
print("5",yhat[5])
print("6",yhat[6])
print("7",yhat[7])
print("8",yhat[8])
print("9",yhat[9])

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
print("19 test_X")
print("0",test_X[0])
print("1",test_X[1])
print("2",test_X[2])
print("3",test_X[3])
print("4",test_X[4])
print("5",test_X[5])
print("6",test_X[6])
print("7",test_X[7])
print("8",test_X[8])
print("9",test_X[9])

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
print("20 inv_yhat")
print("0",inv_yhat[0])
print("1",inv_yhat[1])
print("2",inv_yhat[2])
print("3",inv_yhat[3])
print("4",inv_yhat[4])
print("5",inv_yhat[5])
print("6",inv_yhat[6])
print("7",inv_yhat[7])
print("8",inv_yhat[8])
print("9",inv_yhat[9])

inv_yhat = scaler.inverse_transform(inv_yhat)
print("21 inv_yhat")
print("0",inv_yhat[0])
print("1",inv_yhat[1])
print("2",inv_yhat[2])
print("3",inv_yhat[3])
print("4",inv_yhat[4])
print("5",inv_yhat[5])
print("6",inv_yhat[6])
print("7",inv_yhat[7])
print("8",inv_yhat[8])
print("9",inv_yhat[9])

inv_yhat = inv_yhat[:,0]
print("22 inv_yhat")
print("0",inv_yhat[0])
print("1",inv_yhat[1])
print("2",inv_yhat[2])
print("3",inv_yhat[3])
print("4",inv_yhat[4])
print("5",inv_yhat[5])
print("6",inv_yhat[6])
print("7",inv_yhat[7])
print("8",inv_yhat[8])
print("9",inv_yhat[9])

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
print("23 test_y")
print("0",test_y[0])
print("1",test_y[1])
print("2",test_y[2])
print("3",test_y[3])
print("4",test_y[4])
print("5",test_y[5])
print("6",test_y[6])
print("7",test_y[7])
print("8",test_y[8])
print("9",test_y[9])

inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
print("24 inv_y")
print("0",inv_y[0])
print("1",inv_y[1])
print("2",inv_y[2])
print("3",inv_y[3])
print("4",inv_y[4])
print("5",inv_y[5])
print("6",inv_y[6])
print("7",inv_y[7])
print("8",inv_y[8])
print("9",inv_y[9])

inv_y = scaler.inverse_transform(inv_y)
print("25 inv_y")
print("0",inv_y[0])
print("1",inv_y[1])
print("2",inv_y[2])
print("3",inv_y[3])
print("4",inv_y[4])
print("5",inv_y[5])
print("6",inv_y[6])
print("7",inv_y[7])
print("8",inv_y[8])
print("9",inv_y[9])

inv_y = inv_y[:,0]
print("26 inv_y")
print("0",inv_y[0])
print("1",inv_y[1])
print("2",inv_y[2])
print("3",inv_y[3])
print("4",inv_y[4])
print("5",inv_y[5])
print("6",inv_y[6])
print("7",inv_y[7])
print("8",inv_y[8])
print("9",inv_y[9])

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
print("inv_y,inv_ychat",inv_y[0],inv_yhat[0],abs(inv_y[0]-inv_yhat[0])/inv_y[0])
print("inv_y,inv_ychat",inv_y[1],inv_yhat[1],abs(inv_y[1]-inv_yhat[1])/inv_y[1])
print("inv_y,inv_ychat",inv_y[2],inv_yhat[2],abs(inv_y[2]-inv_yhat[2])/inv_y[2])
print("inv_y,inv_ychat",inv_y[3],inv_yhat[3],abs(inv_y[3]-inv_yhat[3])/inv_y[3])
print("inv_y,inv_ychat",inv_y[4],inv_yhat[4],abs(inv_y[4]-inv_yhat[4])/inv_y[4])
print("inv_y,inv_ychat",inv_y[5],inv_yhat[5],abs(inv_y[5]-inv_yhat[5])/inv_y[5])
print("inv_y,inv_ychat",inv_y[6],inv_yhat[6],abs(inv_y[6]-inv_yhat[6])/inv_y[6])
print("inv_y,inv_ychat",inv_y[7],inv_yhat[7],abs(inv_y[7]-inv_yhat[7])/inv_y[7])
print("inv_y,inv_ychat",inv_y[8],inv_yhat[8],abs(inv_y[8]-inv_yhat[8])/inv_y[8])
print("inv_y,inv_ychat",inv_y[9],inv_yhat[9],abs(inv_y[9]-inv_yhat[9])/inv_y[9])
