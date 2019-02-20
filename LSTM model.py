#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: JONNYFLDN
"""

def create_batch(X,Y,timestep):
    ''' Create batches with 3-dimensions (batch_size,timesteps,features)
    '''
    
    _X,_Y = [],[]
    
    for t in range(X.shape[0]-timestep):
        _X.append(X[t:t+timestep])
        _Y.append(Y[t+timestep]) 
    return np.array(_X), np.array(_Y)


def LSTM_run(model,X,Y,window,timestep):
    ''' Run LSTM model on each window, approximately re-initialising weights via permutations
    '''
    y_pred = []
    y_test = []
    #Get initial weights (default =glorot_uniform)
    initial_weights = model.get_weights()
    
    for t in range(Y.index[window:].shape[0]):
        x_scaler = MinMaxScaler(feature_range=(0,1))
        y_scaler = MinMaxScaler(feature_range=(0,1))
        
        x_train,y_train = x_scaler.fit_transform(X[t:window+t]),y_scaler.fit_transform(Y[t:window+t])
        x_train,y_train = create_batch(x_train,y_train,timestep=timestep)
        
        x_test = np.expand_dims(x_scaler.transform(X.iloc[window-timestep+t:window+t]),axis=0)
        y_test.append(Y.iloc[window+t].values[0])
        
        model.fit(x_train,y_train,epochs=1,batch_size=1,shuffle=False)

        y_pred.append(y_scaler.inverse_transform(model.predict(x_test))[0][0])
        
        initial_weights = [np.random.permutation(w.flat).reshape(w.shape) for w in initial_weights]
        model.set_weights(initial_weights)
    
        
    return y_pred,y_test

def define_model(timestep,features):
    '''LSTM model '''
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(timestep,features),stateful=False))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False, input_shape=(timestep,features),stateful=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1,activation='linear'))
    model.compile(optimizer='rmsprop',loss='mean_squared_error')
    return model 
    

'''Example ''' 

X,Y= FX_model.add_lags(X_set,Y_set,lags = [1,2])
ccy = X.columns.unique('ccy')
features = len(X.columns.unique('variables'))
window = 36
timestep = 5

forecast_idx = Y.index[window:]
df_LSTM_pred = pd.DataFrame(index = forecast_idx,columns =ccy)
df_LSTM_true = pd.DataFrame(index = forecast_idx,columns =ccy)

LSTM_old_pred,LSTM_old_true= df_LSTM_pred,df_LSTM_true

for c in ccy:
    X_ccy = X.loc[:,(slice(None),c)]
    Y_ccy = Y.loc[:,(slice(None),c)]
    
    model = define_model(timestep,features)
    y_pred,y_test = LSTM_run(model,X_ccy,Y_ccy,window,timestep)
    df_LSTM_pred[c],df_LSTM_true[c] = y_pred,y_test
    
df_historical = Y.xs('FX_diff',axis=1,drop_level=True).shift(1).rolling(window).mean().dropna()


m_MSE = (df_LSTM_pred-df_LSTM_true)**2
h_MSE = (df_historical-df_LSTM_true)**2
m_cum_RMSE = np.sqrt(m_MSE.expanding().mean())
h_cum_RMSE = np.sqrt(h_MSE.expanding().mean())
d_cum_RMSE = h_cum_RMSE - m_cum_RMSE
r2oos = 1 - m_MSE.sum(axis=0)/h_MSE.sum(axis=0)

LSTM_pred = d_cum_RMSE,r2oos
