
"""
@author: JONNYFLDN
"""

class FX_model(object):
    
    ''' FX model class to compute out-of-sample predictions ''' 
    
    
    '''Parameters 
       ----------
       model : regressor to be trained 
               e.g Linear Regression or Random Forest
               
       X : array-like input consisting of independent variables / features
       
       Y : array-like input consisting of dependent variables / targets
       
       w_type : expanding or rolling window implementation 
                defaults on expanding window
        
       w_size : window size, defaults on 10 time steps
       
       settings : dictionary including extra parameters
                  
                  hyper-parameters: cv_split, defaults on sklearn's TimeSeriesSplit(n_splits =2)
                                    hyper_param, parameters and grid to be searched
                                    
                  model features: features to be extracted per time-step (e.g. feature-importance)
                  
                  early stopping: specifically included for gradient boosting
                  
                  Example: settings = {"hyper_param":{"cv_split":TimeSeriesSplit(n_splits=2),
                                                      "params":{'min_samples_leaf':[2,3]}},
                                       "early_stopping":{"early_stopping_rounds":3,
                                                         "eval_metric":'rmse','verbose':False},
                                       "features":["feature_importances_"]}
                                       
                  
        lags : lagged amount of X and Y to update feature variables, defaults to None
    
        Attributes
        ----------
        add_lags : static method to return lag X and Y 
        
        hyper_tuning : return best model and hyper-parameters 
                       following tuning
                       
        regress : return model and historical predictions
        
        predict : populate key metrics using regress
        
        
                                 
    '''
    
    def __init__(self,model,X,Y,w_type='expanding',w_size=10,settings=None,lags=None):
                     
        #Inputs
        self.model = model
        self.X = X
        self.Y = Y
        self.settings = settings
        self.lags = lags
        self.w_type = w_type 
        self.w_size = w_size
        self.train_size = w_size
        self.ccy = X.columns.unique(X.columns.names[1])
        self.hyper_param,self.features,self.early_stopping = None, None, None

        if lags:
            self.X,self.Y = self.add_lags(X,Y,lags)

        if settings: 
            self.hyper_param = settings.get('hyper_param',None)
            self.early_stopping = settings.get('early_stopping',None)
            
            if self.early_stopping:
                self.test_size = round(self.w_size*(1-0.8))
            
            if self.hyper_param:
                self.cv_split = self.hyper_param.get('cv_split',TimeSeriesSplit(2))
                
            self.features = settings.get('features',None)
            
        #Output
        self.forecast_index = self.Y.index.intersection(X.index)[w_size:]
        #Historical predictor
        self.h_predict = pd.DataFrame(index=self.forecast_index)
        self.h_MSE = pd.DataFrame(index=self.forecast_index)
        self.h_cum_RMSE = pd.DataFrame(index=self.forecast_index)
        #Model predictor
        self.m_predict = pd.DataFrame(index=self.forecast_index)
        self.m_MSE = pd.DataFrame(index=self.forecast_index)
        self.m_cum_RMSE = pd.DataFrame(index=self.forecast_index)
        #difference and out-of-sample
        self.d_cum_RMSE = pd.DataFrame(index=self.forecast_index)
        self.r2oos = pd.DataFrame(index=self.forecast_index)
        
        self.model_features = pd.DataFrame(index=self.forecast_index)
        self.best_hparam= pd.DataFrame(index=self.forecast_index)
        
    @staticmethod
    def add_lags(X,Y,lags):
        lagged_df = pd.DataFrame(index=X.index)
        levels = X.columns.names[0]
        
        for lag in lags:
                X_lagged = X.shift(lag)
                xcol = X_lagged.columns.unique(level=levels)+" T-"+str(lag)
                X_lagged.columns = X_lagged.columns.set_levels(xcol,level=levels)
                lagged_df= pd.concat([lagged_df,X_lagged],axis=1,sort=False)
                
                Y_lagged = Y.shift(lag)
                ycol = Y_lagged.columns.unique(level=levels)+" T-"+str(lag)
                Y_lagged.columns = Y_lagged.columns.set_levels(ycol,level=levels)
                lagged_df = pd.concat([lagged_df,Y_lagged],axis=1,sort=False)
        
        
        return pd.concat([X,lagged_df],axis=1,sort=False).dropna(axis=0),Y.iloc[len(lags):]
    
    
    def hyper_tuning(self,x,y):

        optimise = GridSearchCV(self.model,param_grid = self.hyper_param['params']
                                ,n_jobs = -1,
                                cv=self.cv_split,scoring='neg_mean_squared_error')\
                                .fit(x,y)
        
        best_params = optimise.best_params_
        best_model = self.model.set_params(**best_params)
        return best_model,best_params
    
    def predict(self):
        
        for ccy in self.ccy:
            X = self.X.loc[:,(slice(None),ccy)]
            Y = self.Y.loc[:,(slice(None),ccy)]
            
            m_predict,h_predict,model_features,hyper_param = zip(*self.regress(X,Y))
            self.m_predict[ccy],self.h_predict[ccy]= m_predict,h_predict
            self.model_features[ccy],self.best_hparam[ccy] = list(model_features),list(hyper_param)
        
        Y_test = self.Y.xs('FX_diff',axis=1,drop_level=True)
        self.m_MSE = self.m_predict.subtract(Y_test,axis='rows',level=1).dropna()**2
        self.h_MSE = self.h_predict.subtract(Y_test,axis='rows',level=1).dropna()**2
        self.m_cum_RMSE = np.sqrt(self.m_MSE.expanding().mean())
        self.h_cum_RMSE = np.sqrt(self.h_MSE.expanding().mean())
        self.d_cum_RMSE = self.h_cum_RMSE - self.m_cum_RMSE
        
        self.r2oos = 1 - self.m_MSE.sum(axis=0)/self.h_MSE.sum(axis=0)
            
        return self.d_cum_RMSE,self.r2oos
        
    
    def regress(self,X,Y):
        best_hparam = None
        model_features = None
        
        for t in range(len(self.forecast_index)):
            i = 0 if self.w_type =='expanding' else t #expanding window
            
            x_train,y_train = X.iloc[i:self.w_size+t].values,Y.iloc[i:self.w_size+t].values
            x_test = X.iloc[self.w_size+t].values
            
            if self.hyper_param:
                self.model,best_param = self.hyper_tuning(x_train,y_train.ravel())
                best_hparam = [v for k,v in best_param.items()]
            
            if self.early_stopping:
                #split into train and validation
                x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = self.test_size)
                self.model = self.model.fit(x_train,y_train.ravel(),eval_set =[(x_val,y_val)],**self.early_stopping)
            else:
                self.model = self.model.fit(x_train,y_train.ravel())
            
            model_prediction = self.model.predict(x_test.reshape(1,-1))[0]
            hist_prediction = np.mean(y_train)
            
            if self.features:
                model_features = [getattr(self.model,f) for f in self.features]
            
            yield model_prediction,hist_prediction,model_features,best_hparam
                
            