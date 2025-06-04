"""
Created on Wed Mar 20 20:51:26 2019
Random forest model
@author: Stamatis
"""

""" seed for reproducibility """
np.random.seed(initial_seed)
random.seed(initial_seed)

""" load model or create new """
load_or_create_model = 'create'
selected_model, model_dict = 'rf', dict()
if need_to_update: version = 22

""" load model """
if load_or_create_model == 'load':
    with open('v' + str(version) + '_' + selected_model + '_regr_model' + '.pkl', 'rb') as f:
        if selected_model == 'poly':
            cost_regressor, train_rmse, cross_val_rmse, test_rmse, poly = pickle.load(f)
        else:
            cost_regressor, train_rmse, cross_val_rmse, test_rmse = pickle.load(f)
        
""" create model """
if load_or_create_model == 'create':

    """ split dataset """
    X_train, X_test, cost_train, cost_test = train_test_split(X, cost, test_size = 0.2, random_state = 0) 
    
    """ train polynomial regression model """
    poly = PolynomialFeatures(degree = 3) 
    X_train_poly = poly.fit_transform(X_train) 
    poly_regressor = linear_model.LinearRegression(normalize = True)
    poly_regressor.fit(X_train_poly, cost_train)  
    poly_test_pred, poly_train_pred = poly_regressor.predict(poly.transform(X_test)), poly_regressor.predict(poly.transform(X_train))
    poly_train_rmse, poly_cross_val_rmse, poly_test_rmse, poly_test_r2, poly_train_r2 = np.sqrt(metrics.mean_squared_error(cost_train, poly_train_pred)), np.sqrt(-np.mean(cross_val_score(poly_regressor, poly.transform(X), cost, cv = 10, scoring = 'neg_mean_squared_error'))), np.sqrt(metrics.mean_squared_error(cost_test, poly_test_pred)), metrics.r2_score(cost_test, poly_test_pred), metrics.r2_score(cost_train, poly_train_pred)
    model_dict['poly'] = (poly_regressor, poly_train_rmse, poly_cross_val_rmse, poly_test_rmse)
    
#    """ train kNN model """
#    scaler = MinMaxScaler(feature_range=(0, 1))
#    X_scaled, X_train_scaled, X_test_scaled = scaler.fit_transform(X), scaler.fit_transform(X_train), scaler.fit_transform(X_test)
#    knn_regressor = neighbors.KNeighborsRegressor(n_neighbors = 10)
#    knn_regressor.fit(X_train_scaled, cost_train)
#    knn_test_pred, knn_train_pred = knn_regressor.predict(X_test_scaled), knn_regressor.predict(X_train_scaled)
#    knn_train_rmse, knn_cross_val_rmse, knn_test_rmse = np.sqrt(metrics.mean_squared_error(cost_train, knn_train_pred)), np.sqrt(-np.mean(cross_val_score(knn_regressor, X_scaled, cost, cv = 10, scoring = 'neg_mean_squared_error'))), np.sqrt(metrics.mean_squared_error(cost_test, knn_test_pred))
#    model_dict['knn'] = (knn_regressor, knn_train_rmse, knn_cross_val_rmse, knn_test_rmse)
    
    """ train random forest model """
    rf_regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)  
    rf_regressor.fit(X_train, cost_train)  
    rf_test_pred, rf_train_pred = rf_regressor.predict(X_test), rf_regressor.predict(X_train)
    rf_train_rmse, rf_cross_val_rmse, rf_test_rmse, rf_test_r2, rf_train_r2 = np.sqrt(metrics.mean_squared_error(cost_train, rf_train_pred)), np.sqrt(-np.mean(cross_val_score(rf_regressor, X, cost, cv = 10, scoring = 'neg_mean_squared_error'))), np.sqrt(metrics.mean_squared_error(cost_test, rf_test_pred)), metrics.r2_score(cost_test, rf_test_pred), metrics.r2_score(cost_train, rf_train_pred)
    model_dict['rf'] = (rf_regressor, rf_train_rmse, rf_cross_val_rmse, rf_test_rmse)
        
#    """ train SVR model """
#    svr_regressor = SVR(kernel = 'poly', degree = 2, C = 1)
#    svr_regressor.fit(X_train, cost_train)
#    svr_test_pred, svr_train_pred = svr_regressor.predict(X_test), svr_regressor.predict(X_train)
#    svr_train_rmse, svr_cross_val_rmse, svr_test_rmse = np.sqrt(metrics.mean_squared_error(cost_train, svr_train_pred)), np.sqrt(-np.mean(cross_val_score(svr_regressor, X, cost, cv = 10, scoring = 'neg_mean_squared_error'))), np.sqrt(metrics.mean_squared_error(cost_test, svr_test_pred))
#    model_dict['svr'] = (svr_regressor, svr_train_rmse, svr_cross_val_rmse, svr_test_rmse)
            
    """ print results """
    print('Polynomial Regression Results --- Training Error')
    print('Root Mean Squared Error:', poly_train_rmse)
    print('R2 Score:', poly_train_r2, '\n')
    print('Polynomial Regression Results --- Cross Validation Error')
    print('Root Mean Squared Error:', poly_cross_val_rmse, '\n')
    print('Polynomial Regression Results --- Testing Error')
    print('Root Mean Squared Error:', poly_test_rmse) 
    print('R2 Score:', poly_test_r2, '\n')
    
    #print('\n\n\nK Nearest Neighbors Results --- Training Error')
    #print('Root Mean Squared Error:', knn_train_rmse, '\n') 
    #print('K Nearest Neighbors Results --- Cross Validation Error')
    #print('Root Mean Squared Error:', knn_cross_val_rmse, '\n')
    #print('K Nearest Neighbors Results --- Testing Error')
    #print('Root Mean Squared Error:', knn_test_rmse)  
    
    print('\n\n\nRandom Forest Results --- Training Error')
    print('Root Mean Squared Error:', rf_train_rmse) 
    print('R2 Score:', rf_train_r2, '\n')
    print('Random Forest Results --- Cross Validation Error')
    print('Root Mean Squared Error:', rf_cross_val_rmse, '\n')
    print('Random Forest Results --- Testing Error')
    print('Root Mean Squared Error:', rf_test_rmse) 
    print('R2 Score:', rf_test_r2, '\n')
    
    #print('\n\n\nSupport Vector Regression Results --- Training Error')
    #print('Root Mean Squared Error:', svr_train_rmse, '\n') 
    #print('Support Vector Regression Results --- Cross Validation Error')
    #print('Root Mean Squared Error:', svr_cross_val_rmse, '\n')
    #print('Support Vector Regression Results --- Testing Error')
    #print('Root Mean Squared Error:', svr_test_rmse)
    
    """ final model choice """
    cost_regressor, train_rmse, cross_val_rmse, test_rmse = model_dict[selected_model]
    with open('v' + str(version) + '_' + selected_model + '_regr_model' + '.pkl', 'wb') as f:
        if selected_model == 'poly':
            pickle.dump([cost_regressor, train_rmse, cross_val_rmse, test_rmse, poly], f)
        else:
            pickle.dump([cost_regressor, train_rmse, cross_val_rmse, test_rmse], f)