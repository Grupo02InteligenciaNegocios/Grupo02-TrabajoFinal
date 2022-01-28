import os
import sys
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras_metrics as km
import math as mt
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)
price_data = pd.read_csv('EmpresasPorAnio.csv')

def model_view(string):
    if string == 'Random Forest':
        price_data = pd.read_csv('EmpresasPorAnio.csv')
        
        st.subheader('Filtrado de columnas a usar')
        price_data = price_data[['Empresa','Date','Close','High','Low','Open','Volume']]
        price_data.sort_values(by = ['Empresa','Date'], inplace = True)
        price_data['change_in_price'] = price_data['Close'].diff()
        
        mask = price_data['Empresa'] != price_data['Empresa'].shift(1)
        price_data['change_in_price'] = np.where(mask == True, np.nan, price_data['change_in_price'])
        price_data[price_data.isna().any(axis = 1)]
        
        st.subheader('Preprocesamiento: Smoothing Data')
        st.text('El objetivo de suavizarlo es eliminar la aleatoriedad y el ruido de nuestros datos \nde precios. En otras palabras, no obtenemos un gráfico puntiagudo hacia arriba y \nhacia abajo, sino uno más suave. Además, esto ayudará al modelo a identificar más \nfácilmente las tendencias a largo plazo.')
        days_out = 30
        price_data_smoothed = price_data.groupby(['Empresa'])[['Close','Low','High','Open','Volume']].transform(lambda x: x.ewm(span = days_out).mean())
        smoothed_df = pd.concat([price_data[['Empresa','Date']], price_data_smoothed], axis=1, sort=False)
        st.write(smoothed_df.head(20))
        
        st.subheader('Preprocesamiento: Signal Flag')
        st.text('Si elige realizar el proceso de suavizado, debemos agregar una columna adicional a \nnuestro marco de datos. Esto servirá como columna de diferencia del marco de datos \noriginal. Sin embargo, en este caso, no queremos un día consecutivo al siguiente, \nqueremos la cantidad de días que queremos predecir. Lo que haremos es tomar la \nventana que usamos arriba para calcular nuestra estadística suavizada y usarla \npara calcular nuestra bandera de señal.')
        days_out = 30
        smoothed_df['Signal_Flag'] = smoothed_df.groupby('Empresa')['Close'].transform(lambda x : np.sign(x.diff(days_out)))
        st.write(smoothed_df.head(20))
        
        st.subheader('Cálculo del indicador: índice de fuerza realtiva (RSI)')
        st.text('RSI es un indicador de impulso popular que determina si la acción está \nsobrecomprada o sobrevendida. Se dice que una acción está sobrecomprada cuando la \ndemanda empuja injustificadamente el precio hacia arriba. Esta condición \ngeneralmente se interpreta como una señal de que la acción está sobrevaluada y es \nprobable que el precio baje. Se dice que una acción está sobrevendida cuando el \nprecio cae bruscamente a un nivel por debajo de su valor real. Este es un \nresultado causado por la venta de pánico. El RSI varía de 0 a 100 y, en general, \ncuando el RSI está por encima de 70, puede indicar que las acciones están \nsobrecompradas y cuando el RSI está por debajo de 30, puede indicar que las \nacciones están sobrevendidas.')
        n = 14
        up_df, down_df = price_data[['Empresa','change_in_price']].copy(), price_data[['Empresa','change_in_price']].copy()
        up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0
        down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0
        down_df['change_in_price'] = down_df['change_in_price'].abs()
        ewma_up = up_df.groupby('Empresa')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
        ewma_down = down_df.groupby('Empresa')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
        relative_strength = ewma_up / ewma_down
        relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))
        price_data['down_days'] = down_df['change_in_price']
        price_data['up_days'] = up_df['change_in_price']
        price_data['RSI'] = relative_strength_index
        st.write(price_data.head(20))
        
        st.subheader('Cálculo del indicador: oscilador estocástico (Stochastic Oscillator)')
        st.text('El oscilador estocástico sigue la velocidad o el impulso del precio. Como regla \ngeneral, el impulso cambia antes de que cambie el precio. Mide el nivel del precio \nde cierre en relación con el rango bajo-alto durante un período de tiempo.')
        n = 14
        low_14, high_14 = price_data[['Empresa','Low']].copy(), price_data[['Empresa','High']].copy()
        low_14 = low_14.groupby('Empresa')['Low'].transform(lambda x: x.rolling(window = n).min())
        high_14 = high_14.groupby('Empresa')['High'].transform(lambda x: x.rolling(window = n).max())
        k_percent = 100 * ((price_data['Close'] - low_14) / (high_14 - low_14))
        price_data['low_14'] = low_14
        price_data['high_14'] = high_14
        price_data['k_percent'] = k_percent
        st.write(price_data.head(20))
        
        st.subheader('Cálculo del indicador: Williams R')
        st.text('Williams R oscila entre -100 y 0. Cuando su valor está por encima de -20, indica \nuna señal de venta y cuando su valor está por debajo de -80, indica una señal de \ncompra.')
        n = 14
        low_14, high_14 = price_data[['Empresa','Low']].copy(), price_data[['Empresa','High']].copy()
        low_14 = low_14.groupby('Empresa')['Low'].transform(lambda x: x.rolling(window = n).min())
        high_14 = high_14.groupby('Empresa')['High'].transform(lambda x: x.rolling(window = n).max())
        r_percent = ((high_14 - price_data['Close']) / (high_14 - low_14)) * - 100
        price_data['r_percent'] = r_percent
        st.write(price_data.head(20))
        
        st.subheader('Cálculo del indicador: media móvil convergencia divergencia (MACD)')
        st.text('EMA significa Media Móvil Exponencial. Cuando el MACD cae por debajo de \nSingalLine, indica una señal de venta. Cuando pasa por encima de la línea de \nseñal, indica una señal de compra.')
        ema_26 = price_data.groupby('Empresa')['Close'].transform(lambda x: x.ewm(span = 26).mean())
        ema_12 = price_data.groupby('Empresa')['Close'].transform(lambda x: x.ewm(span = 12).mean())
        macd = ema_12 - ema_26
        ema_9_macd = macd.ewm(span = 9).mean()
        price_data['MACD'] = macd
        price_data['MACD_EMA'] = ema_9_macd
        st.write(price_data.head(20))

        st.subheader('Cálculo del indicador: tasa de cambio de precio')
        st.text('Mide el cambio de precio más reciente con respecto al precio de n días atrás.')
        n = 9
        price_data['Price_Rate_Of_Change'] = price_data.groupby('Empresa')['Close'].transform(lambda x: x.pct_change(periods = n))
        st.write(price_data.head(20))
        
        st.subheader('Cálculo del indicador: Volumen en equilibrio')
        st.text('El volumen de balance (OBV) utiliza cambios en el volumen para estimar los \ncambios en los precios de las acciones. Este indicador técnico se utiliza para las \ntendencias de compra y venta de una acción, considerando el volumen acumulado: \nsuma acumulativamente los volúmenes en los días en que los precios se agrupan y \nresta el volumen en los días en que los precios bajan, en comparación con los \nprecios de el día anterior.')
        def obv(group):
            volume = group['Volume']
            change = group['Close'].diff()
            prev_obv = 0
            obv_values = []
            for i, j in zip(change, volume):
                if i > 0:
                    current_obv = prev_obv + j
                elif i < 0:
                    current_obv = prev_obv - j
                else:
                    current_obv = prev_obv
                prev_obv = current_obv
                obv_values.append(current_obv)
            return pd.Series(obv_values, index = group.index)
    
        obv_groups = price_data.groupby('Empresa').apply(obv)
        price_data['On Balance Volume'] = obv_groups.reset_index(level=0, drop=True)
        st.write(price_data.head(20))
        
        st.subheader('Construcción del modelo: creación de la columna de predicción')
        st.text('Para crear la columna de predicción, se agrupa el marco de datos por cada \n"Empresa". Una vez que se haya creado los grupos, debemos seleccionar la columna \n"Close", ya que contiene el precio que necesitamos para determinar si las acciones \ncerraron al alza o a la baja en un día determinado. Ahora, se puede usar una \nlógica similar a la que se utilizo para calcular el cambio de precio. Sin embargo, \nen este caso, solo necesitamos saber si el precio es más alto o más bajo en \ncomparación con el día anterior.')
        close_groups = price_data.groupby('Empresa')['Close']
        close_groups = close_groups.transform(lambda x : np.sign(x.diff()))
        price_data['Prediction'] = close_groups
        price_data.loc[price_data['Prediction'] == 0.0] = 1.0
        st.write(price_data.head(20))

        st.subheader('Construcción del modelo: Eliminación de valores de NaN')
        price_data = price_data.dropna()
        st.write(price_data.head(20))
        
        #División de los datos
        X_Cols = price_data[['RSI','k_percent','r_percent','Price_Rate_Of_Change','MACD','On Balance Volume']]
        Y_Cols = price_data['Prediction']
        X_train, X_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, random_state = 0)
        rand_frst_clf = RandomForestClassifier(n_estimators = 100, oob_score = True, criterion = "gini", random_state = 0)
        rand_frst_clf.fit(X_train, y_train)
        y_pred = rand_frst_clf.predict(X_test)
        
        st.subheader('Evaluación del modelo: Precisión')
        st.write('Correct Prediction (%): ', accuracy_score(y_test, rand_frst_clf.predict(X_test), normalize = True) * 100.0)

        st.subheader('Modelo de Evaluación: Informe de Clasificación')
        target_names = ['Down Day', 'Up Day']
        report = classification_report(y_true = y_test, y_pred = y_pred, target_names = target_names, output_dict = True)
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)
        
        st.subheader('Evaluación del modelo: matriz de confusión')
        rf_matrix = confusion_matrix(y_test, y_pred)
        true_negatives = rf_matrix[0][0]
        false_negatives = rf_matrix[1][0]
        true_positives = rf_matrix[1][1]
        false_positives = rf_matrix[0][1]
        accuracy = (true_negatives + true_positives) / (true_negatives + true_positives + false_negatives + false_positives)
        percision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)
        st.text('Accuracy: {}'.format(float(accuracy)))
        st.text('Percision: {}'.format(float(percision)))
        st.text('Recall: {}'.format(float(recall)))
        st.text('Specificity: {}'.format(float(specificity)))
        disp = plot_confusion_matrix(rand_frst_clf, X_test, y_test, display_labels = ['Down Day', 'Up Day'], normalize = 'true', cmap=plt.cm.Blues)
        disp.ax_.set_title('Matriz de confusión')
        st.pyplot()
        
        st.subheader('Evaluación del modelo: importancia de las características')
        feature_imp = pd.Series(rand_frst_clf.feature_importances_, index=X_Cols.columns).sort_values(ascending=False)
        st.write(feature_imp)
        
        st.subheader('Evaluación del modelo: representación gráfica de la importancia de las características')
        x_values = list(range(len(rand_frst_clf.feature_importances_)))
        cumulative_importances = np.cumsum(feature_imp.values)
        plt.plot(x_values, cumulative_importances, 'g-')
        plt.hlines(y = 0.95, xmin = 0, xmax = len(feature_imp), color = 'r', linestyles = 'dashed')
        plt.xticks(x_values, feature_imp.index, rotation = 'vertical')
        plt.xlabel('Variable')
        plt.ylabel('Importancia acumulada')
        plt.title('Random Forest: Gráfico de importancia de características')
        st.pyplot()

        st.subheader('Evaluación del modelo: Curva ROC')
        rfc_disp = plot_roc_curve(rand_frst_clf, X_test, y_test, alpha = 0.8)
        st.pyplot()
        
        st.subheader('Evaluación del modelo: puntaje de error fuera de la bolsa')
        st.write('Random Forest Out-Of-Bag Error Score: {}'.format(rand_frst_clf.oob_score_))
    
    elif string == 'SVR':
        df = pd.read_csv('EmpresasPorAnio.csv')
        
        #Hacemos preprocesamiento
        df.drop("Unnamed: 0",axis = 1, inplace = True)
        pd.unique(df["Empresa"])
        df_mask = df['Empresa']=='AAPL'
        filtered_df = df[df_mask]
        df = filtered_df.head(len(filtered_df)-16)
        
        st.subheader('Preprocesamiento de datos')
        st.write(df.head(20))
        
        # Creamos una lista para almancenar los datos independientes y dependientes
        days = list()
        adj_close_prices = list()
        
        # Obtenemos la fecha y precios de cierre ajustados
        df_days = df.loc[:, 'Date']
        df_adj_close = df.loc[:, 'Adj Close']
        tmp = df['Date'].str.split('-')
        df_days = (tmp.str[2]+tmp.str[0]+tmp.str[1]).astype(int)
        df_days = pd.to_datetime(df['Date']).dt.strftime('%m%d').astype(int)
        
        # Creamos el dataset independiente
        for day in df_days:
            days.append([day])

        # Creamos el dataset dependiente
        for adj_close_price in df_adj_close:
            adj_close_prices.append( float(adj_close_price) )
        
        st.subheader('Creación de tres modelos SVR')
        st.text('Para comprobar este modelo, se comparan tres modelos SVR creados usando:\n- Kerner Lineal\n- Kernel Polinomial\n- Kernel RBF')
        # Creamos 3 modelos SVR
        # Creamos y entrenamos un modelo SVR usando un kernel lineal
        lin_svr = SVR(kernel = 'linear', C=100)
        lin_svr.fit(days, adj_close_prices)
        # Creamos y entrenamos un modelo SVR usando un kernel polinomial
        pol_svr = SVR(kernel = 'poly', C=100, degree = 2)
        pol_svr.fit(days, adj_close_prices)
        # Creamos y entrenamos un modelo SVR usando un kernel rbf
        rbf_svr = SVR(kernel = 'rbf', C=100, gamma = 0.15)
        rbf_svr.fit(days, adj_close_prices)
        
        st.subheader('Grafica del mejor modelo')
        plt.figure(figsize=(16,8))
        plt.scatter(days, adj_close_prices, color='red', label='Data')
        plt.plot(days, rbf_svr.predict(days), color='green', label='Modelo RBF')
        plt.plot(days, pol_svr.predict(days), color='orange', label='Modelo Polinomial')
        plt.plot(days, lin_svr.predict(days), color='blue', label='Modelo Lineal')
        plt.legend()
        st.pyplot()
        
        st.subheader('Mejor precio predecido para el dato dado (1222)')
        daytest = [[1222]]
        st.text('El modelo SVR RBF predijo: {}'.format(rbf_svr.predict(daytest)))
        st.text('El modelo SVR Lineal predijo: {}'.format(lin_svr.predict(daytest)))
        st.text('El modelo SVR Polinomial predijo: {}'.format(pol_svr.predict(daytest)))
        st.text('Siendo el precio real: {}'.format(df['Adj Close'][235]))
        st.text('Concluyendo que el mejor modelo en este caso es RBF')
        
        st.subheader('Metricas de evaluación de SVR con modelo BRF')
        X = days
        y = adj_close_prices
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, test_size=0.30)
        r2_test = r2_score(y_test,rbf_svr.predict(X_test))
        r2_train = r2_score(y_train,rbf_svr.predict(X_train))
        MSE = mean_squared_error(y_test,rbf_svr.predict(X_test))
        MAE = mean_absolute_error(y_test,rbf_svr.predict(X_test))
        st.text("r2_score of train model: {}".format(r2_train))
        st.text("r2_score of test model: {}".format(r2_test))
        st.text("mean absolute error of train model: {}".format(MSE))
        st.text("mean squared error of train model: {}".format(MAE))
        
    elif string == 'LSTM':
        # Preprocesamiento de datos
        st.subheader('Preprocesamiento de datos')
        st.text('Para probar este modelo, se trabajara la columna "Close".')
        df = pd.read_csv('EmpresasPorAnio.csv')
        df1 = df.reset_index()['Close']
        st.write(df1.head(20))
        
        scaler = MinMaxScaler(feature_range=(0,1))
        df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
        
        training_size = int(len(df1)*0.65)
        test_size = len(df1) - training_size
        train_data,test_data = df1[0:training_size,:], df1[training_size:len(df1),:1]
        
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-time_step-1):
                a = dataset[i:(i+time_step), 0] 
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)
        
        time_step = 100
        X_train,y_train = create_dataset(train_data,time_step)
        X_test,y_test = create_dataset(test_data,time_step)
        
        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
        
        model = keras.Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
        
        model.add(LSTM(50,return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        
        model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy', km.binary_precision(), km.binary_recall()])
        model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=64,verbose=1)
        
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        train_predict = scaler.inverse_transform(train_predict)
        
        st.subheader('Grafico de Predicciones')
        look_back=100
        trainPredictPlot = np.empty_like(df1)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        testPredictPlot = np.empty_like(df1)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
        plt.plot(scaler.inverse_transform(df1))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        st.pyplot()
        
def main():
    st.title('Demo - Modelamientos del Grupo 2')
    st.text("En esta demo se aplicara todo lo aprendido en el curso de Inteligencia de Negocios \nmediante la aplicación de los modelos estudiados en clase hacia un determinado \ndataset.")
    
    st.subheader('Dataset a analizar')
    st.text('El dataset posee los valores de empresas registradas en Yahoo Finance, con sus \nrespectivas variaciones en cada fecha y volumenes.')
    st.write(price_data.head())
    
    option = ['Random Forest', 'SVR', 'LSTM']
    st.subheader('Elección del modelo a visualizar')
    model = st.selectbox('¿Que modelo le gustaria visualizar?', option)
    
    model_view(model)
    

if __name__ == '__main__':
    main()