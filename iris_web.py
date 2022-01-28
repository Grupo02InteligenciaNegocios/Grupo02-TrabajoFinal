# Importaciones
import streamlit as st
import pickle
import pandas as pd

# Extraer los archivos pickle
with open('lin_reg.pkl', 'rb') as li:
    lin_reg = pickle.load(li)
    
with open('log_reg.pkl', 'rb') as lo:
    log_reg = pickle.load(lo)
    
with open('svc_m.pkl', 'rb') as sv:
    svc_m = pickle.load(sv)

# Funcion que clasifique plantas
def classify(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolor'
    else:
        return 'Virginica'
    
def main():
    # Titulo
    st.title('Modelamiento de Iris por Sebas')
    
    # Titulo del Sidebar
    st.sidebar.header('User Input Parameters')
    
    # Funcion para poner los parametros en el sidebar
    def user_input_parameters():
        sepal_lenght = st.sidebar.slider('Sepal Lenght', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
        petal_lenght = st.sidebar.slider('Petal Lenght', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
        data = {
            'sepal_lenght': sepal_lenght,
            'sepal_width': sepal_width,
            'petal_lenght': petal_lenght,
            'petal_width': petal_width,
        }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()

    # Escoger el modelo preferido    
    option = ['Linear Regression', 'Logistic_Regression', 'SVM']
    model = st.sidebar.selectbox('Which Model You Like To Use?', option)
    
    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)
    
    if st.button('RUN'):
        if model == 'LinearRegression':
            st.success(classify(lin_reg.predict(df)))
        elif model == 'LogisticRegression':
            st.success(classify(log_reg.predict(df)))
        else:
            st.success(classify(svc_m.predict(df)))

if __name__ == '__main__':
    main()