import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from streamlit_option_menu import option_menu
from plotly import graph_objs as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Forecast", page_icon="image/ico.png", layout="wide")
st.sidebar.image("image/dash.png")
with st.sidebar.image("image/dash.png"):
        selected = option_menu(None,['Beranda', 'Data Aktual', 'Peramalan'], 
                    icons=['house', 'file-earmark-bar-graph', 'graph-up-arrow'])

if (selected == 'Beranda') :
                col1, col2 =st.columns(2)

                with col1 :
                    st.title('Stock Forecasting')
                    st.subheader("Stock Forecasting adalah Aplikasi berbasis Website untuk Peramalan Harga Saham menggunakan metode Holt Winters model Additive")
                with col2 :
                    st.image("image/tes.png", width=500)
                        
if (selected == 'Data Aktual') :
                st.title('Data Aktual')

                # Set the tic
                kode = ("PRDA.JK",)
                selected_ticker =st.selectbox("Kode Saham", kode)

                def load_data(ticker):
                    data = yf.download(ticker)
                    data.reset_index(inplace=True)
                    return data
                data = load_data(selected_ticker)

                #visualisasi data
                def plot_data():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'],  name='Close'))
                    fig.layout.update(title_text="Harga Penutupan Saham Prodia(PRDA.JK)", xaxis_rangeslider_visible=True)
                    fig.update_xaxes(title_text="Tanggal")
                    fig.update_yaxes(title_text="Harga")
                    st.plotly_chart(fig)

                col1, col2 = st.tabs(["Tabel","Grafik"])
                with col1 :
                       st.write(data)
                with col2 :
                       plot_data()
                
if (selected == 'Peramalan') :
                st.title('Peramalan')

                stock = ("PRDA.JK",)
                selected_ticker =st.selectbox("Kode Saham", stock)

                def load_data(ticker):
                    data = yf.download(ticker)
                    data.reset_index(inplace=True)
                    return data
                data = load_data(selected_ticker)

                data = data.drop(['Open','High','Low','Adj Close','Volume'], axis=1)
                train = data[:-365]
                test = data[-365:]
                
                jumlah_hari = st.slider("Pilih jumlah hari peramalan", 1,7, step=1)

                import warnings
                warnings.filterwarnings("ignore")

                #additive
                model = ExponentialSmoothing(data.Close, seasonal_periods=240, trend='add', seasonal='add').fit()
                

                predict_model = model.predict(start=train.Close.shape[0],end=(train.Close.shape[0]+test.Close.shape[0]-1))


            
                def mape(test, predict_model):
                    test.Close, predict_model = np.array(test.Close), np.array(predict_model)
                    mape = np.mean(np.abs((test.Close - predict_model) / test.Close)* 100)
                    return mape


                cast = model.forecast(jumlah_hari)
                cast = pd.DataFrame(cast, columns=['Peramalan'])


                if st.button('Proses'):

                    col1, col2 = st.tabs(["Tabel","Grafik"])
                    with col1:
                        st.subheader("Hasil Peramalan")
                        st.dataframe(cast)
                        st.write("MAPE = ", mape(test, predict_model), "%")
                    with col2:
                        st.subheader("Grafik Hasil Peramalan", )
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=cast['Peramalan'],  name='Peramalan'))
                        fig.layout.update(xaxis_rangeslider_visible=True)
                        fig.update_xaxes(title_text="Hari ke-")
                        fig.update_yaxes(title_text="Harga")
                        st.plotly_chart(fig)
                                       