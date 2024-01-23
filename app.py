import streamlit as st
import pandas as pd 
import plotly.graph_objs as go
import requests
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import numpy as np

st.set_page_config(page_title="Tech Challenger 4🔥")

with st.sidebar:
    st.image('logo-pos-tech.png')
    st.header("Tech Challenger 4")
    st.header("Grupo 49")

    st.write("Thiago Leão")
    st.write("Sarah Fernandes")
    st.write("Naylson")
    st.header("Filtro de Data")
# Obter as datas de entrada do usuário no site
    data_inicio = st.date_input("Digite a data inicial:", min_value=pd.to_datetime("1987-01-01"), max_value=pd.to_datetime("2024-12-31"))
    data_final = st.date_input("Digite a data final:", min_value=data_inicio, max_value=pd.to_datetime("2024-12-31"))
    data_inicio_np = np.datetime64(data_inicio)
    data_final_np = np.datetime64(data_final)

#Titulo do site
st.header("Dashboard Pós Tech")

#Separando as paginas
tab0,tab1,tab2,tab3,tabs4 = st.tabs(["Inicio","Analise 1", "Analise 2", "Datasets","Insights"])



#------------------------------------------------Introdução------------------------------------------------------------------
with tab0:
    pontos = """ 
                - Criar um dashboard a sua escolha
                - Trazer insights sobre a variação do preço do do pretroleo
                - Criar um modelo de Machine learning que faça a previsão diariamente
                - Fazer um deploy
                
"""
    st.header("Tech Challenger 4 👍")
    st.subheader("Pontos abordado")
    st.write(pontos)
 #-----------------------------------------------#Primeira Pagina-------------------------------------------------------------   
with tab1:
    #TItulo da pagina
    st.title('Análise preço do Petróleo')
    st.subheader("Preço por barril do petróleo bruto Brent")

    #importando a base da pasta
    df_tabela = pd.read_csv(dados.csv")
    # Obter as datas de entrada do usuário no site
    # data_inicio = st.date_input("Digite a data inicial:", min_value=pd.to_datetime("1987-01-01"), max_value=pd.to_datetime("2024-12-31"))
    # data_final = st.date_input("Digite a data final:", min_value=data_inicio, max_value=pd.to_datetime("2024-12-31"))

    # Garantir que a coluna "Data" seja do tipo datetime
    df_tabela["Data"] = pd.to_datetime(df_tabela["Data"])

    # # Converter as datas de entrada para datetime64[ns]
    # data_inicio_np = np.datetime64(data_inicio)
    # data_final_np = np.datetime64(data_final)

    # Filtre o DataFrame com base nas datas escolhidas
    df_filtrado = df_tabela[(df_tabela["Data"] > data_inicio_np) & (df_tabela["Data"] < data_final_np)]

    # Criar o gráfico com as datas filtradas
    fig = px.line(df_filtrado, x='Data', y='Preco Petroleo', title='Gráfico de Linhas do Preço do Petróleo')

    # Personalizar o layout do gráfico
    fig.update_xaxes(title_text='Data')
    fig.update_yaxes(title_text='Preço do Petróleo')
    fig.update_xaxes(tickangle=45)
    #plotando o grafico no site
    st.plotly_chart(fig)

    #2° Subtitulo
    st.subheader("Principais casos que influenciaram:")

    texto = """
                -Guerra Irã-Iraque (1980-1988): O conflito levou a interrupções na produção e exportação de petróleo na região, causando volatilidade nos preços.
                
                -Invasão do Kuwait pela Iraque (1990): A invasão resultou na Guerra do Golfo, levando a interrupções significativas na produção de petróleo no Oriente Médio.
                
                -Crise Financeira Asiática (1997-1998): A crise afetou a demanda global por petróleo, resultando em uma queda nos preços.
                
                -Subida do preço do petróleo (1999-2008): A demanda global crescente, juntamente com a produção limitada, levou a um aumento significativo nos preços do petróleo durante este período."""
    st.write(texto)
    #Criando lags para o modelo
    for lag in range(1,3):
        df_tabela[f"Preco_lag_{lag}"] = df_tabela["Preco Petroleo"].shift(lag)
        
    df_tabela.dropna(inplace = True)
    # Filtrando o input do modelo a partir de 2022
    df_novo = df_tabela[df_tabela["Data"]>='2022-01-01'].copy()
    # Dropando a data
    df_novo.drop("Data",axis =1, inplace = True)
    #Resetando o indice
    df_novo.reset_index(inplace = True, drop = True)
    #3° subtitulo
    st.subheader("Previsão")
    #Pegando o modelo
    caminho_modelo = r'C:\tech4\modelo\modelo.joblib'
    #Input de previsão
    qtd_previsao = int(st.slider("Quantos dias você quer prever: ",1,100))

    # Carregar o modelo
    model = joblib.load(caminho_modelo)

    #Fazendo a previsão
    previsao = model.predict(df_novo)

    # Exemplo de array de previsões
    previsao1 = previsao[qtd_previsao*-1:]

    # Criar eixo x (por exemplo, índices)
    eixo_x = np.arange(len(previsao1))

    #Criando o grafico da previsão
    fig1 = px.line(x=eixo_x, y=previsao1, labels={'x': 'Índice', 'y': 'Valor da Previsão'},
                title=(f'Gráfico de Linha das proximas {qtd_previsao} Previsões'))

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig1)
#_______________________________________________2°Pagina_______________________________________________________________________
with tab2:

#1° subtitulo
    
    # Converter as datas de entrada para datetime64[ns]

    st.subheader("Cambio R$ - U$$")


    #importando a base da pasta
    df_cambio = pd.read_csv(dados_cambio.csv",sep=",")

    # Garantir que a coluna "Data" seja do tipo datetime
    df_cambio["Data"] = pd.to_datetime(df_cambio["Data"])
    df_cambio.sort_values(by="Data",inplace=True,ascending=True)
    df_cambio.reset_index(inplace=True,drop=True)
    df_cambio = df_cambio[df_cambio["Data"]>'1995-01-01']
    # Filtre o DataFrame com base nas datas escolhidas
    df_cambio_filtrado = df_cambio[(df_cambio["Data"] > data_inicio_np) & (df_cambio["Data"] < data_final_np)]

    # Criar o gráfico de linhas
    fig_cambio = px.line(df_cambio_filtrado, x='Data', y='Cambio', title='Gráfico de Linhas do Cambio R$ - U$')

    # Personalizar o layout do gráfico
    fig_cambio.update_xaxes(title_text='Data')
    fig_cambio.update_yaxes(title_text='Preço do Cambio')
    fig_cambio.update_xaxes(tickangle=45)

    st.plotly_chart(fig_cambio)



    #Comparaçao com o 1° grafico

    # Criar o gráfico com as datas filtradas
    fig_p = px.line(df_filtrado, x='Data', y='Preco Petroleo', title='Gráfico de Linhas do Preço do Petróleo')

    # Personalizar o layout do gráfico
    fig_p.update_xaxes(title_text='Data')
    fig_p.update_yaxes(title_text='Preço do Petróleo')
    fig_p.update_xaxes(tickangle=45)
    #plotando o grafico no site
    st.plotly_chart(fig_p)

    st.header("Análise")

    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam commodo lectus id felis hendrerit, vitae euismod ex dignissim. Suspendisse potenti. Fusce ultricies ligula et enim scelerisque, non consectetur neque faucibus. Vestibulum commodo risus eget nibh varius, eu aliquet lacus bibendum. Praesent vel augue vel velit lacinia rhoncus a ut odio. Vivamus volutpat, sem et vulputate accumsan, nisi elit venenatis urna, non hendrer")



    
#___________________________________________3°Pagina_______________________________________________________________________________
with tab3:
    st.subheader("Dataset Petroleo")
    df_no_site = df_tabela[["Data","Preco Petroleo"]]
    df_no_site.sort_values(by="Data",inplace=True,ascending=False)
    st.dataframe(df_no_site,hide_index=True)


    st.subheader("Dataset Cambio")
    df_no_site2 = df_cambio[["Data","Cambio"]]
    df_cambio.sort_values(by="Data",inplace=True,ascending=False)
    st.dataframe(df_cambio,hide_index=True)
