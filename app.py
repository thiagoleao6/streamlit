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


    st.write("Naylson Araújo RM 350294")
    st.write("Sarah Fernandes RM 349429")
    st.write("Thiago Leão RM 349791")
    
    st.header("Filtro de Data")
# Obter as datas de entrada do usuário no site
    data_inicio = st.date_input("Digite a data inicial:", min_value=pd.to_datetime("1987-01-01"), max_value=pd.to_datetime("2024-12-31"))
    data_final = st.date_input("Digite a data final:", min_value=data_inicio, max_value=pd.to_datetime("2024-12-31"))
    data_inicio_np = np.datetime64(data_inicio)
    data_final_np = np.datetime64(data_final)

#Titulo do site
st.header("Analise Preditiva Flutuação Preço do Petróleo")

#Separando as paginas
tab0,tab1,tab2,tab3,tab4, tab5 = st.tabs(["Inicio","Histórico e Futuro", "Politicas Globais, Nacionais e dolar", "Analise Final","Datasets e fontes", "Deploy"])



#------------------------------------------------Introdução------------------------------------------------------------------
with tab0:
    pontos = """ 
                - Criar um dashboard interativo
                - Trazer insights sobre a variação do preço do do pretroleo
                - Criar um modelo de Machine learning que faça a previsão diariamente
                - Fazer o deploy do modelo
                
"""
    st.header("Tech Challenger 4 👍")
    st.subheader("Pontos abordado")
    st.write(pontos)
 #-----------------------------------------------#Primeira Pagina-------------------------------------------------------------   
with tab1:
    #TItulo da pagina
    #st.title('Evolução Histórica e Predição')

    st.subheader("Cenário Global")

    st.write("A análise publicada neste perfil, tem como público-alvo investidores do mercado de petróleo, sejam eles iniciantes ou experientes, portanto algumas recomendações e explicações são necessárias para um bom entendimento do estudo e modelo preditivo.")
    st.write("Para analisar o preço do petróleo, é essencial compreender que existem duas cotações para o barril: sendo o WTI, referência para o mercado americano, cotado na bolsa de Nova Iorque, e o Brent, principal referência para o mercado europeu, listado na bolsa de Londres. ")
    st.write("O fator importante a ser considerado é que o mercado internacional de petróleos é influenciado por 3 grandes grupos sendo eles:")
    st.write("**OPEP (Organização dos Países Exportadores de Petróleo):** Composta por países exportadores de petróleo, a OPEP busca coordenar as políticas de produção de petróleo para estabilizar os preços e garantir uma receita estável para os países membros.")
    st.write("**Rússia (em colaboração com outros produtores não pertencentes à OPEP):** A Rússia, não é membra da OPEP,  mas por diversas vezes colabora com o grupo (conhecido como OPEP+). Juntos, esses países coordenam a produção para influenciar os preços globais do petróleo.")
    st.write("**Estados Unidos:** Como maior produtor de petróleo não pertencente à OPEP, os Estados Unidos têm um papel significativo na administração do mercado global de petróleo. A produção de petróleo de xisto nos EUA, em particular, impactou a dinâmica global do setor.")
    st.write("Cabe ressaltar que a China, não está diretamente envolvida na administração global, mas pode influenciar o mercado através de suas políticas e importações. Outros países e regiões, como Canadá, Brasil e países do Golfo Pérsico, também desempenham papéis importantes na produção e exportação de petróleo.")
    st.write("Outros países e regiões, como Canadá, Brasil, Noruega e países do Golfo Pérsico, também têm relevância na produção e exportação de petróleo, embora não estejam necessariamente organizados em blocos formais de gestão como a OPEP ou a OPEP+.")
    st.write("Além disso, existem instituições internacionais, como a Agência Internacional de Energia (AIE), que desempenham papel na monitorização e análise do mercado global de energia, fornecendo informações e recomendações, embora não tenham um papel direto na administração dos preços do petróleo.")
    st.write("A análise a seguir se concentra na evolução dos valores do Brent.")
   
    

    #importando a base da pasta
    df_tabela = pd.read_csv("dados.csv")
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
    st.write("**Para avaliar a evolução dos anos selecione o período na lateral esquerda da tela.**")
    # Criar o gráfico com as datas filtradas
    fig = px.line(df_filtrado, x='Data', y='Preco Petroleo', title='Gráfico de Linhas do Preço do Petróleo')

    # Personalizar o layout do gráfico
    fig.update_xaxes(title_text='Data')
    fig.update_yaxes(title_text='Preço do Petróleo')
    fig.update_xaxes(tickangle=45)
    #plotando o grafico no site
    st.plotly_chart(fig)

    
    st.write("Ao analisar a variação nas últimas décadas, observamos flutuações significativas, impactando a composição do valor do petróleo")

    #2° Subtitulo
    st.subheader("Os principais eventos que influenciaram as oscilações nos preços do petróleo incluem:")

    texto = """
                ➡️ **Guerra Irã-Iraque (1980-1988):** Conflito causou interrupções na produção/exportação, gerando volatilidade nos preços;
                
                ➡️ **Invasão do Kuwait pela Iraque (1990):** Guerra do Golfo interrompeu produção no Oriente Médio, impactando os preços;

                ➡️ **Crise Financeira Asiática (1997-1998):** Redução na demanda global por petróleo levou a uma queda nos preços;

                ➡️ **Subida do preço do petróleo (1999-2008):** Aumento da demanda global (destaque para China e Índia), conflitos e ameaças de fechamento do Estreito de Ormuz (Irã), produção limitada levaram a um significativo aumento nos preços;

                ➡️ **Excesso de Oferta (2014):** Produção no xisto dos EUA e decisões da OPEP levaram a uma queda nos preços;
                
                ➡️ **Tensões EUA-China (2019):** Disputas comerciais contribuíram para incertezas globais nos mercados de petróleo;

                ➡️ **Pandemia da Covid-19 (2020):** Restrições de circulação levaram a uma queda acentuada nos preços, registrando o menor valor desde 2004.
                  """
    


    ##Modelo
    st.write(texto)

    st.subheader("Análise do modelo")

    st.write("Os eventos supracitados refletem, tanto fatores de oferta quanto demanda, além de questões geopolíticas e econômicas. ")
    
    st.write("Desta forma modelos preditivos precisam ser atualizados com frequência, juntamente com uma análise do cenário político global.")

    st.subheader("GRAFICO PREDIÇÃO")

    st.write("O modelo escolhido para a predição foi o GradientBoostingRegressor, por ter um Erro Quadrado Médio(MSE) de 2.87 e o Erro Médio absoluto(MAE) de 1.18")



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
    caminho_modelo = "modelo.joblib"
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


    st.write("Observando o mercado e sites específicos na temática, conseguimos comprovar a aderência do modelo escolhido, no qual acompanha a análise do mercado que prevê, média de 70 a 90 dólares por Barril.")
    st.write("A seguir previsões futuras mencionadas:")
    st.write("Previsões para 2024. Estima-se que os preços continuem a descer dentro de intervalo de trading de 70 a 90 doláres. As tendências no mercado sugerem um equilíbrio entre as restrições de extração de petróleo propostas pela OPEP+ e a demanda estável de energias primarias.")
    st.write("Previsões de longo prazo (2025-2030). Será mantida a incerteza no mercado após de 2025. Com a transição para energias renováveis e veículos elétricos, a dinâmica de preços do USCrude, provavelmente, se alterará. O petróleo será negociado em intervalo de 60 até 100 dólares. Leia mais em: (https://www.litefinance.org/pt/blog/analysts-opinions/a-previsao-de-precos-do-petroleo/#:~:text=Previs%C3%A3o%20para%202023.,uma%20estabiliza%C3%A7%C3%A3o%20gradual%20do%20mercado.)")





#_______________________________________________2°Pagina_______________________________________________________________________
with tab2:

#1° subtitulo
    
    st.write("Aos investidores cabe ressaltar ainda mudanças globais, novos pactos  a exemplo do ESG, agendas globais como 2030,  impactam diretamente nas estratégicas de produção do petróleo, demanda x Oferta")
    st.write("Daremos destaque ao ESG (Ambiental, Social e Governança) que no setor de petróleo tem implicações significativas para o surgimento de novas opções em sua substituição, tais como:")

    texto2 = """
                ➡️**Inovação em Energias Renováveis:** O aumento do ESG tem impulsionado investimentos em P&D, com pesquisa e desenvolvimento de tecnologias de energias renováveis, como solar, eólica e hidrogênio verde, como alternativas mais sustentáveis;

                ➡️**Transição para Veículos Elétricos:** Empresas automotivas estão intensificando seus esforços para produzir e aumentar a comercializar veículos elétricos (VEs), buscando reduzir a dependência de combustíveis fósseis; 

                ➡️**Bioenergia e Biocombustíveis:** A ênfase em práticas agrícolas sustentáveis e a produção de biocombustíveis estão ganhando destaque como opções mais ambientalmente amigáveis para substituir os combustíveis tradicionais derivados do petróleo, ou com composição verde em sua fórmula como o Biodiesel, com crescido globalmente devido às preocupações com a sustentabilidade e redução das emissões de gases de efeito estufa.

                ➡️**Eficiência Energética:** O ESG está incentivando a melhoria da eficiência energética em diversos setores, promovendo a adoção de tecnologias e processos que reduzem a demanda por petróleo;

                ➡️**Pressão por Alternativas Menos Poluentes:** O foco no ESG está gerando pressão dos investidores, consumidores e reguladores para que as empresas busquem alternativas menos poluentes, acelerando a busca por soluções inovadoras.                      
                                     """
    
    st.write(texto2)

    st.write("Em resumo, a atenção crescente ao ESG no setor de petróleo está impulsionando a diversificação e inovação em fontes de energia, promovendo o surgimento de novas opções mais sustentáveis e contribuindo para a transição para uma matriz energética mais limpa.")
    
    st.subheader("Derivados do Petróleo")
    st.write("Os subprodutos do petróleo incluem uma ampla variedade de derivados que são obtidos durante o processo de refino. Alguns dos principais subprodutos do petróleo são:")

    texto3 = """
                ➡️ Gasolina: Utilizada como combustível para veículos.

                ➡️ Diesel: Usado em motores a diesel para veículos e maquinaria.

                ➡️ Querosene: Utilizado em aviação e para produção de combustíveis domésticos.

                ➡️ Óleo Combustível: Usado em indústrias para geração de energia térmica.

                ➡️ Gás de Cozinha (GLP): Utilizado como combustível doméstico.

                ➡️ Asfalto: Utilizado na construção de estradas e pavimentação.

                ➡️ Lubrificantes: Óleos lubrificantes para motores e maquinaria.

                ➡️ Petróleo Cru Residual: Resíduo do processo de refino, usado para produção de outros produtos.

                ➡️ Nafta: Matéria-prima para a indústria petroquímica.

                ➡️ Produtos Petroquímicos: Incluem polímeros, plásticos e produtos químicos.

                    """

    st.write(texto3)

    st.write("Esses subprodutos são essenciais para uma variedade de setores industriais e domésticos, sendo utilizados em várias aplicações cotidianas, demonstração clara aos investidores que o petróleo é hoje um produto essencial para diversos setores, e são impactos permeiam por uma cadeia produtiva diversificada.")

    st.subheader("Impacto do Dólar nos Preços no Brent")

    st.write("Conforme mencionado na introdução deste estudo, o mercado do petróleo é composto por 3 principais entidades, sendo somente uma americana.")

    st.write("Portanto o valor do dólar pode impactar nos preços dos combustíveis, pois muitos países negociam petróleo e derivados em dólares americanos. Um dólar mais fraco em relação a outras moedas pode levar a aumentos nos preços dos combustíveis, enquanto um dólar mais forte pode ter o efeito oposto, isto se aplica aos mercados locais.")

    st.write("No entanto no cenário global, outros fatores, como demanda e oferta de petróleo, instabilidade geopolítica e decisões de produção da OPEP, desempenham papéis importantes nas flutuações dos preços dos combustíveis. Em resumo, conforme demostrado nos gráficos abaixo o valor do dólar é apenas um dos vários fatores que influenciam os preços dos combustíveis, com a oferta e demanda sendo determinantes fundamentais.")

    #Gráfico

    #importando a base da pasta
    df_cambio = pd.read_csv("dados_cambio.csv",sep=",")

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

    st.write("Portanto é nítido observar que o valor do dólar não tem influência global nos preços do petróleo, comparados com a cotação da moeda no cenário nacional, que possui uma moeda fraca frente ao câmbio.")

    
    st.subheader("Fatores Nacionais")
    
    st.write("No Brasil, a política de preços dos combustíveis, especialmente gasolina e diesel, passou de seguir a paridade internacional, com intervenções governamentais diretas desde 2016. Essa abordagem considera variações internacionais e custos de importação, visando controlar a inflação e proteger os consumidores de flutuações abruptas nos preços, que na prática tem o valor impactado pelos tributos e impostos, outras influenciadas na composição dos preços são eventos políticos e econômicos.")
    st.write("O petróleo é negociado na bolsa de valores em forma de contratos no mercado futuro. Estes contratos possuem vencimentos que apresentam grande volatilidade. Por isso são oportunidades muito exploradas por traders. Cada contrato futuro de petróleo é composto por 100 barris. Como o preço do barril no Brasil é negociado em Dólares Americanos, o valor de um contrato futuro de petróleo depende de duas variáveis: o preço do barril e o preço do Dólar.")
    st.write("No cenário internacional como exemplo do site Oil Price, o petróleo do Brasil, recebe o nome Lula, conforme demonstrado abaixo:")

    st.image('Imagem1.png')

    st.write("O gráfico abaixo, reforça que a politica adotada pelo Brasil na composição do valor do petróleo, faz com que os valores sejam mais elevados que os praticados no cenário global, os quais ainda será acrescido os tributos")
    st.image('Imagem2.png',width=500)

    st.write("Os valores para o ano de 2024, estão em alta acompanhando a inflação e aumento dos gastos públicos. Os valores para o ano de 2024, estão em alta acompanhando a inflação e aumento dos gastos públicos.")
    st.image('Imagem3.png')

    st.write("A EPE (empresa de pesquisa energética), publicou um estudo do plano decenal de Expansão de Energia 2032, frente aos preços internacionais de derivados de petróleo")
    st.image('Imagem4.png')

    st.write("Desta forma embora na próxima década ocorra uma pequena queda nos valores dos combustíveis, ainda assim a previsão que o valor permaneça em alta, corroborando com a previsão realizada.")

    st.write("O Brasil, é um grande produtor de petróleo, também realiza importações de petróleo e derivados por diversas razões, como atender a demanda interna específica e garantir uma variedade de produtos refinados. O país exporta e importa petróleo e derivados, mas é geralmente considerado um exportador líquido, exportando mais do que importando. Conforme demonstração abaixo:")
    st.image('Imagem5.jpg')

    st.image('Imagem6.png')

    
#________________________________________3°Pagina_______________________________________________________________________________
with tab3:
    
    st.title("Considerações:")

    st.write("A análise do modelo preditivo prevê uma elevação seguida de uma queda nos valores para o próximo ano acompanhando a tendencia vista em estudos complementares neste documento.")
    st.write("Porém cabe a ressalva que o modelo pode ter interferências diretas limitadas por fatores externos que podem alterar essa previsão. ")

    st.subheader('1.	Cenário Global e Incertezas:')
    st.write('➡️ Mudanças no cenário global, como eventos geopolíticos, guerras, crises econômicas, desastres naturais ou pandemias;')
    
    st.subheader("2.	Falta de Consideração de Todas as Variáveis Mundiais:")
    st.write("➡️ Modelos preditivos geralmente simplificam a realidade, e nem todas as variáveis globais podem ser incluídas. Desta forma fatores complexos, como mudanças na política global, acordos comerciais, avanços tecnológicos ou surtos de doenças, podem influenciar as projeções de maneiras imprevisíveis.")

    st.subheader('3.	Necessidade de Monitoramento Contínuo:')
    st.write("➡️ A análise deve ser dinâmica e sujeita a atualizações conforme novas informações e eventos ocorrem.")


    st.subheader("Cenário Mundial")
    st.write("No cenário mundial, pode ocorrer restrições de produção em alguns países e a dificuldade em aumentar rapidamente a produção para atender à crescente demanda. Países resolverem fazer altas reservas buscando segurança em cenários hostis como guerras.")
    st.write("Outra questão a ser pontuada é a especulação financeira nos mercados de commodities, com investidores financeiros buscando lucros em meio à alta volatilidade, pode contribuir para picos de preços do petróleo.")
    st.write("Conclui-se que, embora o modelo preditivo forneça uma base para previsões, é crucial reconhecer as limitações e a volatilidade inerente ao ambiente global. Podendo ser úteis, mas a capacidade de adaptar e ajustar as previsões com base em mudanças inesperadas é essencial para uma análise mais robusta.")
    st.write("Como vimos a cotação do barril de Brent é calculada em função da produção mundial, mas também da procura global que depende essencialmente da política energética planetária, motivo pelo qual limitamos a previsão de 100 dias. Sendo necessário revisar o modelo, acompanhando as oscilações e atualizações do mercado, desta forma atualizando a base e avaliando novos eventos, imprevistos que podem alterar as condições econômicas e afetar os resultados projetados.")


with tab4:
    st.subheader("Datasets")
    st.write("Variação do Petróleo")
    st.write("link")
    st.dataframe(df_tabela, hide_index=True)

    st.write("Variação do Cambio")
    st.write("link")
    st.dataframe(df_cambio, hide_index=True)

    st.subheader("Referências Bibliograficas")

    links = """
                https://www.eia.gov/opendata/

                https://br.investing.com/commodities/brent-oil-historical-data

                https://www.cnnbrasil.com.br/tudo-sobre/preco-do-petroleo/

                https://www.petromos.com/cotacao

                https://www.fazcomex.com.br/comex/gasolina-importacao-e-exportacao

                https://www.cnnbrasil.com.br/economia/petrobras-anuncia-fim-da-paridade-internacional-de-precos-do-petroleo/

                https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/precos

                https://oilprice.com/oil-price-charts/

                https://www.eia.gov/opendata/browser/petroleum/move

                https://www.gov.br/anp/pt-br

                https://br.investing.com/commodities/brent-oil-advanced-chart

                https://www.epe.gov.br/sites-pt/publicacoes-dados-abertos/publicacoes/PublicacoesArquivos/publicacao-689/topico-640/Caderno%20de%20Pre%C3%A7os%20de%20Derivados_PDE%202032.pdf

                                        """
    

    st.write(links)


with tab5:
    st.title("Documentação do Deploy")
    st.write("Este guia fornece instruções passo a passo para realizar o deploy do seu aplicativo Streamlit com um modelo de Machine Learning no GitHub.")

    st.subheader("1. Criar um repositório no Github")
    st.write("Acesse o Github e siga as instruções para criar um novo repositório. Certifique-se de escolher um nome significativo para o seu projeto.")
    st.write("https://github.com/")
    
    st.subheader("2. Subir os arquivos")

    st.write("Certifique-se de ter os seguintes arquivos no seu diretório local:")

    texto_de = """
                    .Modelo.joblib: Modelo de Machine Learning criado com o GradientBoostingRegressor.
                    
                    .gitignore: Arquivo de configuração para ignorar arquivos e diretórios específicos ao versionar no Git.

                    .app.py: Código-fonte do seu Dashboard Streamlit.

                    .dados.csv: Base de dados do preço do barril do Petróleo.
                    
                    .dados_cambio.csv: Base de dados do câmbio EUA X Brasil.
                    
                    .Logo-pos-tech.png: Imagem do logo da pós tech.

                    .requirements.txt: Lista das bibliotecas e suas versões utilizadas no projeto.

                    .Imagens dos graficos.
                                            """


    st.write(texto_de)

    st.subheader("3. Criar um aplicativo dentro do site do Streamlit")
    st.writer("Acesse o site do Streamlit e siga as instruções para criar uma conta. Após criar a conta, siga os passos para conectar sua conta do GitHub.")
    st.writer("Crie um novo aplicativo no Streamlit, vinculando-o ao seu repositório GitHub. Certifiquese de configurar corretamente o ambiente de execução e as variáveis necessárias para o seu aplicativo.")

    st.writer("https://streamlit.io/")


    


#________________________________________________________________________________________________________________________________________________________________________________
