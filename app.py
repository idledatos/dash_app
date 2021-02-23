import datetime
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
from dash.dependencies import Input, Output
import plotly.express as px
import base64
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import string
import re
from PIL import Image
import spacy
import numpy as np
from io import BytesIO

# Para HerokuSQL
import os
import psycopg2
import warnings
warnings.filterwarnings('ignore')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

colors = {
    'background': '#add8e6',
    'text': '#ff681f'
}

test_png = 'idle-stocklogo.png' # replace with your own image
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
                    html.Div([
                            html.Div(style={'display':'inline-block','vertical-align': 'top','width':'50%'},children = [
                                    html.Div(
                                        className="h-50",
                                        style={"height": "4vh",'width':'10%'}),
                                    html.Img(src='data:image/png;base64,{}'.format(test_base64),
                                        style={'float':'left','margin-left':'5%','height':'8%', 'width':'8%'})]),
                            html.H1(
                                    children='Bitcoin Sentiment Tracker    ',
                                    style={'display':'inline-block','vertical-align': 'top','width':'50%',}),
                            html.H5(
                                    children='Real Time Twitter Based Sentiment Analysis', 
                                    style={'height':'10%','margin-right':'2%'})],                                
                            style={
                                    'textAlign': 'right',
                                    'color': colors['text'],
                                    'backgroundColor':'#005b9f'}),
                    

                    html.Div([
                            html.Div(
                                    className="h-50"
                                    ,style={"height": "1vh",
                                            'backgroundColor':'#5993E5'}),
                            html.Div(
                                    className="h-50",
                                    style = {'display': 'inline-block','width': '2%', 'vertical-align': 'top','backgroundColor':'#5993E5'}),#,className = 'six columns'),

                            html.Div(
                                    dcc.Graph(id='live-update-graph'),
                                    style = {'display': 'inline-block','width': '70%','height': '110%', 'vertical-align': 'top'}),#,className = 'six columns'),

                            html.Div([
                                    # html.Div(
                                    #         className="h-50"
                                    #         ,style={"height": "18vh"}),
                                    html.H4(
                                            children='Trendy tweet in the last ten minutes',
                                            style={
                                                    'textAlign': 'center',
                                                    'color': '#000000',
                                                    "float": "right",
                                                    'backgroundColor':'#FFFFFF',
                                                    'margin-left':'10%',
                                                    'margin-right':'10%'}),
                                    html.Div(
                                            id = 'trendy_tweet',
                                            style={
                                                    'textAlign': 'center',
                                                    "float": "right",
                                                    'color': '#0000000',
                                                    'backgroundColor':'#ff681f',
                                                    'margin-left':'5%',
                                                    'margin-right':'5%'}),
                                    html.H5(
                                            children='RT count',
                                            style={
                                                    'textAlign': 'center',
                                                    'color': '#000000'}),
                                    html.Div(
                                            id = 'count_tweet',
                                            style={
                                                    'textAlign': 'center',
                                                    'color': '#0000000',
                                                    'backgroundColor':'#ff681f',
                                                    'margin-left':'30%',
                                                    'margin-right':'30%'}),
                                    html.Div(
                                            className="h-50"
                                            ,style={"height": "1vh"}),
                                    html.Img(id="image_wc",style = {'height':'50%', 'width':'50%'}),
                                    # html.Div(
                                    #         className="h-50"
                                    #         ,style={"height": "13vh",
                                    #                 'backgroundColor':'#5993E5'})],
                            ],style={
                                   'textAlign': 'center',
                                    'backgroundColor':'#5993E5',
                                    'color': '#000000',
                                    'width': '28%',
                                    "float": "right",
                                    'display': 'inline-block',
                                    'vertical-align': 'top'}),
                            dcc.Interval(
                                    id='interval-component',
                                    interval=60*1000, # in milliseconds
                                    n_intervals=0
                                ),
                            html.Div(
                                    className="h-50"
                                    ,style={"height": "2vh",
                                            'backgroundColor':'#5993E5'}),
                            
                            html.Div([
                                    html.Div(
                                            children = 'Powered by IDLE',className="h-50",
                                            style={
                                                    'margin-left':'90%'}),
                            ],style={
                                     'backgroundColor':'#005b9f',
                                     'color': '#FFFFFF'})
                    ])
             ])

circle_mask = np.array(Image.open("blue.jpg"))
def generate_wordcloud(data):
    # Creating a wordcloud
        wc = WordCloud(width=400, height=330,max_words=200,background_color='#5993E5',mask=circle_mask,colormap="Pastel1").generate_from_frequencies(data)
        
        # Setting Figure Size
        plt.figure(figsize=(14,12))
        
        # Display wordcloud as image
        plt.imshow(wc, interpolation='bilinear')
        
        # Removing all the axes
        plt.axis("off")
        
        return wc.to_image()
nlp=spacy.load('en_core_web_sm')
def clean_text(x):
        x = re.sub("[^a-zA-Z' ]", "", x)
        x = ' '.join([word for word in x.split() if word not in list(string.punctuation)])
        for word in [' btc ',' bitcoin ',' BTC ',' BITCOIN ']:
            x = re.sub(word, "", x)
        #x = ' '.join([word for word in x.split() if word not in ])
        x = x.lower()
        doc=nlp(x)
        new_tokens=[token.text for token in doc if (token.is_stop == False)]
        new_tokens = [x for x in new_tokens if (x!='rt')]
        return ' '.join(new_tokens)


# Multiple components can update everytime interval gets fired.
@app.callback([Output('live-update-graph', 'figure'),
               Output('trendy_tweet', 'children'),
               Output('count_tweet', 'children'),
               Output('image_wc', 'src')],
              Input('interval-component', 'n_intervals')
               )

def update_graph_live(y):
    #data_sent = pd.read_csv('Data_sent.csv', error_bad_lines=False)
    database_url = os.getenv(
    'DATABASE_URL',
    default='postgres://ouhieniktislqg:670cd50a97c032fd1f58880c7b4ad96893c94e7b40706e850a03630a5a7ba8ca@ec2-3-221-49-44.compute-1.amazonaws.com:5432/d7l11l3desflh0',  # E.g., for local dev
    )

    conn = psycopg2.connect(database_url)
    data_sent = pd.read_sql("""
            SELECT *
            FROM twitter
            """, con = conn)
    
    #cursor = connection.cursor()
    data_sent.columns = ['User' if x == 'Us' else x for x in data_sent.columns]
    data_sent['Date'] = pd.to_datetime(data_sent.Date)
    print(data_sent['Date'].max(),data_sent['Date'].min())
    data_sent['Fecha'] = data_sent.Date.apply(lambda x: x.strftime(format = "%Y-%m-%d %H"))
    tweets_data = pd.DataFrame()
    for fecha in set(data_sent.Fecha):
            kpi_pos = data_sent.loc[data_sent.Fecha == fecha, 'Pos'].max()
            kpi_neg = data_sent.loc[data_sent.Fecha == fecha, 'Neg'].min()
            text_pos = data_sent.loc[(data_sent.Fecha == fecha)&(data_sent.Pos == kpi_pos),'Text'].reset_index().Text[0]
            text_neg = data_sent.loc[(data_sent.Fecha == fecha)&(data_sent.Neg == kpi_neg),'Text'].reset_index().Text[0]
            user_pos = data_sent.loc[(data_sent.Fecha == fecha)&(data_sent.Pos == kpi_pos),'User'].reset_index().User[0]
            user_neg = data_sent.loc[(data_sent.Fecha == fecha)&(data_sent.Neg == kpi_neg),'User'].reset_index().User[0]
            tweets_data = tweets_data.append([(fecha,'Neg',text_neg,user_neg)])
            tweets_data = tweets_data.append([(fecha,'Pos',text_pos,user_pos)])
    tweets_data.columns = ['Fecha','Sent','Text','User']
    tweets_data.Fecha = pd.to_datetime(tweets_data.Fecha)
    data_sent_melt = data_sent[['Fecha','Pos','Neg','Neu']].melt(id_vars = 'Fecha',var_name = 'Sent',value_name = 'Val')
    data_plot = data_sent_melt[data_sent_melt.Sent != 'Neu']
    data_plot_2 = data_plot.groupby(['Fecha','Sent']).sum().reset_index()
    data_plot_2['Fecha'] = pd.to_datetime(data_plot_2.Fecha)
    data_plot_2 = data_plot_2.merge(tweets_data,on = ['Fecha','Sent'])
      
    data_plot_2.loc[(data_plot_2.Sent == 'Pos')&(data_plot_2.Val == 0),'Text'] = '-'
    data_plot_2.loc[(data_plot_2.Sent == 'Neg')&(data_plot_2.Val == 0),'Text'] = '-'
    data_plot_2.loc[(data_plot_2.Sent == 'Pos')&(data_plot_2.Val == 0),'User'] = ''
    data_plot_2.loc[(data_plot_2.Sent == 'Neg')&(data_plot_2.Val == 0),'User'] = ''
    # Create the graph with subplots
    from datetime import timedelta

    data_plot_2 = data_plot_2[data_plot_2.Fecha > data_plot_2.iloc[-1,:].Fecha-timedelta(minutes=40)]
    
    fig = px.line(data_plot_2.iloc[:,:], x="Fecha", y="Val", color="Sent",color_discrete_sequence=["red","blue"],
             line_group="Sent",hover_name="User", hover_data = ["Text"])
    fig.update_traces(mode="markers+lines")#, hovertemplate=None)
    fig.update_layout(hovermode="closest",plot_bgcolor= '#4aa8c8')
    fig.update_layout(paper_bgcolor="#add8e6")
    #fig.update_layout( autosize=False,width=1000,height=500,margin=dict(l=50,r=50,b=100,t=100,pad=4))
    fig.update_yaxes(title = '',showticklabels = False,showgrid = False)
    fig.update_xaxes(title = '',showgrid = False)
    fig.update_layout(
        hoverlabel=dict(
        bgcolor="#4aa8c8",
        font_size=10,
        font_family="Courier New",
        font_color = 'white'
        ),
        plot_bgcolor='rgba(0,0,0,0)')

    data_sent['Contador'] = 1
    data_prueba = data_sent[data_sent.Date > data_sent.iloc[-1,:].Date-timedelta(minutes=10)]
    value = data_prueba.groupby('Text').sum().reset_index().sort_values(by='Contador',ascending = False).reset_index()['Contador'][0]
    children = data_prueba.groupby('Text').sum().reset_index().sort_values(by='Contador',ascending = False).reset_index()['Text'][0]

    
    data_prueba_2 = data_sent[data_sent.Date > data_sent.iloc[-1,:].Date-timedelta(minutes=5)]
    data_prueba_2 = data_prueba_2[['Text']]
    data_prueba_2['Clean'] = data_prueba_2.Text.apply(lambda x: clean_text(x))
    
    values = ' '.join(str(v) for v in data_prueba_2['Clean'])
    word_tokens = word_tokenize(values)
    word_tokens = [w.strip(' ') for w in word_tokens]
    word_tokens = [x for x in word_tokens if x != 'bitcoin']
    counts = Counter(word_tokens)
    

    img = BytesIO()
    generate_wordcloud(data=counts).save(img, format='PNG')
    word_cloud = 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


    return fig,children,value,word_cloud

if __name__ == '__main__':
    app.run_server(debug=True)