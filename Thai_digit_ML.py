import base64
import datetime
import dash
import os
from dash import dcc, html,callback_context
from flask import Flask
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
# from aem import app
from PIL import Image
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

from predict import fn_predict
from train_model import fn_train_model



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#compage1
Header_component = html.H1("THAI NUMBER PREDICTION", style={'color': 'FloralWhite', 'text-align': 'Center', 'backgroundColor': 'DarkSlateGray', 'color': 'white','font-family': 'Roboto'})
# test_png = 'D:\Test/2.png'
# test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')


#compage2
df = pd.read_csv('dataset_version.csv')
row_count = df.shape[0]
count_df = df.iloc[:,:].groupby('y').size().reset_index(name='Count')
fig1 = go.Figure(data=go.Bar(
    x=count_df['y'],
    y=count_df['Count'],
    marker_color='rgb(26, 118, 255)',
    text=count_df['Count'],
    textposition='auto',
))
fig1.update_layout(
    xaxis=dict(
        tickmode='linear',
        dtick=1,
        tickfont=dict(size=12)),
    xaxis_title='Label',
    yaxis_title='Numbers of data',
    title= 'Numbers of data by Label (0-9)',
    plot_bgcolor='rgba(0,0,2,0.05)',
    showlegend=False,
)

# compare model
data = pd.read_csv('tmp/model_version.csv')
df = data[data['ModelVersion'] == data['ModelVersion'].max()]






app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])






app.layout = html.Div([
    dcc.Tabs(id='tabs', 
             value='tab-1',
             style={
                'backgroundColor': 'FloralWhite',
                'font-family': 'Roboto',
                # Add more CSS properties as needed
            }
             
             
             
             , children=[
        dcc.Tab(label='User', value='tab-1',
                style={
                'backgroundColor': 'FloralWhite',
                'font-size': '20px',
                'font-weight': 'bold',
                'font-family': 'Roboto',
                # Add more CSS properties as needed
            },
                
                
                
                
                 children=[
            ###Page1####
            
            html.Div([
    dbc.Row([Header_component]),
    dbc.Row([
        dbc.Col(html.Div("IMPORT DATA"), style={'color': 'DimGray', 'font-size': '36px', "margin-left": "167px",'font-family': 'Roboto'}),
        dbc.Col(html.Div("Your upload number is:"), style={'color': 'DimGray', 'font-size': '50px', "margin-left": "110px",'font-family': 'Roboto'})
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Upload(
                id='upload-image',
                children=html.Div(['Drop here!']),
                style={
                    'width': '25%',
                    'height': '45px',
                    'lineHeight': '40px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '30px',
                    'textAlign': 'center',
                    'margin': '0',
                    "margin-left": "160px",
                    'color': 'DarkSlateGray',"backgroundColor": "FloralWhite",
                    'font-family': 'Roboto',
                    
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='output-image-upload'),
            dbc.Col([
                html.P(html.Div([html.Div(id='percent-f1')]), style={'color': 'DarkCyan', 'font-size': '60px', "margin-left": "130px",'font-family': 'Roboto'}),
                # html.P(f'F1 Score show the result ', id='percent-f1', style={'color': 'Teal', 'font-size': '20px', "margin-left": "130px", 'borderWidth': '1px','font-family': 'Roboto'})

            ])
        ])),
        dbc.Col(html.Div([
            html.H1(id="output-number", style={ 'color': 'DarkSlateGray',"backgroundColor": "FloralWhite",'text-align': 'Center' ,'font-size': '450px','font-family': 'Roboto'}),
            dbc.Col(html.Div(children=[
                html.H1(html.Div(id='percent-accu', style={'font-size': '20px','backgroundColor': 'DarkSlateGray', 'color': 'FloralWhite', "margin-right": "`120px",'margin':'0','font-family': 'Roboto'})),
                # dbc.Col('If the number result is wrong, plase provide us with the correct one',style={'font-family': 'Roboto'}),
#                 dbc.Col(html.Div([
#                     dcc.Input(id='text-input', type='text', placeholder='Enter text...'),
#                     html.Div(id='output-div'),
#                 dbc.Col(html.Div([
#                      html.Button('Submit', id='submit-button', n_clicks=0),
#                      html.Div(id='output-div')




# ]))
# ]))
            ]))
        ]))
    ])
])


            



        ]),
         dcc.Tab(label='Modify ', value='tab-2',
                 style={
                'backgroundColor': 'FloralWhite',
                'font-size': '20px',
                'font-weight': 'bold'
                # Add more CSS properties as needed
            }
              ,
                 
                 
                  children=[
           ####Page2### 
           

           html.Div(
    children=[
        html.H1("THAI NUMBER PREDICTION", style={'color': 'FloralWhite', 'text-align': 'Center', 'backgroundColor': 'DarkSlateGray', 'color': 'white','font-family': 'Roboto'}),
        html.Div(
            children=[

                ####-------------upload---------------####
                html.Div(
            html.Div([
    html.Div([
        html.Label('Upload New Data:'),
        dcc.Dropdown(
            id='path-dropdown',
            options=[
                {'label': f'Number {i}', 'value': f'Dataset/{i}'} for i in range(10)
            ],
            value='Dataset/0', style={'width': '100%'}
            ,
        )
    ]),
    dcc.Upload(
        id='upload-data',style={'color': 'DimGray','font-family': 'Roboto'},
        children=html.Button('Upload Files',style={'color': 'DimGray','font-family': 'Roboto'}),
        multiple=True
    ),
    html.Div(id='file-preview', style={'font-size': '18px'}),  # Add a div for file preview with adjusted font size
    html.Button('Submit', id='submit-button'),  # Add a submit button
    html.Div(id='output-data'),
    html.Div(id='confirmation-message', style={'margin-top': '10px'})  # Add a div for the confirmation message
])

        ),
                ####-------------slider1 PCA-----------####
        html.Div(
    style={'text-align': 'center', 'width': '50%'},
    children=[
        html.Div(
            style={'margin-left': '15px'},
            children=[
                html.Label("PCA Value", style={'color': 'DimGray', 'font-family': 'Roboto'}),
                dcc.Slider(
                    id="pca-slider",
                    min=50,
                    max=95,
                    step=5,
                    value=50,
                ),
            ]
        ),
    ]
),
        
        
        
        
        
                ####-------------slider2 Percent of Train data--------##
                 html.Div(
    style={'text-align': 'center', 'width': '50%', 'color': 'red'},
    children=[
        html.Div(
            children=[
                html.Label("Percent of Train Data", style={'margin-left': '15px', 'color': 'DimGray', 'font-family': 'Roboto'}),
                dcc.Slider(
                    id="split-slider",
                    min=50,
                    max=95,
                    step=5,
                    value=50
                ),
            ]
        )
    ]
)

                ####-----------confident interval--------####

                #html.Div(html.Div(
    #children=[
        #html.Label("Confident Interveal", style={"margin-left": "15px",'color': 'DimGray','font-family': 'Roboto'}),
        #dcc.Slider(
            #id="split-slider",
            #min=50,
            #max=95,
            #step=5,
            #value=50
        
            
            
        #)]), style={'text-align': 'center', 'width': '33.33%',}),
            ],
            style={'display': 'flex', 'flex-direction': 'row'}
        ),
        html.Div(
            children=[
                ####-----------Implance--------####
                html.Div(
    children=[
        html.Label("Imbalance Data?", style={'text-align': 'center', 'width': '33.33%', 'font-family': 'Roboto'}),
dcc.RadioItems(
    id="imbalance-radio",
    options=[
        {'label': 'Yes', 'value': True},
        {'label': 'No', 'value': False}
    ],
    value=True,
    style={'font-family': 'Roboto'}
),
        html.Div([
    html.Button("Train data", id="prediction-button", n_clicks=0,
                style={'color': 'FloralWhite', 'text-align': 'Center', 'backgroundColor': 'DarkSlateGray', 'color': 'white', 'font-family': 'Roboto'}),
    html.Div(id="output-div2")
])
                 
    ],
),
                ####-----------card1---------------##### 
                html.Div(html.Div(
                    children=[
                        html.Div(
                            "All",
                            style={"text-align": "center", "margin-bottom": "30px",'font-family': 'Roboto'}
                        ),
                        html.Div(
                            f'{row_count}',
                            style={"border-color": "black", "border-style": "solid", "width": "250px", "height": "100px", "text-align": "center", "line-height": "100px",'color': 'DarkSlateGray',"backgroundColor": "FloralWhite"},
                        )
                    ],
                    style={"margin-left": "20px", "display": "flex", "flex-direction": "column", "align-items": "center"}
                ), style={'text-align': 'center', 'width': '33.33%'}),
                #####-------- card2---------### 
                html.Div(html.Div(
                    children=[
                        html.Div(
                            "Train",
                            style={"text-align": "center", "margin-bottom": "30px",'font-family': 'Roboto'}
                        ),
                        html.Div(
                            id="Train-data",
                            style={"border-color": "black", "border-style": "solid", "width": "250px", "height": "100px", "text-align": "center", "line-height": "100px",'color': 'DarkSlateGray',"backgroundColor": "FloralWhite"},
                        )
                    ],
                    style={"margin-left": "20px", "display": "flex", "flex-direction": "column", "align-items": "center",'font-family': 'Roboto'}
                ), style={'text-align': 'center', 'width': '33.33%'}),
                 #####-------- card3---------### 
                html.Div(html.Div(
                    children=[
                        html.Div(
                            "Test",
                            style={"text-align": "center", "margin-bottom": "30px",'font-family': 'Roboto'}
                        ),
                        html.Div(
                            id='Test-data',
                            style={"border-color": "black", "border-style": "solid", "width": "250px", "height": "100px", "text-align": "center", "line-height": "100px",'color': 'DarkSlateGray',"backgroundColor": "FloralWhite"},
                            children=[
                                html.Span(id="Test-data")
                            ]
                        )
                    ],
                    style={"margin-left": "20px", "display": "flex", "flex-direction": "column", "align-items": "center"}
                ), style={'text-align': 'center', 'width': '33.33%'})
            ],
            style={'display': 'flex', 'flex-direction': 'row'}
        ),
        
html.Div(
         html.H1("THAI NUMBER PREDICTION MODEL RESULT", style={'color': 'white', 'text-align': 'Center', 'backgroundColor': 'white', 'color': 'white'})),        
html.Div(
        html.H1("THAI NUMBER PREDICTION MODEL RESULT", style={'color': 'FloralWhite', 'text-align': 'Center', 'backgroundColor': 'DarkSlateGray', 'color': 'white','font-family': 'Roboto'})), 
        html.Div(
            children=[
                ####------------ predict  ---------------####
                 
                 ##html.Div(dropdown),
                #### ---------- chart 1 ----------------#####
                html.Div(dcc.Graph(
        id='count-graph',
        figure=fig1,
        style={'width': '100%','font-family': 'Roboto'}

        
    ), style={'text-align': 'center','margin':'35px','font-family': 'Roboto'}),
                ####----------- chart 2 ---------------#### 
                html.Div(html.Div([
    dcc.Dropdown(
        id='score-type',
        options=[
            {'label': 'Accuracy', 'value': 'Accuracy'},
            {'label': 'F1 Score', 'value': 'F1'},
            {'label': 'Recall', 'value': 'Recall'},
            {'label': 'Precision', 'value': 'Prec.'},
            {'label': 'AUC Score', 'value': 'AUC'},
                
        ],
        value='Accuracy',
        style={'width': '1100px','font-family': 'Roboto'}
    ),
    dcc.Graph(id='model-scores')
]))
            ],
            style={'display': 'flex', 'flex-direction': 'row','font-family': 'Roboto'}
        )
    ],
    style={'display': 'flex', 'flex-direction': 'column','font-family': 'Roboto'},
    
)




        ])
    ])
    
    
    
    
    ,
])



##call page1##
def parse_contents(contents, filename, date):
    return html.Div([
        html.H6("Your data upload completed", style={'color': 'DarkSlateGrey','font-size': '20px', "margin-left": "140px",}),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, style={'height': '50%', 'width': '40%', "margin-left": "110px"}),
    ])

UPLOAD_DIRECTORY = 'Dataset_tmp/forPredict'
@app.callback(
    [Output('output-image-upload', 'children'),
     Output('output-number', 'children'),
     Output('percent-f1', 'children'),
     Output('percent-accu', 'children')],
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('upload-image', 'last_modified')
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
        
        # Save uploaded files to the specified directory
        for content, name in zip(list_of_contents, list_of_names):
            save_path = os.path.join(UPLOAD_DIRECTORY, name)
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            with open(save_path, 'wb') as f:
                # f.write(content.encode('utf-8'))  #  .encode('utf-8') Assuming content is a string representation of the file
                f.write(decoded)
            result = fn_predict(save_path)
            print (result)
                
        return children,result[0],(f'F1 Score : {result[3]:.1f} %'),(f'Accuracy Score : {result[4]:.1f} %')

## ปุ่มsubmit ## 
@app.callback(
    Output('output-div', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('text-input', 'value')]
)
def update_output_div(n_clicks, input_text):
    if n_clicks > 0:
        return html.H3(f"You entered: {input_text}", )
    else:
        return html.H3("Click the button to submit")


## Call page 2 ## 

uploaded_files = {}  # Global dictionary to store uploaded files

def save_file(contents, filename, save_path):
    # Decode the file contents
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Save the file to the specified path
    file_path = os.path.join(save_path, filename)
    with open(file_path, 'wb') as f:
        f.write(decoded)

    return html.Div([
        f"Saved: {filename} Completed"
        # html.Hr()
    ])

@app.callback(Output('output-data', 'children'),
              Input('submit-button', 'n_clicks'),  # Add an input for the submit button
              State('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('path-dropdown', 'value'),
              prevent_initial_call=True)
def update_output(submit_clicks, contents, filenames, save_path):
    ctx = callback_context  # Get the callback context

    children = []

    # Check if the submit button is clicked and there are uploaded files
    if submit_clicks and contents is not None and filenames is not None:
        # Clear previously uploaded files
        uploaded_files.clear()

        # Get the index of the dropdown value that corresponds to the save_path
        path_index = int(save_path.split('/')[-1])

        # Store the uploaded files in the global dictionary
        uploaded_files[path_index] = {
            'contents': contents,
            'filenames': filenames
        }

        # Iterate over the uploaded files and save them to the appropriate paths
        for path_index, files in uploaded_files.items():
            contents = files['contents']
            filenames = files['filenames']

            for i in range(len(contents)):
                children.append(save_file(contents[i], filenames[i], save_path))

    return children

@app.callback(Output('file-preview', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              prevent_initial_call=True)
def update_file_preview(contents, filenames):
    if contents is not None and filenames is not None:
        file_previews = []

        for content, filename in zip(contents, filenames):
            # Decode the file contents
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)

            # Create a data URL for the preview
            data_url = 'data:{};base64,{}'.format(content_type, base64.b64encode(decoded).decode())

            # Create an image or file preview based on the file type
            if content_type.startswith('image/'):
                file_previews.append(html.Div([
                    html.H5(filename),
                    html.Img(src=data_url, style={'max-width': '100%'}),
                    html.Hr()
                ]))
            else:
                file_previews.append(html.Div([
                    f" {filename} "# html.P('File type: {}'.format(content_type)),
                    # html.Hr()
                ]))

        return file_previews

####--------slider-------####

@app.callback(
    Output('model-scores', 'figure'),
    [Input('score-type', 'value')]
)

def update_graph(score_type):
    if score_type is None:
        raise dash.exceptions.PreventUpdate

    sorted_df = df.sort_values(by=score_type, ascending=False)
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_df['Model'],
            y=sorted_df[score_type],
            marker=dict(color='rgba(0, 0, 140, 0.8)'),
            text=sorted_df[score_type],
            textposition='auto',
        )
    ])

    fig.update_layout(
        xaxis_title='Model',
        yaxis_title=score_type,
        title=f'{score_type} Scores',
        plot_bgcolor='rgba(0,0,2,0.05)',
        showlegend=False,
        yaxis=dict(range=[0, 1.0]),
        xaxis=dict(tickfont=dict(size=9),),
        margin=dict(t=100),height=480,width =780,
       
        
    )

    return fig


@app.callback(
    [Output('Train-data', 'children'),
    Output('Test-data', 'children'),],
    Input("split-slider", "value"),
)
def update_data(split_value):
    x_train = 0
    x_test = 0
    
    if split_value is not None:
        # Use the values in your desired function or logic
        # fn_train_model(pca_value, split_value, imbalance_value)
        x_train = int(row_count * split_value/100)
        x_test  = int(row_count - x_train)
        # print(x_test)
    
    return x_train, x_test



@app.callback(
    Output('output-div2', 'children'),
    Input('prediction-button', 'n_clicks'),
    Input('pca-slider', 'value'),
    Input('split-slider', 'value'),
    Input('imbalance-radio', 'value'),
)
def update_output(n_clicks,pca_value,train_data_percent,imbalance_data):
    if n_clicks > 0:
        # Perform your prediction logic here
        print("Start Train data")
        print(pca_value)
        print(train_data_percent)
        print(imbalance_data)
        fn_train_model(pca_value,train_data_percent,imbalance_data)
        # Return the content for the popup
        return dbc.Alert("Train model successfully", color="success", dismissable=True)






if __name__ == '__main__':
    app.run_server(debug=False)
