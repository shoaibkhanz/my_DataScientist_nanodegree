
import dash
import dash_core_components as dcc
import dash_html_components as html


app = dash.Dash()

app.layout = html.Div(children = [
    html.H1('Dash Example'),
    dcc.Graph(id = 'example',
    figure = {
        'data':[
            {'x':[1,2,3,4,5],'y':[8,2,3,7,4],'type':'line','name':'cars'},
            {'x':[1,2,3,4,5],'y':[8,2,6,1,2],'type':'bar','name':'bikes'}
            ],
        'layout':{
            'title': 'Basix Dash example'
            }
            }) 
            ])

if __name__ == '__main__':
    app.run_server(debug = True)


