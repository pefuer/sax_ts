import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import numpy as np
from scipy.stats import norm
import plotly.graph_objs as go
from numpy.random import default_rng
rng = default_rng()


def get_breakpoints(n):
    b = norm.ppf(np.linspace(0,1, n+1))
    
    # do not slice inf/-inf as they are used later in range comparisons
    return b

def get_sax_representation(T, w, breakpoints, cardinality):
    """ Transforms a time series to sax
    
    Args:
        T (array) time series
        w (int) dimension - time series length must be a multiple of w
        breakpoints (array) boundaries of the breakpoints. Lower and upper bounds must be -inf/inf
        cardinality (int) sax word length
       
    Returns:
        List of sax words
    """
    
    n = len(T)
    sax_words = []

    for i in range(0, w):
       
        start = n / w * i
        end = n / w * (i + 1)
        
        paa = w/n * np.sum(T[int(start):int(end)])
    
        # find the word/range that describes the paa value    
        for k in range(0, cardinality+1):
    
            if(paa > breakpoints[k] and paa <= breakpoints[k+1]):
                sax_word = k
                break
        sax_words.append(sax_word)
        
        #print(f"word from {start} to {end}. PAA: {paa} SAX: {sax_word}" )
    
    return sax_words

def get_sax_figure(T, w, breakpoints, sax_words):
    """ Get sax plot
    
    Args:
        T (array) time series
        w (int) dimension - time series length must be a multiple of w
        breakpoints (list) boundaries of the breakpoints. Lower and upper bounds must be -inf/inf
        sax_words (list) the sax representation
       
    Returns:
        Plotly figure
    """
    
    n = len(T)
    
    fig = go.Figure()

    plot_breakpoints = np.copy(breakpoints)
    plot_breakpoints[0] = np.min(T) - 0.1
    plot_breakpoints[-1] = np.max(T) + 0.1

    shapes = [] 

    for i, sax_word in enumerate(sax_words):

        start = i * (n/w)
        end = (i+1) * (n/w)
        
        fig.add_shape(type="rect", x0=start-.1, y0=plot_breakpoints[sax_word], x1=end-.2, y1=plot_breakpoints[sax_word+1], 
                    line=dict(color=None, width=0), 
                    fillcolor="LightSkyBlue",
                    opacity=0.5, layer="below")


    fig.add_trace(go.Scatter(x=np.arange(0, len(T)), y=T))

    for k in range(1, len(breakpoints)-1):
        fig.add_shape(type="line", x0=0, y0=breakpoints[k], x1=len(T), y1=breakpoints[k], line_dash="dot")


    fig.update_layout(title='SAX representation of T',
                    xaxis_title='Time Series Index',
                    yaxis_title='Value')
    return fig

def get_hist(sax_words, cardinality):
    """ Get a histogram of the sax symbols
    
    Args:
        breakpoints (list) boundaries of the breakpoints. Lower and upper bounds must be -inf/inf
        sax_words (list) the sax representation
       
    Returns:
        List of sax words
    """
    hist, bin_edges = np.histogram(sax_words, bins=cardinality)

    fig = go.Figure(data=go.Bar(x=np.arange(0, cardinality), y=(hist / np.sum(hist))))
    fig.update_layout(title='SAX symbol frequency',
                    xaxis_title='Symbol Index',
                    yaxis_title='Relative Frequency')
    return fig

# This short series approximates the example from the paper
T = np.array([-.6, -.5, -.7, -1, -.8, -1.5, -.75, -.1, -.15, .2, .3, .2, 1.3, 1.7, 1.5, 1.4])

# Synthetic, slightly longer series
x_len = 128
x = np.linspace(0, 16, x_len)
x = np.sin(x) + np.sin(0.25*x) + np.sin(-.5*x) + 0.1*rng.standard_normal(x_len)
x = (x - np.mean(x)) / np.std(x)
T = x


app = dash.Dash(__name__)


app.layout = html.Div([
    dcc.Markdown('''

    # SAX Representation for Time Series
    SAX can be used as a compact representation of time series.
    This demo can be used to experiment with the parameters of the SAX transformation.

    Additional information on the algorithm can be found in _iSAX: Indexing and Mining Terabyte Sized Time Series, Jin Shieh & Eamonn Keogh, SIGKDD, 2008_, section 2.3.
    
    * Dimension **w**: this parameter controls the dimensionality of the resulting vector space. More dimensions increase the temporal resolution of the representation at the cost of increased storage requirements
    * Cardinality **a**: the cardinality controls the number of symbols that are used. The number of symbols is equal to the distinct values that are used to approximate the time series.

    The two plots below illustrate the results of the SAX transformation:

    ### Upper Plot: Transformation Result
    The blue line shows the input time series. Dashed black lines illustrate the breakpoints (thresholds) that are used to determine the underlying PAA representations.
    The box overlays indicate the SAX symbol which is used for a particular section of the series. The bottom-most box represents the first symbol (0). The top-most box represents the last symbol (a-1).

    ### Lower Plot: Histogram of SAX Symbols
    The lower plot shows the relative frequency of the symbols that were used for representing the signal.

    '''),
    html.Label(
            [
                "Dimensions (w)",
                dcc.Dropdown(
                    id='dimensions',
                    options=[{'label': i, 'value': i} for i in [2,4,8,16,32]],
                    value='8',
                ),
            ]
    ),
    html.Label(
            [
                "Cardinality (a)",
                dcc.Dropdown(
                    id='cardinality',
                    options=[{'label': i, 'value': i} for i in [2,4,8,16,32]],
                    value='4',
                ),
            ]
    ),
    dcc.Graph(id='graph-sax'),
    dcc.Graph(id='graph-hist')
])


@app.callback(
    Output('graph-sax', 'figure'),
    Output('graph-hist', 'figure'),
    Input('dimensions', 'value'),
    Input('cardinality', 'value'))
def update_figure(dimensions, cardinality):

    # Length of time series
    n = len(T)

    # Dimensions of the vector space
    w = int(dimensions)

    # Length of the sax symbol
    cardinality = int(cardinality)

    breakpoints = get_breakpoints(cardinality)
    
    sax_words = get_sax_representation(T, w, breakpoints, cardinality)

    return get_sax_figure(T, w, breakpoints, sax_words), get_hist(sax_words, cardinality)

if __name__ == '__main__':
    app.run_server(debug=False)