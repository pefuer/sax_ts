# SAX Demo

This Dash application allows experimenting with the parameters of the SAX transformation. For more information on the transformation, please refer to Sec. 2.3 of _iSAX: Indexing and Mining Terabyte Sized Time
Series, Jin Shieh & Eamonn Keogh, SIGKDD, 2008_

By default, a timeseries of length 128 is used. For comparison, the code also contains a shorter time series which is very similar to the one in the above-mentioned paper (see line sax_dash.py:130).

## Instructions Python 3.9.2
Use `requirements.txt` to install dependencies. Tested with Python 3.9.2 / Windows.

## Instructions Anaconda
After cloning the repository, run the following commands

1. `conda create -n saxts dash numpy scipy`
2. `conda activate saxts`
3. `python sax_dash.py`
4. Use your browser to navigate to `http://127.0.0.1:8050/`
