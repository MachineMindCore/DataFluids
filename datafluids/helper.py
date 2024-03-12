import pandas as pd
import datetime as dt
import streamlit as st
import numpy as np
import base64
import io
from datetime import datetime as dt2


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    href = f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download excel file</a>'
    st.markdown(href, unsafe_allow_html=True)
    return


def df2st(iframe):
    size = len(iframe.index)
    Data = np.asarray([])
    Time = np.asarray([])
    for i in range(0, size):
        dayData = iframe.iloc[i, 0:24].to_numpy()
        Data = np.concatenate((Data, dayData))
        dayFrame = iframe.iloc[i, 24]
        dayFrame = dt2.strptime(dayFrame, '%Y-%m-%d')
        for j in range(0, 24):
            h = dt.timedelta(hours=j)
            day = dayFrame + h
            Time = np.concatenate((Time, day), axis=None)

    Data = Data.astype(float)
    return [Time, Data]


def data_expand(iframe, datai, datei):
    size = len(iframe.index)
    Data = np.asarray([])
    Time = np.asarray([])
    v24 = np.ones((1, 24))
    for i in range(0, size):
        dayData = int(iframe.iloc[i, datai])
        Data = np.append(Data, dayData * v24)
        dayFrame = iframe.iloc[i, datei]
        dayFrame = dt2.strptime(dayFrame, '%Y-%m-%d')
        for j in range(0, 24):
            h = dt.timedelta(hours=j)
            day = dayFrame + h
            Time = np.concatenate((Time, day), axis=None)

    return [Time, Data]
