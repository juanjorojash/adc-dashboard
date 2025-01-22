import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='SAR ADC dashboard',
    page_icon=':level_slider:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data


def SAR(res,vref,vin,pr):
# res = resolucion en formato 0bXXXXX
# vref = voltaje de referencia positivo asumiendo 0 como referencia negativa
# vin = voltaje de entrada, valor a convertir en digital
# pr = imprimir o no
    max = res+1 #0b1111 -> 0b10000
    doc = max>>1 # 0b10000 -> 0b1000
    if pr == True:
        print("Step 0:", "{:04b}".format(doc) )
    n = res.bit_length()
    for index in range(1,n):
        vdac = (doc/max)*vref
        if n > 1:
            doc = doc | (1 << n-2)
        if vin < vdac:
            doc = doc & ~(1 << n-1)
        if pr == True:
            print("Step %d:" % (index), "{:04b}".format(doc) )
        n = n - 1
    vout = (doc/max)*vref
    return vout

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :level_slider: SAR ADC example


Change the analog wave and the resolution of the SAR ADC and observe the result
'''

# Add some spacing
''
''

option = st.selectbox(
    'Select the type of analog wave',
    ['sine','sine + cosine','exponential'])

f = 1000
T = 1/f
t = np.arange(0,T,T/100)
bias = 2 
s = 0*t + bias
'Constants:' 
st.latex(r'''f=1000 \,\mathrm{Hz} \,\,\, T=1/f \, \mathrm{s} \,\,\, A=1 \,\mathrm{V} \,\,\, B=2 \,\mathrm{V}''')

match option:
    case 'sine':
        s = np.sin(2*np.pi*2*f*t) + bias
        equation = r''' A \cdot \sin{(2 \cdot \pi \cdot 2 \cdot f \cdot t)} + B '''
    case 'sine + cosine':
        s = np.sin(2*np.pi*2*f*t)+np.cos(2*np.pi*4*f*t) + bias
        equation = r''' A \cdot \sin{(2 \cdot \pi \cdot 2 \cdot f \cdot t)} + A \cdot \cos{(2 \cdot \pi \cdot 4 \cdot f \cdot t)} + B '''
    case 'exponential':
        s = (t>=0)*(t<=(0.5*T))*(1-np.exp(-(t/(0.5*T)))) + (t>(0.5*T))*(2*np.exp(-((t-0.5*T)/(0.5*T))))
        equation = r''' A \cdot \left( u(t) - u(t - 0.5 \cdot T) \right)  \cdot \left( 1 - e^{\dfrac{-t}{0.5 \cdot T}}\right) +
        B \cdot \left( u(t - 0.5 \cdot T) \right) \cdot \left( e^{\dfrac{- t - 0.5 \cdot T}{0.5 \cdot T}}\right) '''
    case _:
        'ohh no'

'Equation:'
st.latex(equation)

data={'time':t*1000,'wave':s}

df=pd.DataFrame(data)

'Analog wave'
st.line_chart(
    df,
    x='time',
    y='wave',
    x_label='time (ms)',
    y_label='voltage (V)',
)

resolution = st.select_slider(
    "Select the resolution of the ADC in bits",
    options=[
        "2",
        "4",
        "6",
        "8",
        "10",
    ],
)

vout = np.array([])

for index in range(len(s)):
    match resolution:
        case '2':
            vout = np.append(vout, SAR(0b11,5,s[index],False))
        case '4':
            vout = np.append(vout, SAR(0b1111,5,s[index],False))
        case '6':
            vout = np.append(vout, SAR(0b111111,5,s[index],False))
        case '8':
            vout = np.append(vout, SAR(0b11111111,5,s[index],False))
        case '10':
            vout = np.append(vout, SAR(0b1111111111,5,s[index],False))

df['dig'] = vout

mse = mean_squared_error(df['wave'], df['dig'])

'Analog wave with digital approximation'

'RMSE:',mse
st.line_chart(
    df,
    x='time',
    y=['wave','dig'],
    x_label='time (ms)',
    y_label='voltage (V)',
)


