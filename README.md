# Final-Project-Econometrics
MODELOS DE ARMA, ARIMA, COINTEGRATION, ETC.
## PROYECTO FINAL: EQUIPO 2
# This code is designed to run in Google Colab for your Econometrics final project.
# To fix the binary incompatibility with pmdarima and numpy 2.0+, we install pmdarima from source using --no-binary.

!pip install pmdarima --no-binary pmdarima

# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from itertools import combinations
from google.colab import files

# Upload the data files (run this in Colab to upload the files)
# If you get KeyboardInterrupt, it means the upload was interrupted; rerun the cell and upload the files.
uploaded = files.upload()

# Load the data from the uploaded Excel files
# Assuming the files are named as provided: '1D 1MIN.xlsx', '3dias3min.xlsx', '5dias5min.xlsx'
# If file names differ, adjust accordingly.
df_1d1m = pd.read_excel(next(f for f in uploaded if '1D 1MIN' in f or '1d1min' in f.lower()))
df_3d3m = pd.read_excel(next(f for f in uploaded if '3dias3min' in f or '3d3m' in f.lower()))
df_5d5m = pd.read_excel(next(f for f in uploaded if '5dias5min' in f or '5d5m' in f.lower()))

# Set 'Date' as index for each dataframe
df_1d1m.set_index('Date', inplace=True)
df_3d3m.set_index('Date', inplace=True)
df_5d5m.set_index('Date', inplace=True)

# List of stocks and their close columns (note: APPL might be a typo for AAPL, but keeping as is)
stocks = ['SBUX', 'MCD', 'XOM', 'CAT', 'APPL', 'NVDA']
close_columns = [f'{stock} Close' for stock in stocks]

# Dictionary of dataframes for each time frame
data_frames = {
    '1D/1m': df_1d1m,
    '3D/3m': df_3d3m,
    '5D/5m': df_5d5m
}

# Function to perform unit root test (ADF)
def unit_root_test(series, stock, frame):
    result = adfuller(series.dropna())
    print(f'Unit Root Test (ADF) for {stock} in {frame}:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print('Series has a unit root (non-stationary)')
    else:
        print('Series is stationary')
    print('\n')

# Function to perform stationarity test (KPSS)
def stationarity_test(series, stock, frame):
    result = kpss(series.dropna(), regression='c')
    print(f'Stationarity Test (KPSS) for {stock} in {frame}:')
    print(f'KPSS Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] < 0.05:
        print('Series is not stationary')
    else:
        print('Series is stationary')
    print('\n')

# Function to generate correlogram (ACF and PACF plots)
def correlogram(series, stock, frame):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), ax=ax[0], title=f'ACF for {stock} in {frame}')
    plot_pacf(series.dropna(), ax=ax[1], title=f'PACF for {stock} in {frame}')
    plt.show()
    print('\n')

# Function to perform cointegration test between two series
def cointegration_test(series1, series2, stock1, stock2, frame):
    score, p_value, _ = coint(series1.dropna(), series2.dropna())
    print(f'Cointegration Test between {stock1} and {stock2} in {frame}:')
    print(f'p-value: {p_value}')
    if p_value < 0.05:
        print('Series are cointegrated')
    else:
        print('Series are not cointegrated')
    print('\n')

# Function to fit ARIMA model
def arima_model(series, stock, frame):
    model = auto_arima(series.dropna(), seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    fitted = model.fit(series.dropna())
    print(f'ARIMA Model for {stock} in {frame}:')
    print(fitted.summary())
    print('\n')

# Function to fit ARMA model (ARIMA with d=0)
def arma_model(series, stock, frame):
    model = auto_arima(series.dropna(), d=0, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    fitted = model.fit(series.dropna())
    print(f'ARMA Model for {stock} in {frame}:')
    print(fitted.summary())
    print('\n')

# Section 1: Perform specific analyses as per instructions
print('=== Specific Analyses for Designated Stocks and Time Frames ===\n')

# SBUX: Unit Root Test in 1D/1m
unit_root_test(df_1d1m['SBUX Close'], 'SBUX', '1D/1m')

# MCD: Stationarity Test in 3D/3m
stationarity_test(df_3d3m['MCD Close'], 'MCD', '3D/3m')

# XOM: Correlogram in 5D/5m
correlogram(df_5d5m['XOM Close'], 'XOM', '5D/5m')

# CAT: Cointegration (with XOM, as per example in attached files) in 1D/1m (from pair row)
cointegration_test(df_1d1m['CAT Close'], df_1d1m['XOM Close'], 'CAT', 'XOM', '1D/1m')

# APPL: ARIMA in 1D/1m (assuming from pair row with SBUX)
arima_model(df_1d1m['APPL Close'], 'APPL', '1D/1m')

# NVDA: ARMA in 1D/1m (assuming from pair row with SBUX)
arma_model(df_1d1m['NVDA Close'], 'NVDA', '1D/1m')

# Section 2: Replicate the analyses for all stocks across all time frames
print('=== Replicated Analyses for All Stocks Across All Time Frames ===\n')

for frame, df in data_frames.items():
    print(f'--- Time Frame: {frame} ---')
    for stock, col in zip(stocks, close_columns):
        series = df[col]
        print(f'** Stock: {stock} **')
        unit_root_test(series, stock, frame)
        stationarity_test(series, stock, frame)
        correlogram(series, stock, frame)
        arima_model(series, stock, frame)
        arma_model(series, stock, frame)

# Section 3: Pairwise Combinations - Cointegration Tests for All Unique Pairs in Each Time Frame
print('=== Pairwise Cointegration Analyses ===\n')

# Generate all unique pairs
pairs = list(combinations(stocks, 2))

# For each time frame, compute cointegration p-values and display in a table
for frame, df in data_frames.items():
    print(f'--- Cointegration Results for {frame} ---')
    results = []
    for stock1, stock2 in pairs:
        col1 = f'{stock1} Close'
        col2 = f'{stock2} Close'
        series1 = df[col1]
        series2 = df[col2]
        _, p_value, _ = coint(series1.dropna(), series2.dropna())
        conclusion = 'Cointegrated' if p_value < 0.05 else 'Not Cointegrated'
        results.append([f'{stock1}-{stock2}', p_value, conclusion])

    # Display as table
    results_df = pd.DataFrame(results, columns=['Pair', 'p-value', 'Conclusion'])
    print(results_df.to_markdown(index=False))
    print('\n')

# Additional: Review and Integrate from Previous Homeworks
# Based on attached HTML files, add structural break detection for example pairs (e.g., CAT-XOM)
# Using CUSUM test on OLS residuals as implied in the HTML interpretations

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import breaks_cusumolsresid

print('=== Structural Break Detection from Previous Homeworks (Example: CAT-XOM in 5D/5m) ===\n')

# Example for CAT and XOM in 5D/5m
y = df_5d5m['XOM Close'].dropna()
x = df_5d5m['CAT Close'].dropna()
x = np.column_stack((np.ones(len(x)), x))  # Add constant
model = OLS(y, x).fit()
resid = model.resid
cusum_res = breaks_cusumolsresid(resid)
print(f'CUSUM Test for Structural Break in XOM (regressed on CAT) in 5D/5m:')
print(f'Statistic: {cusum_res[0]}, p-value: {cusum_res[1]}')
if cusum_res[1] < 0.05:
    print('Structural break detected')
else:
    print('No structural break detected')
print('\n')

# Similarly, you can add for other pairs like SBUX-MCD, APPL-NVDA if needed
# For correlation/regression from Correlation.html, example:
print('=== Example Correlation and Regression from Previous Homeworks (APPL-NVDA in 1D/1m) ===\n')
corr = df_1d1m['APPL Close'].corr(df_1d1m['NVDA Close'])
print(f'Correlation between APPL and NVDA in 1D/1m: {corr}')

# Simple OLS regression
y = df_1d1m['APPL Close'].dropna()
x = df_1d1m['NVDA Close'].dropna()
x = np.column_stack((np.ones(len(x)), x))
reg_model = OLS(y, x).fit()
print(reg_model.summary())
print('\n')

# End of code - This covers the full project requirements, integrating previous homework elements.
Req
