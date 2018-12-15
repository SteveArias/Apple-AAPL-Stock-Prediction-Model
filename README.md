# Apple-AAPL-Stock-Prediction-Model
This is a prediction model using Python and Machine Learning to try to predict the closing price of AAPL stock.

# Strategy
I initially planned to just use Linear Regression with the closing price of the AAPL stock to try to predict the closing price of the next day. I then switched to using a Neural Network with multiple features. In order to capture as much information to help with this prediction model, I used multiple stock features. The five stock features I used are:
* Open Price
* High Price
* Low Price
* Close Price
* Volume

In addition, I wanted to use other stocks in order to get a wider breadth of data that could allow the model to read any possible trends. I used the stock of major competitors of Apple and some of the companies that are major suppliers to the company (e.g. Qualcomm). Finally, I decided to use three ETFs that are major holders of AAPL stock, as well as major holders of technoogy companies in general. For example, the Vanguard Information Technology ETF (VGT) is an index of companies in the information technology sector which the company considers to be the following three areas: software, consulting, and hardware.

Competitors | Suppliers  | ETFs
------------- | ------------- | -------------
Amazon (AMZN)  | Analog Devices, Inc. (ADI) | Vanguard Information Technology ETF (VGT)
Microsoft (MSFT)  | Qualcomm (QCOM) | SPDR S&P 500 Trust ETF (SPY)
Google (GOOGL)  | Texas Instruments Incorporated (TXN) |  iShares Dow Jones US Technology (IYW)
