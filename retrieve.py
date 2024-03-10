import yfinance as yf
import pandas as pd
data = yf.download('TSLA','2019-01-01','2019-02-01')
data.to_csv("VERIFY.csv")
