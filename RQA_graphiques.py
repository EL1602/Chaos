import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import yfinance as yf
import sympy as smp
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.image_generator import ImageGenerator
from pyrqa.computation import RPComputation
from PIL import Image
from numpy import asarray
import matplotlib.dates as mdates

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days= 6000)

#endDate1 = dt.datetime.now()- dt.timedelta(days= 7)
#startDate1 = endDate - dt.timedelta(days= 13)

btc_data = yf.download("^BSESN", startDate, endDate, interval="1d")
#btc_data1 = yf.download("BTC-USD", startDate1, endDate1, interval="1m")
closing_prices = btc_data["Close"]
#closing_prices1 = btc_data1["Close"]

returns = closing_prices.pct_change().tolist()
tseries = closing_prices.tolist() 



#returns1 = closing_prices.pct_change().tolist()
#tseries1 = closing_prices.tolist()

#returns2 = returns + returns1
#tseries2 = tseries + tseries1


x_axis = range(len(tseries)) 
plt.figure(figsize=(12, 6)) 
plt.plot(closing_prices.index.to_numpy(), closing_prices.to_numpy())
plt.xlabel("Date")
plt.ylabel("IBOVESPA (USD)")
plt.title("Prix selon le temps")
plt.xticks(rotation=45)  
plt.show()

x_axis2 = range(len(returns)) 
plt.figure(figsize=(12, 6)) 
plt.plot(closing_prices.index.to_numpy(), returns)
plt.xlabel("Date")
plt.ylabel("retours (%)")
plt.title("Retours selon le temps")
plt.xticks(rotation=45)  
plt.show()



a = [1, 2, 3, 4,5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2]*100

#a = np.random.rand(1000)
ts = TimeSeries(tseries, embedding_dimension=4, time_delay=1)  
settings = Settings(time_series=ts,  
                    analysis_type=Classic,
                    similarity_measure=EuclideanMetric,
                    neighbourhood=FixedRadius(radius= 1500) 
)

computation = RPComputation.create(settings)
result = computation.run()
ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse,
                                    'recurrence_plotA.png')



computation = RQAComputation.create(settings,
                                    verbose=True)
result = computation.run()
result.min_diagonal_line_length = 2
result.min_vertical_line_length = 2
result.min_white_vertical_line_length = 2
print(result)

#print(ts, type(ts))
#print(closing_prices, type(closing_prices))
#print(returns, type(returns))
#print(meanReturns, type(meanReturns))

def mettre_axe(path_image_png,time_series) :
    image = Image.open(path_image_png)
    numpydata = asarray(image)
    parse_dates = []
    for i in time_series:
        original_date = dt.datetime.strptime(i, "%Y-%m-%dT%H:%M:%S.%f000")
        formatted_Date = original_date.strftime("%Y-%m-%d")

        parse_dates.append(formatted_Date)

    parse_dates = mdates.date2num(parse_dates)
    fig, ax = plt.subplots()
    ax.imshow(numpydata, extent= [parse_dates[0],parse_dates[-1],parse_dates[0],parse_dates[-1]])
    ax.xaxis_date()
    ax.yaxis_date()
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    ax.yaxis.set_major_formatter(date_format)
    plt.xlabel('Temps [AAAA-MM-JJ]',fontsize = 21)
    plt.ylabel('Temps [AAAA-MM-JJ]', fontsize = 21)
    plt.tick_params(axis = 'both', which = 'both', direction = 'in', length = 6)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.show()

x_lims = list(map(dt.datetime.fromtimestamp, [982376726, 982377321]))
# Parse the original date string into a datetime object
time_series = []
for i in closing_prices.index.to_numpy():
    time_series.append(str(i))


mettre_axe('recurrence_plotA.png', time_series)