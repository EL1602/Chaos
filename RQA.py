import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import pyrqa
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.image_generator import ImageGenerator
from pyrqa.computation import RPComputation


#choix du symbole de l'action à analyser 
ticker = "^GSPC"


#choix de l'intervalle par nombre de jours
endDate = dt.datetime.now() - dt.timedelta(days= 4500)
startDate = endDate - dt.timedelta(days= 2000)
intervalle = f"de {startDate.year}-{startDate.month}-{startDate.day} à {endDate.year}-{endDate.month}-{endDate.day}"


#extraction des données de l'action par yfinance
stock_data = yf.download(ticker, startDate, endDate, interval="1d")
closing_prices = stock_data["Close"]
returns = closing_prices.pct_change().tolist()
tseries = closing_prices.tolist() 

#figure du prix selon le temps
plt.figure() 
plt.plot(closing_prices.pct_change().index, closing_prices)
plt.xlabel("Date (année)", fontsize=16)
plt.ylabel(f"{ticker} (USD)", fontsize=16)
plt.ylabel(f"{ticker} (USD)", )
plt.title(f"Prix de {ticker} à {intervalle} ")
plt.xticks(rotation=45, fontsize=16) 
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()

#figure des retours selon le temps 
plt.figure() 
plt.plot(closing_prices.index, returns)
plt.xlabel("Date (année)", fontsize=16)
plt.ylabel("retours (%)", fontsize=16)
plt.ylabel(f"retours {ticker}(%)")
plt.title(f"Retours de {ticker} {intervalle}")
plt.xticks(rotation=45, fontsize=16) 
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()


#sélection de la fenetre de temps sur laquelle on caclcule les coefficients
window = 300
startDate1 = startDate + dt.timedelta(days= window)
intervalle2 = f"de {startDate1.year}-{startDate1.month}-{startDate1.day} à {endDate.year}-{endDate.month}-{endDate.day}"

#calcul des coefficients en fonction du temps
lams = []
det = []
rr = []
tt = []

for i in range(window, len(returns)):
    ret = returns[i-window: i]
    ts = TimeSeries(ret, embedding_dimension=6, time_delay=1)  
    settings = Settings(time_series=ts,  
                    analysis_type=Classic,
                    similarity_measure=EuclideanMetric,
                    neighbourhood=FixedRadius(radius= 0.05))
    computation = RQAComputation.create(settings,
                                    verbose=True)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2
    lams += [result.laminarity]
    det += [result.determinism]
    rr += [result.recurrence_rate]
    tt += [result.trapping_time]


plt.figure() 
plt.plot(closing_prices[window:].index, lams)
plt.xlabel("Date (AAAA-MM)", fontsize=16)
plt.ylabel("Laminarité", fontsize=16)
plt.title(f"Laminarité  de {ticker} {intervalle2}")
plt.xticks(rotation=45, fontsize=16) 
plt.yticks(fontsize=16)
plt.tight_layout() 
plt.show()

plt.figure() 
plt.plot(closing_prices[window:].index, det)
plt.xlabel("Date (AAAA-MM)", fontsize=16)
plt.ylabel("Déterminisme", fontsize=16)
plt.title(f"Déterminisme  de {ticker} {intervalle2}")
plt.xticks(rotation=45, fontsize=16) 
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()

plt.figure() 
plt.plot(closing_prices[window:].index, rr)
plt.xlabel("Date (AAAA-MM)", fontsize=16)
plt.ylabel("Taux de récurrence", fontsize=16)
plt.title(f"Taux de récurrence de {ticker} {intervalle2}")
plt.xticks(rotation=45, fontsize=16) 
plt.yticks(fontsize=16) 
plt.tight_layout()
plt.show()

plt.figure() 
plt.plot(closing_prices[window:].index, tt)
plt.xlabel("Date (AAAA-MM)", fontsize=16)
plt.ylabel("Temps de piègement", fontsize=16)
plt.title(f"Temps de piègement de {ticker} {intervalle2}")
plt.xticks(rotation=45, fontsize=16) 
plt.yticks(fontsize=16) 
plt.tight_layout()
plt.show()



#calcul des coefficients de la RQA et tracé du diagramme de récurrence
ts = TimeSeries(tseries, embedding_dimension=3, time_delay=1)  
settings = Settings(time_series=ts,  
                    analysis_type=Classic,
                    similarity_measure=EuclideanMetric,
                    neighbourhood=FixedRadius(radius= 0.1) 
)

computation = RPComputation.create(settings)
result = computation.run()
ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse,
                                    'RQAsin_sin.png')



computation = RQAComputation.create(settings,
                                    verbose=True)
result = computation.run()
result.min_diagonal_line_length = 2
result.min_vertical_line_length = 2
result.min_white_vertical_line_length = 2
print(result)

