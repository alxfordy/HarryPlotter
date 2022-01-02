from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
import yfinance as yf
from sklearn.cluster import AgglomerativeClustering, KMeans
import mplfinance as mpf
from datetime import date,timedelta
from kneed import KneeLocator

"""
TODO
Include Volume in the calculation
Test it with rolling durations and without
"""

class SupRes():
    def __init__(self, ticker='MSFT', period="1y", interval="1h", duration=90):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.duration = duration

   
    def run(self):
        print(f"Collecting data for ticker [{self.ticker}]")
        data_df = self.get_ticker_info()
        print(data_df.head(5))
        optimal_clusters = self.find_elbow(data_df)
        print(f"Optimal Clusers = {optimal_clusters}")
        # close_of_clusters = self.agglomerative_sup_res(data_df, clusters=optimal_clusters)
        close_of_clusters = self.k_means_sup_res(data_df, clusters=optimal_clusters)

    def get_ticker_info(self):
        df = yf.download(self.ticker, period=self.period, interval=self.interval)
        df.index = pd.to_datetime(df.index)
        return df

    @staticmethod
    def find_elbow(data, increment=0, decrement=0):
        print(f"Identifying the Elbow")
        sum_of_squared_distances = dict()
        sum_of_squared_distances_l = list()
        K = range(1,15)
        #TODO Need to sort this as it's working against the whole yahoo DF not just close
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(data)
            sum_of_squared_distances[k] = km.inertia_
            sum_of_squared_distances_l.append(km.inertia_)
        ## https://practicaldatascience.co.uk/machine-learning/how-to-use-knee-point-detection-in-k-means-clustering
        """
        Usually the Kneedle algorithm will be optimimal but it's best to allow for increase/decrease
        So optional arugments added for that reason
        """
        kn = KneeLocator(x=list(sum_of_squared_distances.keys()),
                y=list(sum_of_squared_distances.values()),
                curve='convex',
                direction='decreasing')
        k = kn.knee + increment - decrement
        # Here you can plot the graph to find the elbow using eye but the Kneed Package does this programmatically
        """
        plt.plot(K, sum_of_squared_distances_l, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()
        """
        return k

    def k_means_sup_res(self, df, clusters=4, plot=True):
        date = df.index
        print(df)
        # Reset_Index for merging
        df.reset_index(inplace=True)
        df['max_smooth'] = df['High'].rolling(self.duration).max()
        df['min_smooth'] = df['Low'].rolling(self.duration).min()

        smooth_df = df.drop(['High', 'Low', 'Close', 'Open', 'Adj Close', 'index', 'Volume'], axis=1).dropna()
        print(smooth_df)
        km = KMeans(n_clusters=clusters)
        km = km.fit(smooth_df)
        smooth_df['clusters'] = km.labels_

        # Get index of the max wave for each cluster
        print(smooth_df)
        W2 = smooth_df.loc[smooth_df.groupby('clusters')['max_smooth'].idxmax()]
        print(df)
        df.index = date 
        if plot == True:
            mpf.plot(df, type='candle', style='yahoo', volume=True,
                    hlines=dict(hlines=W2.max_smooth.to_list(), colors='g', linestyle='dashed', alpha=0.7),figsize= (20, 10), savefig='testsave.png')
       
        W2.waves.drop_duplicates(keep='first', inplace=True)

    def agglomerative_sup_res(self, df, clusters=4, plot=True):
        date = df.index
        print(df)
        # Reset_Index for merging
        df.reset_index(inplace=True)

        # Build rolling stats
        # This is basically just flattening the graph out to 90 hour points instead of a data point every hour
        mx = df.High.rolling(self.duration).max().rename('waves')
        mn = df.Low.rolling(self.duration).min().rename('waves')

        print(mx)
        print(mn)

        # This is making the one column series above into two, -1 for low data points and +1 for high to identify them
        mx_waves = pd.concat([mx,pd.Series(np.zeros(len(mx))+1)],axis = 1)
        mn_waves = pd.concat([mn,pd.Series(np.zeros(len(mn))+-1)],axis = 1)    

        W = mx_waves.append(mn_waves).sort_index().dropna()
        """
        3523  181.330002  1.0
        3524  167.720093 -1.0
        3524  181.330002  1.0
        3525  181.330002  1.0
        3525  167.720093 -1.0
        """
        X = np.concatenate((W.waves.values.reshape(-1,1),
                    (np.zeros(len(W))+1).reshape(-1,1)), axis = 1 )
        # Clustering Algo
        cluster = AgglomerativeClustering(n_clusters=clusters,
                affinity='euclidean', linkage='ward')
        cluster.fit_predict(X)
        W['clusters'] = cluster.labels_

        # Get index of the max wave for each cluster
        W2 = W.loc[W.groupby('clusters')['waves'].idxmax()]

        print(W2)
        df.index = date 
        # Plotting
        if plot == True:
            mpf.plot(df, type='candle', style='yahoo', volume=True,
                    hlines=dict(hlines=W2.waves.to_list(), colors='g', linestyle='dashed', alpha=0.7),figsize= (20, 10))
       
        W2.waves.drop_duplicates(keep='first', inplace=True)
   
        return W2.reset_index().waves

        # kmeans = KMeans(n_clusters= kn.knee).fit(X.reshape(-1,1))
        # c = kmeans.predict(X.reshape(-1,1))
        # minmax = []
        # for i in range(kn.knee):
        #     minmax.append([-np.inf,np.inf])
        # for i in range(len(X)):
        #     cluster = c[i]
        #     if X[i] > minmax[cluster][0]:
        #         minmax[cluster][0] = X[i]
        #     if X[i] < minmax[cluster][1]:
        #         minmax[cluster][1] = X[i]

    def bearish_fractal(self, df):
        #method 1: fractal candlestick pattern
        # determine bullish fractal 
        def is_support(df,i):  
            cond1 = df['Low'][i] < df['Low'][i-1]   
            cond2 = df['Low'][i] < df['Low'][i+1]   
            cond3 = df['Low'][i+1] < df['Low'][i+2]   
            cond4 = df['Low'][i-1] < df['Low'][i-2]  
            return (cond1 and cond2 and cond3 and cond4) 
        # determine bearish fractal
        def is_resistance(df,i):  
            cond1 = df['High'][i] > df['High'][i-1]   
            cond2 = df['High'][i] > df['High'][i+1]   
            cond3 = df['High'][i+1] > df['High'][i+2]   
            cond4 = df['High'][i-1] > df['High'][i-2]  
            return (cond1 and cond2 and cond3 and cond4)
        # to make sure the new level area does not exist already
        def is_far_from_level(value, levels, df):    
            ave =  np.mean(df['High'] - df['Low'])    
            return np.sum([abs(value-level)<ave for _,level in levels])==0

        # a list to store resistance and support levels
        levels = []
        for i in range(2, df.shape[0] - 2):  
            if is_support(df, i):    
                low = df['Low'][i]    
                if is_far_from_level(low, levels, df):      
                    levels.append((i, low))  
            elif is_resistance(df, i):    
                high = df['High'][i]    
                if is_far_from_level(high, levels, df):      
                    levels.append((i, high))

    def window_shifting(self, df):
        #method 2: window shifting method
        #using the same symbol as the first example above
        symbol = 'COST'
        df = get_stock_price(symbol)
        pivots = []
        max_list = []
        min_list = []
        for i in range(5, len(df)-5):
            # taking a window of 9 candles
            high_range = df['High'][i-5:i+4]
            current_max = high_range.max()
            # if we find a new maximum value, empty the max_list 
            if current_max not in max_list:
                max_list = []
            max_list.append(current_max)
            # if the maximum value remains the same after shifting 5 times
            if len(max_list)==5 and is_far_from_level(current_max,pivots,df):
                pivots.append((high_range.idxmax(), current_max))
                
            low_range = df['Low'][i-5:i+5]
            current_min = low_range.min()
            if current_min not in min_list:
                min_list = []
            min_list.append(current_min)
            if len(min_list)==5 and is_far_from_level(current_min,pivots,df):
                pivots.append((low_range.idxmin(), current_min))
        # plot_all(pivots, df)                

if __name__ == "__main__":
    hp = SupRes()
    hp.run()