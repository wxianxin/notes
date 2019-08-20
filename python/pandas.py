# 20170119
########################################
# Data Structure
# pandas has 3 major data structures
# pd.Series    pd.DataFrame    pd.Panel(less used)
# Series is a one-dimensional labeled array like 'ndarray' in numpy
# DataFrame is a 2-dimensional labeled data structure like a spreadsheet or SQL table
# pandas has powerful time series tools (see documents)
########################################

import pickle
import quandl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from statistics import mean

api_key = 'vV25yLapyX2eA4w-48f8'

dates = pd.date_range('20130101', periods=6)
# Create date sequence
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
# index is like row name, columns is column names
df.index
df.columns
df.values
# Display the index, columns, and the underlying numpy data
df.describe()
df.T
# transpose
df.mean()
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())
# Applying functions to the data

df.loc(0)
# Select 0th row

################################
# pandas IO

df = pd.read_csv("ZILL-Z07310_MLP.csv")
print(df.head())

df.set_index('Date', inplace=True)
df.to_csv('newcsv2.csv')

df = pd.read_csv("ZILL-Z07310_MLP.csv", index_col = 0)
print(df.head())

# Reference a specific column
print(df['Visitors'])
print(df.Visitors)

# convert to list
print(df.Visitors.tolist())
print(np.array(df[['Bounce_Rate','Visitors']]))
df2 = pd.DataFrame(np.array(df[['Bounce_Rate','Visitors']]))
print(df2)

HPI_data['New Column'] = HPI_data['NY'] * 2
# Add new column to dataframe

df.columns = ['JerseyCity']
# rename column

df.to_csv('newcsv3.csv', header=False)
df = pd.read_csv('newcsv4.csv', names=['Date','JerseyCity'], index_col=0)

# use pandas as converter of different file formats
df.to_html('newhtml.html')

df.rename(columns={'JersetCity':'07310_HPI'})
print(df.head())

################################

# Concatenation & Append & Merge
df1 = pd.DataFrame({'HPI':[80,85,88,85], 'Int_rate':[2, 3, 2, 2], 'US_GDP_Thousands':[50, 55, 65, 55]}, index = [2001, 2002, 2003, 2004])
df2 = pd.DataFrame({'HPI':[80,85,88,85], 'Int_rate':[2, 3, 2, 2], 'US_GDP_Thousands':[50, 55, 65, 55]}, index = [2005, 2006, 2007, 2008])
df3 = pd.DataFrame({'HPI':[80,85,88,85], 'Int_rate':[2, 3, 2, 2], 'Unemployment':[7, 8, 9, 6], 'Low_tier_HPI':[50, 52, 50, 53]}, index = [2001, 2002, 2003, 2004])

concat=pd.concat([df1,df2,df3])
print(concat)

df4 = df1.append(df3)
print(df4)

merged = pd.merge(df1, df3, on='Year', how = 'left')
# 'left': merge according to the left table
# 'right': merge according to the right table
# 'outer': union of the key, reserve every key of the joined tables
# 'inner': (DEFALUT) intersection of keys of the joined tables
print(merged)

################################

def grab_initial_states_data():

    fifty_states = pd.read_html('https://simple.wikipedia.org/wiki/U.S._postal_abbreviations')
    # print(fifty_states)
    # we get a list of dataframes and we need to extract the information we need
    fifty_states = fifty_states[1]

    main_df = pd.DataFrame()

    for abbv in fifty_states[0][1:]:
        print("FMAC/HPI_" + str(abbv))
        query="FMAC/HPI_"+str(abbv)
        df = quandl.get(query, authtoken=api_key)
        df.columns = [str(abbv)]

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    # pickle

    pickle_out = open('fiddy_states.pickle', 'wb')   # write binary
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

################################

pickle_in = open('fiddy_states.pickle','rb')
HPI_data = pickle.load(pickle_in)
print(HPI_data)

# pandas own version
HPI_data.to_pickle('pickle.pickle')
HPI_data2 = pd.read_pickle('pickle.pickle')
print(HPI_data2)

################################

def HPI_Benchmark():
    df = Quandl.get("FMAC/HPI_USA", authtoken=api_key)
    df["United States"] = (df["United States"]-df["United States"][0]) / df["United States"][0] * 100.0
    return df

def mortgage_30y():
    df = Quandl.get("FMAC/MORTG", trim_start="1975-01-01", authtoken=api_key)
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('1D')
    df=df.resample('M') # If you do month directly, you would get NAN, because of lack of data
    return df

# grab_initial_states_data()
HPI_data = pd.read_pickle('fiddy_states.pickle')

HPI_data.plot()
plt.savefig('fig.png')


# Correlation
HPI_States_Correlation = HPI_data.corr()
print(HPI_States_Correlation)



# resampling
NY_1yr = HPI_data['NY'].resample('A', how='mean')
# by default it is 'mean', others like 'ohlc'; 'A' means yearly
# for more, see help documents
print(NY_1yr.head())

NY_1yr.plot()
plt.savefig('fig2.png')

################################

# Handling missing data
# pandas primarily uses the value np.nan to represent missing data.
HPI_data.dropna()
# remove any row with NAN; returns a new dataframe
HPI_data.dropna(inplace=True)

HPI_data.dropna(how='all', inplace=True)
# remove rows which are all NAN
# thres(threshold): pass a certain number of NANs then we drop it.
# for more, check for help documents

HPI_data.fillna(method='ffill', inplace=True)
# ffill : forward fill, take previous data
# bfill : back fill
HPI_data.fillna(value=-9999, inplace=True)
# for more, check for help documents

print(HPI_data.isnull().values.sum())
# show how many NAN in the dataframe


# Percent change
HPI_data.pct_change()


# Rolling statistics
HPI_data['NY12MA'] = pd.rolling_mean(HPI_data['NY'], 12)
HPI_data['NY12STD'] = pd.rolling_std(HPI_data['NY'], 12)
NY_NJ_12corr = pd.rolling_corr(HPI_data['NY'], HPI_data['NJ'], 12)

# rolling_xxx, for more see help documents
# rolling_apply, Generic, you can write your own code


# Boolean Indexing(Comparison)
bridge_height = {'meters':[10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}
df = pd.DataFrame(bridge_height)

print(df)
df['STD'] = pd.rolling_std(df['meters'], 2)
df_std = df.describe()['meters']['std']
df = df[ (df['STD'] < df_std) ]
print(df)

################################

M30 = mortgage_30y()
HPI_Bench = HPI_Benchmark()
M30.columns=['M30']
HPI = HPI_Bench.join(m30)
rint(HPI.head())

# shift
HPI_data.shift(-1)

# data slicing
df["adult"][df.age >= 18] = 1

# count & proportion
df["adult"].value_counts()
df["adult"].value_counts(normalize=True)