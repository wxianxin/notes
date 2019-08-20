python has no constant variables. We can acheive it by setting a class for constant.
python tuple: like a list, difference is you cannot modify tuple
python data type: complex: a=2.5+6.3j; a.real; a.imag; a.conjugate
python data type: string: 'hello'; "hello"; '''hello''';
python data type: string: mapping date type: dict
python computation: divisible \\
python expression "3 < 4 < 7" is ok.
r: r'hello': cancel escape character
u: u'hello': convert into Unicode

"range()" in python3 is "xrange()"in python2
"list(range())" in python3 == "range()" in python2

while xx:
    xxxx
else:
    xxxx
#

DocString
def fcn():
    'explain what does this function do'

print fcn.__doc__
#check DocString of the function

def fcn(x=True)
#set default value of a argument

def fcn(y, x=True)
#OK
def fcn(x=True, y)
#Not OK
#default argument(s) must follow non-default argument(s)

#keyword argument
fcn(x = 1, y = False)

#python pass function as argument
def fcn(x):
    return(x+x)
def fcn2(fcn,y):
    print fcn(y)

#anonymous function
def fcn(x,y):
    return x + y

fcn = lambda x, y : x + y

->fcn(1,2)=3

# files
file_obj = open(filename, mode='r', buffering=-1)
# r a r+ a+
file_obj.read()
file_obj.read(size)
file_obj.write()
file_obj.seek()
file_obj.close()

# Dictionary
my_dict = {}.fromkeys(("key_1","Key_2","Key_3","Key_4"), 3000)

my_dict = dict(zip(list_1, list_2))

del my_dict
#delete my_dict
my_dict.keys()
my_dict.values()

my_dict.update(my_dict_2)

my_dict.clear()
my_dict.get()
my_dict.items()
my_dict.iter()
my_dict.setdefault()
my_dict.fromkeys()
my_dict.keys()
my_dict.pop()
my_dict.update()

my_dict=dict((('a',1),('b',2)))
a = my_dict
my_dict=dict((('a',1),('b',3)))
a
# >>>{'b': 3, 'a': 3}
# just a reference

def fcn(args1, *args, **kwargs):
    print(args1, argst, argsd)
fcn('a','b','c','d',a1=1,a2=2,a3=3)

# *args for list arguments
# **kwargs for dictionary arguments

# sets — Unordered collections of unique elements
names = ['a','b','c','d','a','b']
namesSet = set(names)
namesSet
set(['a','b','c','d'])

aSet = set('hello')
fSet = frozenset('hello')

# set comparison

# in      # belong
# not in  # not belong
# ==      # =
# !=      # !=
# <       # ⊂ proper subset
# <=      # ⊆ subset
# >       # ⊃ proper superset
# >=      # ⊇ superset

# & A∩B intersection
# | A∪B union
# - (- or \) relative complement; A-B=(belongs to A but not b)
# ^ ∆ symmetric difference; A^B=(elements that only belong to either A or B)

my_set.issubset(bSet)

# Scipy
from scipy import linalg

linalg.det(arr)

# ndarray
from numpy import *
my_array = array([(1,2,3),(4,5,6)])
my_array = arange(1,5,0.5)

my_array.sum()
my_array.sum(axis = 0)
my_array.sum(axis = 1)

my_array = array([(1,2,3,4),(5,6,7,8)])
my_array[:1]
# ??????
my_array = array([1,2,3])
my_arr_1 = array([2,4,6])
my_arr_2 = array([7,8,9])
where(my_array > 2, my_arr_1, my_arr_2)

def fun(x,y):
    return (x+1)*(y+1)
arr = fromfunction(fun,(9,9)) # starts from 0

# ufunc
help(ufunc)

# Pandas
from pandas import Series
# Series is 1-dimensional data with index and order
my_series = pd.Series([1,2.0,'a'])
my_series = pd.Series(['a','b','c'], index = [1,2,3])
my_series.index
my_series.values

my_series = pd.Series(['a','b','c'], index = [1,2,3])
my_index = [1,3]
new_series = pd.Series(my_series, index = my_index)

my_series_1 = pd.Series(['a','b','c'], index = [1,2,3])
my_series_2 = pd.Series(['b','c','d'], index = [2,3,4])

# Data Alignment
my_series_1 + my_series_2
# only take '2' and '3'

# Series name property (column names)
my_series.name = 'letters'
my_series.index.name = 'numbers'

# DataFrame, 2-dimensional table data structure (A collection of Series that share same index)
data = {'name':['wang','zhang','li'], 'age':[18,19,20]}
frame = pd.DataFrame(data)

frame.wang
frame.age
frame['name']

frame.ix[1]
#index
frame['name']='admin'
del frame['pay']

frame.index.name = "No."

# nltk (natural language tool kit)

# matplotlib
from matplotlib.finance import quotes_historical_yahoo
from datatime import date
import pandas as pd
today = date.today()
start = (today.year - 1, today.month, today.day)
quotes = quotes_historical_yahoo('AAPL', start, today)
df = pd.DataFrame(quotes)
print df

# add column names
fields=['date','open','close','high','low','volume']
quotesdf = pd.DataFrame(quotes, columns=fields)
print quotesdf

# add index that starts from one rather than 0
quotesdf = pd.DataFrame(quotes, index=range(1,len(quotes)+1), columns=fields)

# convert numerical date into readable date
from datetime import date
day1 = date.fromordinal(735190)
day2 = date.fromordinal(735551)

# create data series
import pandas as pd
dates=pd.date_range("20161001", periods=7)

# show data
xxdf.index
xxdf.columns
xxdf.values
xxdf.describe

# show the first 5 items or last 5
xxdf.head(5)
xxdf.tail(5)

# data selection
# select crow(s)
quotesdf[u'2013-12-25':u'2014-01-01']
# select column(s)
quotesdf['code']
quotesdf.code

# tag in dataframe
xxdf.loc[1:5,]
# loc : location
xxdf.loc[:,['code','lasttrade']]
xxdf.loc[1:5,['code','lasttrade']]

xxdf.iloc[1:5,[0,2]]

quotesdf[quotesdf.index >= u'2016-01-01']
quotesdf[(quotesdf.index >= u'2016-01-01')and(quotesdf.close >= 95)]

# mean
xxdf.mean(columns = 'lasttrade')

# up days
len(quotesdf[quotesdf.close > quotesdf.open])

# sort
xxdf.sort(columns = 'lasttrade')

# groupby
xxdf.groupby('month').count()
xxdf.groupby('month').sum().volume
# waste computation effort
g = xxdf.groupby('month')
g_volume = g['volume']
g_volume.sum()

# Merge
a
b.append(a)
pd.concat([a,b])
pd.comcat([1,b], ignore_index = True)
# join(Just like SQL, based on key)
pd.merge(xxdf, yydf, on='code')