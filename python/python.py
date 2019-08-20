# Author: Steven Wang	Date:20151221
# python 3

{} #curly braces
no datatype
use white space to structure(no ";", use indent)
"True" "False"
comments:
#comments

"""
docstring
"""
# These strings can be extracted automatically through the __doc__ member of the object and are used by pydoc.

################################################################################
# string
print(text)
print("a string")
print("a" + "big" + "hat") #concatenate
print("Line", lineNum, " :", line, end="")
# do not end with giving a new line
input("Please input:") #input function with hint message
variable.isalpha() #False if there is non-letter characters

"cat"[0] == c
"cat"[2] == t
"cat"[1:2] == at
"cat"[0:len("cat")] == cat

len(variable)
variable.lower()
variable.upper()
variable.capitalize()
variable.lstrip()
variable.rstrip()
variable.strip()
str(variable)    #explicit string conversion
#methods that use dot notation only work with strings, or other words; len() and str() can work on other datatypes

my_list = ['a','b','c']
#turn list into string
' '.join[my_list] #=> a b c
"---".join[my_list] #=> a---b---c
# '' and "" are interchangeable

################################################################################
# formatting
print("{} is {} {}".format("He", "a", "boy"))
print("The time is {:02d}:{:02d}".format(9,7))

print ("Hello, %s") %(variable)     # %s, placeholder
print ("Hello, %s and %s") %(variable1,variable2)
# python 2

'Alpha' "Bravo" "Thailand's people" #no need for backslash

################################################################################
# data & time:
from datetime import datetime #inport "datetime" module
now = datetime.now()
now.year now.month now.day now.hour now.minute now.second

################################################################################
# logicl operators
boolean operator:
not, and, or
-----------> priority

if __:
elif __: #else if, elif is only checked if the original "if" statement is False
else __:

	"**" #to the power of

5/2 = 2
float(5)/ 2 = 2.5
"/" #continuation charater, next line is following up

max()
min()
abs()
type()

sum(variable) # summation

################################################################################
# LOOP:
for key in dict_name:
    print(dict_name[key])
#CAUTION: dictionary is unordered

enumerate: # return index
    for index, item in enumerate(list):
    print(index,) # output the index
    print(item) # output the list element

zip: #create pairs of elements, stops at the end of shorter list
    for a,b in zip(list_a, list_b):
        print(a, b)


################################################################################
# function:
1.header: 
def function_name(arguments):
# One can also pass function as argument.
2. optional comment
3.  body    #indented

# Function annotation
# Only for annotation use, no effect on code at all.
def my_func(name: str, age: int, weight: float) -> str:
    return "{} is {} and weighs{}".format(name, age, weight)

# Dynamic function generation
def generate_function(num):
    def power_of_num(value):
        pass
        print("blaaaa")
    return power_of_num

desired_funcion = generate_function(3)
desired_funcion(5)

# List of functions
list_of_functions = [func_1, func_2]
list_of_functions[1](args)

import module
1.generic import:  import math
	call: math.sqrt()
2.function import: from math import sqrt  #pulling in just a single function from a module
    call: sqrt()
3.universal import: from math import *    #CAUTION: fill the program with many variable and function names
	call: sqrt()
# best to stick with module name

import math
everything = dir(math)
print(everything)
#show everything available in math

list
list_name = [item_1, item_2]
empty_list = []

list_name[0]    #call by index
list_name.append()
len(list_name)
list_name[1:3]   #return the 2nd to the 4th
#think of string as a list of characters
list_name[:3]
list_name[6:]

list_name.index() # search the first corresponding element
list_name.insert('index', value)

for variable in list_name:
    pass
else:
    # (not compulsory) if no break occurred

list_name.sort()   #modify the list so it is in order

################################################################################
# dictionary, like map in c++ ?
d = {'key1':1, 'key2':2, 'key3':3}
dict_name[new_key] = new value
empty_dict = {}
my_var = my_dict.get('Steve', 'Steve not in the dictionary')
# return the corresponding value if 'Steve' is in the dict; Otherwise return the second parameter value.
del dict_name[key_name]
list_name.remove()


list_name.pop(index) # return value and remove
list_name.remove(item) # find the item and remove
del(list_name[index]) # like 'pop' but no return

range # a shortcut to generate a list
range(6) #=> [0,1,2,3,4,5]  range(stop)
range(1,6) #=> [1,2,3,4,5]  range(start,stop)
range(1,6,3) #=> [1,4]  range(start,stop,step)

for item in my_list:
	print(item)

for i in range(len(my_list)):
	print(my_list[i])

["0"] * 5 #=> ['0','0','0','0','0']

################################################################################
form random import randint
int(raw_imput("hint")) #raw_input always returns string type

import random
rand_num = random.randrange(1, 51)

my_list = ["Peter", "Paul","Amy"]
random.choice(my_list)

################################################################################

if x not in range(8) or y not in range(5)
break # to get out of the loop

print(a,b)

while loop_condition:

while/else: #else executes as long as while loop condition is evaluated False; if the loop exits as the result of break, [else] will not be executed

for/else: # [else]excutes after [for] only if [for] ends normally(not because of [break])

print(char,) #[,] character after [print char] keeps next [print] keep its output in the same line

for key in my_dict: # you get key


"*" * 4 # return "****"

my_string.split() #split the string into a list of strings

if a not in b:

sorted(list) # return the ordered list

sum[my_list] #returns sum of all list elements

!!! # always devide by 2.0
float(len(my_list))

#dictionary is unordered key/value pairs
print(my_dict.items()) #=> print all keys and its values in random order
my_dict.keys() # returns an array of dictionary's keys
my_dict.items() # returns an array of dictionary's values

# tuple: a sequence of immutable objectl; a Tuple is like a list, but their values can't be changed
# tuples are surrounded by "()"
my_tuple = (1,2,3,4)

x, y = y, x
# This is doable in python. Because on the RHS it first generate a tuple of y and x, then the LHS unpack the tuple.
print(item,) # the trailing comma keeps printing on the same line

#list comprehension: for/in if
my_list = [ i for i in range(51) if i % 2 == 0]
my_list = [x*2 for x in range(1,6)]
my_list = [x*2 for x in range(1,6) if (x*3) % 3 == 0]

#list slicing:
[start:end:step] #start is inclusive; end is exclusive
#omitting indices:
Python will pick up default:
#default start index is 0
#default ending index is the end
#default step is 1
my_list = [1,2,3,4,5]
my_list[3:] #=>[4,5]
my_list[:2] #=>[1,2]
my_list[::2] #=>[1,3,5]
my_list[::-1] #=>[5,4,3,2,1] {reversing a list)

################################################################################
#functional programming(anonymous function):
# lambda arg1, arg2,...:expression use the args
sum = lambda x, y : x + y

################################################################################
#bitwise operator:
#In python: to write numbers in bianry format by starting the number with [0b].
	0b100
	0b1101
bin() # takes an integer as input and returns its binary
oct() # like bin()
hex() # like bin()

int("42") #=> 42
int("110") #=> 6
int("0b110") #=> 6

# floor division in python is integer division;
# in python 3: 5/2 = 2.5; 5//2 = 2

0b0001 << 2 #=> 0b0100
0b0100 >> 2 #=> 0b0001

#bitwise compare
a = 0b0101010
b = 0b0001111

& #AND
0&0=0
0&1=0
1&0=0
1&1=1
a&b = 0b1010

| #OR
0|0=0
0|1=1
1|0=1
1|1=1
a|b = 0b0101111

^ #XOR: exclusive or operator; for a bit, either of the two is 1 but not both
0^0=0
0^1=1
1^0=1
1^1=0

a^b = 0b0100101

~ #NOT opeartor: add one the number and then make it negative

# bit mask
# a bit mask can help you turn specific bits on, turn others off, or just collect data from an integer

data = 0b010011010
mask = 0b000010000
print(data&mask)

& # using[&] to turn on bits
| # using[|] to turn a corresponding bit on if it is off and leave it on if it is already on
^ # using[^] to flip bits
<< >> # using [<<] and [>>] to slide mask into place

################################################################################
# class:
# when a class has its own functions, those functions are called methods
# ["Eric"] and [my_dict] are instances of the [str] and [dict] class.
class NewClass(object):	# in the parenthesis is the class from which the new class inherits
						# [object] is the simplest, most basic class
						# by convention, user-defined Python class names start with a capital letter
    """This is my new class"""      # docstring
	def __init__(self, age, name):	# [__intit__()] exist by default
		self.age = age				# Python will use the 1st parameter that [__init__()] receives to refer to the object being created.
		self.name = name			# Python convention: it's overwhelming common to use [self] as the 1st parameter in [__init__()]

	def method_1(self):	# for any method in a class, you need to provide [self] as the 1st argument
		pass						# [pass] statement is a null operation; nothing happens when it executes; works as a placeholder
	
	def method_2(self,a,b):
		pass
	
	member_variable_1 = True
	member_variable_2 = "test"

# Class instantiation
my_object = NewClass(18, "python")	# parameter list starts from the parameter after [self] (you don't need to give [self])
print(my_object.name)				# dot notation: to access attributes of out objects

# class scope:
# needs further attention
# static field(variable)    #like [public] in c++
# global field(variable)	
# member field(variable)	#like [protected] in c++
# instance veriable	#like [private] in c++

# Inheritance: "is-a" relationship
superclass	subclass
baseclass	derivedclass
parentclass	childclass

class BaseClass(object):
	def __init__(self,a,b):
		pass

class DerivedClass(BaseClass):

# override(overload): # in a new Derived class, override a function is to define a function with the same function name

super # [super] directly access the attributes or methods of a superclass
	def method(self,a,b):
		return super(DerivedClass, self).method(a,b)
	
# 2 ways of initialization
#1.
	def __init__(self,a,b,c,d):
		BaseClass.__init__(self,a,b)
		self.c = c
		self.d = d

#2.
	def __init__(self,a,b,c,d):
		super(DerivedClass, self).__init__(a,b)	# no [self] in the 2nd parenthesis
		self.c = c
		self.d = d

__repr__() # a built-in methods: representation, tell Pyhon how to represent an object of our class
	def __repr__(self):
		return 'DerivedClass(a=%s, b=%s)' % (self.a, self.b)

DerivedClass #=> "DerivedClass(a=1, b=2)" (assume a="1" and b="2")

#__repr__ goal is to be unambiguous
#__str__ goal is to be readable
#Container’s __str__ uses contained objects’ __repr__


################################################################################
# Magic methods
class Time(hour, minute, second):
    
    def __init__(self):
        self.hour = hour
        self.minute = minute
        self.second = second

    def __add__(self, other_time):
        pass

def main():
    time1 + time2

# Static method(public method)
class sum:

    @staticmethod
    def getSum(*args):
        sum = 0
        for i in args:
            sum += i

        return sum

def main():
    Sum.getSum(1,2,3)

main()

################################################################################
# File I/O:
my_file = open("output.txt", "w")	# open "output.txt" in "w" mode("w" stands for "write"), store the result of this operation in a file object [my_file]
# "w": write only mode
# "r": read only mode
# "r+": read and write mode
# "a": append mode
# default without mode will be read

my_file.write("abcd\n")
my_file.read()
my_file.close() # During I/O process, data is buffered. Python doesn't flush the buffer, that is, write data to the file until it's sure you've done writing. One way to do this is to close the file.
# 1. read() reads everything into 1 string
# 2. readline() reads everything including the first newline
# 3. readlines() returns a list of every line which includes each newline

#automatically close the files: file objects contain a pair of built-in methods: 
__enter__()
__exit__()	#when [__exit__()] is invoked, the file will be closed

# use [with][as] to invoke [__exit__()]
with open("text.txt","w") as my_file:
	my_file.write("abcd")
# By using with, it will automatically close the file stream.

# eg.
with open("mydata.txt", encoding="utf-8") as myFile:
# encoding="...."

#check if the file is closed:
print(my_file.closed) #Python file objects have a [closed] attribute, [True] for closed, [False] for not closed

################################################################################
# try
try:

except ValueError:
# EG. Int expected while having a string
except IndexError:
# EG. list index exceeds defined limit
except:
# Any other error
else:
# if no except encountered
finally:
# always be executed; executed before the code is stopped


# Make custom exceptions
class MiracleError(Exception):

    __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
        

################################################################################

# Global variable
var_1 = 1
def fcn():
    global var_1 =2


# os module
import os
os.getcwd()
os.rename("my_file.txt","my_file_2.txt")
os.mkdir("mydir")
os.chdir("..")
os.rmdir("mydir")

# unpacking argument lists/spalt
def sumAll(A, *args):
    sum = 0
    for _ in args:
        sum += 1

    return sum
sumAll(1,2,3,4,5)
# Number of arguments is not limited.

# unpacking named arguments
def my_func(**kwargs):
    for name, value in kwargs.items():
        print(name, value)

# You can use these along with named arguments too. The explicit arguments get values first and then everything else is passed to *args and **kwargs. The named arguments come first in the list. 
def multi_func(arg1, *args, **kwargs):
	print(arg1)
	for arg in args:
		print(arg)

	for key, value in kwargs:
		print(key, value)

x = 1
eval("x + 1")
# eval() interprets a string as code



# Built-in function map
map(function, iterable, ...)
map(my_func, [0,1,2,3])
map((lambda x: x*3), [0,1,2,3])
map((lambda x: x*y), [0,1,2,3],[3,2,1,0])

# filter
# filter for which function returns true.
filter(function, iterable)
my_list = range(16)
filter(lambda x: x%3 == 0, my_list)


# Generator expressions
double = (x * 2 for x in range(10))
print("Double :", next(double))
print("Double :", next(double))

# List comprehension
print([x for x in range(1, 11)])
print([x for x in range(1, 11) if x % 2 != 0])
print([x * y for x in range(1, 3) for y in range(4, 8)])
# use list comprehension inside list comprehension
# eg.1
print([x for x in [i * 2 for i in range(10)] if x % 8 == 0])
# eg.2
import random
print([x for x in [random.randint(1, 1001) for i in range(50)] if x % 9 == 0])
# eg.3
multiList = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
print([col[1] for col in multiList])
print([multiList[i][i] for i in range(len(multiList))])

################################################################################
# Iterables
# Iterable is a list of values
# eg.1
sampStr = iter("Sample")
 
print("Char :", next(sampStr))
print("Char :", next(sampStr))
# eg.2
class FibGenerator:
    def __init__(self):
        self.first = 0
        self.second = 1
 
    def __iter__(self):
        return self
 
    def __next__(self):
        fibNum = self.first + self.second
        self.first = self.second
        self.second = fibNum
        return fibNum
 
fibSeq = FibGenerator()
for i in range(10):
    print("Fib :", next(fibSeq))

################################################################################
# Generator functions (yield)
# A generator function returns a result generator when called. They can be suspended and resumed during execution of your program to create results over time rather than all at once
# We use generators when we want to big result set, but we don't want to slow down the program by creating it all at one time

def isprime(num):
    # This for loop cycles through primes from 2 to
    # the value to check
    for i in range(2, num):
 
        # If any division has no remainder we know it
        # isn't a prime number
        if (num % i) == 0:
            return False
    return True
 
# This is the generator
def gen_primes(max_number):
 
    # This for loop cycles through primes from 2 to
    # the maximum value requested
    for num1 in range(2, max_number):
 
        if isprime(num1):
 
            # yield is what makes this a generator
            # When called by next it will return the
            # next result
            yield num1
 
# Create a reference to the generator
prime = gen_primes(50)

################################################################################
# python scope
def scope_test():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_local()
    print("After local assignmentmagic:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam)

scope_test()
print("In global scope:", spam)

# output of previous sample code
# After local assignment: test spam
# After nonlocal assignment: nonlocal spam
# After global assignment: nonlocal spam
# In global scope: global spam


# to avoid unintentional execution of the code in a module when importing this module
if __name__ == '__main__':
	main()

# sys.argv is a list in Python, which contains the command-line arguments passed to the script.
from sys import argv
print(sys.argv)

variable type annotation

->
# PEP-0484
# python >= 3.5: -> is used to indicate the type that the function returns.

def my_func(arg: str="pubg") -> my_type:



EOF