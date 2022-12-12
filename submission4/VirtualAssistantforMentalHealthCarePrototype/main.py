import re
import random
import pandas
import numpy

pandas.set_option("mode.chained_assignment", None)

data = pandas.read_csv("Data/mentalhealthquestionandanswer.csv")

head = data.head(10)
print(head)





