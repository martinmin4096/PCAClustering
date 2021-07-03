from math import *
import sys
import math

def erf(z,x):
   series = []
   series2 = []
   for n in xrange(sys.maxint):
      series.append(((-1)**n + z**(2*n+1))/(math.factorial(n)*(2*n+1)))
      sumseries = sum(series)
      series2.append(((-1)**n + x**(2*n+1))/(math.factorial(n)*(2*n+1)))
      sumseries2 = sum(series2)
      if ((2/sqrt(pi) * sumseries2)-(2/sqrt(pi) * sumseries)/(2/sqrt(pi) * sumseries2)) < .00001:
         print(n)
         break

