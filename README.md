# Sequences
Fibonacci and other sequences

# fibonaccy.py

Iterative Functions O(n):
 - fibo(N)        Gets the Nth fibonacci number 
 - fiboN(V)       Counts fibonacci numbers <= V
 - sumFibo(N)     returns the sum of the first N fibonacci numbers
 - sumEvenFibo(N) returns the sum of the first N EVEN fibonacci numbers

Iterative Functions with Exponential progress O(logN) or O(LogN^2):
 - fastFibo(N)        Gets the Nth fibonacci number
 - fastSumFibo(N)     returns the sum of the first N fiboinacci numbers O(LogN^2)
 - fastEvenFibo(N)    Gets the Nth EVEN fibonacci number (every third number is even)
 - fastSumEvenFibo(N) returns the sum of the first N EVEN fibonacci numbers O(LogN^2)
 
 Sequence Class: Generalized fibonacci sequences (including variants)
 - Optimized calculation in O(logN) or O(logN2)
 - Customizable base and factors:  f(n) = f(n-2)*F0 = f(n-1)*f1,  f(1) = B0,  f(2) = B1
 - Access values by index, ranges and slices
 - Iterators to stride through values, including infinite iterator
 - sum and count functions
 - selection/count of values within ranges
 
 NOTE: Optimizations only work when f0=1. I'm still working out the math for f0>1.
 
