# Nth Fibonacci number (iterative)
def fibo(N):
    a = b = 1
    for _ in range(1,N): a,b = b,a+b
    return a,b

# find starting values (greater than zero)
# from two sequenctial values and factors
from math import gcd
def findBase(f0,f1,a,b):
    c = f0*a+f1*b
    while a>0:
        a,b,c = (b-f1*a)//f0,a,b
    return b,c

"""
   a*f0 + b*f1
   (a+b)*f0 + b*f1-b*f0
   (a+b)*f0 + b*(f1-f0) = c
"""
# Determines factors and base from 4 consecutive numbers of the sequence
# (math)
#
# returns A,B,f(1),f(2)  where: f(n) = A*f(n-2) + B*f(n+1)
#
# e.g.    isfibo(2, 8, 34, 144) --> (1, 4, 2, 8)T
#
#            meaning: f(n)= 1*f(n-2) + 4*f(n-1),  f(1)=2, f(2)=8
#
#            Sequence(1,4,2,8) if the sequence of even fibonacci numbers
#
def isFibo(a,b,c,d):
    n  = (a*d - b*c)
    m  = (a*c - b**2)
    if m == 0 or n%m : return None
    f1 = n//m
    n,m  = (c-f1*b),a
    if m == 0: n,m = (d-f1*c),b    
    if n%a: return None
    f0 = n//a
    a,b = findBase(f0,f1,a,b)
    return f0,f1,a,b

# find the factors and base from 3 consecutive terms
# (iterative)
#
def findFibo(a,b,c):
    m = (c % a)
    for k in range(m,c+1,b):
        f1 = (k-m)//b
        f0 = (c - f1*b)//a
        if f0<=0: break
        if a*f0+b*f1 != c: continue
        a,b = findBase(f0,f1,a,b)
        return f0,f1,a,b
    

        
# infinite iterator of fibonacci numbers
#
#  starting from zero:  0,1,1,2,3,5,8,...
#
# example use: sum of first 10 even fibonacci:
#
#   sum(islice(iFibo(),3,10*3,3)) == 257114
#
def iFibo(): 
    a,b = 0,1
    while True:
        yield a
        a,b = b,a+b

# List of N fibonacci numbers
def listFibo(N):
    result = [1,1][:N]
    a = b = 1
    for _ in range(2,N):
        a,b = b,a+b
        result.append(b)
    return result

# Index of fibonacci number
def fiboN(f):
    N = 0
    a,b = 0,1
    while a < f:
        a,b = b,a+b
        N+=1
    return N

# sum of firt N fibonacci numbers (iterative)
def sumFibo(N):
    result = min(2,N)
    a = b = 1
    for _ in range(2,N):        
        a,b = b,a+b
        result += b        
    return result

# sum of the first N even fibonacci numbers iterative
def sumEvenFibo(N):
    total = 0
    a,b = 2,3
    for _ in range(N):
        total += a
        #print(a)
        a,b = a+2*b,2*a+3*b
    return total

# Nth Fibonacci number (exponential iterations) O(log(N)) time
def fastFibo(N):
    a,b   = 1,1
    f0,f1 = 0,1
    r,s   = (1,1) if N&1 else (0,1)
    N   //=2
    while N > 0:
        a,b   = f0*a+f1*b, f0*b+f1*(a+b)
        f0,f1 = b-a,a
        if N&1: r,s = f0*r+f1*s, f0*s+f1*(r+s)
        N //= 2        
    return r,s

# sum of first N fibonacci numbers (exponential recursive) O(Log(N)^2) time
def fastSumFibo(N,rec=False):
    if N < 2: return [(0,1),(1,1)][N] if rec else [0,1][N]
    m     = N//2
    f0,f1 = fastFibo(m-1)
    a,b   = fastSumFibo(m,True)
    r,s   = a + f0*a+f1*b, b + f0*b+f1*(a+b)
    if N&1:
        d0,d1 = fastFibo(N)
        r,s = r+d0, s+d1
    return (r,s) if rec else r

# Nth Even Fibonacci number (based on the O(log(N)) function)
def fastEvenFibo(N):
    a,b = fastFibo(3*N)
    return a,a+2*b

# sum of the first N even fibinacci numbers O(log(N)^2) time

def fastSumEvenFibo(N,rec=False):
    if N < 2: return [(0,2),(2,8)][N] if rec else [0,2][N]
    m     = N//2
    f0,f1 = fastEvenFibo(m-1) 
    p0,p1 = f0//2,f1//2
    a,b   = fastSumEvenFibo(m,True)
    r     =  a + p0*a+p1*b
    s     =  b + p0*b+p1*(a+4*b)
    if N&1:
        d0,d1 = fastEvenFibo(N)
        r,s = r+d0, s+d1
    return (r,s) if rec else r

"""
[2, 8, 34]
144, 610, 2584, 10946, 46368, 196418, 832040]

"""


# Sequence: Generic class for all fibonacci like sequences (of 2 terms)
# ---------
# * All functions are optimized to respond in O(logN) time
# * Sequence values are (infinitely) long integers
#
# CREATING A SEQUENCE
# -------------------
# ** Sequence parameter must produce an increasing sequence
#    because internal optimizations rely on this caracteristic
#
# f = Sequence()         Creates a basic fibonacci sequence
# f = Sequence(2,9)      Sets base for sequence 2,9,11,20,...
# f = Sequence(2,8,1,4)  Sets base and factors 2,8,34,144,...
#                        f(n) = f(n-2)*1 + f(n-1)*4  (even fibinacci)
# FUNCTIONS
# ---------
#
# based on indexes in the sequence
# --------------------------------
# NOTE: indexing is one-based and ranges are always inclusive
#
# f.next(5,8)   --> 13           next value (int)
# f.next(5,8,3) --> [13,21,34]   next values (list)
#
# f.previous(5,8)    --> 3       previous value (int)
# f.previous(5,8,3)  --> [1,2,3] previous values (list)
#
# f[6]      --> 8         returns nth number in the sequence
# f[:10]    --> 1,1,2,... iterator for firt 10 values
# f[3:6]    --> 2,3,5,8   iterator for values in range (inclusive)
# f[3:12:3] --> 2,8,34    iterator for values in striding range
# f[1::2]   --> 1,2,5,... infinite iterator (i.e. no stop specified) 
#
# f.list(3,6)    --> [2,3,5,8] same as subscript bu returns a list
# f.list(3,12,3) --> [2,8,34]  same as subscript bu returns a list
# f.sum(10)      --> 143       sum of first 10  values
# f.sum(10,20)   --> 17567     sum values from 10th to 20th inclusively
#                              NOTE: sum() responds in O(logN^2) time
#
# based on sequence values:
# ------------------------
#
# f.before(100)      --> 89   highest value before 100
# f.after(100)       --> 144  lowest value after 100 
# f.inRange(50,144)  --> [55, 89, 144] values between 50 and 144
# f.count(100)       --> 11    number of values from 1 to 100 (inclusively)
# f.count(3,100)     --> 8     number of values fro 3 to 100 (inclusively)
# f.has(144)         --> True  Indicates if value is present in sequence
# f.sumRange(50,144) --> 288   Sum of values in the range
#                              like: sum(f.inRange(50,144)) but faster
# 
from math import log
class Sequence:

    def __init__(self,factor0=1,factor1=1,base0=1,base1=1):
        self.f0 = factor0
        self.f1 = factor1
        self.b0 = base0
        self.b1 = base1
        self.factors = [(0,1),(factor0,factor1)] # powers of 2 (0,1,...)

    def next(self,a=0, b=None, count=None):
        result = []
        steps  = count or 1
        if b is None:
            a,b = self.b0,self.b1
            result=[a,b]
            steps -= 2
        for _ in range(steps):
            a,b = b, self.f0*a + self.f1*b
            result.append(b)
        return result[:count] if count else result[0]

    def previous(self,a,b,count=None):
        result = []
        for _ in range(count or 1):
            a,b = (b - self.f1*a)//self.f0,a
            result.insert(0,a)
        return result if count else result[0]

    def __getitem__(self,index):
        if isinstance(index,slice):
            def generate():
                step  = index.step or 1
                m0,m1 = self.stepFactors(step)
                a,b   = self.nth(index.start or 1,count=2)
                n     = index.start or 1
                while index.stop is None or n*step <= index.stop*step:                    
                    yield a
                    n += step
                    if self.f0 == 1:
                        a,b = m0*a+m1*b,m0*b+m1*self.next(a,b)
                    else:
                        a,b = self.nthFrom(a,b,step+1)
                return
            return generate()
        return self.nth(index)

    def list(self,start,stop=None,step=None):
        if stop == None: start,stop = 1,start
        return list(self[start:stop:step])

    def inRange(self,start,stop=None,step=None):
        if stop == None: start,stop = 1,start
        startN = self.count(start)
        stopN  = self.count(stop)
        rangeN = range(start,stop+1,step or 1)
        return [ n for n in self[startN:stopN+1] if n in rangeN ]

    def has(self,N):
        c = self.count(N)
        return N == self[c]

    def after(self,N):
        c = self.count(N)
        return self[c+1]

    def before(self,N):
        c = self.count(N)
        value = self[c]
        return value if value < N else self[c-1]

    def binFactors(self,p):
        if p < len(self.factors): return self.factors[p]
        F0,B0  = self.f1,self.b0
        while p >= len(self.factors):
           f0,f1 = self.factors[-1]
           a,b   = self.b0,self.b1
           a,b   = f0*a+f1*b,f0*b+f1*self.next(a,b)
           a,b,c = self.previous(a,b,2)+[a]
           f0,f1 = (f0*a+f1*b)//B0, (f0*b+f1*c)//B0
           self.factors.append((f0,f1))            
        return f0,f1

    def stepFactors(self,N):
        r,s = 1,0
        p   = -1
        while N > 0:
            pBit,N = N&1,N//2
            p     += 1
            if not pBit: continue
            f0,f1 = self.binFactors(p)
            if s==0: r,s = f0,f1
            else:    r,s = f0*r+f1*s, f0*s+f1*self.next(r,s)
        return r,s        

    def nthFrom(self,a,b,N):
        if N==0: return self.previous(a,b),a
        if N==1: return a,b
        if self.f0 != 1:
            for _ in range(N-1):
                a,b=b,self.next(a,b)
            return a,b
        c = self.next(a,b)
        if N==2: return b,c
        f0,f1 = self.stepFactors(N-1)        
        return f0*a+f1*b, f0*b+f1*c
    
    def nth(self,N,count=None):
        r,s = self.nthFrom(self.b0,self.b1,N)
        if count is None: return r
        if count < 3    : return [r,s][:count]
        return [r,s]+self.next(r,s,count-2)
        
    def count(self,start,stop=None):
        if stop is None: start,stop = 0,start
        def getCount(N):
            lo,hi  = 0,int(log(N+1,2))*2
            while hi >= lo:
                mid = (hi+lo)//2
                x   = self[mid]
                if  x > N : hi = mid - 1
                elif x < N: lo = mid + 1
                else: return mid
            return lo-1
        return getCount(stop)-getCount(start)
             
    def sum(self,start,stop=None):
        if stop is None: start,stop = 0,start
        if self.f0 != 1: return sum(self[start:stop])
        
        def accum(N):
            if N < 1 : return 0, self.b0
            if N == 1: return self.b0, self.b1
            m     = N//2
            a,b   = accum(m)
            f0,f1 = self.stepFactors(m)
            r,s   = a+f0*a+f1*b, b+f0*b+f1*self.next(a,b)
            if N&1:
                d0,d1 = self.nth(N,2)
                r,s   = r+d0, s+d1
            return (r,s)
        return accum(stop)[0]-accum(start-1)[0]

    def sumRange(self,start,stop=None):
        if stop is None: start,stop=0,start
        startN = self.count(start)
        if self[startN]<start: startN += 1
        stopN  = self.count(stop)
        return self.sum(startN,stopN)
    
if __name__ == "__main__":
    f = Sequence()
    f2 = Sequence(1,4,2,8) 
    from itertools import accumulate,islice

    #############  TESTS ###################
       
    f20 = [fibo(i)[0] for i in range(1,21)]
    f30 = [fibo(i)[0] for i in range(1,31)]

    """
    f(n)   = f(n-1)+f(n-2)
    f(n+1) = f(n)+f(n-1)
    f(n+2) = f(n) + f(n+1)
    f(n+3) = f(n+1) + f(n+2)
    f(n+4) = 2f(n) + 3f(n+1)
    f(n+5) = 3f(n) + 5f(n+1)
    f(n+6) = 5f(n) + 8f(n+1)
    [1,   1,   2,    3,    5,    8,    13]     ∑ = 33  ==> 33*8+53*13 = 953
    [1,   2,    3,    5,    8,   13,   21]     ∑ = 53  ==> 53*8+86*13 = 1542 

    [21,  34,  55,   89,   144,  233,  377]    ∑ = 953 
    [34,  55,   89,   144,  233,  377, 610]    ∑ = 1542 
    ...13*f(n-2)+21*f(n-1)...
    [610, 987, 1597, 2584, 4181, 6765, 10946]  ∑ = 27670

    f0,f1 = 0,1
    [1,1] ∑ = 2  
    [1,2] ∑ = 3  

    f0,f1 = 1,2 ==> stride by 2
    [2,3] ∑ = 5  [1,1,2,3] T = 7  
    [3,5] ∑ = 8  [1,2,3,5] T = 11

    f0,f1 = 2,3 ==> stride by 4
    [5, 8,  13, 21] ∑ = 47  [1,1,2,3,5,8,13,21]  = 54 
    [8, 13, 21, 34] ∑ = 76  [1,2,3,5,8,13,21,34] = 87

    f0,f1 = 13,21
    [34, 55, 89,  144, 233, 377, 610, 987]  ∑ = 2529  Total = 2583
    [55, 89, 144, 233, 377, 610, 987, 1597] ∑ = 4092  Total = 4179

    Even Fibo : f(n) = 2*f(n-2) + 3*f(n-1)
    [2, 8]    ∑ = 10
    [34, 144] ∑ = 178


    [2,   8,   34]
    [8,  34,  144]

    [144, 610, 2584]  f(n)*4 + 17*f(n+1)

    f(n+3) = f(n)   + 2*f(n+1)  ==> f(n+1) = (f(n+3)-f(n))/2
    f(n+4) = 2*f(n) + 3*f(n+1)
    f(n+5) = 3*f(n) + 5*f(n+1)
    f(n+6) = 5*f(n) + 8*f(n+1)

    f(n+6) = 5*f(n) + 8*(f(n+3)-f(n))/2
    f(n+6) = 5*f(n) + 4*f(n+3)-4*f(n)
    f(n+6) = f(n) + 4*f(n+3)

    EVEN Fibo: f(n+2) = f(n) + 4*f(n+1)

    f(n+1) = f(n)*4 + f(n-1)
    f(n+2) = f(n) + 4*f(n+1)
    f(n+3) = 4*f(n) + 17*f(n+1)
    """


    from timeit import timeit
    count = 1
    N = 40000

    t= timeit(lambda:Sequence().sum(N), number=count) 
    print(f"{count} x Sequence().sum({N})",t)

    t= timeit(lambda:fastSumFibo(N), number=count) 
    print(f"{count} x fastSumFibo({N})",t)

    t= timeit(lambda:sumFibo(N), number=count) 
    print(f"{count} x sumFibo({N})",t)

    t= timeit(lambda:Sequence(1,4,2,8).sum(N), number=count) 
    print(f"{count} x Sequence(1,4,2,8).sum({N})",t)

    t= timeit(lambda:fastSumEvenFibo(N), number=count) 
    print(f"{count} x fastSumEvenFibo({N})",t)

    t= timeit(lambda:sumEvenFibo(N), number=count) 
    print(f"{count} x sumEvenFibo({N})",t)

    t= timeit(lambda:fastFibo(N), number=count) 
    print(f"{count} x fastFibo({N})",t)

    t= timeit(lambda:fibo(N), number=count) 
    print(f"{count} x fibo({N})",t)
