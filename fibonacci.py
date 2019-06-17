# Nth Fibonacci number (iterative)
def fibo(N):
    a = b = 1
    for _ in range(1,N): a,b = b,a+b
    return a,b

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
    N = 1
    a = b = 1
    while a < f:
        a,b = b,a+b
        N+=1
    return N

# Nth Fibonacci number (exponential iterations) O(log(N)) time
def binFibo(N):
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

# sum of firt N fibonacci numbers (iterative)
def sumFibo(N):
    result = min(2,N)
    a = b = 1
    for _ in range(2,N):        
        a,b = b,a+b
        result += b        
    return result

# sum of first N fibonacci numbers (exponential recursive) O(Log(N)^2) time
def binSumFibo(N,rec=False):
    if N < 2: return [(0,1),(1,1)][N] if rec else [0,1][N]
    m     = N//2
    f0,f1 = binFibo(m-1)
    a,b   = binSumFibo(m,True)
    r,s   = a + f0*a+f1*b, b + f0*b+f1*(a+b)
    if N&1:
        d0,d1 = binFibo(N)
        r,s = r+d0, s+d1
    return (r,s) if rec else r


# Nth Even Fibonacci number (based on the O(log(N)) function)
def binEvenFibo(N):
    a,b = binFibo(3*N)
    return a,a+2*b

# sum of the first N even fibinacci numbers O(log(N)^2) time
def binSumEvenFibo(N,rec=False):
    if N < 2: return [(0,2),(2,8)][N] if rec else [0,2][N]
    m     = N//2
    f0,f1 = binEvenFibo(m-1)
    p0,p1 = f0//2,f1//2
    a,b   = binSumEvenFibo(m,True)
    r     =  a + p0*a+p1*b
    s     =  b + p0*b+p1*(a+4*b)
    if N&1:
        d0,d1 = binEvenFibo(N)
        r,s = r+d0, s+d1
    return (r,s) if rec else r

# sum of the first N even fibonacci numbers iterative
def sumEvenFibo(N):
    total = 0
    a,b = 2,3
    for _ in range(N):
        total += a
        #print(a)
        a,b = a+2*b,2*a+3*b
    return total

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
N = 4000000

##t= timeit(lambda:binSumFibo(N), number=count) 
##print(f"{count} x binSumFibo({N})",t)
##
##t= timeit(lambda:sumFibo(N), number=count) 
##print(f"{count} x sumFibo({N})",t)

t= timeit(lambda:binSumEvenFibo(N), number=count) 
print(f"{count} x binSumEvenFibo({N})",t)

t= timeit(lambda:sumEvenFibo(N), number=count) 
print(f"{count} x sumEvenFibo({N})",t)

##t= timeit(lambda:binFibo(N), number=count) 
##print(f"{count} x binFibo({N})",t)
##
##t= timeit(lambda:fibo(N), number=count) 
##print(f"{count} x fibo({N})",t)
