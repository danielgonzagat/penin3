import math, random
PHI = (1 + 5 ** 0.5) / 2
GOLDEN_ANGLE = 2*math.pi*(1 - 1/PHI)  # ~2.399963...

def fib(n:int)->int:
    if n<=2: return 1
    a,b=1,1
    for _ in range(3,n+1): a,b=b,a+b
    return b

def fib_seq(k:int)->list:
    s=[]; a,b=1,1
    for _ in range(max(0,k)): s.append(a); a,b=b,a+b
    return s

def clamp(x,a,b): return a if x<a else (b if x>b else x)

_rng = random.Random(1234)
def rng()->random.Random: return _rng