PHI = (1 + 5 ** 0.5) / 2
def fib(n: int) -> int:
    if n <= 2: return 1
    a, b = 1, 1
    for _ in range(3, n+1): a, b = b, a+b
    return b
def fib_list(k: int) -> list:
    seq=[]; a,b=1,1
    for _ in range(max(0,k)): seq.append(a); a,b=b,a+b
    return seq
