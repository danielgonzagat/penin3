from typing import List
import math
def euclidean(a:List[float], b:List[float])->float:
    n=min(len(a),len(b)); 
    if n==0: return 0.0
    s=0.0
    for i in range(n): d=(a[i]-b[i]); s+=d*d
    return math.sqrt(s)
class NoveltyArchive:
    def __init__(self,k:int=10,max_size:int=2000): self.k=k; self.max_size=max_size; self.data:List[List[float]]=[]
    def add(self,b:List[float]):
        if b: self.data.append(list(b))
        if len(self.data)>self.max_size: self.data.pop(0)
    def score(self,b:List[float])->float:
        if not self.data: return 0.0
        d=sorted(euclidean(b,x) for x in self.data)
        k=min(self.k,len(d)); 
        return sum(d[:k]) / float(max(1,k))
