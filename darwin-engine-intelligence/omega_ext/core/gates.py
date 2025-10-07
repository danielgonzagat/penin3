from typing import Dict, Any, Tuple
class SigmaGuard:
    def __init__(self, th:Dict[str,float]): self.th=th
    @staticmethod
    def _ece(probs, labels, bins:int=10)->float:
        if not probs or not labels: return 0.0
        bsize=1.0/bins; gap=0.0
        for i in range(bins):
            lo,hi=i*bsize,(i+1)*bsize
            buck=[(p,y) for p,y in zip(probs,labels) if lo<=p<hi]
            if not buck: continue
            conf=sum(p for p,_ in buck)/len(buck)
            acc=sum(1 for p,y in buck if (p>=0.5)==bool(y))/len(buck)
            gap += abs(acc-conf)*(len(buck)/len(probs))
        return gap
    def evaluate(self, m:Dict[str,Any])->Tuple[bool,Dict[str,float]]:
        emax=float(self.th.get("ece_max",0.1)); bmax=float(self.th.get("rho_bias_max",1.05)); rmax=float(self.th.get("rho_max",0.99))
        ece=self._ece(m.get("probs"), m.get("labels")); rho_bias=float(m.get("rho_bias",1.0)); rho=float(m.get("rho",0.5))
        eco_ok=bool(m.get("eco_ok",True)); consent=bool(m.get("consent",True))
        ok=(ece<=emax) and (rho_bias<=bmax) and (rho<rmax) and eco_ok and consent
        return ok, {"ece":ece,"rho_bias":rho,"rho":rho,"eco_ok":float(eco_ok),"consent":float(consent)}
