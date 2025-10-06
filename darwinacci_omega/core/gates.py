class SigmaGuard:
    def __init__(self, ece_max=0.10, rho_bias_max=1.05, rho_max=0.99):
        self.ece_max=ece_max; self.rho_bias_max=rho_bias_max; self.rho_max=rho_max
    def evaluate(self, m:dict)->tuple[bool,dict]:
        ece=float(m.get("ece",0.0)); rho_bias=float(m.get("rho_bias",1.0)); rho=float(m.get("rho",0.5))
        eco_ok=bool(m.get("eco_ok",True)); consent=bool(m.get("consent",True))
        ok=(ece<=self.ece_max) and (rho_bias<=self.rho_bias_max) and (rho<self.rho_max) and eco_ok and consent
        return ok, {"ece":ece,"rho_bias":rho_bias,"rho":rho,"eco_ok":float(eco_ok),"consent":float(consent)}