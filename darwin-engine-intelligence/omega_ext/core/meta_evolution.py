from dataclasses import dataclass
@dataclass
class MetaParams: mut_rate:float; cx_rate:float; mut_scale:float; pop_size:int
class MetaEvolution:
    def __init__(self, mut_rate=0.08, cx_rate=0.75, mut_scale=0.2, pop_size=40):
        self.params = MetaParams(mut_rate, cx_rate, mut_scale, pop_size)
    def step(self, progress_delta:float, novelty_mean:float, f_step:int):
        if progress_delta < 1e-4:  # estagnou
            self.params.mut_rate  = min(0.45, self.params.mut_rate*1.06)
            self.params.mut_scale = min(0.60, self.params.mut_scale*1.06)
        else:  # progrediu
            self.params.mut_rate  = max(0.02, self.params.mut_rate*0.98)
            self.params.mut_scale = max(0.05, self.params.mut_scale*0.98)
        self.params.cx_rate = max(0.2, min(0.95, self.params.cx_rate + (0.006 if (f_step%2==0) else -0.006)))
        if novelty_mean > 0.6: self.params.pop_size=min(160, self.params.pop_size+3)
        else:                  self.params.pop_size=max(24,  self.params.pop_size-1)
        return self.params
