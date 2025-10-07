import sys, random
from omega_ext.core.bridge import DarwinOmegaBridge
from omega_ext.plugins.adapter_darwin import autodetect

def main():
    init_fn, eval_fn = autodetect()
    engine = DarwinOmegaBridge(init_genome_fn=init_fn, eval_fn=eval_fn,
                               seed=123, max_cycles=7,
                               thresholds={"ece_max":0.10,"rho_bias_max":1.05,"rho_max":0.99},
                               canary_fn=None, breed_fn=None)
    engine.run(max_cycles=7)

if __name__=="__main__":
    sys.exit(main())
