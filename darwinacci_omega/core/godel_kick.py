def godel_kick(ind, rng, severity=0.35, new_genes=2):
    # perturba fortemente + injeta "axiomas"
    for _ in range(new_genes):
        ind["axiom_"+str(rng.randint(1,9))] = rng.gauss(0.0, severity*2)
    for k in list(ind.keys()):
        if rng.random()<0.4: ind[k] += rng.gauss(0.0, severity)
    return ind