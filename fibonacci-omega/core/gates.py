import yaml
from typing import Dict, Any, Tuple, List, Optional

class Policy:
    def __init__(self, config: Dict[str, Any]):
        self.name = config["name"]
        self.type = config["type"]
        self.metric = config["metric"]
        self.condition = config.get("condition")
        self.value = config.get("value")
        self.must_be = config.get("must_be")
        self.fail_message = config["fail_message"]

    def evaluate(self, metrics: Dict[str, Any]) -> Optional[str]:
        metric_value = metrics.get(self.metric)
        if metric_value is None:
            return f"Métrica '{self.metric}' necessária para a política '{self.name}' não foi encontrada."
        if self.type == "threshold":
            if self.condition == "less_than_or_equal" and not (metric_value <= self.value): return self.fail_message
            if self.condition == "less_than" and not (metric_value < self.value): return self.fail_message
        elif self.type == "boolean":
            if metric_value != self.must_be: return self.fail_message
        return None

class SigmaGuard:
    def __init__(self, policy_file: str = "fibonacci_omega/configs/policies.yaml"):
        try:
            with open(policy_file, 'r') as f:
                config = yaml.safe_load(f)
            self.policies = [Policy(p_config) for p_config in config.get("policies", [])]
            print(f"[SigmaGuard] {len(self.policies)} políticas carregadas de '{policy_file}'.")
        except FileNotFoundError:
            print(f"[SigmaGuard] Aviso: Arquivo de políticas '{policy_file}' não encontrado. Operando sem gates de segurança.")
            self.policies = []

    def evaluate(self, metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        failures = [msg for policy in self.policies if (msg := policy.evaluate(metrics)) is not None]
        return not failures, failures
