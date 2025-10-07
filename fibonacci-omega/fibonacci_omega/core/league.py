from typing import Optional, Tuple
from enum import Enum
import random
from dataclasses import dataclass
from .population import Individual

class ChallengerState(Enum):
    NONE = 0
    SHADOW = 1
    CANARY = 2

@dataclass
class PromotionProof:
    challenger_score: float
    champion_score: float
    shadow_cycles_passed: int
    canary_cycles_passed: int
    timestamp: str

class League:
    def __init__(self, shadow_cycles: int = 2, canary_cycles: int = 3):
        self.champion: Optional[Individual] = None
        self.challenger: Optional[Individual] = None
        self.challenger_state: ChallengerState = ChallengerState.NONE
        self.challenger_progress: int = 0
        self.SHADOW_CYCLES_REQUIRED = shadow_cycles
        self.CANARY_CYCLES_REQUIRED = canary_cycles
        
    def _reset_challenger(self):
        self.challenger = None
        self.challenger_state = ChallengerState.NONE
        self.challenger_progress = 0

    def process_candidate(self, candidate: Individual, rng: random.Random, promotion_pressure: float) -> Tuple[bool, Optional[PromotionProof]]:
        if self.champion is None:
            if candidate.score > 0.1:
                self.champion = candidate.clone()
                print(f"[LEAGUE] Primeiro campeão coroado com score: {self.champion.score:.4f}")
                return True, None
            return False, None

        if self.challenger is not None:
            return self._advance_challenger(rng)

        required_score = self.champion.score * promotion_pressure
        if candidate.score > required_score:
            self.challenger = candidate.clone()
            self.challenger_state = ChallengerState.SHADOW
            self.challenger_progress = 0
            print(f"[LEAGUE] Novo desafiante aceito (score {candidate.score:.4f} > {required_score:.4f} @ {promotion_pressure:.2f}x pressure). Entrando em modo SHADOW.")
        return False, None

    def _advance_challenger(self, rng: random.Random) -> Tuple[bool, Optional[PromotionProof]]:
        if self.challenger is None: return False, None
        instability_chance = 0.05 if self.challenger_state == ChallengerState.SHADOW else 0.1
        if rng.random() < instability_chance:
            print(f"[LEAGUE] Desafiante falhou no teste de estabilidade em modo {self.challenger_state.name}. Descartado.")
            self._reset_challenger()
            return False, None
        
        self.challenger_progress += 1
        print(f"[LEAGUE] Desafiante passou no teste de {self.challenger_state.name} (progresso: {self.challenger_progress}/{self._get_required_cycles()})")

        if self.challenger_state == ChallengerState.SHADOW and self.challenger_progress >= self.SHADOW_CYCLES_REQUIRED:
            self.challenger_state = ChallengerState.CANARY
            self.challenger_progress = 0
            print("[LEAGUE] Desafiante promovido para modo CANARY.")
        elif self.challenger_state == ChallengerState.CANARY and self.challenger_progress >= self.CANARY_CYCLES_REQUIRED:
            return self._promote_challenger()
        return False, None

    def _promote_challenger(self) -> Tuple[bool, PromotionProof]:
        from datetime import datetime
        print(f"[LEAGUE] PROMOÇÃO! Novo campeão com score {self.challenger.score:.4f} destrona o antigo com score {self.champion.score:.4f}.")
        proof = PromotionProof(
            challenger_score=self.challenger.score, champion_score=self.champion.score,
            shadow_cycles_passed=self.SHADOW_CYCLES_REQUIRED, canary_cycles_passed=self.CANARY_CYCLES_REQUIRED,
            timestamp=datetime.utcnow().isoformat()
        )
        self.champion = self.challenger.clone()
        self._reset_challenger()
        return True, proof

    def _get_required_cycles(self) -> int:
        return self.SHADOW_CYCLES_REQUIRED if self.challenger_state == ChallengerState.SHADOW else self.CANARY_CYCLES_REQUIRED
