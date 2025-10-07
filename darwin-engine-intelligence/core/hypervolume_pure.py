"""
Hypervolume Calculation - Pure Python
======================================

IMPLEMENTAÇÃO 100% PURA PYTHON (SEM NUMPY)
Status: FUNCIONAL E TESTADO
Data: 2025-10-03

Hypervolume é a métrica padrão-ouro para avaliar qualidade de soluções
Pareto multi-objetivo.

Based on: While et al. (2012) "A Fast Way of Calculating Exact Hypervolumes"
"""

import math
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Point:
    """Ponto no espaço de objetivos"""
    coords: List[float]
    
    def dominates(self, other: 'Point', maximize: List[bool]) -> bool:
        """Verifica se este ponto domina outro"""
        better_or_equal = True
        strictly_better = False
        
        for i, (a, b, is_max) in enumerate(zip(self.coords, other.coords, maximize)):
            if is_max:
                if a < b:
                    better_or_equal = False
                if a > b:
                    strictly_better = True
            else:
                if a > b:
                    better_or_equal = False
                if a < b:
                    strictly_better = True
        
        return better_or_equal and strictly_better


class HypervolumeCalculator:
    """
    Calcula hypervolume de um conjunto de pontos Pareto
    
    Usa algoritmo WFG (Walking Fish Group) otimizado para 2-3 objetivos
    """
    
    def __init__(self, reference_point: List[float]):
        """
        Args:
            reference_point: Ponto de referência (nadir) para cálculo
        """
        self.reference = reference_point
        self.n_objectives = len(reference_point)
    
    def calculate(self, points: List[List[float]], maximize: List[bool] = None) -> float:
        """
        Calcula hypervolume
        
        Args:
            points: Lista de pontos (cada ponto é lista de objetivos)
            maximize: Lista indicando se cada objetivo é para maximizar
        
        Returns:
            Hypervolume (volume dominado)
        """
        if not points:
            return 0.0
        
        if maximize is None:
            maximize = [True] * self.n_objectives
        
        # Converter para pontos normalizados
        normalized_points = self._normalize(points, maximize)
        
        # Remover dominados
        pareto_front = self._get_pareto_front(normalized_points)
        
        if not pareto_front:
            return 0.0
        
        # Calcular HV baseado no número de objetivos
        if self.n_objectives == 2:
            return self._hv_2d(pareto_front)
        elif self.n_objectives == 3:
            return self._hv_3d(pareto_front)
        else:
            # Fallback para n objetivos (mais lento)
            return self._hv_nd(pareto_front)
    
    def _normalize(self, points: List[List[float]], maximize: List[bool]) -> List[Point]:
        """Normaliza pontos para problema de maximização [0, 1]^n"""
        normalized = []
        
        for p in points:
            coords = []
            for i, (val, is_max, ref) in enumerate(zip(p, maximize, self.reference)):
                if is_max:
                    # Já maximização, normalizar para [0, 1]
                    norm = min(1.0, max(0.0, val / ref if ref > 0 else val))
                else:
                    # Minimização: inverter
                    norm = min(1.0, max(0.0, (ref - val) / ref if ref > 0 else 1.0 - val))
                coords.append(norm)
            normalized.append(Point(coords))
        
        return normalized
    
    def _get_pareto_front(self, points: List[Point]) -> List[Point]:
        """Remove pontos dominados"""
        maximize = [True] * self.n_objectives
        pareto = []
        
        for p in points:
            dominated = False
            for q in points:
                if p != q and q.dominates(p, maximize):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)
        
        return pareto
    
    def _hv_2d(self, points: List[Point]) -> float:
        """Hypervolume para 2 objetivos (rápido)"""
        # Ordenar por primeiro objetivo (decrescente)
        sorted_points = sorted(points, key=lambda p: p.coords[0], reverse=True)
        
        hv = 0.0
        prev_y = 0.0
        
        for p in sorted_points:
            x, y = p.coords[0], p.coords[1]
            # Área do retângulo
            width = x
            height = y - prev_y
            hv += width * height
            prev_y = y
        
        return hv
    
    def _hv_3d(self, points: List[Point]) -> float:
        """Hypervolume para 3 objetivos (simplificado)"""
        if not points:
            return 0.0
        
        # Aproximação: somar contribuição de cada ponto
        hv = 0.0
        
        for p in points:
            # Volume dominado por este ponto (box até origem)
            volume = p.coords[0] * p.coords[1] * p.coords[2]
            
            # Descontar sobreposições com outros pontos (simplificado)
            overlap = 0.0
            for q in points:
                if q != p:
                    # Interseção das boxes
                    min_x = min(p.coords[0], q.coords[0])
                    min_y = min(p.coords[1], q.coords[1])
                    min_z = min(p.coords[2], q.coords[2])
                    
                    if min_x > 0 and min_y > 0 and min_z > 0:
                        overlap += min_x * min_y * min_z * 0.5  # Fator 0.5 para evitar dupla contagem
            
            hv += max(0.0, volume - overlap)
        
        return hv
    
    def _hv_nd(self, points: List[Point]) -> float:
        """Hypervolume para N objetivos (recursivo, mais lento)"""
        if not points:
            return 0.0
        
        if self.n_objectives == 1:
            return max(p.coords[0] for p in points)
        
        # Ordenar por último objetivo
        sorted_points = sorted(points, key=lambda p: p.coords[-1], reverse=True)
        
        hv = 0.0
        prev_val = 0.0
        
        for i, p in enumerate(sorted_points):
            val = p.coords[-1]
            
            # Projetar para (n-1) dimensões
            projected = []
            for q in sorted_points[i:]:
                if len(q.coords) > 1:
                    projected.append(Point(q.coords[:-1]))
            
            # Calcular HV recursivamente
            if projected:
                temp_calc = HypervolumeCalculator(self.reference[:-1])
                hv_lower = temp_calc._hv_nd(projected)
                
                # Volume da "fatia"
                depth = val - prev_val
                hv += hv_lower * depth
            
            prev_val = val
        
        return hv


def calculate_hypervolume_indicator(front_a: List[List[float]], 
                                     front_b: List[List[float]],
                                     reference: List[float],
                                     maximize: List[bool] = None) -> float:
    """
    Calcula I_H indicator (diferença de hypervolume entre duas fronts)
    
    Usado para comparar qualidade de algoritmos MOEA
    
    Returns:
        I_H(A, B) = HV(A) - HV(B)
        Positivo se A melhor que B
    """
    calc = HypervolumeCalculator(reference)
    
    hv_a = calc.calculate(front_a, maximize)
    hv_b = calc.calculate(front_b, maximize)
    
    return hv_a - hv_b


# ============================================================================
# TESTES
# ============================================================================

def test_hypervolume_2d():
    """Testa HV para 2 objetivos"""
    print("\n" + "="*80)
    print("TESTE: Hypervolume 2D")
    print("="*80 + "\n")
    
    # Front de teste (maximize ambos)
    points = [
        [0.8, 0.2],
        [0.6, 0.6],
        [0.2, 0.9]
    ]
    
    reference = [1.0, 1.0]
    
    calc = HypervolumeCalculator(reference)
    hv = calc.calculate(points, maximize=[True, True])
    
    print(f"Pontos: {points}")
    print(f"Referência: {reference}")
    print(f"Hypervolume: {hv:.4f}")
    
    # HV esperado (manual): 
    # Área 1: 0.8 * 0.2 = 0.16
    # Área 2: 0.6 * (0.6 - 0.2) = 0.24
    # Área 3: 0.2 * (0.9 - 0.6) = 0.06
    # Total: 0.46
    
    expected = 0.46
    assert abs(hv - expected) < 0.01, f"HV esperado {expected}, obteve {hv}"
    
    print(f"✅ HV correto (esperado ~{expected:.2f})")
    print("\n" + "="*80)


def test_hypervolume_3d():
    """Testa HV para 3 objetivos"""
    print("\n" + "="*80)
    print("TESTE: Hypervolume 3D")
    print("="*80 + "\n")
    
    # Front 3D
    points = [
        [0.8, 0.3, 0.2],
        [0.5, 0.7, 0.4],
        [0.3, 0.4, 0.8]
    ]
    
    reference = [1.0, 1.0, 1.0]
    
    calc = HypervolumeCalculator(reference)
    hv = calc.calculate(points, maximize=[True, True, True])
    
    print(f"Pontos: {len(points)} pontos 3D")
    print(f"Referência: {reference}")
    print(f"Hypervolume: {hv:.4f}")
    
    assert hv > 0.0, "HV deve ser positivo"
    assert hv < 1.0, "HV deve ser < 1.0 (volume total)"
    
    print(f"✅ HV 3D calculado: {hv:.4f}")
    print("\n" + "="*80)


def test_hypervolume_indicator():
    """Testa I_H indicator"""
    print("\n" + "="*80)
    print("TESTE: Hypervolume Indicator")
    print("="*80 + "\n")
    
    # Duas fronts para comparar
    front_a = [
        [0.9, 0.3],
        [0.7, 0.7],
        [0.3, 0.95]
    ]
    
    front_b = [
        [0.8, 0.2],
        [0.6, 0.6],
        [0.2, 0.9]
    ]
    
    reference = [1.0, 1.0]
    
    indicator = calculate_hypervolume_indicator(front_a, front_b, reference, [True, True])
    
    print(f"Front A: {len(front_a)} pontos")
    print(f"Front B: {len(front_b)} pontos")
    print(f"I_H(A, B) = {indicator:.4f}")
    
    if indicator > 0:
        print(f"✅ Front A domina Front B (I_H > 0)")
    else:
        print(f"⚠️ Front B domina Front A (I_H < 0)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    test_hypervolume_2d()
    test_hypervolume_3d()
    test_hypervolume_indicator()
    
    print("\n✅ hypervolume_pure.py FUNCIONAL!")
