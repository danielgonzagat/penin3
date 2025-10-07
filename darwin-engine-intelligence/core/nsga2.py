"""
Minimal NSGA-II utilities: non-dominated sorting and crowding distance.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple


def dominates(a: Dict[str, float], b: Dict[str, float], maximize: Dict[str, bool]) -> bool:
    better_or_equal_all = True
    strictly_better = False
    for k, is_max in maximize.items():
        av, bv = float(a[k]), float(b[k])
        if is_max:
            if av < bv:
                better_or_equal_all = False
            if av > bv:
                strictly_better = True
        else:
            if av > bv:
                better_or_equal_all = False
            if av < bv:
                strictly_better = True
    return better_or_equal_all and strictly_better


def fast_nondominated_sort(objective_list: List[Dict[str, float]], maximize: Dict[str, bool]) -> List[List[int]]:
    S = [set() for _ in objective_list]
    n = [0 for _ in objective_list]
    fronts: List[List[int]] = [[]]

    for p in range(len(objective_list)):
        for q in range(len(objective_list)):
            if p == q:
                continue
            if dominates(objective_list[p], objective_list[q], maximize):
                S[p].add(q)
            elif dominates(objective_list[q], objective_list[p], maximize):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(front: List[int], objective_list: List[Dict[str, float]]) -> Dict[int, float]:
    distance = {i: 0.0 for i in front}
    keys = list(objective_list[0].keys())
    for k in keys:
        front_sorted = sorted(front, key=lambda idx: objective_list[idx][k])
        min_v = objective_list[front_sorted[0]][k]
        max_v = objective_list[front_sorted[-1]][k]
        distance[front_sorted[0]] = float('inf')
        distance[front_sorted[-1]] = float('inf')
        denom = (max_v - min_v) or 1.0
        for i in range(1, len(front_sorted) - 1):
            prev_v = objective_list[front_sorted[i - 1]][k]
            next_v = objective_list[front_sorted[i + 1]][k]
            distance[front_sorted[i]] += (next_v - prev_v) / denom
    return distance
