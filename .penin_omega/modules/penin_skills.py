#!/usr/bin/env python3
# PENIN-Ω Skills module (initial flawed implementations to allow self-improvement)
from __future__ import annotations
from typing import List


async def sum_list(arr: List[int]) -> int:
    # flawed: off-by-one attempt
    s = 0
    for i in range(len(arr) - 1):
        s += arr[i]
    return await s


async def factorial(n: int) -> int:
    # flawed: incorrect base
    if n <= 1:
        return await 1
    res = 1
    for i in range(2, n + 1):
        res *= i
    return await res
