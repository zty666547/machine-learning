"""Guaranteed-candidate global edit-distance screening for fixed-length DNA."""
from __future__ import annotations
from collections import defaultdict

try:
    import edlib
except ImportError:
    edlib = None


def bounded_edit_distance(a: str, b: str, limit: int, *, force_python: bool = False) -> int | None:
    """Return exact Levenshtein distance if <= limit, otherwise None."""
    if limit < 0: raise ValueError("limit must be nonnegative")
    if abs(len(a) - len(b)) > limit: return None
    if edlib is not None and not force_python:
        distance = edlib.align(a, b, mode="NW", task="distance", k=limit)["editDistance"]
        return None if distance < 0 or distance > limit else int(distance)
    n, m = len(a), len(b)
    inf = limit + n + m + 1
    previous = list(range(m + 1))
    for i in range(1, n + 1):
        current = [inf] * (m + 1)
        if i <= limit: current[0] = i
        for j in range(max(1, i - limit), min(m, i + limit) + 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a[i - 1] != b[j - 1]))
        previous = current
    return previous[m] if previous[m] <= limit else None


class EditDistanceIndex:
    """Find all <=k references using k+1 disjoint reference blocks and bounded shifts.

    With at most k edits, at least one of k+1 reference blocks is untouched and
    appears in the query displaced by at most k bases. The final dynamic
    programming check decides membership; this is not a heuristic seed filter.
    """
    def __init__(self, references: list[str], limit: int):
        if not references or limit < 0:
            raise ValueError("Expected nonempty references and a nonnegative limit")
        self.references = [s.upper() for s in references]
        self.length = len(self.references[0])
        if limit >= self.length or any(len(s) != self.length or set(s) - set("ACGT") for s in self.references):
            raise ValueError("References must be equal-length A/C/G/T with limit < length")
        self.limit = limit
        self.blocks = [(i * self.length // (limit + 1), (i + 1) * self.length // (limit + 1))
                       for i in range(limit + 1)]
        self.index = defaultdict(set)
        for i, reference in enumerate(self.references):
            for block, (a, b) in enumerate(self.blocks):
                self.index[(block, reference[a:b])].add(i)

    def find(self, query: str) -> dict[int, int]:
        query = query.upper()
        if len(query) != self.length or set(query) - set("ACGT"):
            raise ValueError("Query must match reference length and contain A/C/G/T")
        possible = set()
        for block, (a, b) in enumerate(self.blocks):
            for delta in range(-self.limit, self.limit + 1):
                start, end = a + delta, b + delta
                if 0 <= start < end <= self.length:
                    possible.update(self.index.get((block, query[start:end]), ()))
        hits = {}
        for i in possible:
            distance = bounded_edit_distance(query, self.references[i], self.limit)
            if distance is not None: hits[i] = distance
        return hits
