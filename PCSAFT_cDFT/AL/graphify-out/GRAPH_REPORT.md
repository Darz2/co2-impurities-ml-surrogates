# Graph Report - .  (2026-04-22)

## Corpus Check
- Corpus is ~17,545 words - fits in a single context window. You may not need a graph.

## Summary
- 11 nodes · 10 edges · 4 communities detected
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Kernel AL Algorithm|Kernel AL Algorithm]]
- [[_COMMUNITY_Selection & Sampling|Selection & Sampling]]
- [[_COMMUNITY_PC-SAFT Bubble Point|PC-SAFT Bubble Point]]
- [[_COMMUNITY_Pool Generation|Pool Generation]]

## God Nodes (most connected - your core abstractions)
1. `compute_U_T()` - 3 edges
2. `run_algorithm1()` - 3 edges
3. `stratified_select()` - 2 edges
4. `compute_pbubble_single()` - 2 edges
5. `Kernel uncertainty: U_T = diag(K_TT - K_TS K_SS⁻¹ K_ST)     X_T : (|T|, d) — can` - 1 edges
6. `Returns a list of pool indices in the order they were added to S.     Starts wit` - 1 edges
7. `Proportional CO2-bin × mixture_size stratified sample of size n.` - 1 edges
8. `PC-SAFT bubble-point. Returns P in bar, or NaN on failure.` - 1 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities

### Community 0 - "Kernel AL Algorithm"
Cohesion: 0.5
Nodes (4): compute_U_T(), Kernel uncertainty: U_T = diag(K_TT - K_TS K_SS⁻¹ K_ST)     X_T : (|T|, d) — can, Returns a list of pool indices in the order they were added to S.     Starts wit, run_algorithm1()

### Community 1 - "Selection & Sampling"
Cohesion: 0.5
Nodes (2): Proportional CO2-bin × mixture_size stratified sample of size n., stratified_select()

### Community 2 - "PC-SAFT Bubble Point"
Cohesion: 1.0
Nodes (2): compute_pbubble_single(), PC-SAFT bubble-point. Returns P in bar, or NaN on failure.

### Community 3 - "Pool Generation"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **4 isolated node(s):** `Kernel uncertainty: U_T = diag(K_TT - K_TS K_SS⁻¹ K_ST)     X_T : (|T|, d) — can`, `Returns a list of pool indices in the order they were added to S.     Starts wit`, `Proportional CO2-bin × mixture_size stratified sample of size n.`, `PC-SAFT bubble-point. Returns P in bar, or NaN on failure.`
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `PC-SAFT Bubble Point`** (2 nodes): `compute_pbubble_single()`, `PC-SAFT bubble-point. Returns P in bar, or NaN on failure.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pool Generation`** (1 nodes): `01_generate_pool.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `compute_U_T()` connect `Kernel AL Algorithm` to `Selection & Sampling`?**
  _High betweenness centrality (0.178) - this node is a cross-community bridge._
- **Why does `run_algorithm1()` connect `Kernel AL Algorithm` to `Selection & Sampling`?**
  _High betweenness centrality (0.178) - this node is a cross-community bridge._
- **What connects `Kernel uncertainty: U_T = diag(K_TT - K_TS K_SS⁻¹ K_ST)     X_T : (|T|, d) — can`, `Returns a list of pool indices in the order they were added to S.     Starts wit`, `Proportional CO2-bin × mixture_size stratified sample of size n.` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._