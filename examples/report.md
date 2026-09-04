# Attractor Analysis — capital_berlin_multilingual

## Data
model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
d_model=5120, n_layers=64, n_heads=40
prompts: 12 (multilingual), expected answer: Berlin
point cloud per layer: 81 token hidden states x 5120 dims
metric=euclidean, max_dim=2, eps_max=1.5 x mean-nn

## Persistent features across depth (attractors)
| layer | n | nsimp | eps_max | b0_essential | b0_long_lived | b0_persist | b1_essential | b1_long_lived | b1_persist | b2_essential | b2_long_lived | b2_persist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| layer_000 | 81 | 22586 | 2.224 | 1 | 72 | 134.5 | 0 | 0 | 0 | 19263 | 1087 | 1584 |
| layer_016 | 81 | 26047 | 275.3 | 12 | 77 | 1.442e+04 | 0 | 0 | 156.2 | 22775 | 7795 | 5.289e+05 |
| layer_032 | 81 | 10228 | 294.3 | 12 | 77 | 1.585e+04 | 0 | 1 | 265.1 | 8136 | 1141 | 1.373e+05 |
| layer_048 | 81 | 2009 | 334.2 | 12 | 77 | 1.893e+04 | 7 | 0 | 203.3 | 1088 | 279 | 2.793e+04 |
| layer_064 | 81 | 594 | 149 | 16 | 77 | 8926 | 2 | 1 | 83.06 | 200 | 166 | 1.69e+04 |

## Findings
b1_essential peaks at layer_048 with 7 persistent feature(s).
b2_essential peaks at layer_016 with 22775 persistent feature(s).

## Betti function, layer_000
| eps | b0 | b1 | b2 |
| --- | --- | --- | --- |
| 1.335 | 72 | 0 | 0 |
| 1.416 | 72 | 0 | 0 |
| 1.496 | 71 | 0 | 0 |
| 1.577 | 69 | 0 | 0 |
| 1.658 | 64 | 0 | 0 |
| 1.739 | 55 | 0 | 1 |
| 1.82 | 44 | 0 | 11 |
| 1.901 | 34 | 0 | 164 |
| 1.982 | 17 | 0 | 732 |
| 2.063 | 8 | 0 | 2599 |
| 2.143 | 1 | 0 | 7483 |
| 2.224 | 1 | 0 | 19263 |
