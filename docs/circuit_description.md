# EMT Circuit — Biology & Parameters

## State vector: x = [miR200, mZEB, ZEB, SNAIL, mSNAIL, miR34, I]

| Index | Species | Role |
|-------|---------|------|
| 0 | miR200 | miRNA — represses ZEB (6 binding sites) |
| 1 | mZEB   | ZEB mRNA |
| 2 | ZEB    | EMT transcription factor |
| 3 | SNAIL  | EMT transcription factor |
| 4 | mSNAIL | SNAIL mRNA |
| 5 | miR34  | miRNA — represses SNAIL (2 binding sites) |
| 6 | I      | External signal (TGF-β proxy) |

## Tristability
Three coexisting steady states: E (Epithelial), M (Mesenchymal), Hybrid E/M.

## Diffusion coefficients (RD model)
D = 0.1 a.u. for all species; channel 6 (I) is pinned (D = 0).
