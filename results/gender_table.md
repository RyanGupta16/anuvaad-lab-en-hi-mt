| System                    |   Accuracy |   ΔG (M−F) |   ΔS (pro−anti) |   Ambiguous → masc % |
|:--------------------------|-----------:|-----------:|----------------:|---------------------:|
| Transformer (scratch)     |       50   |      100   |             0   |                100   |
| mT5-small (fine-tuned)    |       45.8 |       91.7 |             8.3 |                100   |
| LLaMA-3.2-1B (zero-shot)  |       83.3 |       16.7 |             0   |                 91.7 |
| LLaMA-3.2-1B (kNN 4-shot) |       54.2 |       75   |            25   |                100   |
| LLaMA-3.2-1B (LoRA)       |       62.5 |       75   |             8.3 |                 83.3 |
| Gemma-3-1B (zero-shot)    |       75   |       16.7 |            16.7 |                 66.7 |
| Gemma-3-1B (kNN 4-shot)   |       75   |       16.7 |            16.7 |                 66.7 |
| Gemma-3-1B (LoRA)         |       79.2 |       25   |            25   |                 83.3 |