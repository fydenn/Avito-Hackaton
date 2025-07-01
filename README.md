# 86th Place Solution (RecSys)
## Hackathon of Avito
The task for this Hackaton involve create recomender system. This system should make top-40 predictions for every user. The metric of this competition is Recall.
## Tech stack
- PyTorch
- Hnswlib
- Feature extraction
- Polars
- Pandas
## Architecture:
- Stage 1: Retrieve 150 nearest items (Euclidean distance)  
- Stage 2: Score candidates via multi-head attention network  
- Stage 3: Rank and return top-40 recommendations

