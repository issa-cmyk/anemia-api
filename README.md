# anemia-api

Smartphone screening for **iron-deficiency anaemia** from a photo of the eye's conjunctiva (inner eyelid). Given an image, the API returns a triage level (green / orange / red) — a low-cost, non-invasive flag for who most needs a confirmatory blood test.

## Why
Anaemia is common and under-diagnosed where lab access is limited. A conjunctiva photo from any phone can help prioritise testing, without a needle.

## Approach
- **Feature:** conjunctival colour (a* channel) with a **skin-anchored white balance**, to stay robust across lighting conditions and phone models.
- **Model:** logistic regression on engineered colour features.
- **Data:** cohort recruited on **prescribed blood tests**, not on appearance — to avoid verification bias.

## Results (honest)
- **AUC ≈ 0.74** under **nested cross-validation** (N ≈ 100) — the figure that survives, not an optimistic leave-one-out score.
- I traced and rejected a tempting AUC ≈ 0.80 once I found it came from data snooping.
- **Known limits:** colour-based signal has a ceiling; performance varies with skin phototype and device; severe-anaemia strata are small. Screening aid, not a diagnosis.

## Quickstart
```bash
pip install -r requirements.txt
uvicorn [module]:app --reload
# POST an image to [endpoint] → returns triage level + score
