# Model card: Yako eye screening

## Intended use

- **For**: a first triage step, to decide who should get a blood test, in pharmacies or community settings.
- **Not for**: diagnosing anemia, measuring hemoglobin, ruling out anemia in a pregnant woman or in a person with
  warning signs (breathlessness, fainting, heavy bleeding). Those people need a blood test whatever the result.

## Data

- Côte d'Ivoire, 2 sites, 152 patients (34 with anemia, 118 without).
- Reference: hemoglobin from a blood count; anemia defined with the WHO thresholds by age, sex and pregnancy.
- Photos of the lower eyelid pulled down, taken with several smartphones (Xiaomi, Apple, Tecno, Honor), without
  flash, in indoor light.
- Skin: dark phototypes (mostly V and VI).
- The data are not published: they are health data covered by the patients' consent.

## Model

- Segmentation: U-Net, ResNet-50 encoder (segmentation-models-pytorch), trained on manually annotated photos,
  input 384×384.
- Feature: `a_norm`, the CIELAB a* of the conjunctiva minus that of the surrounding skin, after a white balance on
  that skin. Many other colour and texture features were tested and did not improve on it.
- Classifier: logistic regression on `a_norm` alone, balanced class weights.

## Performance (full pipeline, automatic segmentation)

| Metric | Value |
|---|---|
| AUC | 0.70 (95% CI 0.60 to 0.79, bootstrap) |
| AUC, 5-fold cross-validation repeated 20 times | 0.69 |
| Sensitivity at the green threshold (0.40) | 0.87 |
| Specificity at the green threshold (0.40) | 0.41 |
| Severe anemia (Hb < 9 g/dL) classified green | 0 of 6 |
| Anemia among green results | 9% (prevalence 22%) |

Sensitivity and specificity are out of sample (5-fold cross-validation repeated 50 times).

Performance is better on wide shots (eye and surrounding face visible) than on close-ups.

## Repeatability (same eye, photos seconds apart)

419 usable photos in 90 series. SD of a\* within a series 1.1, between patients 4.0, ICC 0.93. Two single photos give
the same colour in 83 to 90% of pairs; green and red together in 2 pairs out of 837. Median of 3 photos (what the app
does): agreement 91% instead of 83% (simulated on the 28 series with at least 6 photos). Same session and same light
only: test-retest across sessions remains to be measured.

## Known limitations

- **Small cohort**: 34 anemic patients, so wide confidence intervals. These numbers need a prospective validation.
- **One country, dark skin**: the skin anchor removes most of the effect of the phone and the light, but the offset it
  corrects depends on the skin. The model is not expected to transfer as is to lighter skin; it needs recalibration.
- **Light**: under a ±20% change of lighting, about a third of borderline results can switch category. The capture
  guidance, the quality checks and the median of 3 photos reduce this, they do not remove it.
- **Phones**: small systematic differences between phone models were observed; the sample per model is too small to
  correct them yet.
- **Moderate specificity**: more than half of non-anemic people get orange or red. Orange and red carry a similar
  risk today, so the red level does not yet add information beyond "get a blood test".
- Conjunctival pallor is a sign of anemia, not of its cause.

## Ethics and safety

- Patients gave informed consent for the study. No photo or individual data is in this repository.
- The demo server keeps no image.
- A result must always come with the advice to get a blood test when it is orange or red, and with the warning signs
  that need urgent care.
