# Yako eye screening: anemia triage from a photo of the eyelid

> I built this project at Meditect. Official, canonical repository: **[Meditect/yako-anemia-eye](https://github.com/Meditect/yako-anemia-eye)**. This is my personal copy.

Anemia makes the inside of the lower eyelid (the palpebral conjunctiva) paler. This project measures that pallor on a
smartphone photo and turns it into a simple triage: **green** (low probability of anemia), **orange** (uncertain,
blood test advised) or **red** (high probability, blood test strongly advised).

It was developed and calibrated in Côte d'Ivoire, on patients with dark skin, and it runs today in the
[Yako](https://yako.app) health app. The goal is to bring a first screening step to women in places with no
laboratory nearby, starting with pharmacies.

**Live scan (Yako app, in French):** https://api.34-77-18-188.sslip.io/3 shows the live guidance, the automatic
photo and the analysed area. In the app, the result then comes back in the conversation; to see the colour result,
run the demo of this repository (one command, see [Quick start](#quick-start)).

> **Research prototype.** It is not a medical device and does not give a diagnosis. Only a blood test measures
> hemoglobin. Do not use it to rule out anemia in a pregnant woman or in someone with warning signs.

## How it works

```
photo ──► quality check ──► U-Net segmentation ──► skin-anchored colour ──► logistic ──► green / orange / red
          (blur, light)      (conjunctiva mask)     a_norm = a*(conj) − a*(skin)
```

1. **Quality check**: blur, exposure and resolution.
2. **Segmentation**: a U-Net with a ResNet-50 encoder finds the conjunctiva, with geometric checks on the mask
   (position, size, horizontal shape).
3. **Colour measure**: the white balance is set on the skin around the eye (grey-world), then
   `a_norm` = mean CIELAB a* of the conjunctiva minus mean a* of the skin. Anchoring on the patient's own skin makes the
   measure much less sensitive to the phone and the light. Specular reflections are removed, and skin pixels are
   filtered by colour so that a finger or hair in the frame does not bias the reference.
4. **Score**: a one-feature logistic regression gives the probability of anemia. Thresholds: green < 0.40 ≤ orange
   < 0.65 ≤ red.

The overlay shows what was measured: the conjunctiva in green and the skin reference in blue.

## Why this measure

**Why the conjunctiva.** Hemoglobin is what makes blood red. When it is low, the inside of the lower eyelid gets paler.
Clinicians already look at it at the bedside. It is one of the few places where blood shows through without skin
pigment in the way, which matters for patients with dark skin.

**Why a\* (CIELAB).** The CIELAB colour space separates lightness (L\*) from colour; its a\* axis goes from green to red.
Redness of the conjunctiva is what changes with anemia, and a\* measures it while being less sensitive to how bright the
photo is. We compared many other features on the cohort: other colour spaces (HSV, YCrCb), colour statistics, texture,
and the features proposed in the literature. None did better than a\* once the normalisation below was applied, so the
model keeps this single measure.

**Why normalise on the patient's own skin.** The same eye does not have the same colour on two phones or under two
lights: each phone applies its own white balance and exposure. In our cohort, part of the colour differences came from
the phone model rather than from the patients. The skin around the eye is in the same photo, under the same light, taken
by the same phone, so it is used as a reference: the white balance is set on this skin (grey world), then
`a_norm = a*(conjunctiva) - a*(skin)`. What remains is how much redder the conjunctiva is than the patient's own skin.
This made the measure more robust to the place and the phone. The other side of it: the offset that the skin corrects
depends on the skin itself, so the model is calibrated for dark skin and needs recalibration for other populations.

**Details that matter.**

- *A skin ring away from the eyelid.* The reference is a ring 10 to 40 pixels around the conjunctiva (on a 512×512
  image), not glued to it. Glued to it, the ring falls on the rim of the eyelid, which also gets paler with anemia:
  the subtraction then erases the signal. On close-up photos, the glued ring gave an AUC of 0.43, worse than chance.
- *Colour filter on the ring.* On wide shots the ring can reach a finger, hair or the wall. Only pixels whose colour
  matches the skin right next to the eye are kept. In nested cross-validation, this filter was selected in every fold.
- *Reflections removed.* A reflection shows the colour of the light, not of the tissue, and pulls a\* towards zero, that
  is towards "anemic". A simulated reflection on 10% of the conjunctiva raised the probability by 7 points on average;
  with the filter, by less than 2.
- *One feature, simple model.* With 34 anemic patients, a model with more inputs learns noise. A logistic regression on
  a single measure is easy to check, and its output is monotonic: a paler conjunctiva never lowers the risk.

## Repeatability

A screening tool must give the same answer when the photo is taken again. We ran the full pipeline on every photo of
the cohort: several photos of the same eye taken seconds apart (419 usable photos, 90 series).

| | Value |
|---|---|
| Spread of a\* between patients (SD) | 4.0 |
| Spread of a\* from one photo to the next, same eye (SD) | 1.1 |
| Intraclass correlation (ICC) | 0.93, "excellent" on the usual scale (above 0.90, Koo & Li 2016) |
| Two single photos give the same colour | 83 to 90% |
| Two photos give green and red | 2 pairs out of 837 |

When the colour changes, it moves to the next level (green to orange, orange to red), for patients close to a
threshold. Averaging repeated measurements makes a measure more reliable (Spearman-Brown formula), so **the app and the
demo take 3 photos 0.25 s apart and keep the median**: two such results agree in 91% of cases instead of 83%. The median
rather than the mean, so that one failed photo (blink, glare) does not move the result. When the 3 photos do not all
give the same level, the result says that the person is close to a threshold.

These photos were taken in one session, under the same light. Between two separate sessions the variation is larger:
light is the main source of variation of conjunctival colour on photos, and normalisation reduces it without removing
it (Collings et al. 2016). Measuring this test-retest variation is part of the planned validation.

## Results

Cohort from Côte d'Ivoire, 2 sites, **152 patients, 34 with anemia** (hemoglobin from a blood count, WHO thresholds by
age and sex), photos taken with several Android and iPhone models. The full pipeline was run end to end, with
automatic segmentation, exactly as in this repository.

| Metric | Value |
|---|---|
| AUC | 0.70 (95% CI 0.60 to 0.79) |
| AUC, 5-fold cross-validation (20 repeats) | 0.69 |
| Sensitivity (anemic patients not classified green) | 0.87 |
| Specificity (non-anemic patients classified green) | 0.41 |
| Severe anemia (Hb < 9 g/dL) classified green | 0 of 6 |

Sensitivity and specificity are measured out of sample (5-fold cross-validation repeated 50 times).

What each colour means in this cohort (22% of patients with anemia):

| Result | Share of patients | Of whom anemic | Advice |
|---|---|---|---|
| Green | 34% | 9% | No test needed now, unless symptoms or pregnancy |
| Orange | 49% | 29% | Blood test advised |
| Red | 17% | 30% | Blood test strongly advised |

A green result divides the risk of anemia by about 2.4, so one patient in three can be spared a blood test. Orange and
red carry a similar risk today; both lead to a blood test.

**Choosing the threshold.** The green threshold sets the balance between missed anemia and unnecessary tests:

| Green threshold | Sensitivity | Specificity | Anemic patients classified green (of 34) |
|---|---|---|---|
| 0.30 | 0.96 | 0.27 | 1 |
| **0.40 (default)** | **0.87** | **0.41** | **4 to 5** |
| 0.50 | 0.67 | 0.57 | 11 |

0.40 is the highest threshold that keeps every severe case out of green. It can be changed in
`anemia_eye/model_calibration.json` (`t_green`). See [MODEL_CARD.md](MODEL_CARD.md) for the details and the
limitations.

## Quick start

### Try it in one command

You only need [Python 3.10 or later](https://www.python.org/downloads/). Open a terminal and paste:

**macOS / Linux** (Terminal)

```bash
curl -fsSL https://raw.githubusercontent.com/Meditect/yako-anemia-eye/main/run.sh | bash
```

**Windows** (PowerShell)

```powershell
irm https://raw.githubusercontent.com/Meditect/yako-anemia-eye/main/run.ps1 | iex
```

It downloads the code into a `yako-anemia-eye` folder in your home directory, installs everything in a private Python
environment (nothing is installed system-wide), downloads the model and opens the demo in your browser. The first run
takes a few minutes; the next ones start in seconds. Press `Ctrl+C` in the terminal to stop.

The demo works like the Yako app: the camera opens in the page, live checks guide the person (light, reflections,
movement, position of the pink inner eyelid, blur), and when everything is right a 3-2-1 countdown starts and the
photo takes itself. The camera closes right away, then the result is shown with the analysed area.

### Manual installation

Python 3.10 or later (the commands below use `python3`, as on macOS and Linux; on Windows type `python`). CPU is enough (about one second per photo).

```bash
git clone https://github.com/Meditect/yako-anemia-eye.git && cd yako-anemia-eye
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 predict.py my_eye_photo.jpg --overlay results/
```

The 125 MB segmentation weights download on the first run. On Windows, activate the environment with
`.venv\Scripts\activate`. Web demo:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000.

The web demo opens the camera directly in the page, on a computer (webcam) as on a phone, with the live guidance and
automatic photo described above. Choosing an existing photo is also possible.

On a phone, browsers allow the live camera only over https. To test from a phone on the same Wi-Fi:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 30 -subj "/CN=yako-demo" -keyout key.pem -out cert.pem
uvicorn server:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

Then open `https://<your computer's IP>:8000` on the phone and accept the certificate warning. Over plain http, the page falls back to the phone's own camera app, which also works.

With Docker:

```bash
docker build -t yako-eye . && docker run -p 8000:8000 yako-eye
```

From Python:

```python
import cv2
from anemia_eye import AnemiaEye

eye = AnemiaEye()
r = eye.predict(cv2.imread('photo.jpg'))
print(r.ok, r.triage, r.probability, r.error)
```

Tests: `python3 -m pytest tests`.

### Taking a good photo

- Daylight or a bright room, no flash.
- Pull the lower eyelid down with one finger to show the pink inner part.
- Eye in the middle of the frame, with some skin visible around it.

When a photo cannot be used, the tool says why (too dark, blurry, conjunctiva not visible, off-centre) instead of
giving a result.

## Repository

| Path | Content |
|---|---|
| `anemia_eye/pipeline.py` | The whole pipeline: quality, segmentation, measure, score |
| `anemia_eye/model_calibration.json` | Logistic regression parameters (mean, scale, coefficient, intercept) |
| `predict.py` | Command line |
| `server.py`, `static/index.html` | Web demo (FastAPI) with live guidance and automatic photo |
| `run.sh`, `run.ps1` | One-command installation and launch (macOS/Linux, Windows) |
| `tests/` | Unit tests |

The segmentation weights are hosted on Hugging Face:
[pk180/anemiaeye-models](https://huggingface.co/pk180/anemiaeye-models) (`unet_resnet50_v2_fullframe.pt`). Their
SHA-256 is checked after download.

## Privacy

- No patient photo and no patient-level data are included in this repository.
- The demo server processes images in memory and does not store them.
- Only aggregate model parameters are published (four numbers in `model_calibration.json`).

## Roadmap

- Prospective validation in pharmacies, with blood tests for every participant.
- Larger cohort of women of reproductive age and pregnant women.
- Combination with other signals (nail bed, symptoms) to raise specificity.

## References

- Collings S. et al. Non-invasive detection of anaemia using digital photographs of the conjunctiva. *PLoS ONE*, 2016.
  [PMC4829242](https://pmc.ncbi.nlm.nih.gov/articles/PMC4829242/)
- Koo T.K., Li M.Y. A guideline of selecting and reporting intraclass correlation coefficients for reliability
  research. *Journal of Chiropractic Medicine*, 15(2), 2016. doi:10.1016/j.jcm.2016.02.012
- Mannino R.G. et al. Smartphone app for non-invasive detection of anemia using only patient-sourced photos. *Nature
  Communications*, 9, 2018.
- Prediction of anemia in real-time using a smartphone camera processing conjunctival images, 2024.
  [PMC11090304](https://pmc.ncbi.nlm.nih.gov/articles/PMC11090304/)

## Authors

Developed by [Meditect](https://meditect.com). Research, data collection and model by Issa Cissé.

## License

[MIT](LICENSE), © 2026 Meditect. Contributions and independent validations are welcome.

---

## En français

Yako analyse une photo de l'intérieur de la paupière inférieure pour repérer une pâleur liée à l'anémie, et donne un
tri simple : vert, orange ou rouge. Le modèle a été construit en Côte d'Ivoire, sur 152 patients à peau foncée, et
fonctionne aujourd'hui dans l'application Yako. C'est un prototype de recherche pour le dépistage, pas un diagnostic :
seule une prise de sang mesure l'hémoglobine.

Sur 152 patients (dont 34 anémiques) : AUC 0,70, sensibilité 0,87, spécificité 0,41, aucun cas sévère (Hb < 9 g/dL)
classé vert. Détails et limites dans [MODEL_CARD.md](MODEL_CARD.md).

**Scan en direct (application Yako) :** https://api.34-77-18-188.sslip.io/3 montre le guidage, la photo automatique et
la zone analysée. Dans l'application, le résultat arrive ensuite dans la conversation ; pour voir la couleur du
résultat, lancer la démonstration de ce dépôt (une commande, ci-dessous).

### Pourquoi cette mesure

**La conjonctive.** C'est l'hémoglobine qui rend le sang rouge. Quand elle baisse, l'intérieur de la paupière du bas
pâlit. Les soignants le regardent déjà au lit du malade. C'est aussi l'un des rares endroits où le sang se voit sans
le pigment de la peau, ce qui compte pour les peaux foncées.

**La valeur a\* (espace CIELAB).** CIELAB sépare la luminosité (L\*) de la couleur ; l'axe a\* va du vert au rouge. C'est
le rouge de la conjonctive qui change avec l'anémie, et a\* le mesure en étant moins sensible à la luminosité de la photo.
Nous avons comparé de nombreuses autres caractéristiques sur la cohorte : autres espaces de couleur (HSV, YCrCb),
statistiques de couleur, texture, caractéristiques de la littérature. Aucune n'a fait mieux que a\* une fois la
normalisation ci-dessous appliquée : le modèle garde donc cette seule mesure.

**La normalisation sur la peau du patient.** Un même œil n'a pas la même couleur sur deux téléphones ou sous deux
lumières : chaque téléphone applique sa propre balance des blancs et sa propre exposition. Dans notre cohorte, une partie
des écarts de couleur venait du modèle de téléphone, pas des patients. La peau autour de l'œil est dans la même photo,
sous la même lumière, prise par le même téléphone : elle sert de référence. On règle la balance des blancs sur cette peau
(monde gris), puis `a_norm = a*(conjonctive) - a*(peau)`. Il reste ce qui compte : de combien la conjonctive est plus
rouge que la peau du patient. La mesure devient ainsi plus robuste au lieu et au téléphone. Revers : l'écart corrigé par
la peau dépend de la peau elle-même, donc le modèle est calibré pour les peaux foncées et doit être recalibré ailleurs.

**Les détails qui comptent.**

- *Un anneau de peau écarté de la paupière.* La référence est un anneau de 10 à 40 pixels autour de la conjonctive (sur
  une image de 512×512), pas collé à elle. Collé, il tombe sur le bord de la paupière, qui pâlit lui aussi avec l'anémie,
  et la soustraction efface le signal : sur les gros plans, l'anneau collé donnait une AUC de 0,43, pire que le hasard.
- *Un filtre de couleur sur l'anneau.* Sur un plan large, l'anneau peut toucher un doigt, des cheveux ou le mur. On ne
  garde que les pixels dont la couleur ressemble à la peau juste à côté de l'œil. En validation croisée imbriquée, ce
  filtre a été retenu dans tous les plis.
- *Les reflets retirés.* Un reflet renvoie la couleur de la lumière, pas celle du tissu, et tire a\* vers zéro, donc vers
  « anémique ». Un reflet simulé sur 10 % de la conjonctive augmentait la probabilité de 7 points en moyenne ; avec le
  filtre, de moins de 2.
- *Une seule mesure, un modèle simple.* Avec 34 patients anémiques, un modèle à plusieurs entrées apprend du bruit. Une
  régression logistique sur une seule mesure se vérifie facilement, et elle est monotone : une conjonctive plus pâle ne
  fait jamais baisser le risque.

### Répétabilité

Un outil de dépistage doit donner la même réponse quand on refait la photo. Nous avons passé chaque photo de la cohorte
dans le modèle : plusieurs photos du même œil prises à quelques secondes d'intervalle (419 photos utilisables, 90
séries). L'écart de a\* d'une photo à l'autre (écart-type 1,1) est petit devant l'écart entre patients (4,0) : ICC 0,93,
« excellent » selon l'échelle habituelle (Koo & Li 2016). Deux photos donnent la même couleur dans 83 à 90 % des cas ;
sinon, c'est presque toujours la couleur voisine, chez des patients proches d'un seuil (vert et rouge : 2 paires sur
837). Comme la moyenne de mesures répétées est plus fiable (formule de Spearman-Brown), **l'application et la démo
prennent 3 photos à 0,25 s d'intervalle et gardent la médiane** : deux résultats concordent alors dans 91 % des cas au
lieu de 83 %. Quand les 3 photos ne donnent pas toutes le même niveau, le résultat indique que la personne est proche d'une limite. Ces photos viennent d'une
même séance : entre deux séances, la variation est plus grande, la lumière étant la principale source de variation
(Collings et al. 2016). La mesurer fait partie de la validation prévue.

### Démarrage rapide

#### Essayer en une commande

Il suffit d'avoir [Python 3.10 ou plus récent](https://www.python.org/downloads/). Ouvrir un terminal et coller :

**Mac / Linux** (Terminal)

```bash
curl -fsSL https://raw.githubusercontent.com/Meditect/yako-anemia-eye/main/run.sh | bash
```

**Windows** (PowerShell)

```powershell
irm https://raw.githubusercontent.com/Meditect/yako-anemia-eye/main/run.ps1 | iex
```

La commande télécharge le code dans un dossier `yako-anemia-eye` du dossier personnel, installe tout dans un
environnement Python séparé (rien n'est installé sur le reste de l'ordinateur), télécharge le modèle et ouvre la
démonstration dans le navigateur. La première fois prend quelques minutes, les suivantes quelques secondes. `Ctrl+C`
dans le terminal pour arrêter.

La démonstration fonctionne comme l'application Yako : la caméra s'ouvre dans la page, un guidage en direct aide la
personne (lumière, reflets, mouvement, position de la partie rose, flou), puis un compte à rebours 3-2-1 démarre et la
photo se prend toute seule. La caméra se ferme aussitôt, et le résultat s'affiche avec la zone analysée.

#### Installation manuelle

Il faut Python 3.10 ou plus récent (les commandes utilisent `python3`, comme sur Mac et Linux ; sous Windows, taper `python`). Un ordinateur ordinaire suffit (environ une seconde par photo), sans carte
graphique.

```bash
git clone https://github.com/Meditect/yako-anemia-eye.git && cd yako-anemia-eye
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 predict.py photo_oeil.jpg --overlay resultats/
```

Les poids du modèle (125 Mo) se téléchargent au premier lancement. Sous Windows, activer l'environnement avec
`.venv\Scripts\activate`. Page de démonstration :

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Puis ouvrir http://localhost:8000 dans le navigateur.

La page ouvre directement la caméra (webcam sur ordinateur, caméra du téléphone), avec le guidage et la photo
automatique décrits plus haut. On peut aussi choisir une photo existante.

Sur téléphone, le navigateur n'autorise la caméra en direct qu'en https. Pour tester depuis un téléphone sur le même
Wi-Fi :

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 30 -subj "/CN=yako-demo" -keyout key.pem -out cert.pem
uvicorn server:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

Puis ouvrir `https://<adresse IP de l'ordinateur>:8000` sur le téléphone et accepter l'avertissement du certificat.
En http simple, la page ouvre l'appareil photo du téléphone à la place, ce qui marche aussi.

Avec Docker :

```bash
docker build -t yako-eye . && docker run -p 8000:8000 yako-eye
```

### Bien prendre la photo

- À la lumière du jour ou dans une pièce bien éclairée, sans flash.
- Tirer la paupière du bas vers le bas avec un doigt pour montrer la partie rose.
- L'œil au milieu de l'image, avec un peu de peau visible autour.

Quand une photo est inutilisable, l'outil dit pourquoi (trop sombre, floue, partie rose non visible, mal centrée) au
lieu de donner un résultat.

### Que veut dire chaque couleur

| Résultat | Conseil |
|---|---|
| Vert | Pas de prise de sang nécessaire pour l'instant, sauf symptômes ou grossesse |
| Orange | Prise de sang conseillée |
| Rouge | Prise de sang fortement conseillée |

Une femme enceinte, ou une personne avec des signes d'alerte (essoufflement, malaise, saignements abondants), doit
faire une prise de sang quel que soit le résultat.
