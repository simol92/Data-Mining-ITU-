12/12-2023

##### Hvad har vi lavet siden sidst?

Malte

- fik problemer med duplikater ifbm Hopkins nearest neighbour
- ca 4000 gengangere
- get_df() beholder første udgave af sangen, fjerner alle andre
- song recommendation
  - dropper "mode" fordi kun placeret i ekstremer
  - scaler ikke længere, men bruger min og max værdier
  - silhouette scores for de forskellige clusters
  - Vi skal se på hvordan vi bedst laver clustering
  - kan vi bruge feature engineering til at skabe moods, som vi kan bruge til at lave clusters
  - Maja: vi kan prøve at encode playlist_genre til som feature, når vi laver clusters
  - mangler at prøve flere clustering modeller

Simon

- Måske prøve PCA uden "mode"
- har lavet rythm_related udvalg af features, men model performer dårligere for R2 end med all ordinal features ift at predicte danceability
- danceability er korreleret med energy
- laver to modeller: random forest og gradient boosting (begge baseret på decision trees)
- minimal forskel i performance
- kan få højere R2 ved at skrue på learning rate
- vi skal måske bruge de algoritmer vi har haft om, hvis der ikke er den store forskel i performance

Emil

- ikke noget fornuftigt

Maja

- to eksperimenter:

1.

- dictionary med valence og energy og mood intervaller
- hver sang for assignet et mood
- halvdelen bliver klassificeret som "other"

2.

- kmean for at tjekke kombination af med gridsearch (chatGPT)
- prøv evt at plotte hver feature stdd

##### Opsummering af hvad der skal indgå i rapporten

- Metoder
- forklaring af algoritmerne
- hvor gode er resultaterne
- compare with baseline

5 ting:

- eda
- klassificering d-tree (årstal, genre, subgenre)
- regression (Simon)
- mood klassificering
- song recommendation (cluster)
  skal implementeres og prøves af

Simons forslag til struktur:

- eda
- model exploration
  - supervised
  - unsupervised
- recommendation engine (med bedste model)

men modellerne svarer på forskellige spørgsmål
vi skal fokusere på de forskellige research spørgsmål

TODO

- find ud af datetime format bug (utils.py, ISO...)

fordeling af opgaver:

Simon

- skriv om decision tree og regression på danceability

Maja

- fnuller (feature engineering)
- skriv på eda

Malte

- færdiggør modelling
- find ud af antal clusters
- skriv afsnit om recommendation i rapporten
- skriv intro om data
- afsnit om research questions

Emil

- gør kode læsbar, skriv kommentarer
- færdiggør eda i rapporten
- skriv om classification af genre

##### tidsplan

19. kl 14:00 gennemgang af alt

20. deadline
