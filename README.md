# machine-and-deep-learning
PROJET COUVRANT LES MODULES : - Optimisation pour le Machine Learning - Deep Learning : Théorie &amp; Pratique
Partie I : Machine Learning 
Considérons l’ensemble de données sur une maladie cardiaque, heart_disease.csv (téléchargé
depuis Kaggle), contenant des données médicales sur n = 303 patients. Comme le montre la
Figure 1, heart_disease.csv contient 303 lignes (exemples) et 14 colonnes (variables) qui sont :
— age : âge du patient
— sex : sexe du patient (0 : féminin, 1 : masculin)
— cp : type de douleur angineuse (1 : typique, 2 : atypique, 3 : non angineuse, 4 : asymptomatique)
— trestbps : tension artérielle au repos (en mm Hg)
— chol : quantité de cholestérol (en mg/dl) récupérée via le capteur IMC
— fbs : variable binaire regardant si la glycémie à jeun est > 120 mg/dl (1 : vrai, 0 : faux)
— restecg : résultats électrocardiographiques (ECG) au repos
— thalach : fréquence cardiaque maximale atteinte (en battement par minute (bpm))
— exang : angine induite par l’effort (1 : oui, 0 : non)
— oldpeak : dépression ST (anomalie dans un ECG) induite par l’exercice par rapport à l’état de repos
— slope : pente du segment ST (partie d’un ECG) d’effort maximal
— ca : variable catégorielle sur le nombre de grands vaisseaux
— thal : variable catégorielle sur le type de défaut
— target : variable binaire sur la maladie du coeur (0 : pas de maladie, 1 : maladie).
1
Figure 1 – Aperçu sur heart_disease.csv
Partie I/A : Prédiction de la fréquence cardiaque maximale
Dans cette première partie, nous nous concentrons sur l’étude de la fréquence cardiaque maximale atteinte, la variable thalach dans le jeu de données (voir Figure 1).
1. Questions préliminaires
(a) Quelle est la nature de la variable thalach (qualitative ordinale/nominale ? quantitative continue/discrète ? autre ?) ? Même question pour age, sex et chol.
(b) Selon vous, comment peut-on vérifier l’existence d’un lien et la force de ce lien entre la
variable chol et thalach ?
2. A présent, on souhaite mettre en place un modèle qui nous permettra de prédire la
fréquence cardiaque maximale (thalach) d’un patient à partir de différentes variables
identifiées comme ayant un effet / lien significatif sur thalach. Dans la suite, on considérera que ces variables identifiées sont: age, sex, trestbps, chol, oldpeak et thal. Un Data
scientist nous suggère un modèle de régression linéaire pour ce problème de prédiction.
(a) Est-ce que la régression linéaire est réellement le modèle adapté pour ce problème ?
Justifiez votre réponse.
(b) Identifiez la variable étiquette de ce problème. On la notera par y dans la suite.
(c) Identifiez la ou les variables caractéristiques de ce problème (on les notera dans la suite
par x
[1], x[2], x[3], x[4], x[5], x[6]).
(d) Écrivez l’expression mathématique du modèle de régression pour le patient i (avec
i ∈ {0, . . . , n = 302}) ayant l’étiquette yi et les caractéristiques x
[1]
i
, . . . , x
[6]
i
, sans
oublier le terme d’erreur (résidu) i
. On note β0, β1, . . . , β6 les paramètres du modèle.
(e) Afin de pouvoir utiliser notre modèle de régression pour des fins de prédiction, nous
devons d’abord l’entraîner.
i. Quel est le but de l’entrainement selon vous ? En quoi cela consiste concrètement ?
ii. Quelle fonction perte considérez vous pour ce problème ? Justifiez votre choix.
iii. Quel algorithme d’optimisation suggérez vous pour l’entrainement de notre modèle ?
Comment fonctionne cet algorithme ?
iv. Quelles peuvent être, selon vous, les limites de cet algorithme d’optimisation ?
Connaissez-vous une autre alternative ?
(f) Notre Data scientist nous suggère de séparer nos données en données d’entrainement et
données de test avant de procéder à l’entrainement de notre modèle. Expliquez en quoi
cette séparation des données est importante et à quoi servent les données d’entrainement
et à quoi servent les données de test.
2
(g) Après l’entraînement, vous devez communiquer à l’hôpital le pouvoir prédictif de votre
modèle. Quel indicateur (exprimant la qualité d’un modèle de régression) choisissezvous et sur quelles données vous le calculez ? Pourquoi ?
(h) L’hôpital vous demande de prédire la fréquence cardiaque maximale d’un nouveau
patient i = 317 absent des données initiales (Figure 1). Comment procédez-vous (en
utilisant votre modèle entrainé dont les paramètres estimés sont notés βˆ
0, βˆ
1, . . . , βˆ
6) ?
Partie I/B : Prédiction de la présence de la maladie cardiaque
Dans cette deuxième partie, nous nous concentrons sur la prédiction de la survenue
de la maladie du coeur, la variable target dans le jeu de données (voir Figure 1).
3. Notre Data scientist nous suggère ici d’utiliser un modèle de classification pour prédire si
un nouveau patient est malade du coeur ou non.
(a) Êtes vous d’accord avec lui ? Pourquoi ?
(b) Ce problème fait partie de quelle classe de problèmes de Machine Learning (apprentissage par renforcement, supervisé, non supervisé) ? Justifiez.
On considère dans la suite qu’on a identifié les variables suivantes comme les caractéristiques pertinentes
de notre problème: age, sex, cp, chol, thalach, oldpeak, thal.
4. Suggérez un algorithme de classification pour ce problème de prédiction de la variable
target. Pourquoi cet algorithme ? Décrivez brièvement le principe de cet algorithme.
5. Soit ysubset = (1, 1, 0, 0, 1, 0, 1, 0, 0, 1) et yˆ subset = (1, 0, 1, 0, 1, 1, 1, 1, 0, 1) respectivement,
une partie des vraies étiquettes et des étiquettes prédites (par votre algorithme de classification) correspondantes.
(a) Proposez un pseudo code (décrivez la procédure algorithmique) permettant de calculer
rapidement la précision (“accuracy”) de votre modèle en partant de ysubset et yˆ subset.
(b) Reproduisez et Complétez la matrice de confusion ci-dessous à partir des valeurs des
vecteurs ysubset et yˆsubset.
True 0 True 1
Predicted 0 TN = FN =
Predicted 1 FP = TP =
où TN, FN, FP et TP correspondent respectivement aux nombres de : vrais négatifs, faux négatifs,
faux positifs et vrais positifs.
(c) Déduisez de la matrice de confusion : l’“accuracy”, la sensibilité (taux d’individus
positifs bien prédits) et la spécificité (taux d’individus négatifs bien prédits) de votre
modèle de classification.
Partie II : Deep Learning - Théorie & Pratique
Partie II/A : Deep Learning - Théorie (Optimisation)
Soit un réseau de neurones multi-couches dense (un Perceptron multi-couches ou MLP) à une
seule couche cachée ayant h neurones (ou unités) et une couche de sortie de taille q.
— On considère un mini-batch X ∈ R
n×d
(n exemples ayant chacun d caractéristiques) pour
constituer sa couche d’entrée.
— Pour établir les connexions entre la couche d’entrée et la couche cachée, nous définissons
la matrice de poids W(1) ∈ R
d×h
et le vecteur de biais b
(1) ∈ R
1×h
.
3
— Pour établir les connexions entre la couche cachée et la couche de sortie, nous définissons
la matrice de poids W(2) ∈ R
h×q
et le vecteur de biais b
(2) ∈ R
1×q
.
Les calculs ci-dessous entre la couche d’entrée et la couche cachée, puis entre la couche cachée et
la couche de sortie ont été faits :
— Pre-activation et activation au niveau de la couche cachée :
Z
(cachée) = XW(1) + b
(1) et H = a
(1)(Z
(cachée)), ici a
(1)(z) = 1
1+e−z
— Pre-activation et activation au niveau de la couche de sortie :
Z
(sortie) = HW(2) + b
(2) et Y = a
(2)(Z
(sortie)), ici a
(2)(z) = 1
1+e−z
— Fonction de perte :
E(W, b) = 1
n
Xn
i=1
1
2
