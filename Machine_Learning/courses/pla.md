# Perceptron Learning Algorithm (PLA)
- Nous introduisons l'algorithme d'apprentissage du Perceptron (PLA), qui vise à ajuster les poids du Perceptron pour minimiser les erreurs de classification sur l'ensemble de données d'entraînement.
- L'algorithme PLA consiste à parcourir itérativement les exemples d'entraînement, à identifier les erreurs de classification et à mettre à jour les poids du Perceptron pour corriger ces erreurs.

Explorons l'Algorithme d'Apprentissage du Perceptron (PLA).

### **Objectif :**
   - L'objectif du PLA est d'ajuster itérativement les poids du Perceptron afin de trouver une frontière de décision qui sépare les exemples positifs des exemples négatifs.

### **Initialisation :**
   - Au début de l'algorithme, les poids du Perceptron sont généralement initialisés à des valeurs aléatoires ou à zéro.
   - Le nombre d'itérations $ t $ est également initialisé à zéro.

### **Boucle d'Apprentissage :**
   - L'algorithme PLA itère jusqu'à ce qu'il ne reste plus d'erreurs de classification sur l'ensemble des données d'entraînement.
   - À chaque itération, l'algorithme parcourt tous les exemples d'entraînement et vérifie s'ils sont correctement classés par le Perceptron.

### **Correction des Erreurs :**
   - Si un exemple est mal classé (la sortie du Perceptron ne correspond pas à l'étiquette réelle), les poids du Perceptron sont ajustés pour corriger cette erreur.
   - L'ajustement des poids est effectué en ajoutant ou en soustrayant les caractéristiques de l'exemple mal classé pondérées par l'étiquette de l'exemple (plus pour les exemples positifs, moins pour les exemples négatifs).

### **Mise à Jour des Poids :**
   - Les poids du Perceptron sont mis à jour à chaque itération selon la règle de mise à jour suivante :
     $$ w_{t+1} = w_t + y_n(t) \cdot x_n(t) $$
     où $ w_t $ et $ w_{t+1} $ sont les poids avant et après la mise à jour, respectivement, $ x_n(t) $ est le vecteur de caractéristiques de l'exemple mal classé, et $ y_n(t) $ est l'étiquette de l'exemple.
   - Cette mise à jour force le Perceptron à mieux classer l'exemple mal classé lors des itérations suivantes.

### **Critère d'Arrêt :**
   - L'algorithme s'arrête lorsque tous les exemples sont correctement classés, c'est-à-dire lorsque le Perceptron ne commet plus d'erreurs sur l'ensemble des données d'entraînement.

### **Sortie :**
   - Une fois que l'algorithme a convergé et qu'il n'y a plus d'erreurs, les poids du Perceptron sont considérés comme appris et peuvent être utilisés pour classer de nouveaux exemples.

En résumé, l'algorithme PLA est un algorithme simple mais efficace pour entraîner un Perceptron à classer des données linéairement séparables. Il ajuste itérativement les poids du Perceptron en corrigeant les erreurs de classification jusqu'à ce que toutes les données d'entraînement soient correctement classées.