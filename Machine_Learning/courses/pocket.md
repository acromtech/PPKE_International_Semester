## Pocket Algorithm
- Pour pallier les limites du PLA lorsque les données ne sont pas linéairement séparables, nous présentons l'algorithme de poche (Pocket Algorithm). Cet algorithme modifie le PLA en conservant les meilleurs poids rencontrés jusqu'à présent dans une "poche" (ou "pocket").
- L'idée est de garder les poids qui produisent le moins d'erreurs de classification sur l'ensemble de données, même si le Perceptron ne parvient pas à atteindre une classification parfaite.

Explorons l'Algorithme de Poche (Pocket Algorithm).

### **Objectif :**
   - L'objectif du Pocket Algorithm est similaire à celui du PLA : entraîner un Perceptron à classer des données. Cependant, le Pocket Algorithm cherche à trouver le meilleur ensemble de poids plutôt que de simplement corriger les erreurs de classification.

### **Initialisation :**
   - Initialise les poids du Perceptron$ w $ à des valeurs aléatoires ou à zéro.
   - Initialise un ensemble de poids de poche (pocket)$ \hat{w} $ à zéro.

### **Boucle d'Apprentissage :**
   - Itère à travers tous les exemples d'entraînement jusqu'à ce qu'un critère d'arrêt soit atteint.
   - Pour chaque exemple$ (x_n, y_n) $ dans l'ensemble de données d'entraînement :
     - Calcule la prédiction du Perceptron$ h(x_n) = \text{sign}(w^T x_n) $.
     - Si la prédiction est incorrecte ($ h(x_n) \neq y_n $), met à jour les poids du Perceptron selon la règle de mise à jour du PLA :
       $$ w \leftarrow w + y_n x_n $$

### **Mise à Jour du Pocket :**
   - Après chaque mise à jour des poids du Perceptron, compare les performances actuelles du Perceptron avec celles des poids stockés dans le pocket.
   - Si les performances actuelles sont meilleures que celles du pocket (c'est-à-dire que le nombre d'erreurs de classification est plus faible) :
     - Met à jour les poids du pocket avec les poids actuels du Perceptron :
       $$ \hat{w} \leftarrow w $$

### **Critère d'Arrêt :**
   - L'algorithme s'arrête lorsque tous les exemples d'entraînement ont été parcourus ou lorsqu'un critère d'arrêt spécifique est atteint (par exemple, un nombre maximum d'itérations).

### **Sortie :**
   - Une fois que l'algorithme a convergé, les poids du pocket représentent le meilleur ensemble de poids trouvés jusqu'à présent, c'est-à-dire ceux qui donnent les meilleures performances de classification sur les données d'entraînement.

### **Comparaison avec le PLA :**
   - Contrairement au PLA qui se concentre uniquement sur la correction des erreurs de classification, le Pocket Algorithm cherche activement à trouver les meilleurs poids possibles en comparant les performances des poids actuels avec ceux stockés dans le pocket.

En résumé, l'algorithme de poche est une variante améliorée du PLA qui vise à trouver les meilleurs poids pour un Perceptron en conservant les poids qui donnent les meilleures performances sur les données d'entraînement. Cela permet d'obtenir une meilleure généralisation et une meilleure capacité de classification sur de nouvelles données.
