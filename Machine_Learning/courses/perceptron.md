# Learning to Answer Yes/No

Dans cette deuxième partie du cours sur les fondements de l'apprentissage automatique, nous nous concentrons sur l'apprentissage de répondre par oui ou par non. Voici les points clés abordés :

## 1. Perceptron Hypothesis Set
- Dans cette section, nous introduisons le concept d'un ensemble d'hypothèses appelé Perceptron, qui constitue une méthode simple mais puissante pour résoudre des problèmes de classification binaires.
- Le Perceptron évalue les caractéristiques d'un exemple donné en leur attribuant des poids, puis en calculant une somme pondérée. Si cette somme dépasse un certain seuil, le Perceptron émet une réponse positive (oui), sinon, il émet une réponse négative (non).
- Les hypothèses du Perceptron sont des fonctions linéaires qui séparent l'espace des caractéristiques en deux régions, une pour chaque classe de sortie.

Explorons en détail le Perceptron Hypothesis Set.

### 1.1. **Concept :**
- Le Perceptron est un modèle simple d'apprentissage automatique utilisé pour la classification binaire. Son ensemble d'hypothèses est basé sur des fonctions linéaires qui définissent des frontières de décision dans l'espace des caractéristiques.

### 1.2. **Représentation Mathématique :**
- Soit $ x = (x_1, x_2, \dots, x_d) $ le vecteur de caractéristiques d'un exemple donné, où $ d $ est le nombre de caractéristiques.
- Le Perceptron calcule une somme pondérée des caractéristiques, pondérée par des poids $ w = (w_0, w_1, \dots, w_d) $, où $ w_0 $ est le biais (ou le seuil).
- La décision est prise en comparant cette somme pondérée à un seuil. Si la somme dépasse le seuil, le Perceptron émet une sortie positive (oui), sinon il émet une sortie négative (non).

### 1.3. **Formulation de l'Hypothèse :**
- L'hypothèse du Perceptron est une fonction d'activation basée sur la somme pondérée des caractéristiques :
    $$ h(x) = sign\left( \left( \sum_{i=1}^{d} w_i x_i \right) - \text{seuil} \right) $$
- La fonction de signe $ sign(\cdot) $ retourne +1 si l'argument est positif, -1 si l'argument est négatif, et 0 si l'argument est nul.

### 1.4. **Interprétation Géométrique :**
- Dans un espace bidimensionnel (R2), les caractéristiques $ x_1 $ et $ x_2 $ peuvent être représentées comme des coordonnées sur un plan cartésien.
- Les poids $ w_1 $ et $ w_2 $ du Perceptron définissent une ligne dans ce plan, qui agit comme une frontière de décision entre les exemples positifs (classe +1) et les exemples négatifs (classe -1).
- Les exemples d'entraînement sont alors classés en fonction de leur position par rapport à cette ligne.

### 1.5. **Linearité :**
- Comme les hypothèses du Perceptron sont des fonctions linéaires, les frontières de décision qu'elles définissent sont des hyperplans. Dans des espaces de dimensions supérieures, ces frontières sont des hyperplans linéaires.

### 1.6. **Utilisation Pratique :**
- Bien que simple, le Perceptron est un modèle puissant pour les problèmes de classification binaire. Il peut être utilisé dans de nombreux domaines, tels que la détection de spam, la reconnaissance de caractères, et plus encore.

En résumé, le Perceptron Hypothesis Set est un ensemble d'hypothèses basées sur des fonctions linéaires qui définissent des frontières de décision pour la classification binaire. Ce concept forme la base du modèle de Perceptron et de ses algorithmes d'apprentissage associés.

## 2. Vector Form of Perceptron Hypothesis
- Cette section montre comment représenter le calcul du Perceptron de manière plus compacte en utilisant la notation vectorielle. Cela implique l'utilisation de produits scalaires entre le vecteur de poids et le vecteur de caractéristiques, rendant les calculs plus efficaces.

Explorons la Forme Vectorielle de l'Hypothèse du Perceptron.

### 2.1. **Introduction :**
   - La représentation vectorielle de l'hypothèse du Perceptron permet une formulation plus concise et efficace du calcul de la sortie du Perceptron.
   - En utilisant des vecteurs, nous pouvons regrouper les caractéristiques et les poids associés pour simplifier les calculs.

### 2.2. **Notation Vectorielle :**
   - Soit $ x = (x_1, x_2, \dots, x_d) $ le vecteur de caractéristiques d'un exemple donné, et $ w = (w_0, w_1, \dots, w_d) $ le vecteur de poids du Perceptron, où $ w_0 $ est le biais (ou seuil).
   - Nous pouvons également ajouter une caractéristique constante $ x_0 = 1 $ au vecteur $ x $, ce qui simplifie les calculs et nous permet de traiter le biais de manière uniforme.

### 2.3. **Calcul de l'Hypothèse :**
   - En utilisant la notation vectorielle, l'hypothèse du Perceptron peut être formulée de la manière suivante :
     $$ h(x) = sign(w^T x) $$
     où $ w^T $ représente la transposée du vecteur de poids $ w $, et $ sign(\cdot) $ est la fonction de signe.
   - Cette formulation calcule le produit scalaire entre les vecteurs de poids et de caractéristiques, ce qui revient à la somme pondérée des caractéristiques.

### 2.4. **Interprétation :**
   - Géométriquement, le produit scalaire $ w^T x $ mesure la projection du vecteur de caractéristiques $ x $ sur le vecteur de poids $ w $.
   - Si cette projection dépasse un certain seuil, le Perceptron émet une sortie positive (classe +1), sinon il émet une sortie négative (classe -1).

### 2.5. **Avantages de la Formulation Vectorielle :**
   - La notation vectorielle simplifie les calculs et permet une représentation plus concise de l'hypothèse du Perceptron.
   - Elle facilite également l'extension du Perceptron à des espaces de dimensions supérieures, car les opérations vectorielles restent les mêmes.

En conclusion, la Forme Vectorielle de l'Hypothèse du Perceptron permet une représentation plus efficace du calcul de la sortie du Perceptron en utilisant des opérations vectorielles telles que le produit scalaire. Cette formulation simplifie les calculs et facilite l'extension du Perceptron à des problèmes dans des espaces de dimensions supérieures.

## 3. Perceptrons in R2
- On explore ici l'application des Perceptrons dans un espace bidimensionnel (R2). Les exemples d'entraînement sont des points dans le plan, et les Perceptrons définissent des lignes qui séparent les points en deux classes différentes.
- Les Perceptrons fonctionnent comme des classificateurs linéaires binaires dans cet espace, où chaque ligne de séparation définit une frontière entre les deux classes.

Examinons les Perceptrons en R2.

### 3.1. **Contexte :**
   - Dans un espace bidimensionnel (R2), les Perceptrons sont utilisés pour effectuer une classification binaire en fonction de deux caractéristiques.
   - Les exemples d'entraînement sont représentés comme des points dans le plan cartésien, avec chaque point ayant deux coordonnées (x1, x2).

### 3.2. **Frontières de Décision :**
   - Les Perceptrons définissent des frontières de décision linéaires dans l'espace R2. Ces frontières séparent l'espace en deux régions, une pour chaque classe de sortie.
   - Mathématiquement, une frontière de décision est représentée par une ligne droite dans le plan, où un côté de la ligne est associé à une classe positive et l'autre à une classe négative.

### 3.3. **Forme de l'Hypothèse :**
   - L'hypothèse d'un Perceptron en R2 est une fonction linéaire qui évalue la somme pondérée des caractéristiques (coordonnées x1 et x2) pour chaque exemple d'entraînement.
   - L'expression générale de l'hypothèse est de la forme :
     $$ h(x) = sign(w_0 + w_1x_1 + w_2x_2) $$
   - Cette expression représente une équation d'une ligne droite dans le plan cartésien, où $ w_0 $, $ w_1 $, et $ w_2 $ sont les poids du Perceptron.

**Note importante** :
Dans l'expression $ h(x) = \text{sign}(w_0 + w_1x_1 + w_2x_2) $, $ w_0 $ représente le biais du Perceptron. Le biais est un terme constant ajouté à la somme pondérée des caractéristiques $ w_1x_1 + w_2x_2 $. Il est également parfois appelé le seuil.
Le rôle du biais dans un Perceptron est de contrôler le point de décision, c'est-à-dire le point où la somme pondérée $ w_1x_1 + w_2x_2 $ passe de la classe négative à la classe positive (ou vice versa) lorsqu'elle est comparée au seuil. En ajustant la valeur du biais, on peut déplacer la position de la ligne de décision dans l'espace R2, ce qui permet au Perceptron de mieux s'adapter aux données et de trouver une frontière de décision optimale.

### 3.4. **Interprétation Géométrique :**
   - Les exemples d'entraînement sont classés en fonction de leur position par rapport à la ligne de décision définie par le Perceptron.
   - Les points du côté positif de la ligne sont classés comme appartenant à la classe positive, tandis que ceux du côté négatif sont classés comme appartenant à la classe négative.

### 3.5. **Flexibilité :**
   - Les Perceptrons en R2 peuvent définir différentes frontières de décision en ajustant les valeurs des poids $ w_0 $, $ w_1 $, et $ w_2 $.
   - Les différentes lignes de décision séparent les exemples de manière différente, ce qui permet au Perceptron de s'adapter à des distributions de données variées.

### 3.6. **Limitations :**
   - Les Perceptrons en R2 sont limités aux frontières de décision linéaires. Ils ne peuvent pas capturer des relations non linéaires entre les caractéristiques et les classes de sortie.
   - Cependant, dans de nombreux cas, des frontières de décision linéaires suffisent pour résoudre efficacement des problèmes de classification.

En résumé, les Perceptrons en R2 sont des modèles simples mais puissants pour la classification binaire dans un espace bidimensionnel. Ils définissent des frontières de décision linéaires qui séparent les exemples en fonction de leurs caractéristiques, permettant ainsi la classification des données.


**Notes :** 
Un perceptron est un modèle de classification binaire qui prend un ensemble de données en entrée et les sépare en deux classes, souvent désignées comme classe positive et classe négative, en fonction d'une frontière de décision linéaire.

Voici quelques points clés à retenir sur les perceptrons :

1. **Entrées et Poids :** Un perceptron prend un vecteur d'entrée $ x $ qui représente les caractéristiques d'un exemple donné. Chaque élément de ce vecteur est pondéré par un poids correspondant $ w $.

2. **Fonction de Sommation Pondérée :** Les entrées sont pondérées par les poids et sommées. Cette somme pondérée est ensuite passée à travers une fonction d'activation qui produit la sortie du perceptron.

3. **Fonction d'Activation :** La fonction d'activation la plus couramment utilisée dans les perceptrons est la fonction d'activation de seuil, qui prend la somme pondérée en entrée et produit une sortie binaire en fonction d'un seuil fixé. Si la somme pondérée dépasse le seuil, la sortie est activée (généralement +1 ou 1), sinon la sortie est désactivée (généralement -1 ou 0).

4. **Apprentissage :** Les poids du perceptron sont ajustés pendant le processus d'apprentissage pour minimiser les erreurs de classification sur un ensemble de données d'entraînement. Cela se fait généralement en utilisant des algorithmes d'apprentissage supervisé comme le Perceptron Learning Algorithm (PLA) ou le Pocket Algorithm.

En résumé, un perceptron est un modèle de réseau de neurones simple mais puissant qui peut être utilisé pour effectuer des tâches de classification binaire en séparant les données avec une frontière de décision linéaire.