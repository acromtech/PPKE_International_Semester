## Lecture 2: Learning to Answer Yes/No
- Perceptron Hypothesis Set
- Perceptron Learning Algorithm (PLA)
- Guarantee of PLA
- Non-Separable Data

## Lecture 3: Types of Learning
focus: binary classification or regression from a
batch of supervised data with concrete features

## Lecture 4: Feasibility of Learning
- Learning is Impossible?
- Probability to the Rescue
- Connection to Learning
- Connection to Real Learning
Conclusion : 
- learning is PAC-possible
- if enough statistical data and finite |H|

## Lecture ?? : Noise and Error
Where do these error measures come from?
How to introduce uncertainty?

## Lecture 5: Training versus Testing
- Recap and Preview
- Effective Number of Lines
- Effective Number of Hypotheses
- Break Point
Conclusion :
- Recap and Preview: two questions: Eout(g) ≈Ein(g), and Ein(g) ≈0
- Effective Number of Lines at most 14 through the eye of 4 inputs
- Effective Number of Hypotheses at most mH(N) through the eye of N inputs
- Break Point when mH(N) becomes ‘non-exponential’

## Lecture 6: Theory of Generalization
- Restriction of Break Point : break point ‘breaks’ consequent points
- Bounding Function: Basic Cases B(N ,k ) bounds mH(N ) with break point k
- Bounding Function: Inductive Cases B(N ,k ) is poly (N )
- A Pictorial Proof: mH(N ) can replace M with a few changes

## Lecture 7: The VC Dimension
- Definition of VC Dimension : maximum non-break point
- VC Dimension of Perceptrons : dVC (H) = d + 1
- Physical Intuition of VC Dimension : dVC ≈#free parameters
- Interpreting VC Dimension : loosely: model complexity & sample complexity

## Lecture 9 : Linear Regression
analytic solution wLIN = X†y with
linear regression hypotheses and squared error

## Lecture 10: Logistic Regression
- Logistic Regression Problem : P (+1|x) as target and θ(wT x) as hypotheses
- Logistic Regression Error : cross-entropy (negative log likelihood)
- Gradient of Logistic Regression Error : θ-weighted sum of data vectors
- Gradient Descent : roll downhill by −∇Ein(w)

## Lecture 11: Linear Models for Classification
binary classification via (logistic) regression;
multiclass via OVA/OVO decomposition

## Lecture 12: Nonlinear Transformation
- Quadratic Hypotheses : linear hypotheses on quadratic-transformed data
- Nonlinear Transform : happy linear modeling after Z = Φ(X)
- Price of Nonlinear Transform : computation/storage/[model complexity]
- Structured Hypothesis Sets : linear/simpler model first

## Lecture 13: Hazard of Overfitting
- What is Overfitting? : lower Ein but higher Eout
- The Role of Noise and Data Size : overfitting ‘easily’ happens!
- Deterministic Noise : what Hcannot capture acts like noise
- Dealing with Overfitting : data cleaning/pruning/hinting, and more

## Lecture 14: Regularization
- Regularized Hypothesis Set : original H+ constraint
- Weight Decay Regularization : add λ/N*w^(T)*w in Eaug
- Regularization and VC Theory : regularization decreases dEFF
- General Regularizers target-dependent, [plausible], or [friendly]

## Lecture 15: Validation
- Model Selection Problem : dangerous by Ein and dishonest by Etest
- Validation : select with Eval(Am(Dtrain)) while returning Am∗(D)
- Leave-One-Out Cross Validation : huge computation for almost unbiased estimate
- V-Fold Cross Validation : reasonable computation and performance

## Lecture ?? : Rademacher complexity
...

## Lecture 12: Neural Network
- Motivation : multi-layer for power with biological inspirations
- Neural Network Hypothesis : layered pattern extraction until linear hypothesis
- Neural Network Learning : backprop to compute gradient efficiently
- Optimization and Regularization : tricks on initialization, regularizer, early stopping

## Lecture ?? : Support Vector Machines (SVM)
...

## Lecture 13: Deep Learning
- Deep Neural Network difficult hierarchical feature extraction problem
- Autoencoder : unsupervised NNet learning of representation
- Denoising Autoencoder : using noise as hints for regularization
- Principal Component Analysis : linear autoencoder variant for data processing

## Lecture ??: Reinforcement Learning
...




# Theory of Generalization

La théorie de la généralisation explore comment un modèle appris sur un ensemble d'entraînement peut se généraliser à de nouveaux exemples non vus. Cette section couvre les fonctions de bornage, les points de rupture et une preuve visuelle pour comprendre les fondements théoriques de la généralisation.

## Bounding Function: Inductive Cases
### Fonction de Bornage : Cas Inductifs
La fonction de bornage est utilisée pour limiter la complexité d'un modèle en fonction du nombre de points d'entrée. Le bornage se fait en utilisant une approche inductive.

### Théorème de la Fonction de Bornage
Pour une hypothèse ayant un point de rupture \(k\), la fonction de croissance \(B(N, k)\) est bornée par :
\[ B(N, k) \leq \sum_{i=0}^{k-1} \binom{N}{i} \]
Cette borne montre que \(m_H(N)\) est une fonction polynomiale de \(N\) si un point de rupture existe.

### Points de Rupture
Le point de rupture est le plus petit entier \(k\) pour lequel \(B(N, k)\) est strictement inférieur à \(2^N\).

#### Exemples de Points de Rupture
- **Rayons positifs** : \(m_H(N) = N + 1\), point de rupture à 2
- **Intervalles positifs** : \(m_H(N) = \frac{1}{2}N^2 + \frac{1}{2}N + 1\), point de rupture à 3
- **Perceptrons 2D** : \(m_H(N) = \frac{1}{6}N^3 + \frac{5}{6}N + 1\), point de rupture à 4

## A Pictorial Proof
### Étape 1: Remplacement de Eout par Ein'
On remplace \(Eout\) par \(E'_{in}\) pour éviter les biais infinis. \(E'_{in}\) est calculé à partir d'un sous-ensemble de vérification de taille \(N\).

### Étape 2: Décomposition de H par Type
On décompose l'ensemble des hypothèses \(H\) en fonction des types de données, ce qui permet d'appliquer une borne d'union sur \(m_H(2N)\).

### Étape 3: Utilisation de l'Inégalité de Hoeffding sans Remplacement
On utilise l'inégalité de Hoeffding sans remplacement pour prouver que :
\[ P\left[ \exists h \in H \text{ t.q. } \left| E_{in}(h) - E_{out}(h) \right| > \epsilon \right] \leq 4m_H(2N) \exp \left( -\frac{1}{8} \epsilon^2 N \right) \]
Ce résultat est connu sous le nom de borne de Vapnik-Chervonenkis (VC).

[Retour à la table des matières](./README.md)
