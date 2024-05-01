**Introduction à CUDA**

Dans cette présentation, nous abordons CUDA, une plateforme de calcul parallèle développée par NVIDIA pour l'accélération de calcul sur les GPU.

**Système CPU-GPU**

La première partie de la diapositive met en évidence la relation entre le CPU (unité centrale de traitement) et le GPU (unité de traitement graphique). Ces deux composants sont essentiels dans les systèmes informatiques modernes. L'illustration montre une carte mère représentant le CPU, équipée de DDR3, et une liaison à une carte graphique représentant le GPU, avec GDDR5. Cette distinction met en évidence les différentes architectures de mémoire utilisées par ces deux composants.

**Vue matérielle**

Ensuite, nous plongeons dans les détails de l'architecture matérielle du GPU. L'unité de base est ce qu'on appelle un "multiprocesseur de flux" (SM). Voici ses caractéristiques :

- **64 cœurs et 256 Ko de registres** : Les cœurs sont les unités de calcul de base, capables de traiter des opérations simultanées. Les registres sont utilisés pour stocker des données temporaires pendant les calculs.
  
- **192 Ko de mémoire partagée + cache L1** : La mémoire partagée est une mémoire rapide utilisée par les threads d'un même bloc pour communiquer et partager des données. Le cache L1 est une mémoire cache de premier niveau, très rapide, utilisée pour stocker des données et instructions fréquemment utilisées.
  
- **Jusqu'à 2048 threads par SM** : Les threads sont des unités d'exécution de base, des tâches individuelles pouvant être exécutées simultanément sur les cœurs du GPU.

Pour donner un exemple concret, la carte P100 est mentionnée, avec ses spécifications :

- **108 SMs -> 6912 cœurs** : Ceci indique le nombre de multiprocesseurs et de cœurs disponibles sur la carte.
  
- **40 Mo de cache L2 partagé** : Le cache L2 est une mémoire cache de deuxième niveau, également utilisée pour stocker des données fréquemment utilisées, mais avec une capacité plus grande que le cache L1.
  
- **40 Go de mémoire HBM, bande passante de 1555 Go/s** : HBM (High Bandwidth Memory) est une technologie de mémoire empilée qui offre une bande passante très élevée, ce qui est crucial pour les opérations de calcul intensives.

La représentation schématique montre un GPU avec plusieurs blocs "SM" à l'intérieur, ainsi qu'un bloc "cache L2". De plus, un zoom sur un bloc SM met en évidence les petits blocs individuels ainsi que le bloc "cache L1/mémoire partagée". Cela illustre la hiérarchie de mémoire et la structure interne d'un SM.

Cette diapositive fournit une vue d'ensemble des composants matériels essentiels d'un GPU CUDA et de leurs spécifications clés, jetant ainsi les bases pour une compréhension plus approfondie de la plateforme CUDA et de son fonctionnement.


**Multithreading**

Cette section explique le concept de multithreading et comment il est utilisé dans le contexte de CUDA.

- Chaque thread exécute une opération avant que la suivante ne commence, mais grâce à la présence de nombreux threads et de nombreux cœurs de traitement, plusieurs opérations peuvent être exécutées simultanément.
  
- L'accès à la mémoire de l'appareil (GPU) comporte un délai de 400 à 600 cycles. Les threads doivent donc attendre ce délai, ce qui souligne l'importance d'avoir de nombreux threads qui peuvent effectuer d'autres tâches pendant ce temps d'attente.

**Vue logicielle**

Cette partie présente la vue logicielle de CUDA et décrit les étapes typiques dans la programmation CUDA.

1. **Initialiser la carte** : Prépare la carte GPU pour l'utilisation.
2. **Allouer la mémoire** : Réserve de l'espace mémoire sur l'ordinateur hôte (CPU) et sur l'appareil (GPU).
3. **Copier les données** : Transférer les données depuis l'ordinateur hôte vers l'appareil GPU.
4. **Lancer plusieurs instances d'un "kernel" de calcul** : Démarre l'exécution du code de calcul sur le GPU.
5. **Copier les données de retour** : Transférer les résultats du calcul depuis le GPU vers l'ordinateur hôte.
6. **Répéter** : Cette séquence peut être répétée pour effectuer des calculs supplémentaires.
7. **Libérer la mémoire** : Libérer l'espace mémoire alloué et terminer l'exécution.

**Programmation CUDA**

- CUDA est une extension de C/C++ permettant d'implémenter du code SIMT (Single Instruction, Multiple Threads) qui s'exécute sur le GPU.

- Abstraction SIMT : Le code est écrit du point de vue d'un seul thread. Les threads sont organisés en groupes appelés blocs, où les threads d'un même bloc peuvent communiquer entre eux. De nombreux blocs peuvent être lancés, mais ils ne peuvent pas interagir directement.

- Il existe également des appels API pour gérer le GPU et effectuer des transferts de mémoire.

**Lancement d'une opération GPU**

En termes de programmation CUDA, le lancement d'une opération GPU est effectué à l'aide d'une syntaxe spécifique :

```cuda
kernel_routine<<<gridDim, blockDim>>>(args);
```

- `gridDim` est le nombre d'instances du noyau : le nombre de groupes de threads.
- `blockDim` est le nombre de threads dans chaque instance : la taille des groupes.
- `args` est un certain nombre d'arguments à passer au noyau GPU.

**Niveau inférieur de programmation CUDA**

Chaque instance (groupe) du noyau est attribuée à un SM et est exécutée par un certain nombre de threads. Chaque thread a accès à diverses informations, telles que des variables passées en arguments, des pointeurs vers des tableaux en mémoire de l'appareil, des constantes globales, et des types de mémoire tels que la mémoire partagée et la mémoire privée (registres). De plus, chaque thread connaît des variables spéciales qui décrivent sa position dans la grille de threads et dans son bloc respectif.

**Exemple de code hôte (CPU)**

Le code hôte (CPU) est également illustré, montrant comment interagir avec le GPU pour exécuter des calculs CUDA. L'exemple montre l'allocation de mémoire, le lancement d'un noyau CUDA, le transfert des résultats de retour vers le CPU et la libération de la mémoire après utilisation.

```cpp
int main(int argc, char **argv) {

    // DECLARATION DES VARIABLES
    float *h_x, *d_x; // h=host, d=device, utilisés pour stocker des données sur l'hôte (CPU) et sur le périphérique (GPU) respectivement
    int nblocks=2, nthreads=8, nsize=2*8; // nombre de blocs, le nombre de threads par bloc, et la taille des données à traiter

    // ALLOCATION DE MEMOIRE
    h_x = (float *)malloc(nsize*sizeof(float)); //  Alloue de la mémoire sur l'hôte pour stocker les données. La fonction malloc est utilisée pour allouer de la mémoire dynamique.
    cudaMalloc((void **)&d_x,nsize*sizeof(float)); // cudaMalloc- Alloue de la mémoire sur le périphérique (GPU) pour stocker les données.

    // APPEL DU KERNEL CUDA
    my_first_kernel<<<nblocks,nthreads>>>(d_x); // Lance l'exécution du kernel CUDA appelé my_first_kernel. Les chevrons <<< >>> sont utilisés pour spécifier le nombre de blocs et de threads à utiliser pour exécuter le kernel. Dans cet exemple, nous utilisons 2 blocs et 8 threads par bloc.

    // TRANSFERT DES DONNES DE RETOUR
    cudaMemcpy(h_x,d_x,nsize*sizeof(float), cudaMemcpyDeviceToHost); // Copie les données résultantes depuis le périphérique vers l'hôte. La fonction cudaMemcpy est utilisée pour effectuer des transferts de données entre l'hôte et le périphérique. cudaMemcpyDeviceToHost indique que les données sont copiées depuis le périphérique vers l'hôte

    // AFFICHAGE DES RESULTATS
    for (int n=0; n<nsize; n++)
    printf(" n, x = %d %f \n",n,h_x[n]);

    //LIBERATION DE MEMOIRE
    cudaFree(d_x); free(h_x); // Libère la mémoire allouée précédemment sur le périphérique et sur l'hôte respectivement. Les fonctions cudaFree et free sont utilisées pour libérer la mémoire sur le GPU et sur le CPU.
}
```

**Appels API clés**

Cette partie du cours met en lumière les principales fonctions d'API utilisées dans la programmation CUDA :

- `cudaMalloc(void** ptr, size_t size)`: Alloue de la mémoire sur l'appareil (GPU).
  
- `cudaMemcpy(void * to, void *from, size_t size, direction)`: Copie les données entre l'ordinateur hôte et l'appareil (GPU). Les directions typiques sont de l'hôte vers le périphérique (`cudaMemcpyHostToDevice`) et du périphérique vers l'hôte (`cudaMemcpyDeviceToHost`).
  
- `cudaFree(void *ptr)`: Libère la mémoire allouée sur l'appareil (GPU).
  
- `cudaDeviceSynchronize()`: Synchronise le périphérique (GPU) avec le CPU. Les lancements de noyaux sont asynchrones, donc cette fonction garantit que toutes les opérations précédentes sur le périphérique sont terminées avant de continuer l'exécution sur le CPU.
  
- Les lancements de noyaux sont asynchrones, ce qui signifie que les appels aux fonctions ci-dessus peuvent être retardés en raison de l'acheminement asynchrone des noyaux.

**Code du noyau**

Le code du noyau est également abordé dans cette section. Voici les points saillants :

- L'identificateur `__global__` indique qu'il s'agit d'une fonction de noyau, c'est-à-dire une fonction qui s'exécutera sur le GPU.
  
- Chaque thread définit un élément du tableau `x`.
  
- Dans chaque bloc de threads, `threadIdx.x` varie de 0 à `blockDim.x - 1`, il est donc nécessaire d'ajouter `blockDim.x * blockIdx.x` pour obtenir un identifiant unique globalement (`tid`).

**Exemple de code du noyau**

```cuda
#include <helper_cuda.h>

__global__ void my_first_kernel(float *x)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    x[tid] = (float) threadIdx.x;
}
```

- `__global__` indique que c'est une fonction de noyau.
  
- Chaque thread affecte une valeur de type `float` à un élément du tableau `x`. Cette valeur est simplement l'indice du thread dans son bloc.
  
- `tid` est calculé comme l'indice du thread dans le bloc (`threadIdx.x`) plus le produit de la taille du bloc (`blockDim.x`) et de l'indice du bloc (`blockIdx.x`), ce qui garantit que chaque thread obtient un identifiant global unique dans l'ensemble du grid de threads.

Ce code illustre la manière dont les opérations sont exécutées simultanément par les threads sur le GPU et comment les identifiants uniques sont utilisés pour accéder aux données de manière ordonnée et cohérente.


**Programmation CUDA - Suite**

**Limite de threads par bloc**

Il est important de noter qu'il y a une limite de 1024 threads par bloc. Pour comprendre comment ces threads sont exécutés, prenons un exemple : supposons que nous ayons 1000 blocs, chacun avec 128 threads. Sur du matériel Pascal, vous auriez probablement 8 à 16 blocs s'exécutant simultanément sur chaque SM (multiprocesseur de flux), et chaque bloc comporte 4 warps, totalisant de 32 à 64 warps. À chaque cycle d'horloge, l'ordonnanceur de warps du SM choisit un warp prêt à être exécuté, sans attendre la mémoire ni les instructions précédentes. L'objectif pour le programmeur est de s'assurer qu'il y a suffisamment de warps disponibles à tout moment pour l'exécution.

**Importance de la mémoire - Localité**

La mémoire est un aspect crucial des performances. Considérez un GPU avec une puissance de calcul de 9.7 téraflops/s mais une bande passante mémoire de 1555 Go/s. Les performances dépendent de la manière dont les données sont accédées en mémoire. Les modèles d'accès à la mémoire spatiale et temporelle sont importants pour les performances :

- La localité spatiale : si nous accédons à `x[i]`, nous accéderons probablement à `x[i+1]` aussi.
  
- La localité temporelle : si nous accédons à `x[i]`, il est probable que nous y accédions à nouveau.

L'existence de caches vise à accélérer ces accès. Cependant, il y a peu de cache sur le GPU. L'exploitation de la localité spatiale est essentielle, en espérant que si un thread accède à `x[i]`, alors le thread suivant accédera à `x[i+1]`. Cela est dû au fait que les threads `p` et `p+1` exécutent toujours la même instruction (comme un accès mémoire) mais avec un index différent.

**Exemple de bon et mauvais noyau**

Un "bon" noyau assure que les threads dans un warp accèdent à des éléments de tableau voisins, couvrant une ligne de cache complète et exploitant une localité spatiale parfaite. Un "mauvais" noyau, en revanche, implique que différents threads dans un warp accèdent à des données sur différentes lignes de cache, générant un modèle d'accès stridé, ce qui est lent et inefficace.

**Variables globales, constantes et registres**

Les tableaux globaux sur l'appareil sont maintenus dans la mémoire de l'appareil, alloués via le code hôte et existent jusqu'à ce que le code hôte les libère. Les variables globales peuvent également être déclarées dans le code du noyau avec une portée globale, comme les variables C normales. De plus, il est possible de déclarer des variables constantes, qui ne peuvent pas être modifiées par les noyaux. Les variables constantes sont idéales pour la diffusion : tous les threads lisent la même constante. Enfin, tout ce qui est déclaré dans la portée locale (dans la fonction `__global__`) est mappé sur des registres par défaut.


**Registres**

Si votre application nécessite plus de registres que ce que le matériel peut fournir, ils seront alloués dans la mémoire globale par le compilateur. Cependant, cette allocation dans la mémoire globale est coûteuse et ces registres peuvent être mis en cache dans le cache L1 pour améliorer les performances. Vous pouvez compiler avec l'option `-Xptxas=-v` pour obtenir un rapport du compilateur sur le nombre de registres que votre application utilise.

**Blocs 2D**

Dans le cas simple que nous avons vu précédemment, nous avions une grille de blocs à une dimension et un ensemble de threads à une dimension dans chaque bloc. Si nous voulons utiliser un ensemble de threads à deux dimensions, nous utilisons `blockDim.x` et `blockDim.y`, ainsi que `threadIdx.x` et `threadIdx.y`, etc. Pour lancer une grille de blocs 2D, les threads sont assignés aux warps de manière contiguë par rangée.

```cuda
dim3 threads(16,4);
dim3 blocks(8,22);
my_new_kernel<<<blocks,threads>>>(d_x);
```

Voici un exemple de code avec une grille de blocs 2D :

```cuda
__global__ void lap(int I, int J, float *u1, float *u2) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int id = i + j * I;
    if (i==0 || i==I-1 || j==0 || j==J-1) {
        u2[id] = u1[id]; // Conditions aux limites de Dirichlet
    }
    else {
        u2[id] = 0.25f * (u1[id-1] + u1[id+1] + u1[id-I] + u1[id+I]);
    }
}
```

**Cache de texture**

La mémoire de texture, initialement destinée aux graphiques, est efficace pour les données en lecture seule, en particulier pour les accès non uniformes. Elle dispose de 24 Ko sur Pascal et utilise des lignes de cache de 32 octets, ce qui signifie qu'il y a plus de lignes de cache par thread. Pour utiliser la mémoire de texture, les pointeurs doivent être déclarés avec `const type * __restrict__ var`.


**Mémoire partagée**

Dans un noyau CUDA, le préfixe `__shared__` comme dans :

```cuda
__shared__ int x_dim;
__shared__ float x[128];
```

déclare des données partagées entre tous les threads d'un bloc de threads. N'importe lequel de ces threads peut le lire ou l'écrire. Cependant, la portée de cette mémoire partagée est limitée aux threads du même bloc ; chaque bloc possède sa propre portion de mémoire partagée.

**Avantages de la mémoire partagée :**

- Essentiel pour les opérations nécessitant une communication entre les threads.
  
- Utile pour la réutilisation des données (peut être utilisée comme un cache !).
  
- Réduit l'utilisation des registres lorsque la variable a la même valeur pour tous les threads.

**Synchronisation**

Tous les threads dans un bloc ne s'exécutent pas simultanément. Si un thread N doit lire la valeur écrite par un thread M, il faut garantir l'ordonnancement. Ainsi, nous avons besoin de synchronisation pour assurer une utilisation correcte de la mémoire partagée pour la communication. La fonction `__syncthreads()` insère une barrière ; aucun thread/warp n'est autorisé à continuer au-delà de ce point tant que les autres ne l'ont pas atteint.

**Mémoire partagée & syncthreads**

Dans cet exemple, la mémoire partagée est utilisée pour stocker des valeurs lues à partir d'un tableau global. Chaque thread charge une valeur dans la mémoire partagée, puis `__syncthreads()` est utilisé pour s'assurer que tous les threads ont terminé le chargement avant de poursuivre l'exécution. Cela évite les conflits lors de la lecture de valeurs écrites par d'autres threads.

**Opérations atomiques**

Dans certains cas, une application nécessite que plusieurs threads mettent à jour un compteur en mémoire partagée ou globale. Les opérations atomiques permettent de garantir que ces mises à jour se déroulent de manière sûre et cohérente, même lorsque plusieurs threads tentent de les effectuer simultanément. Par exemple, `atomicAdd` ajoute une valeur à une adresse spécifiée et renvoie l'ancienne valeur.

Les opérations atomiques prennent en charge plusieurs types de données, principalement des entiers et des flottants, et comprennent des opérations telles que l'addition, le minimum/maximum, l'incrémentation/décrémentation, l'échange/comparer-et-échanger.


**Divergence des warps**

Les threads sont exécutés en warps de 32, avec tous les threads du warp exécutant la même instruction en même temps. Mais que se passe-t-il si différents threads d'un warp doivent faire des choses différentes ? C'est ce qu'on appelle la divergence des warps dans CUDA.

Sur les GPU NVIDIA, les instructions prédicatives sont utilisées pour exécuter des instructions uniquement si un indicateur logique est vrai. Par exemple, dans un cas où différents threads dans un warp ont besoin d'exécuter différentes instructions en fonction d'une condition, tous les threads calculent le prédicat logique et ensuite deux instructions prédicatives sont exécutées en fonction de ce prédicat.

La divergence des warps peut entraîner une perte de performance significative car le coût d'exécution est la somme des deux branches, même si une seule branche est réellement exécutée. Cette perte de performance peut être très coûteuse, surtout si une branche est très coûteuse, comme le calcul de la racine carrée.

La divergence des warps peut également entraîner une perte d'efficacité parallèle, en particulier dans les pires cas où un thread nécessite une branche coûteuse, ce qui peut entraîner une perte de 32x dans le pire des cas.

**Réductions**

Les opérations de réduction sont couramment utilisées pour calculer la somme, le minimum ou le maximum d'un ensemble de valeurs. Pour qu'un opérateur de réduction fonctionne correctement, il doit être commutatif et associatif, ce qui signifie que les éléments peuvent être réarrangés et combinés dans n'importe quel ordre.

Une approche courante pour la réduction consiste à effectuer d'abord une réduction locale au sein de chaque bloc de threads, puis à combiner les résultats partiels des blocs pour obtenir le résultat final. Cette réduction locale peut être réalisée en utilisant de la mémoire partagée, où chaque bloc stocke ses résultats partiels et les combine en utilisant une opération de réduction. Ensuite, les résultats partiels de chaque bloc sont combinés pour obtenir le résultat final.

**Warp shuffles**

Les warp shuffles sont un mécanisme permettant de déplacer des données entre les threads dans le même warp sans utiliser de mémoire partagée. Cela fonctionne pour des données de 32 bits.

Il existe quatre variantes de warp shuffles :

- `__shfl_up` : copie à partir d'une voie avec un ID inférieur par rapport à l'appelant.
- `__shfl_down` : copie à partir d'une voie avec un ID supérieur par rapport à l'appelant.
- `__shfl_xor` : copie à partir d'une voie basée sur le XOR bit à bit de son propre ID.
- `__shfl` : copie à partir d'une voie indexée.

Par exemple, `__shfl_up` prend comme arguments une variable de registre local et un décalage dans le warp. Si la voie à décaler n'existe pas, la valeur est prise à partir du thread actuel.

Les warp shuffles peuvent être utilisés pour des opérations telles que la sommation de tous les éléments dans un warp sans utiliser de mémoire partagée, comme illustré dans l'exemple où la valeur est additionnée avec la valeur décalée vers le bas à chaque itération d'une boucle.

**Opérations atomiques**

Les opérations atomiques sont essentielles pour garantir l'intégrité des données partagées entre les threads. Par exemple, `atomicCAS` est une opération de comparaison-et-échange qui permet de mettre à jour une variable atomiquement. Si la valeur actuelle est égale à la valeur attendue, alors une nouvelle valeur est stockée à la place et l'ancienne valeur est renvoyée. Ceci est souvent utilisé pour implémenter des verrous pour des régions critiques du code.

Dans l'exemple de verrouillage global atomique, une variable `lock` est utilisée pour indiquer si une ressource est verrouillée ou non. Le verrou est acquis en utilisant `atomicCAS` et libéré en remettant `lock` à 0.

Cependant, un problème potentiel est que lorsque les threads écrivent des données dans la mémoire du périphérique, l'ordre d'achèvement n'est pas garanti. Pour résoudre ce problème, `__threadfence()` peut être utilisé pour attendre que toutes les écritures dans la mémoire globale soient visibles par tous les threads sur le périphérique. Il existe également `__threadfence_block()` pour attendre que toutes les écritures dans la mémoire globale et partagée soient visibles par tous les threads dans le bloc.
