Description of Representations Used
The code developed for this project aimed to find the best fitting function given a random function. Three optimization strategies were used: Random Search, Hill Climbing, and Genetic Programming. 
The Random Search is the most straightforward algorithm. It generates random mathematical expressions to find the best fit for a dataset. The accuracy of each expression is measured by root mean squared error, and the most accurate one is retained and saved throughout a set number of iterations.
The Hill Climbing Algorithm starts with a random solution and then starts to make adjustments to its search. This algorithm mutates the solution each generation, keeping the variant with improved fitness. This helps maximize the algorithm to find the best-fit function for the data set. 
The Genetic Program is a generational evolutionary algorithm where programs are represented as trees. The fitness was evaluated as a root mean squared error with a penalty applied to the size of the tree. We represent programs as trees and evaluate fitness as root mean squared error with a penalty applied to the size of the tree. Although a maximum size limit isn’t applied, a hard limit on the depth of the tree is applied.
Description of Random Search Algorithm
The random search code is designed to iteratively generate and evaluate random expressions, aiming to minimize a certain fitness function. It enters a loop that persists until a predefined number of generations is reached. Each iteration produces a random expression, evaluates its fitness based on the random mean squared error and compares this to the best fitness achieved thus far. If the current expression is superior, it updates the best-known fitness and expression. 
Description of the Random Mutation Hill Climber Algorithm
The hill-climbing algorithm begins with an initial random solution and iteratively improves it to fit a given dataset. In each generation, it creates a copy of the current best solution, mutates it, and evaluates the new solution's fitness. If the mutated solution offers better fitness (lower root mean squared error), it replaces the current best solution. The process repeats for a specified number of generations, with periodic checkpoints saving the best solution to a file. 
Description of the Evolutionary Algorithm, including the variation and selection methods used
	The algorithm used six mutations, a node add mutation which selects a random leaf node in the function and creates a random expression of depth 1 (depth one is like a single function and its operands, up to 3 nodes). Then, one of the random expression's children is deleted and replaced with the leaf node from the tree and attached to the newly formed expression where the terminal was. For node removal, it selects the parent of a leaf node and replaces it with a random terminal. These have a default probability of 0.05. We also use a subtree add and subtree remove, which are similar but can add a larger tree or remove a larger tree and have a default probability of 0.03. Finally, we have node change and subtree change, which changes the operand of a node or replaces a subtree with a random expression of the same depth, which has probabilities of 0.05 and 0.02, respectively. It is possible to have 0 or more mutations applied to an offspring of two parents after a tournament. We do apply both crossover and mutation in the same generation. 
Subtree crossovers were used where and did not distinguish between inner nodes and leaf nodes during operation. The crossover is undone if it creates trees that violate our constraints on max or min depth. We use a crossover probability of 0.9; without crossover (the other 0.1), individuals are copied directly into the next generation. We also use elitism to keep the best individual from each generation to ensure monotonicity. 
There are 3 possible stopping conditions: max generation, no improvement in many generations, and too long according to clock time. Parsimony pressure is applied with the penalty based on the tree's size (number of nodes), and do not prune the trees at all. Individuals store their fitness for use within a single generation for logging or selection of elites, etc., but we do not use any fitness caching (so we do not allow individuals to skip fitness evaluation in any round). Trees are initialized with the ramped half-and-half method, which uses half FULL and half GROW initialization methods. FULL results in a full tree of a given depth, and constants or variables are only chosen at the max depth for initialization, GROW chooses between variable/constant and function with some probability at all tree depth levels and results in trees of various shapes. Ramped half and half uses a 50/50 strategy based on these two. 
Tournament selection is used with the tournament size configurable. The tournament size directly controls the selection pressure, so with a tournament size of 2, the population remains relatively diverse, whereas, with higher tournament sizes, the population is drawn more strongly toward solutions with better fitness. 
Analysis of the performance of the Algorithms
The Random Search and Hill Climber outperformed expectations regarding finding the best function. The genetic program faced issues regarding which variables to choose, including parsimony coefficient, tournament sizes, crossover probability, and mutation probability. In the first iteration of the Genetic Program, the optimal values of (0.1, 10, 0.9, and 1) were chosen in order to induce more Diversity among solutions and increase the tournament selection results. This, however, resulted in the worst results when searching for the correct function. To remedy this, a grid search was implemented to test the effectiveness of the various different coefficients. 
The analysis found that these values (0.01, 2, 0.9, and 1 )were more optimal, and they were applied to the genetic program. This resulted in the best-fit function. This genetic program also found the solution extremely quickly, but it took many generations to make any kind of slightly more accurate changes. The Random Search and Hill Climber, however, were able to catch the genetic program results within 100,000 generations. 
Likely, if the program could be run for 10^6 generations and not have memory issues, the Genetic program with the grid search results would’ve likely outperformed all other algorithms.

