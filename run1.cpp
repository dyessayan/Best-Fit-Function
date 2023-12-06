#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <random>
#include <thread>
#include <unordered_set>
#include <utility>
#include <chrono>


const double CONSTANT_RANGE = 5.0;  
const int MAX_INITIAL_SIZE  = 5;
const int MIN_TREE_DEPTH    = 1;
const int MAX_TREE_DEPTH    = 6;
const int POPULATION_SIZE   = 250;

//Probabilities to mutate single nodes
double NODE_CHANGEOP_PROBABILITY        = 0.05;
double NODE_ADD_PROBABILITY             = 0.05;
double NODE_DEL_PROBABILITY             = 0.05;

//probabilities to mutate subtrees
double SUBTREE_MUTATION_PROBABILITY     = 0.03;
double ADD_SUBTREE_MUTATION_PROBABILITY = 0.02;
double REM_SUBTREE_MUTATION_PROBABILITY = 0.02;

//crossover probability
double CROSSOVER_PROBABILITY             = 0.90;
const double RAND_MAX_DOUBLE = static_cast<double>(RAND_MAX);


const int NUM_ARITY_2 = 4;
const int NUM_ARITY_1 = 6;
const int NUM_ARITY_0 = 2;

bool DEBUG = false; //set to true to enable assertions for identifying causes of invalid trees.

enum fns { ADD, SUB, MUL, DIV, COS, SIN, SQR, CUB, SQRT, CBRT,  ID, CONST};
std::vector<double> R;

class Node {
  public:
  fns fn;
  long double fitness;
  long double unpenalized_fitness;
  double value;
  uint64_t size;
  Node * leftchild; // left subtree
  Node * rightchild; //right subtree
  Node * parent;

  Node() : fn(ADD), value(0), leftchild(nullptr), rightchild(nullptr), parent(nullptr), fitness(999999.99), size(9999999) {};
  Node(fns fn) : fn(fn), value(0), leftchild(nullptr), rightchild(nullptr), parent(nullptr), fitness(999999.99), size(9999999)  {};
  Node(fns t, double value) : fn(t), value(value), leftchild(nullptr), rightchild(nullptr), parent(nullptr), fitness(999999.99), size(9999999)  {};
  Node(fns fn, Node* parent) : fn(fn), value(0), leftchild(nullptr), rightchild(nullptr), parent(parent), fitness(999999.99), size(9999999)  {};
  Node(fns t, double value, Node* parent) : fn(t), value(value), leftchild(nullptr), rightchild(nullptr), parent(parent), fitness(999999.99), size(9999999)  {};
  Node(fns t, double value, Node* parent, double fitness, uint64_t size) : fn(t), value(value), leftchild(nullptr), rightchild(nullptr), 
                parent(parent), fitness(fitness), size(size) {};
};

//bool tree_ok(Node* n, Node* parent=nullptr);
int get_arity(Node* n);
int count_nodes(Node * n, int numnodes);
bool tree_ok(Node* n, Node* parent = nullptr) {
    if(!n) return true;

    if(n->parent != parent) return false;                                // A node's parent should be the same as its parent's child
    if((get_arity(n) > 0) && (n->leftchild == nullptr)) return false;    // if node has arity 1 or 2 its leftchild should not be nullptr
    if((get_arity(n) == 2) && (n->rightchild == nullptr)) return false;  // if node has arity 2 its rightchild should not be nullptr
    return (tree_ok(n->leftchild, n) && tree_ok(n->rightchild, n));
}
bool tree_ok_newroot(Node* n) {
    if(!n) return true;

    //if(n->parent != parent) return false;                                // A node's parent should be the same as its parent's child
    if((get_arity(n) > 0) && (n->leftchild == nullptr)) return false;    // if node has arity 1 or 2 its leftchild should not be nullptr
    if((get_arity(n) == 2) && (n->rightchild == nullptr)) return false;  // if node has arity 2 its rightchild should not be nullptr
    return (tree_ok(n->leftchild, n) && tree_ok(n->rightchild, n));
}
Node* update_root(Node * n) {
    if (n->parent == nullptr) {
        return n;
    }
    else {
        return update_root(n->parent);
    }
}

std::string node_to_string(Node* n) {
    if(!n) return "";
    std::string result;
    if(n->fn == ADD) {
        result.push_back('+');
        result += node_to_string(n->leftchild);
        result += node_to_string(n->rightchild);
    }
    if(n->fn == SUB) {
        result.push_back('~'); //Use ~ instead of - for subtraction since - is for negative numbers
        result += node_to_string(n->leftchild);
        result += node_to_string(n->rightchild);
    }
    if(n->fn == MUL) {
        result.push_back('*');
        result += node_to_string(n->leftchild);
        result += node_to_string(n->rightchild);
    }
    if(n->fn == DIV) {
        result.push_back('/');
        result += node_to_string(n->leftchild);
        result += node_to_string(n->rightchild);
    }
    if(n->fn == SQR) {
        result.push_back('@');
        result += node_to_string(n->leftchild);
    }
    if(n->fn == CUB) {
        result.push_back('#');
        result += node_to_string(n->leftchild);
    }
    if(n->fn == SQRT) {
        result.push_back('!');
        result += node_to_string(n->leftchild);
    }
    if(n->fn == CBRT) {
        result.push_back('$');
        result += node_to_string(n->leftchild);
    }
    if(n->fn == SIN) {
        result.push_back('S');
        result += node_to_string(n->leftchild);
    }
    if(n->fn == COS) {
        result.push_back('C');
        result += node_to_string(n->leftchild);
    }
    if(n->fn == ID) {
        result = "x";
    }
    if(n->fn == CONST) {
        result = "d" + std::to_string(n->value);
    }
    return result;
}

Node* string_to_node(const std::string &s, uint16_t &pos, Node* parent = nullptr) {
    if(pos >= s.size()) return nullptr;
    if(s.at(pos) == 'd') pos++;

    Node * n = new Node();
    n->parent = parent;
    if (s.at(pos) == 'x') {
        n->fn = ID;
        pos++;
    }
    else if (isdigit(s.at(pos)) || (s.at(pos) == '-')) {
        uint16_t digit_end = s.find_first_not_of("0123456789.", pos+1); //get the first non-numeric, non-decimal element's position
        n->fn = CONST;
        n->value = std::stod(s.substr(pos, digit_end));
        pos = digit_end;
    }
    else {
        if(s.at(pos) == '+') {
            n->fn = ADD;
            n->leftchild  = string_to_node(s, ++pos, n);
            n->rightchild = string_to_node(s, pos, n);
        }
        else if(s.at(pos) == '~') {
            n->fn = SUB;
            n->leftchild  = string_to_node(s, ++pos, n);
            n->rightchild = string_to_node(s, pos, n);
        }
        else if(s.at(pos) == '*') {
            n->fn = MUL;
            n->leftchild  = string_to_node(s, ++pos, n);
            n->rightchild = string_to_node(s, pos, n);
        }
        else if(s.at(pos) == '/') {
            n->fn = DIV;
            n->leftchild  = string_to_node(s, ++pos, n);
            n->rightchild = string_to_node(s, pos, n);
        }
        else if(s.at(pos) == 'S') {
            n->fn = SIN;
            n->leftchild  = string_to_node(s, ++pos, n);
        }
        else if(s.at(pos) == 'C') {
            n->fn = COS;
            n->leftchild  = string_to_node(s, ++pos, n);
        }
        else if(s.at(pos) == '@') {
            n->fn = SQR;
            n->leftchild  = string_to_node(s, ++pos, n);
        }
        else if(s.at(pos) == '#') {
            n->fn = CUB;
            n->leftchild  = string_to_node(s, ++pos, n);
        }
        else if(s.at(pos) == '!') {
            n->fn = SQRT;
            n->leftchild  = string_to_node(s, ++pos, n);
        }
        else if(s.at(pos) == '$') {
            n->fn = CBRT;
            n->leftchild  = string_to_node(s, ++pos, n);
        }
    }
    return n;
    // if (pos >= s.size()) {
    //     assert(tree_ok(n));
    //     std::cout << print_fn(n) << std::endl;
    //     std::cout << s << std::endl;
    //     assert(print_fn(n) == s);
    // }
}

void checkpoint(std::vector<Node*> population, std::string filename) {
    std::ofstream f(filename);
    if(!f.is_open()) {
        std::cout << "Could not open " << filename << std::endl;
        return;
    }
    for(auto genome : population) {
        f << node_to_string(genome) << std::endl;
    }
    f.close();
}

std::vector<Node*> load_checkpoint(std::string filename) {
    std::string line;
    std::ifstream f(filename);
    std::vector<Node*> population;
    if(!f.is_open()) {
        std::cout << "Could not open " << filename << std::endl;
    }
    while(std::getline(f, line)) {
        uint16_t start = 0;
        Node* n = string_to_node(line, start);
        population.push_back(n);
    }
    return population;
}

Node* deepcopy(Node* tree, Node* parent = nullptr) {
    if(!tree) {
        return nullptr;
    }
    Node* treecopy = new Node(tree->fn, tree->value, parent, tree->fitness, tree->size);
    treecopy->leftchild = deepcopy(tree->leftchild, treecopy);
    treecopy->rightchild = deepcopy(tree->rightchild, treecopy);
    return treecopy;
}

double evaluate(Node* node, double x) {

    // We need to make sure to pass the value of x to this function
    // Make sure evaluate works fine with unary operators where 
    switch(node->fn) {
        case ADD: return evaluate(node->leftchild, x) + evaluate(node->rightchild, x);
        case SUB: return evaluate(node->leftchild, x) - evaluate(node->rightchild, x);
        case MUL: return evaluate(node->leftchild, x) * evaluate(node->rightchild, x);
        case DIV: 
        {
            double y = evaluate(node->rightchild, x);
            if (y != 0)  return evaluate(node->leftchild, x) / y; // safe division makes the program runnable 
            else return 99999999.99;
        }
        case COS: return std::cos(evaluate(node->leftchild, x));
        case SIN: return std::sin(evaluate(node->leftchild, x));
        case SQR: return std::pow(evaluate(node->leftchild, x), 2.0);
        case CUB: return std::pow(evaluate(node->leftchild, x), 3.0);
        case SQRT: {
            double y = evaluate(node->leftchild, x);
            if (y<0) return 99999999.99;
            else return std::sqrt(evaluate(node->leftchild, x));
        }
        case CBRT: return std::cbrt(evaluate(node->leftchild, x));
        case ID: return x;
        case CONST: return node->value;
    }
    return 0;
}

std::string print_fn(Node* node) {
    switch(node->fn) {
        case ADD: return  "(" + print_fn(node->leftchild) + " + " + print_fn(node->rightchild) + ")";
        case SUB: return  "(" + print_fn(node->leftchild) + " - " + print_fn(node->rightchild) + ")";
        case MUL: return  "(" + print_fn(node->leftchild) + " * " + print_fn(node->rightchild) + ")";
        case DIV: return  "(" + print_fn(node->leftchild) + " / " + print_fn(node->rightchild) + ")";
        case COS: return  "Cos(" + print_fn(node->leftchild) + ")";
        case SIN: return  "Sin(" + print_fn(node->leftchild) + ")";
        case SQR: return  "(" + print_fn(node->leftchild) + ")^2";
        case CUB: return  "Sin(" + print_fn(node->leftchild) + ")^3";
        case SQRT: return  "Sqrt(" + print_fn(node->leftchild) + ")";
        case CBRT: return  "Cubert(" + print_fn(node->leftchild) + ")";
        case ID: return "x";
        case CONST: return std::to_string(node->value);
    }
    return "INVALID_FUNCTION";
} 

double test_function(double x) {
    //x**3 - 4*x**2 +0.4*x + 1
    return x*x*x - 4*x*x + 0.4*x + 1;
}

long double rmse_fitness(Node * n, const std::vector<double> &cases, const std::vector<double> &targets, const int&timeout = 300) {
    long double error = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < cases.size(); i++) {
        long double res = evaluate(n, cases.at(i)) - targets.at(i);
        long double res_sq = res*res;
        error += (1/(cases.size() * 1.0) * res_sq);
        auto _loop_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = _loop_time - start_time;
        if (elapsed_time.count() > timeout) return 1e200;

    }
    if (error < 0) return 1E20; 
    else return sqrt(error);
}

long double rmse_fitness(Node * n, const std::vector<double> &cases, const std::vector<double> &targets, double parsimony_coefficient, const int& timeout=300) {
    auto start_time = std::chrono::high_resolution_clock::now();
    long double nonpenalized_fitness = rmse_fitness(n, cases, targets, timeout);
    auto _loop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = _loop_time - start_time;
    long double fitness;
    if (elapsed_time.count() < timeout) fitness = nonpenalized_fitness + parsimony_coefficient*count_nodes(n, 0);
    else fitness = nonpenalized_fitness;
    return fitness;
}

const double VARIABLE_BIAS = 0.6;
const double TERMINAL_PROBABILITY = 0.3;
Node * randomExpression(int depth, int type, Node* previous_node) {
    // This uses the FULL initialization method i.e, every tree will be full and terminals will only be chosen at the max depth
    // It would be a good idea to add the Grow method too and use a half-ramped initialization 
    if (type == 0) {
        //FULL INITIALIZATION
        if (depth == 0) {
            if ((rand() / RAND_MAX_DOUBLE) < VARIABLE_BIAS) {
                return new Node(ID, previous_node);
            }
            else {
                Node* n = new Node(CONST, previous_node);
                int rn = rand() % R.size();
                n-> value = R[rn];
                return n;
            }
        }

        uint16_t op_num = rand() % (NUM_ARITY_2 + NUM_ARITY_1);
        fns op = static_cast<fns>(op_num);
        Node * n = new Node(op, previous_node);
        n->leftchild = randomExpression(depth-1, 0, n);
        if (op_num < NUM_ARITY_2) {
            n->rightchild = randomExpression(depth-1, 0, n);
        }
        return n;
    }
    else if (type == 1) {
        //GROW INITIALIZATION

        depth = depth * 2;
        double random = rand();
        double rn = random / RAND_MAX_DOUBLE;
        // std::cout << "random: " << random << std::endl << "RAND_MAX: " << RAND_MAX << std::endl;
        // std::cout << "Random Number: " << std::to_string(rn) << std::endl;
        if((depth == 0) || (rn < TERMINAL_PROBABILITY)) {
            //with 50% probability sample terminal or constant
            if ((rand() / RAND_MAX_DOUBLE) < VARIABLE_BIAS) {
                return new Node(ID, previous_node);
            }
            else {
                Node* n = new Node(CONST, previous_node);
                int rn = rand() % R.size();
                n-> value = R[rn];
                return n;
            }
        }
        else if (rn >= TERMINAL_PROBABILITY) {
            uint16_t op_num = rand() % (NUM_ARITY_2 + NUM_ARITY_1);
            fns op = static_cast<fns>(op_num);
            Node * n = new Node(op, previous_node);
            n->leftchild = randomExpression(depth-1, 0, n);
            if (op_num < NUM_ARITY_2) {
                n->rightchild = randomExpression(depth-1, 0, n);
            }
            return n;
        }
    }
    //Getting rid of the warning
}

Node * randomExpression(int depth, int type) {
    // This uses the FULL initialization method i.e, every tree will be full and terminals will only be chosen at the max depth
    // It would be a good idea to add the Grow method too and use a half-ramped initialization 
    if (type == 0) {
        //FULL INITIALIZATION
        if (depth == 0) {
            if ((rand() / RAND_MAX_DOUBLE) < VARIABLE_BIAS) {
                return new Node(ID);
            }
            else {
                Node* n = new Node(CONST);
                int rn = rand() % R.size();
                n-> value = R[rn];
                return n;
            }
        }

        uint16_t op_num = rand() % (NUM_ARITY_2 + NUM_ARITY_1);
        fns op = static_cast<fns>(op_num);
        Node * n = new Node(op);
        n->leftchild = randomExpression(depth-1, 0, n);
        if (op_num < NUM_ARITY_2) {
            n->rightchild = randomExpression(depth-1, 0, n);
        }
        return n;
    }
    else if (type == 1) {
        //GROW INITIALIZATION

        depth = depth * 2;
        double random = rand();
        double rn = random / RAND_MAX_DOUBLE;
        // std::cout << "random: " << random << std::endl << "RAND_MAX: " << RAND_MAX << std::endl;
        // std::cout << "Random Number: " << std::to_string(rn) << std::endl;
        if((depth == 0) || (rn < (1/(depth*1.0) / 2.0 ) )) {
            //with 50% probability sample terminal or constant
            if ((rand() / RAND_MAX_DOUBLE) < VARIABLE_BIAS) {
                return new Node(ID);
            }
            else {
                Node* n = new Node(CONST);
                int rn = rand() % R.size();
                n-> value = R[rn];
                return n;
            }
        }
        else {
            uint16_t op_num = rand() % (NUM_ARITY_2 + NUM_ARITY_1);
            fns op = static_cast<fns>(op_num);
            Node * n = new Node(op);
            n->leftchild = randomExpression(depth-1, 0, n);
            if (op_num < NUM_ARITY_2) {
                n->rightchild = randomExpression(depth-1, 0, n);
            }
            return n;
        }
    }
    //Getting rid of the warning
}

void delete_subtree(Node* n) {
    if(n == nullptr) {
        return;
    }
    if(n->leftchild != nullptr) {
        delete_subtree(n->leftchild);
    }
    if(n->rightchild != nullptr) {
        delete_subtree(n->rightchild);
    }

    delete(n);
    n = nullptr;
}
/*
Validate that each tree node has 2 children if arity 2, 1 child if arity 1, and one parent if not root.
Then use assertions to determine where the string_to_node is creating Nodes with nullptr as children.
*/



int count_nodes(Node * n, int numnodes) {
    int numnodes_n = 0;
    if(n == nullptr) return 0;
    if(n!= nullptr) numnodes_n += 1; // for n itself
    numnodes_n += count_nodes(n->leftchild, 0);
    numnodes_n += count_nodes(n->rightchild, 0);
    return numnodes_n;
}

std::vector<Node*> label_nodes(Node * n, std::vector<Node*> &node_labels, bool root_ok) {
    if(root_ok) {
        node_labels.push_back(n); // for n itself
    }
    if(n->leftchild != nullptr) {
        label_nodes(n->leftchild, node_labels, true);
    }
    if(n->rightchild != nullptr) {
        label_nodes(n->rightchild, node_labels, true);
    }
    return node_labels;
}

int get_depth(Node * n) {
    int Ldepth = 0, Rdepth = 0;
    if((n->leftchild == nullptr) && (n->rightchild == nullptr)) {
        return 0;
    }
    if(n->leftchild != nullptr) {
        Ldepth = get_depth(n->leftchild);
    }
    if(n->rightchild != nullptr) {
        Rdepth = get_depth(n->rightchild);
    }
    int depth = 1 + std::max(Ldepth, Rdepth);
    return depth;
}

std::vector<Node*> label_terminals(Node * n, std::vector<Node*> &terminal_labels, bool root_ok) {
    // node_labels.push_back(n); // for n itself
    if((!root_ok) && (get_depth(n) == 0)) { 
        // Return an empty vector if root is not ok and the tree has depth 0
        return terminal_labels;
    }
    if((n->leftchild == nullptr) && (n->rightchild == nullptr)) {
        //If node has no children then it is a terminal (leaf node)
        terminal_labels.push_back(n);
    }
    if(n->leftchild != nullptr) {
        //If node has a leftchild, label the leftchild's terminals
        label_terminals(n->leftchild, terminal_labels, true);
    }
    if(n->rightchild != nullptr) {
        //If node has a rightchild, label the rightchild's terminals
        label_terminals(n->rightchild, terminal_labels, true);
    }
    return terminal_labels;
}



Node* get_random_node(Node * n, bool root_ok) {
    Node * return_val = n;
    if(get_depth(n) == 0) return n;
    std::vector<Node*> nodes;
    int numnodes = count_nodes(n, 0);
    std::vector<Node*> node_labels;
    node_labels = label_nodes(n, nodes, false);
    int random_index = rand() % (node_labels.size());
    return node_labels.at(random_index);
}

Node* get_random_terminal(Node *n, bool root_ok) {
    Node * return_val = n;
    if (get_depth(n) == 0) return n;
    std::vector<Node*> nodes;
    int numnodes = count_nodes(n, 0);
    std::vector<Node*> terminal_labels;
    terminal_labels = label_terminals(n, nodes, false);
    int random_index = rand() % (terminal_labels.size());
    return terminal_labels.at(random_index);
}

int get_arity(Node * n) {
    switch(n->fn) {
        case ADD:
        case SUB:
        case MUL:
        case DIV: return 2;
        case SIN: 
        case SQR:
        case CUB:
        case SQRT:
        case CBRT:
        case COS: return 1;
        case ID:
        case CONST: return 0;
    }
}

bool is_leftchild(Node* n) {
    if(n->parent == nullptr) {
        return false;
    }
    if(n->parent->leftchild == n) {
        return true;
    }
    return false;
}

void crossover(Node *&p1, Node *&p2) {
    Node* subtree_1;
    Node* subtree_2;
    bool s1left, s2left;
    if((get_depth(p1) < std::max(MIN_TREE_DEPTH, 1)) || (get_depth(p2) < std::max(1, MIN_TREE_DEPTH))) {
        return;
    }
    if (rand() / RAND_MAX_DOUBLE > CROSSOVER_PROBABILITY) return;

    

    subtree_1 = get_random_node(p1, false);
    subtree_2 = get_random_node(p2, false);


    Node* subtree_1_parent = subtree_1->parent;
    Node* subtree_2_parent = subtree_2->parent;

    // std::cout << "=========   DURING CROSSOVER   =========" <<std::endl;
    // std::cout << "PARENT 1 (PRE-CROSS): " << print_fn(p1) << std::endl;
    // std::cout << "PARENT 2 (PRE-CROSS): " << print_fn(p2) << std::endl;
    // std::cout << "SUBTREE 1: " << print_fn(subtree_1) << std::endl;
    // std::cout << "SUBTREE 2: " << print_fn(subtree_2) << std::endl;
    if(subtree_1_parent->leftchild == subtree_1) s1left = true;
    else s1left = false;
    if(subtree_2_parent->leftchild == subtree_2) s2left = true;
    else s2left = false;
    subtree_2->parent = subtree_1_parent;
    subtree_1->parent = subtree_2_parent;
    if(s1left) subtree_1_parent->leftchild = subtree_2;
    else subtree_1_parent->rightchild = subtree_2;
    if(s2left) subtree_2_parent->leftchild = subtree_1;
    else subtree_2_parent->rightchild = subtree_1;

    // std::cout << "PARENT 1 (POST-CROSS): " << print_fn(p1) << std::endl;
    // std::cout << "PARENT 2 (POST-CROSS): " << print_fn(p2) << std::endl;

    int new_depth_p1 = get_depth(p1);
    int new_depth_p2 = get_depth(p2);
    if((new_depth_p1 < MIN_TREE_DEPTH+1) || (new_depth_p1 > MAX_TREE_DEPTH) || (new_depth_p2 < MIN_TREE_DEPTH+1) || (new_depth_p1 > MAX_TREE_DEPTH)) {
        //If we're here we have violated a max size constraint, so we undo the crossover
        subtree_2->parent  = subtree_2_parent;
        subtree_1->parent  = subtree_1_parent;
        if(s1left) subtree_1_parent->leftchild = subtree_1;
        else subtree_1_parent->rightchild = subtree_1;
        if(s2left) subtree_2_parent->leftchild = subtree_2;
        else subtree_2_parent->rightchild = subtree_2;
    }
    if (DEBUG) {
        assert(tree_ok(p1));
        assert(tree_ok(p2));
    }
}




void mutate(Node * n) {
    Node* new_root;
    // Node Mutation Change a given node to a new node with the same arity
    double rn = rand() / RAND_MAX_DOUBLE;
    //std::cout << "rn Node: " << rn << std::endl;
    if(rn < NODE_CHANGEOP_PROBABILITY) {
        //std::cout << "NODE CHANGE " << std::endl;
        Node * m = get_random_node(n, true);
        int arity = get_arity(m);
        if(arity == 2) {
            fns op = static_cast<fns>(rand() % NUM_ARITY_2);
            m->fn = op;
        }
        else if (arity ==1) {
            int index = rand() % NUM_ARITY_1;
            fns op = static_cast<fns>(NUM_ARITY_2 + index);
            m->fn = op;
        }
        else if (arity == 0) {
            int index = rand() % 2;
            fns op = static_cast<fns>((NUM_ARITY_1+NUM_ARITY_2) + index);
            m->fn  = op;
        }
        if (DEBUG) assert(tree_ok(n));
    }    
    rn = rand() / RAND_MAX_DOUBLE;
    if(rn < NODE_DEL_PROBABILITY) {
        if(get_depth(n) > MIN_TREE_DEPTH+1) {
            //std::cout << "NODE DELETE " << std::endl;
            Node * term = get_random_terminal(n, false);
            Node * term_parent = term->parent;
            bool termleft = is_leftchild(term);
            Node * term_grandparent = term_parent->parent;
            if(term_grandparent != nullptr) {
                if (is_leftchild(term_parent)) term_grandparent->leftchild = term;
                else term_grandparent->rightchild = term;
                term->parent = term_grandparent;
                if (termleft) {
                    delete_subtree(term_parent->rightchild);
                    delete term_parent;
                }
                else {
                    delete_subtree(term_parent->leftchild);
                }
            }
        }
        if (DEBUG) assert(tree_ok(n));
    }
    rn = rand() / RAND_MAX_DOUBLE;
    //std::cout << "RN Replace Subtree: " << rn << std::endl;
    if(rn < SUBTREE_MUTATION_PROBABILITY) {
        //std::cout << "SUBTREE MUTATION " << std::endl;

        Node* m = get_random_node(n, false);
        if((get_depth(m) > 0) && (m->parent != nullptr)) {
            int m_depth  = get_depth(m);
            Node * m_parent = m-> parent;
            bool left_subtree = (m == (m_parent->leftchild)) ? 1 : 0;
            Node * new_m;
            delete_subtree(m);
            double subtree_init_type = rand() / RAND_MAX_DOUBLE;
            new_m = randomExpression(m_depth, subtree_init_type, m_parent);
            if(left_subtree) {
                m_parent->leftchild = new_m;
            }
            else {
                m_parent->rightchild = new_m;
            }
        }
        if (DEBUG) assert(tree_ok(n));
    }
    rn = rand() / RAND_MAX_DOUBLE;
    //std::cout << "RN Add Subtree: " << rn << std::endl;
    if(rn < ADD_SUBTREE_MUTATION_PROBABILITY) {
        int max_depth = MAX_TREE_DEPTH - get_depth(n);
        if((max_depth > 0) && get_depth(n) > 0) {
            //std::cout << "ADD SUBTREE MUTATION " << std::endl;

            Node* rand_terminal = get_random_terminal(n, true);
            Node* parent_n      = rand_terminal->parent;
            Node* new_n;
            int subtree_init_type = rand() % 2;
            bool left_subtree;
            if(parent_n != nullptr) {
                left_subtree = (rand_terminal == parent_n->leftchild) ? 1 : 0;
            } else left_subtree = false;
            delete_subtree(rand_terminal);
            double rn;
            rn = rand() / RAND_MAX_DOUBLE;
            if(rn < 0.4) {
                new_n = randomExpression(1, subtree_init_type, parent_n);
            }
            else if(rn < 0.7) {
                new_n = randomExpression(std::min(2, max_depth), subtree_init_type, parent_n);
            }
            else if(rn < 0.9) {
                new_n = randomExpression(std::min(3, max_depth), subtree_init_type, parent_n);
            }
            else{
                new_n = randomExpression(std::min(4, max_depth), subtree_init_type, parent_n);
            }
            if(left_subtree) {
                parent_n ->leftchild = new_n;
            }
            else if(parent_n != nullptr) {
                parent_n ->rightchild = new_n;
            }
        }
        if (DEBUG) assert(tree_ok(n));
    }
    rn = rand() / RAND_MAX_DOUBLE;
    //std::cout << "RN Rem Subtree: " << rn << std::endl;
    if(rn < REM_SUBTREE_MUTATION_PROBABILITY) {
        //std::cout << "REMOVE SUBTREE MUTATION " << std::endl;
        if(get_depth(n) > std::max(2, MIN_TREE_DEPTH)) {
            Node* m = get_random_node(n, true);
            if(get_depth(m) > 0) {
                Node * m_parent = m->parent;
                Node * replacement;
                if(m_parent->leftchild == m) {
                    if ((rand() % 2) == 0) {
                        replacement = new Node(ID, m_parent);
                        delete_subtree(m);
                        m_parent->leftchild = replacement;
                    }
                    else {
                        replacement = new Node(CONST, m_parent);
                        int rn = rand() % R.size();
                        replacement-> value = R[rn];
                        delete_subtree(m);
                        m_parent->leftchild = replacement;
                    }
                }
                if(m_parent->rightchild == m) {
                    if ((rand() % 2) == 0) {
                        replacement = new Node(ID, m_parent);
                        delete_subtree(m);
                        m_parent->rightchild = replacement;
                    }
                    else {
                        replacement = new Node(CONST, m_parent);
                        int rn = rand() % R.size();
                        replacement-> value = R[rn];
                        delete_subtree(m);
                        m_parent->rightchild = replacement;
                    }
                }
            }
        }
        if (DEBUG) assert(tree_ok(n));
    }
    rn = rand() / RAND_MAX_DOUBLE;
    if(rn < NODE_ADD_PROBABILITY) {
        //Node * save_for_debugging = deepcopy(n);
        //std::cout << "NODE ADDITION " << std::endl;
        
        // std::cout << "DEBUG" << std::endl;
        // std::cout << "f: " << print_fn(n) << "\nm: " << print_fn(m) << "\nterm: " << print_fn(term) << std::endl;

        if(get_depth(n) < MAX_TREE_DEPTH) {
            Node* m = randomExpression(1, 0);
            Node* term = get_random_terminal(n, true);

            Node* term_parent = term->parent;
            if((rand() % 2 == 0) || (m-> rightchild == nullptr)){ //Rightchild in a random expression of depth 1 could be null if function's arity is 1
                Node * temp = m->leftchild;
                m->leftchild = term;
                if (term_parent != nullptr) {
                    if (is_leftchild(term)) {
                        term_parent->leftchild = m;
                        m->parent = term_parent;
                    }
                    else {
                        term_parent->rightchild = m;
                        m->parent = term_parent;
                    }
                }
                else new_root = m;
                term-> parent = m;
                delete_subtree(temp);
            }
            else {
                Node * temp = m->rightchild;
                m->rightchild = term;
                if (term_parent != nullptr) {
                    if (is_leftchild(term)) {
                        term_parent->leftchild = m;
                        m->parent = term_parent;
                    }
                    else {
                        term_parent->rightchild = m;
                        m->parent = term_parent;
                    }
                }
                else new_root = m;
                term-> parent = m;
                delete_subtree(temp);
            }
        }
        //n = update_root(n);
        if (DEBUG) assert(tree_ok_newroot(n));
    }
}

void output_vals(Node* n, const std::vector<double> &test_pts, const std::string &searchtype, const std::string &info, const std::string &id) {
    std::ofstream f;
    std::string filename;
    filename = "Outputs\\" + searchtype + "\\" + id + "_" + info + ".csv";
    f.open(filename);
    if(!f.is_open()) {
        std::cout << "Could not open file: " << filename;
        return;
    }
    for (int i = 0; i < test_pts.size(); i++){
        double out_val = evaluate(n, test_pts.at(i));
        f << out_val;
        if( i < test_pts.size() - 1) {
            f << ',';
        }
    }
    f << std::endl;
    f.close();
    return;
}

std::vector<double> read_eval_pts(std::string filename, uint8_t coord) {
    std::ifstream f;
    std::vector<double> pts;
    std::string line;
    f.open(filename);
    if(!f.is_open()) { 
        std::cout << "COULD NOT OPEN FILE: " << filename;
        return pts;
    }
    while(std::getline(f, line)) {
        std::string x, y;
        std::stringstream ss(line);
        std::getline(ss, x, ',');
        std::getline(ss, y);
        // May need to strip whitespace
        if (coord == 0) {
            double x_val = stod(x);
            pts.push_back(x_val);
        }
        else if (coord == 1) {
            double y_val = stod(y);
            pts.push_back(y_val);
        }
    }
    return pts;
}

void write_id(int id) {
    std::ofstream f("id.txt");
    f << std::to_string(id);
    f.close();
}

int read_id() {
    std::ifstream f("id.txt");
    std::string ids;
    int id;
    std::getline(f, ids);
    id = stoi(ids);
    return id;
}

void save_all_fitnesses(const std::vector<Node*> &population, std::string filename) {
    std::ofstream f(filename);
    for(int i = 0; i < population.size(); i++) {
        f << std::to_string(population.at(i)->fitness) << std::endl;
    }
    f.close();
}
void save_all_sizes(const std::vector<Node*> &population, std::string filename) {
    std::ofstream f(filename);
    for(int i = 0; i < population.size(); i++) {
        f << std::to_string(population.at(i)->size) << std::endl;
    }
    f.close();
}
void log_coefficients(const std::vector<double> &coefficients, std::string filename) {
    std::ofstream f(filename);
    for(int i = 0; i < coefficients.size(); i++) {
        f << coefficients.at(i) << std::endl;
    }
    f.close();
}

void track_best_size(const std::vector<int> &sizes, std::string filename) {
    std::ofstream f(filename);
    for(int i = 0; i < sizes.size(); i++) {
        f << sizes.at(i) << std::endl;
    }
    f.close();
}

void write_fn(Node* n, std::string filename)  {
    std::ofstream f(filename);
    f << print_fn(n) <<std::endl;
    f.close();
}

void evaluate_fitness(std::vector<Node*>::iterator start, std::vector<Node*>::iterator end, const std::vector<double> &cases, const std::vector<double> &targets, const double &parsimony_coefficient = 0, const int&timeout=300) {
    for (auto it = start; it != end; it++) {
        long double fitness = rmse_fitness(*it, cases, targets, parsimony_coefficient, timeout);
        int size = count_nodes(*it, 0);
        
        (*it)->fitness = fitness;
        (*it)->size    = size;
        (*it)->unpenalized_fitness = fitness - parsimony_coefficient*size;
    }
}

void evaluate_fitness_population(std::vector<Node*> &population, const std::vector<double> &cases, const std::vector<double> &targets, const double &parsimony_coefficient = 0, const int&timeout = 300) {
    int numThreads = std::thread::hardware_concurrency();
    int chunkSize  = POPULATION_SIZE/numThreads;

    std::vector<std::thread> threads;
    for(int i = 0; i < numThreads; i++) {
        std::vector<Node*>::iterator start = population.begin() + i*chunkSize;
        std::vector<Node*>::iterator end;
        if(i == numThreads - 1) end = population.end();
        else end = start + chunkSize;
        threads.push_back(std::thread(evaluate_fitness, start, end, cases, targets, parsimony_coefficient, timeout));
    }
    for(auto &thread : threads) {
        thread.join();
    }
}

void track_best_fitness(std::vector<long double> best_fitnesses, std::string filename) {
    std::ofstream f(filename);
    if(!f.is_open()) {
        std::cout << "COULD NOT OPEN " << filename << std::endl;
        return;
    }
    for(int i = 0; i < best_fitnesses.size(); i++) {
        f << i << ", " << best_fitnesses.at(i) << std::endl;
    }
    f.close();
}

long double mean(std::vector<Node*> &population, const uint8_t &t) {
    long double m = 0;
    if(t == 0) {
        //mean of fitness
        for(auto &ind: population) {
            m+= ind->fitness;
        }
    }
    else if (t == 1) {
        //mean of size
        for(auto &ind: population) {
            ind->size = count_nodes(ind, 0);
            m += ind->size;
        }
    }
    return m/(population.size() * 1.0);
}

double correlation_of_size_and_fitness(std::vector<Node*> &population) {
    long double mean_fitness, mean_size, varsize, covar_size_fitness;
    mean_fitness = mean(population, 0);
    mean_size    = mean(population, 1); //all sizes are stored after this call
    long double s = 0;
    for(auto &ind : population) {
        s += (ind->size-mean_size)*(ind->size-mean_size);
    }
    varsize = s/(POPULATION_SIZE * 1.0);
    s = 0;
    for(auto &ind: population) {
        s += (ind->size-mean_size)*(ind->fitness - mean_fitness);
    }
    covar_size_fitness = s/(POPULATION_SIZE * 1.0);
    return covar_size_fitness/varsize;
}

std::vector<Node*> tournament_selection(std::vector<Node*> &population, const int &total_size, const int &tournament_size=2) {
    std::vector<Node*> next_generation;
    while(next_generation.size() < total_size) {
        Node* w1;
        Node *w2; 
        std::vector<int> idxs1, idxs2;
        std::vector<Node*> competitors1, competitors2;
        for(int i = 0; i < tournament_size; i++) {                                  
            int idx1 = rand() % POPULATION_SIZE;
            int idx2 = rand() % POPULATION_SIZE;
            competitors1.push_back(population.at(idx1));
            idxs1.push_back(idx1);
            competitors2.push_back(population.at(idx2));
            idxs2.push_back(idx2);
        }
        int w1idx=0, w2idx=0;
        long double w1best = competitors1.at(0)->fitness;
        long double w2best = competitors2.at(0)->fitness;
        for(int i = 1; i < tournament_size; i++) {
            if(competitors1.at(i)->fitness < w1best) {
                w1best = competitors1.at(i)->fitness;
                w1idx  = idxs1.at(i);
            }
            if(competitors2.at(i)->fitness < w2best) {
                w2best = competitors2.at(i)->fitness;
                w2idx  = idxs2.at(i);
            }
        }
        w1 = deepcopy(population.at(w1idx));
        w2 = deepcopy(population.at(w2idx));
        crossover(w1, w2);
        mutate(w1);
        w1 = update_root(w1);
        mutate(w2);
        w2 = update_root(w2);
        next_generation.push_back(w1);
        next_generation.push_back(w2);
    }
    while(next_generation.size() > total_size) {
        delete_subtree(next_generation.back());
        next_generation.pop_back();
    }
    return next_generation;
}

struct parameters {
    double crossover_probs;
    double mutation_probs;
    long double coefficient_values;
    int tournament_sizes; 
};

int main() {
    //_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    srand(time(0));
    for (int i = 0; i < 100; i++) {
        double rand_num = rand();
        double rand_num_01 = rand_num / RAND_MAX_DOUBLE;
        double rand_num_02 = rand_num_01 * 2;
        double rand_num_neg1_1 = rand_num_02-1;
        double rand_num_final = rand_num_neg1_1 * CONSTANT_RANGE;
        R.push_back(rand_num_final); // Get Random Ephemeral Constants between -CONSTANT_RANGE and CONSTANT RANGE
    }

    //std::string DATASET = "Bronze";
    //std::string DATASET = "Silver";
    std::string DATASET   = "Gold";
    std::vector<double> pts = read_eval_pts(DATASET  + ".txt", 0);
    std::vector<double> tgts = read_eval_pts(DATASET + ".txt", 1);

    int run_id = read_id();
    write_id(run_id + 1);

    const int CHECKPOINT_FREQUENCY = 100;

    int num_repeats    = 1;
    long double PARSIMONY_COEFFICIENT = 0.01; //You may want to use smaller values if the function is more complex than this one
    std::vector<long double> coefficient_values {1, 0.1, 0.01, 0.001, 0.0001}; //0.010.01
    std::vector<int> tournament_sizes {2, 5, 10, 15, 20 }; //2
    std::vector<double> crossover_probs = {0.99, 0.9, 0.8, 0.6, 0.3}; //0.9
    std::vector<double> mutation_probs = {0.5, 1, 2, 5, 10, 15 }; //1
    std::vector<std::vector<int>> generations_at_termination;// {{{100, 100, 100}, {100,200, 300}}};


    bool DO_GENETIC = true;
    bool DO_RANDOM  = true;
    bool DO_HILLCLIMBER = true;
    bool grid_search = false;
    int parameter_count = 5; // Number of parameter combinations to search over
    bool use_my_params = false;

    if(!grid_search) {
        parameter_count = 1;
    }
    if(DO_GENETIC) {
    std::vector<std::vector<std::vector<long double>>> learning_curves;
    std::vector<std::vector<std::vector<long double>>> best_fitnesses;
    std::vector<std::vector<std::vector<long double>>> best_fair_fitnesses;
    std::vector<std::vector<std::vector<int>>> max_program_sizes;
    std::vector<std::vector<std::vector<long double>>> mean_fitnesses;
    std::vector<std::vector<std::vector<std::vector<long double>>>> all_fitnesses;
    std::vector<std::vector<std::vector<double>>> mean_program_sizes;
    std::vector<std::vector<std::vector<int>>> min_program_sizes;
    //std::vector<std::vector<int>> generations_at_termination;

    std::vector<parameters*> param_values;

    std::mt19937 engine(std::random_device{}());
    std::uniform_int_distribution<int> distribution1(0,4);
    std::uniform_int_distribution<int> distribution2(0,5);
for(int p = 0; p < parameter_count; p++) {

    std::vector<std::vector<long double>> best_fitnesses_param;
    std::vector<std::vector<long double>> best_fair_fitnesses_param;
    std::vector<std::vector<int>> max_program_sizes_param;
    std::vector<std::vector<long double>> mean_fitnesses_param;
    std::vector<std::vector<std::vector<long double>>> all_fitnesses_param;
    std::vector<std::vector<double>> mean_program_sizes_param;
    std::vector<std::vector<int>> min_program_sizes_param;
    std::vector<int> generations_at_termination_param;

    parameters *params = new parameters {crossover_probs.at(distribution1(engine)), mutation_probs.at(distribution2(engine)), coefficient_values.at(distribution1(engine)), 
                            tournament_sizes.at(distribution1(engine))};
    
    
    param_values.push_back(params);

    for(int repeat = 0; repeat < num_repeats; repeat++){
        std::cout << "=========================== BEGINNING REPEAT " << repeat << " =============================================";
        if (grid_search) PARSIMONY_COEFFICIENT = coefficient_values.at(repeat);
        int MAX_GENERATION = 100000;
        long MAX_TIME      = 1e20; //10800;                // make this a smaller number to enforce a maximum running time in wall time (seconds)
        const int patience = 100000;                  // If no improvement after this many generations end the run. (can set max generation higher)
        bool DYNAMIC_PARSIMONY = false;
        std::vector<Node*> population;
        int generation = 0;
        std::vector<long double> best_fitnesses;
        std::vector<long double> fair_fitnesses;
        std::vector<int> best_sizes;
        bool warm_start = false;
        long double best_fitness=9999999;
        long double best_fair_fitness = 9999999;
        int best_index=0;
        int tournament_size = 2;
        int single_fitness_timeout = 300; // 5 minutes for a single tree. Change as you see fit.
        double singleton_replace_probability = 0.03;

        std::ofstream logfile(".\\Outputs\\Genetic\\" + std::to_string(run_id) + "_Repeat_" + std::to_string(repeat) + "_logfile.txt");



        // Set params for the repeats
        if(grid_search) {
            PARSIMONY_COEFFICIENT = params->coefficient_values;
            double multiplier = params->mutation_probs;
            NODE_ADD_PROBABILITY = multiplier * 0.05;
            NODE_CHANGEOP_PROBABILITY = multiplier * 0.05;
            NODE_DEL_PROBABILITY = multiplier * 0.05;
            SUBTREE_MUTATION_PROBABILITY = multiplier * 0.03;
            ADD_SUBTREE_MUTATION_PROBABILITY = multiplier * 0.02;
            REM_SUBTREE_MUTATION_PROBABILITY = multiplier * 0.02;
            CROSSOVER_PROBABILITY = params->crossover_probs;
            tournament_size = params->tournament_sizes;
        }

        if(!warm_start) {
            for(int i = 0; i < POPULATION_SIZE; i++) {
                Node* ind = randomExpression(rand() % (MAX_INITIAL_SIZE - MIN_TREE_DEPTH) + MIN_TREE_DEPTH, rand() % 2);
                assert(tree_ok(ind));
                population.push_back(ind);
            }
        }
        else {
            const std::string CHECKPOINT_ID     = 0;
            const int CHECKPOINT_NUMBER         = 90;
            generation = CHECKPOINT_NUMBER+1;
            population = load_checkpoint(".\\Checkpoints\\Genetic\\" + CHECKPOINT_ID + "_generation_" + std::to_string(CHECKPOINT_NUMBER) + ".txt");
            for(int i = 0; i < 5; i++) {
                std::cout << "Checkpoint Loaded Population " + std::to_string(i) + print_fn(population.at(i)) << std::endl;
            }
            best_fitness = rmse_fitness(population.at(0), pts, tgts, PARSIMONY_COEFFICIENT, single_fitness_timeout);
        }
        //std::vector<std::vector<long double>> learning_curves_repeat;
        std::vector<long double> best_fitnesses_repeat;
        std::vector<long double> best_fair_fitnesses_repeat;
        std::vector<int> max_program_sizes_repeat;
        std::vector<long double> mean_fitnesses_repeat;
        std::vector<std::vector<long double>> all_fitnesses_repeat;
        std::vector<double> mean_program_sizes_repeat;
        std::vector<int> min_program_sizes_repeat;
        int generations_at_termination_repeat;
        Node* best_function = nullptr;


        auto start_time = std::chrono::high_resolution_clock::now();

        //EVOLUTIONARY LOOP BEGINNING
        while(generation < MAX_GENERATION) {

            std::cout << "======= Begin Generation: " << generation << std::endl;
            std::vector<Node*> next_generation;
            std::vector<long double> fitnesses;
            std::vector<int> sizes;
            std::vector<double> coefficients;
            coefficients.push_back(PARSIMONY_COEFFICIENT);
            evaluate_fitness_population(population, pts, tgts, PARSIMONY_COEFFICIENT, single_fitness_timeout);


            //Save fitnesses and get elite population - may want to modify to save k elite 
            int max_size = -1; int min_size = 99999999; int best_fn_size;
            long double best_fitness_this_round = 999999999;
            long double best_fair_fitness_this_round = 999999999;
            int best_fn_size_this_round = -1;
            int best_index_this_round = -1;
            for(int i = 0; i < POPULATION_SIZE; i++) {
                long double fitness_i  = population.at(i)->fitness;//rmse_fitness(population.at(i), pts, tgts, PARSIMONY_COEFFICIENT);
                long double fair_fitness_i  = population.at(i)->unpenalized_fitness;//rmse_fitness(population.at(i), pts, tgts, PARSIMONY_COEFFICIENT);
                int prog_size = population.at(i)->size;
                if ((prog_size == 1) && (generation > 5) && ((fitness_i > best_fitnesses_repeat.at(best_fitnesses_repeat.size()-1)) || (mean_fitnesses_repeat.at(mean_fitnesses_repeat.size()-1) < 3))) {
                    double rn = rand() /RAND_MAX_DOUBLE;
                    if (rn < singleton_replace_probability) {
                        Node * temp = population.at(i);
                        population.at(i) = randomExpression(rand() % (MAX_INITIAL_SIZE - MIN_TREE_DEPTH) + MIN_TREE_DEPTH, rand() % 2);
                        population.at(i)->fitness = rmse_fitness(population.at(i), pts, tgts, PARSIMONY_COEFFICIENT, single_fitness_timeout);
                        fitness_i = population.at(i)->fitness;
                        population.at(i)->size = count_nodes(population.at(i), 0);
                        prog_size = population.at(i)->size;
                        fair_fitness_i = fitness_i - PARSIMONY_COEFFICIENT*prog_size;
                        delete_subtree(temp);
                    }
                }
                fitnesses.push_back(fitness_i);
                fair_fitnesses.push_back(fair_fitness_i);
                sizes.push_back(prog_size);
                if(prog_size < min_size) min_size = prog_size;
                if(prog_size > max_size) max_size = prog_size;
                if (fitness_i < best_fitness) {
                    std::cout << "New Best Fitness: " << fitness_i << std::endl;
                    best_fitness = fitness_i;
                    best_index   = i;
                    if (best_function != nullptr) delete_subtree(best_function);
                    best_function = deepcopy(population.at(i));
                    best_fn_size = prog_size;
                }
                if (fitness_i < best_fitness_this_round) {
                    best_fitness_this_round = fitness_i;
                    best_index_this_round   = i;
                    best_fn_size_this_round = prog_size;
                }
                if (fair_fitness_i < best_fair_fitness) {
                    std::cout << "New Best Fair Fitness: " << fitness_i << std::endl;
                    best_fair_fitness = fair_fitness_i;
                }
                if (fair_fitness_i < best_fair_fitness_this_round) {
                    best_fair_fitness_this_round = fair_fitness_i;
                }
            }
            mean_fitnesses_repeat.push_back(mean(population, 0));
            best_fitnesses_repeat.push_back(best_fitness);
            best_fair_fitnesses_repeat.push_back(best_fair_fitness);
            mean_program_sizes_repeat.push_back(mean(population, 1));
            max_program_sizes_repeat.push_back(max_size);
            min_program_sizes_repeat.push_back(min_size);
            
            if (DYNAMIC_PARSIMONY) {
                PARSIMONY_COEFFICIENT = correlation_of_size_and_fitness(population);
                std::cout << "Mean Fitness: " << mean(population, 0) << std::endl;
                std::cout << "Mean Size: " << mean(population, 1) << std::endl;
                std::cout << "Parsimony Coefficient: " << PARSIMONY_COEFFICIENT << std::endl;
            }
            //save elite
            next_generation.push_back(deepcopy(population.at(best_index_this_round)));
            best_fitnesses.push_back(best_fitness);
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = current_time - start_time;
            bool patience_exceeded = (generation > patience) && (best_fitnesses.at(generation) >= best_fitnesses.at(generation - patience));
            bool timeout_exceeded  = elapsed_time.count() > MAX_TIME;
            if((patience_exceeded) || (timeout_exceeded)) {
                //stopping condition
                if (patience_exceeded) std::cout << "No Improvement in " + std::to_string(patience) + " generations. Ending Run" << std::endl;
                else if (timeout_exceeded) std::cout << "Runtime of " + std::to_string(elapsed_time.count()) + " Exceeds Max Runtime (" << std::to_string(
                    MAX_TIME) << " s). Ending Run" << std::endl;
                for (auto& p : next_generation) {
                    delete_subtree(p);
                }
                break;
            }
            std::vector<Node*> tournament_results = tournament_selection(population, POPULATION_SIZE-1, tournament_size);
            next_generation.insert(next_generation.end(), tournament_results.begin(), tournament_results.end());

            std::cout << "=========================*************STATISTICS*************=======================================" << std::endl;
            std::cout << "=========================*** ID: " << std::to_string(run_id) << " Repeat: " << std::to_string(repeat) << "***=======================================" << std::endl;
            std::cout << "Generation: \t\t\t" << generation << "\nBest Fitness (Unpenalized): \t" << best_fair_fitness  <<"\nBest Fitness (True): \t\t" << best_fitness << std::endl;
            std::cout << "Best Function: \t\t\t" << print_fn(population.at(best_index_this_round)) << std::endl;
            std::cout << "Best Fair Fitness This Round: \t" << best_fair_fitness_this_round << std::endl;
            std::cout << "Best Fitness This Round: \t" << best_fitness_this_round << std::endl;
            std::cout << "Best Function Size: \t\t" << best_fn_size_this_round << std::endl;
            std::cout << "Mean Fitness: \t\t\t" << mean_fitnesses_repeat.at(mean_fitnesses_repeat.size() - 1) << std::endl;
            std::cout << "Mean Program Size: \t\t" << mean_program_sizes_repeat.at((mean_program_sizes_repeat.size() - 1)) << std::endl;
            std::cout << "Max Program Size: \t\t" << std::to_string(max_size) << std::endl;
            std::cout << "Min Program Size: \t\t" << std::to_string(min_size) << std::endl;
            std::cout << "Parsimony Coefficient: \t\t" << PARSIMONY_COEFFICIENT << std::endl;
            std::cout << "Elapsed Seconds: \t\t" << std::to_string(elapsed_time.count()) << std::endl << std::endl;
            std::cout << "=========================************END**************=======================================" << std::endl;

            logfile << "=========================*************STATISTICS*************=======================================" << std::endl;
            logfile << "=========================*** ID: " << std::to_string(run_id) << " Repeat: " << std::to_string(repeat) << "***=======================================" << std::endl;
            logfile << "Generation: \t\t\t" << generation << "\nBest Fitness (Unpenalized): \t" << best_fair_fitness  <<"\nBest Fitness (True): \t\t" << best_fitness << std::endl;
            logfile << "Best Function: \t\t\t" << print_fn(population.at(best_index_this_round)) << std::endl;
            logfile << "Best Function Size: \t\t" << best_fn_size_this_round << std::endl;
            logfile << "Mean Fitness: \t\t\t" << mean_fitnesses_repeat.at(mean_fitnesses_repeat.size() - 1) << std::endl;
            logfile << "Mean Program Size: \t\t" << mean_program_sizes_repeat.at((mean_program_sizes_repeat.size() - 1)) << std::endl;
            logfile << "Max Program Size: \t\t" << std::to_string(max_size) << std::endl;
            logfile << "Min Program Size: \t\t" << std::to_string(min_size) << std::endl;
            logfile << "Parsimony Coefficient: \t\t" << PARSIMONY_COEFFICIENT << std::endl;
            logfile << "Elapsed Seconds: \t\t" << std::to_string(elapsed_time.count()) << std::endl << std::endl;
            logfile << "=========================************END**************=======================================" << std::endl;


            if(generation % CHECKPOINT_FREQUENCY == 0) {
                std::string filename = ".\\Checkpoints\\Genetic\\" + std::to_string(run_id) + "_Repeat_" + std::to_string(repeat) + "_Gen" + std::to_string(generation) + "checkpoint.txt";
                std::string best_solution_fn    = ".\\Checkpoints\\Genetic\\"+ std::to_string(run_id)+ "_Repeat_" + std::to_string(repeat) + "_Gen" + std::to_string(generation) + "best_solution_fn.txt";
                checkpoint(population, filename);
                write_fn(population.at(best_index_this_round), best_solution_fn);
                std::string all_fitnesses       = ".\\Checkpoints\\Genetic\\"+ std::to_string(run_id)+ "_Repeat_" + std::to_string(repeat) + "_Gen" + std::to_string(generation) +  "_all_fitnesses.txt";
                std::string all_sizes           = ".\\Checkpoints\\Genetic\\"+  std::to_string(run_id)+ "_Repeat_" + std::to_string(repeat) + "_Gen" + std::to_string(generation) +  "_all_sizes.txt";
                output_vals(population.at(best_index_this_round), pts, "Genetic", DATASET +  "_Repeat_" + std::to_string(repeat)+ "_Gen_" + std::to_string(generation), std::to_string(run_id));
                save_all_fitnesses(population, all_fitnesses);
                save_all_sizes(population, all_sizes);
            }

            for(auto &p : population) {
                    delete_subtree(p);
            }
            all_fitnesses_repeat.push_back(fitnesses);
            population = next_generation;
            best_index = 0;
            generation++;
        }
        generations_at_termination_repeat = generation;
        std::vector<long double> fitnesses;
        evaluate_fitness_population(population, pts, tgts, PARSIMONY_COEFFICIENT, single_fitness_timeout);
        best_index = -1; best_fitness = 999;
        for(int i = 0; i < POPULATION_SIZE; i++) {
            long double fitness_i = population.at(i)->fitness;//rmse_fitness(population.at(i), pts, tgts);
            fitnesses.push_back(fitness_i);
            if (fitness_i < best_fitness) {
                std::cout << "New Best Fitness: " << fitness_i << std::endl;
                best_fitness = fitness_i;
                best_index   = i;
            }
        }
        std::string filename            = ".\\Checkpoints\\" + std::to_string(run_id) + "_Repeat_" + std::to_string(repeat) + "generation_" + std::to_string(generation) + ".txt";
        std::string learning_curve      = ".\\Outputs\\Genetic\\"+ std::to_string(run_id) + "_Repeat_" + std::to_string(repeat)+ "_learning_curve.txt";

        std::string program_sizes       = ".\\Outputs\\Genetic\\"+ std::to_string(run_id)+ "_Repeat_" + std::to_string(repeat) + "best_program_sizes.txt";

        std::string all_fitnesses       = ".\\Outputs\\Genetic\\"+ std::to_string(run_id)+ "_Repeat_" + std::to_string(repeat) + "all_fitnesses.txt";
        std::string learning_curve_ckpt = ".\\Checkpoints\\"+ std::to_string(run_id)+ "_Repeat_" + std::to_string(repeat) + "_learning_curve.txt";
        std::string best_solution_fn    = ".\\Outputs\\Genetic\\"+ std::to_string(run_id)+ "_Repeat_" + std::to_string(repeat) + "best_solution_fn.txt";
        std::string coefficients        = ".\\Outputs\\Genetic\\" + std::to_string(run_id)+ "_Repeat_" + std::to_string(repeat) + "parsimony_coefficients.txt";

        checkpoint(population, filename);
        output_vals(population.at(best_index), pts, "Genetic", DATASET + "_Repeat_" + std::to_string(repeat) + "_Gen_" + std::to_string(generation), std::to_string(run_id));
        write_fn(population.at(best_index), best_solution_fn);
        track_best_fitness(best_fitnesses, learning_curve);
        save_all_fitnesses(population, all_fitnesses);
        track_best_fitness(best_fitnesses, learning_curve_ckpt);
        track_best_size(best_sizes, program_sizes);
        //learning_curves_repeat.push_back(best_fitnesses);
        //std::vector<long double> best_fitnesses_repeat;
        best_fitnesses_param.push_back(best_fitnesses_repeat);
        best_fair_fitnesses_param.push_back(best_fair_fitnesses_repeat);
        max_program_sizes_param.push_back(max_program_sizes_repeat);
        mean_fitnesses_param.push_back(mean_fitnesses_repeat);
        all_fitnesses_param.push_back(all_fitnesses_repeat);
        mean_program_sizes_param.push_back(mean_program_sizes_repeat);
        min_program_sizes_param.push_back(min_program_sizes_repeat);
        generations_at_termination_param.push_back(generations_at_termination_repeat);
        for (auto &p : population) {
            delete_subtree(p);
        }
    }
    best_fitnesses.push_back(best_fitnesses_param);
    best_fair_fitnesses.push_back(best_fair_fitnesses_param);
    max_program_sizes.push_back(max_program_sizes_param);
    mean_fitnesses.push_back(mean_fitnesses_param);
    all_fitnesses.push_back(all_fitnesses_param);
    mean_program_sizes.push_back(mean_program_sizes_param);
    min_program_sizes.push_back(min_program_sizes_param);
    generations_at_termination.push_back(generations_at_termination_param);

    
    if (grid_search) {
    //find the best parameters from our parameter choices;
    std::ofstream pf(std::to_string(run_id) + "ParamSearch_" + std::to_string(p) + ".txt");
    long double best_b = 99999999;
    long double best_fb = 99999999;
    parameters *best_params_for_best_fitness = nullptr;
    parameters *best_params_for_fair_fitness = nullptr;
    for(int i = 0; i < param_values.size(); i++) {
        double b    = 0;
        double fb   = 0;
        double mf   = 0;
        double maxS = 0;
        double minS = 0;
        double avgS = 0;
        for(int j = 0; j < num_repeats; j++) {
            std::vector<long double> current_run_b    = best_fitnesses.at(i).at(j);
            std::vector<long double> current_run_fb    = best_fair_fitnesses.at(i).at(j);
            std::vector<long double> current_run_mf   = mean_fitnesses.at(i).at(j);
            std::vector<int> current_run_maxS = max_program_sizes.at(i).at(j);
            std::vector<int> current_run_minS = min_program_sizes.at(i).at(j);
            std::vector<double> current_run_avgS = mean_program_sizes.at(i).at(j);
            b += current_run_b.at(current_run_b.size() -1);
            fb += current_run_fb.at(current_run_fb.size() -1);
            mf +=current_run_mf.at(current_run_mf.size() -1);
            maxS +=current_run_maxS.at(current_run_maxS.size() -1);
            minS +=current_run_minS.at(current_run_minS.size() -1);
            avgS +=current_run_avgS.at(current_run_avgS.size() -1);
        }
        b /= num_repeats;
        fb /= num_repeats;
        if (b < best_b) {
            best_b = b;
            best_params_for_best_fitness = param_values.at(i);
        }
        if (fb < best_fb) {
            best_fb = fb;
            best_params_for_fair_fitness = param_values.at(i);
        }
        mf /= num_repeats;
        maxS /= num_repeats;
        minS /= num_repeats;
        avgS /= num_repeats;

        std::cout << "Params: \n\tParsimony Coefficient Value: " << param_values.at(i)->coefficient_values << "\n\tTournament Size: "
                            << param_values.at(i)->tournament_sizes << "\n\tMutation Probability Multiplier: " << param_values.at(i)->mutation_probs
                            << "\n\tCrossover Probability: " << param_values.at(i)->crossover_probs << "\n\n\tAvg Best Fitness: " << std::to_string(b)
                            << "\n\tMean Fitness at Finish: " << std::to_string(mf) << "\n\tMax Program Size in Final Generation: " << std::to_string(maxS) 
                            << "\n\tMin Program Size in Final Generation: " << std::to_string(minS) << "\n\tMean Program Size in Final Generation: " 
                            << std::to_string(avgS) << std::endl;

        pf << "Params: \n\tParsimony Coefficient Value: " << param_values.at(i)->coefficient_values << "\n\tTournament Size: "
                            << param_values.at(i)->tournament_sizes << "\n\tMutation Probability Multiplier: " << param_values.at(i)->mutation_probs
                            << "\n\tCrossover Probability: " << param_values.at(i)->crossover_probs << "\n\n\tAvg Best Fitness: " << std::to_string(b)
                            << "\n\tMean Fitness at Finish: " << std::to_string(mf) << "\n\tMax Program Size in Final Generation: " << std::to_string(maxS) 
                            << "\n\tMin Program Size in Final Generation: " << std::to_string(minS) << "\n\tMean Program Size in Final Generation: " 
                            << std::to_string(avgS) << std::endl;
    }
        std::cout << "======================= BEST PARAMETERS BY BEST FITNESS AVERAGED OVER REPEATS ==============================" << std::endl;
        std::cout << "Params: \n\tParsimony Coefficient Value: " <<  best_params_for_best_fitness->coefficient_values << "\n\tTournament Size: "
                            <<  best_params_for_best_fitness->tournament_sizes << "\n\tMutation Probability Multiplier: " <<  best_params_for_best_fitness->mutation_probs
                            << "\n\tCrossover Probability: " << best_params_for_best_fitness->crossover_probs << std::endl << std::endl;

        pf << "======================= BEST PARAMETERS BY BEST FITNESS AVERAGED OVER REPEATS ==============================" << std::endl;
        pf << "Params: \n\tParsimony Coefficient Value: " <<  best_params_for_best_fitness->coefficient_values << "\n\tTournament Size: "
                            <<  best_params_for_best_fitness->tournament_sizes << "\n\tMutation Probability Multiplier: " <<  best_params_for_best_fitness->mutation_probs
                            << "\n\tCrossover Probability: " << best_params_for_best_fitness->crossover_probs << std::endl << std::endl;

        std::cout << "======================= BEST PARAMETERS BY FAIR FITNESS AVERAGED OVER REPEATS ==============================" << std::endl;
        std::cout << "Params: \n\tParsimony Coefficient Value: " <<  best_params_for_fair_fitness->coefficient_values << "\n\tTournament Size: "
                            <<  best_params_for_fair_fitness->tournament_sizes << "\n\tMutation Probability Multiplier: " <<  best_params_for_fair_fitness->mutation_probs
                            << "\n\tCrossover Probability: " << best_params_for_fair_fitness->crossover_probs << std::endl << std::endl;

        pf << "======================= BEST PARAMETERS BY FAIR FITNESS AVERAGED OVER REPEATS ==============================" << std::endl;
        pf << "Params: \n\tParsimony Coefficient Value: " <<  best_params_for_fair_fitness->coefficient_values << "\n\tTournament Size: "
                            <<  best_params_for_fair_fitness->tournament_sizes << "\n\tMutation Probability Multiplier: " <<  best_params_for_fair_fitness->mutation_probs
                            << "\n\tCrossover Probability: " << best_params_for_fair_fitness->crossover_probs << std::endl << std::endl;
    }
}
}  
    //EVOLUTIONARY LOOP END

    // Potential Optimizations
    // Stochastic Fitness Evaluation
    //   - Evaluate each function on only a portion of the fitness cases each generation
    // Keep track of changes to genetic material
    //   - If an indiviual's genetic material has not changed since the last generation, then do not evaluate fitness again  
    // Parallelization of portions of the code
    //   - Fitness evaluation
    //   - Tournament Selection
    // Memory Management
    //   - Languages like Python and Java take care of memory management for you
    //   - C++ requires you to manage your own memory - it's important to delete objects if you create them with the new keyword
    //   - We will want to be careful about that. 
    // Optimization AND robustness 
    //   - Controlling the complexity of solutions
    //   - It's common to use methods like parsimoy pressure (an additional penalty term in the fitness function), or other other complexity control mechanisms
    // Checkpointing and warm starts 
    //   - To make sure the population is saved and you can continue from a previously evolved population
    //        - Node toString method
    //        - Node from tring Method
    // Implementing Demes (Island model)
    // Adaptive or Self-adaptive parameter
int random_fitness_timeout = 300;
if(DO_RANDOM) {
    parameters param;
    int j = 0; //this should loop over parameters if grid searching here, instead just use 0
    for(int repeat = 0; repeat < num_repeats; repeat++) {
        if (grid_search) PARSIMONY_COEFFICIENT = coefficient_values.at(repeat);
        std::vector<long double> random_fitnesses;
        Node * random = randomExpression(rand() % MAX_TREE_DEPTH, rand() % 2);
        Node * best_random = deepcopy(random);
        long double best_random_fitness = rmse_fitness(best_random, pts, tgts, PARSIMONY_COEFFICIENT, random_fitness_timeout); //Still leaving Parsimony coefficient for fair comparison
        random_fitnesses.push_back(best_random_fitness);
        int random_generation = 0;
        std::cout << "===== Starting Random Search =======" << std::endl;
        while(random_generation < generations_at_termination.at(j).at(repeat)) {
            random = randomExpression(rand() % MAX_TREE_DEPTH, rand() % 2);             //Instead of a new random expression every time, mutate the previous one until no fitness improvement
            long double fitness = rmse_fitness(random, pts, tgts, PARSIMONY_COEFFICIENT, random_fitness_timeout);
            if (fitness < best_random_fitness) {
                best_random_fitness = fitness;
                delete_subtree(best_random);
                best_random = deepcopy(random);
            }
            if(random_generation % CHECKPOINT_FREQUENCY == 0) {
                std::string random_output = ".\\Checkpoints\\Random\\" + std::to_string(run_id) + "_Repeat_" + std::to_string(repeat)+ "_Gen" + std::to_string(random_generation) + "best_solution_fn.txt";
                write_fn(best_random, random_output);
                output_vals(best_random, pts, "Random", DATASET+ "_Repeat_" + std::to_string(repeat) + "_Gen_" + std::to_string(random_generation), std::to_string(run_id));
            }
            random_fitnesses.push_back(best_random_fitness);
            std::cout << "Generation: "  << random_generation << std::endl;
            std::cout << "Best Fitness: " << best_random_fitness << std::endl;
            random_generation++;
            delete_subtree(random);
        }
        std::string random_learning_curve = ".\\Outputs\\Random\\" + std::to_string(run_id) + + "_Repeat_" + std::to_string(repeat) + "_learning_curve.txt";
        track_best_fitness(random_fitnesses, random_learning_curve);
    }
}


if (DO_HILLCLIMBER) {
    int j = 0;
        for(int repeat = 0; repeat < num_repeats; repeat++) {
            NODE_ADD_PROBABILITY      = 0.333;
            NODE_CHANGEOP_PROBABILITY = 0.333;
            NODE_DEL_PROBABILITY      = 0.333;
            SUBTREE_MUTATION_PROBABILITY     = 0.167;
            ADD_SUBTREE_MUTATION_PROBABILITY = 0.167;
            REM_SUBTREE_MUTATION_PROBABILITY = 0.167;
            if (grid_search) PARSIMONY_COEFFICIENT = coefficient_values.at(repeat);
            std::vector<long double> hill_climber_fitnesses;
            Node * hc = randomExpression(rand() % MAX_TREE_DEPTH, rand() % 2);
            long double best_hc_fitness = rmse_fitness(hc, pts, tgts, PARSIMONY_COEFFICIENT, random_fitness_timeout); //Still leaving Parsimony coefficient for fair comparison
            hill_climber_fitnesses.push_back(best_hc_fitness);
            int hc_generation = 0;
            std::cout << "===== Starting Hill Cliber Search =======" << std::endl;
            while(hc_generation < generations_at_termination.at(j).at(repeat)) {
                Node* temp = deepcopy(hc);
                mutate(temp);
                long double fitness = rmse_fitness(temp, pts, tgts, PARSIMONY_COEFFICIENT, random_fitness_timeout);
                if (fitness < best_hc_fitness) {
                    best_hc_fitness = fitness;
                    delete_subtree(hc);
                    hc = deepcopy(temp);
                }
                if(hc_generation % CHECKPOINT_FREQUENCY == 0) {
                    std::string hc_output = ".\\Checkpoints\\HillClimber\\" + std::to_string(run_id) + "_Gen_" + std::to_string(hc_generation) + ".txt";
                    write_fn(hc, hc_output);
                    output_vals(hc, pts, "HillClimber", DATASET + "_Repeat_" + std::to_string(repeat) + "_Gen_" + std::to_string(hc_generation), std::to_string(run_id));
                }
                hill_climber_fitnesses.push_back(best_hc_fitness);
                std::cout << "Generation: "  << hc_generation << std::endl;
                std::cout << "Best Fitness: " << best_hc_fitness << std::endl;
                hc_generation++;
                delete_subtree(temp);
            }
            std::string hc_learning_curve = ".\\Outputs\\HillClimber\\" + std::to_string(run_id) +  "_Repeat_" + std::to_string(repeat) + "_learning_curve.txt";
            track_best_fitness(hill_climber_fitnesses, hc_learning_curve);
        }
    }
    return 0;
}