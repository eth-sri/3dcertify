#include "domains/polyhedra.h"
#include "gurobi_c++.h"
#include "utils/lipschitz.h"
#include <vector>

std::pair<std::vector<double>, double>
findLower(const GRBEnv &env, const LipschitzFunction &,
          std::default_random_engine, int, double, int, Statistics &counter);
std::pair<std::vector<double>, double>
findUpper(const GRBEnv &env, const LipschitzFunction &,
          std::default_random_engine, int, double, int, Statistics &counter);
