#include "abstraction/abstraction.h"
#include "domains/polyhedra.h"
#include "gurobi_c++.h"
#include "transforms/parser.h"
#include "transforms/transformation_3d.h"
#include "utils/constants.h"
#include "utils/lipschitz.h"
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

void getSplitPoints(vector<vector<double>> &splitPoints, string value) {
  string token;

  int idx = 0;
  vector<double> v;
  size_t pos;

  while ((pos = value.find(',')) != string::npos) {
    token = value.substr(0, pos);
    if (token == "*") {
      cout << "pushing vector in" << endl;
      splitPoints.push_back(v);
      ++idx;
      v.clear();
    } else {
      cout << "pushing token to vector: " << stod(token) << endl;
      v.push_back(stod(token));
    }
    value.erase(0, pos + 1);
  }
  v.push_back(stod(value));
  splitPoints.push_back(v);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  string out_dir = argv[1];

  size_t num_tests = 1, num_splits = 1, num_inside_splits = 1, num_points;
  std::string set = "test", dataset = "mnist", spatialTransformName;
  std::vector<std::vector<double>> splitPoints;
  std::string name, value;
  bool debug = false;

  std::ifstream config(out_dir + "/config.txt");
  while (config >> name >> value) {
    if (name.empty() && value.empty()) {
      continue;
    }
    cout << "Setting property " << name << " to value: " << value << endl;
    if (name == "set") {
      set = value;
    } else if (name == "split_points") {
      getSplitPoints(splitPoints, value);
      cout << "Total split points: " << splitPoints.size() << endl;
    } else if (name == "num_threads") {
      Constants::NUM_THREADS = stoi(value);
    } else if (name == "max_coeff") {
      Constants::MAX_COEFF = stod(value);
    } else if (name == "lp_samples") {
      Constants::LP_SAMPLES = stoi(value);
    } else if (name == "num_poly_check") {
      Constants::NUM_POLY_CHECK = stoi(value);
    } else if (name == "dataset") {
      dataset = value;
    } else if (name == "chunks") {
      num_splits = stoul(value);
    } else if (name == "inside_splits") {
      num_inside_splits = stoul(value);
    } else if (name == "spatial_transform") {
      spatialTransformName = value;
    } else if (name == "num_tests") {
      num_tests = stoul(value);
    } else if (name == "debug") {
      debug = true;
    } else if (name == "poly_degree") {
      Constants::POLY_DEGREE = stoi(value);
    } else if (name == "poly_eps") {
      Constants::POLY_EPS = stod(value);
    } else if (name == "split_mode") {
      Constants::SPLIT_MODE = value;
    } else if (name == "ub_estimate") {
      Constants::UB_ESTIMATE = value;
    } else if (name == "num_points") {
      num_points = stoul(value);
    } else {
      cout << "Property not found: " << name << endl;
      return 1;
    }
  }

  assert(dataset == "modelnet40" || dataset == "shapenet");
  SpatialTransformation3D &spatialTransformation =
      *getSpatialTransformation3D(spatialTransformName);
  string point_clouds_path = "datasets/" + dataset + "_" + set + ".csv";

  std::ifstream fin(point_clouds_path);
  std::vector<PointCloud> point_clouds;
  std::string line;
  while (getline(fin, line) && point_clouds.size() < num_tests) {
    point_clouds.emplace_back(num_points, line);
  }

  auto verificationChunks =
      spatialTransformation.domain.split(num_splits, splitPoints);

  if (debug) {
    cout << "All verification chunks:" << endl;
    for (HyperBox &hbox : verificationChunks) {
      cout << "hbox: " << hbox << endl;
    }
  }

  GRBEnv *env;
  try {
    env = new GRBEnv("gurobi.log");
  } catch (GRBException &e) {
    std::cout << "Error code = " << e.getErrorCode() << std::endl;
    std::cout << e.getMessage() << std::endl;
    exit(1);
  }

  double totalBoxRuntime = 0, totalPolyRuntime = 0;
  Statistics counter;

  for (size_t j = 0; j < std::min(num_tests, point_clouds.size()); ++j) {
    std::cout << "PointCloud #" << j << endl;

    std::string out_file = out_dir + "/" + to_string(j) + ".csv";
    std::ofstream fou(out_file);
    fou.precision(12);
    fou.setf(ios_base::fixed);

    for (const HyperBox &hbox : verificationChunks) {
      for (const auto &it : hbox.it) {
        fou << it.inf << " " << it.sup << endl;
      }

      spatialTransformation.domain = hbox;

      std::chrono::system_clock::time_point beginBox =
          std::chrono::system_clock::now();

      std::cout << "Chunk: " << hbox << endl;
      std::cout << "Interval box: " << endl;
      PointCloud transformedPointCloud = abstractWithSimpleBox(
          hbox, point_clouds[j], spatialTransformation, num_inside_splits);
      fou << transformedPointCloud;

      std::chrono::system_clock::time_point endBox =
          std::chrono::system_clock::now();
      auto durationBox = std::chrono::duration_cast<std::chrono::milliseconds>(
          endBox - beginBox);
      std::cout << "Box runtime (sec): " << durationBox.count() / 1000.0
                << std::endl;
      totalBoxRuntime += durationBox.count() / 1000.0;

      std::cout << "Abstracting with DeepG" << std::endl;
      std::chrono::system_clock::time_point beginDeepG =
          std::chrono::system_clock::now();

      std::vector<Polyhedra> polys = abstractWithPolyhedra(
          hbox, *env, Constants::POLY_DEGREE, Constants::POLY_EPS,
          point_clouds[j], spatialTransformation, transformedPointCloud,
          counter);
      for (const auto &poly : polys) {
        fou << poly << std::endl;
      }

      std::chrono::system_clock::time_point endDeepG =
          std::chrono::system_clock::now();
      auto durationDeepG =
          std::chrono::duration_cast<std::chrono::milliseconds>(endDeepG -
                                                                beginDeepG);
      cout << "Poly runtime (sec): " << durationDeepG.count() / 1000.0 << endl;
      totalPolyRuntime += durationDeepG.count() / 1000.0;

      fou << "SPEC_FINISHED" << std::endl;
    }
  }

  std::string stats_file = out_dir + "/log.txt";
  std::ofstream fstats(stats_file);

  fstats << "Avg poly runtime (s): " << totalPolyRuntime / (double)(num_tests)
         << std::endl;
  fstats << "Avg polyhedra distance: " << counter.getAveragePolyhedra()
         << std::endl;

  return 0;
}
