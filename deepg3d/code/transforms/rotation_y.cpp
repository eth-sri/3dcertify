#include "rotation_y.h"

Point<double>
RotationYTransformation3D::transform(const Point<double> &point,
                                     const std::vector<double> &params) const {
  assert(params.size() == 1);
  double beta = params[0];
  return {cos(beta) * point.x + sin(beta) * point.z, point.y,
          -sin(beta) * point.x + cos(beta) * point.z};
}

Point<Interval>
RotationYTransformation3D::transform(const Point<Interval> &point,
                                     const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval beta = params[0];
  return {cos(beta) * point.x + sin(beta) * point.z, point.y,
          -sin(beta) * point.x + cos(beta) * point.z};
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
RotationYTransformation3D::gradTransform(const Point<Interval> &point,
                                         const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval beta = params[0];

  return {{-sin(beta) * point.x + cos(beta) * point.z},
          {{0., 0.}},
          {-cos(beta) * point.x - sin(beta) * point.z}};
}

std::tuple<Interval, Interval, Interval>
RotationYTransformation3D::dx(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval beta = params[0];
  return {cos(beta), {0, 0}, sin(beta)};
}

std::tuple<Interval, Interval, Interval>
RotationYTransformation3D::dy(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 1);
  return {{0, 0}, {1, 1}, {0, 0}};
}

std::tuple<Interval, Interval, Interval>
RotationYTransformation3D::dz(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval beta = params[0];
  return {-sin(beta), {0, 0}, cos(beta)};
}

SpatialTransformation3D *RotationYTransformation3D::getInverse() {
  HyperBox new_domain =
      HyperBox({{-this->domain[0].sup, -this->domain[0].inf}});
  return new RotationYTransformation3D(new_domain);
}
