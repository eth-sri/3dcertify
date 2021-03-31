#include "rotation_x.h"

Point<double>
RotationXTransformation3D::transform(const Point<double> &point,
                                     const std::vector<double> &params) const {
  assert(params.size() == 1);
  double gamma = params[0];
  return {point.x, cos(gamma) * point.y - sin(gamma) * point.z,
          sin(gamma) * point.y + cos(gamma) * point.z};
}

Point<Interval>
RotationXTransformation3D::transform(const Point<Interval> &point,
                                     const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval gamma = params[0];
  return {point.x, cos(gamma) * point.y - sin(gamma) * point.z,
          sin(gamma) * point.y + cos(gamma) * point.z};
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
RotationXTransformation3D::gradTransform(const Point<Interval> &point,
                                         const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval gamma = params[0];
  return {{{0, 0}},
          {-sin(gamma) * point.y - cos(gamma) * point.z},
          {cos(gamma) * point.y - sin(gamma) * point.z}};
}

std::tuple<Interval, Interval, Interval>
RotationXTransformation3D::dx(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 1);
  return {{1, 1}, {0, 0}, {0, 0}};
}

std::tuple<Interval, Interval, Interval>
RotationXTransformation3D::dy(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval gamma = params[0];
  return {{0, 0}, cos(gamma), -sin(gamma)};
}

std::tuple<Interval, Interval, Interval>
RotationXTransformation3D::dz(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval gamma = params[0];
  return {{0, 0}, sin(gamma), cos(gamma)};
}

SpatialTransformation3D *RotationXTransformation3D::getInverse() {
  HyperBox new_domain =
      HyperBox({{-this->domain[0].sup, -this->domain[0].inf}});
  return new RotationXTransformation3D(new_domain);
}
