#include "rotation_z.h"

Point<double>
RotationZTransformation3D::transform(const Point<double> &point,
                                     const std::vector<double> &params) const {
  assert(params.size() == 1);
  double alpha = params[0];
  return {cos(alpha) * point.x - sin(alpha) * point.y,
          sin(alpha) * point.x + cos(alpha) * point.y, point.z};
}

Point<Interval>
RotationZTransformation3D::transform(const Point<Interval> &point,
                                     const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval alpha = params[0];
  return {cos(alpha) * point.x - sin(alpha) * point.y,
          sin(alpha) * point.x + cos(alpha) * point.y, point.z};
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
RotationZTransformation3D::gradTransform(const Point<Interval> &point,
                                         const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval alpha = params[0];
  return {{-sin(alpha) * point.x - cos(alpha) * point.y},
          {cos(alpha) * point.x - sin(alpha) * point.y},
          {{0., 0.}}};
}

std::tuple<Interval, Interval, Interval>
RotationZTransformation3D::dx(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval alpha = params[0];
  return {cos(alpha), -sin(alpha), {0, 0}};
}

std::tuple<Interval, Interval, Interval>
RotationZTransformation3D::dy(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval alpha = params[0];
  return {sin(alpha), cos(alpha), {0, 0}};
}

std::tuple<Interval, Interval, Interval>
RotationZTransformation3D::dz(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 1);
  return {{0, 0}, {0, 0}, {1, 1}};
}

SpatialTransformation3D *RotationZTransformation3D::getInverse() {
  HyperBox new_domain =
      HyperBox({{-this->domain[0].sup, -this->domain[0].inf}});
  return new RotationZTransformation3D(new_domain);
}
