#include "rotation_1d.h"

Point<double>
RotationTransformation1D::transform(const Point<double> &point,
                                    const std::vector<double> &params) const {
  assert(params.size() == 1);
  double alpha = params[0];
  return {cos(alpha) * point.x - sin(alpha) * point.y,
          sin(alpha) * point.x + cos(alpha) * point.y, point.z};
}

Point<Interval>
RotationTransformation1D::transform(const Point<Interval> &point,
                                    const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval alpha = params[0];
  return {cos(alpha) * point.x - sin(alpha) * point.y,
          sin(alpha) * point.x + cos(alpha) * point.y, point.z};
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
RotationTransformation1D::gradTransform(const Point<Interval> &point,
                                        const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval alpha = params[0];
  return {{-sin(alpha) * point.x - cos(alpha) * point.y},
          {cos(alpha) * point.x - sin(alpha) * point.y},
          {{0., 0.}}};
}

std::tuple<Interval, Interval, Interval>
RotationTransformation1D::dx(const Point<Interval> &point,
                             const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval alpha = params[0];
  return {cos(alpha), -sin(alpha), {0, 0}};
}

std::tuple<Interval, Interval, Interval>
RotationTransformation1D::dy(const Point<Interval> &point,
                             const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval alpha = params[0];
  return {sin(alpha), cos(alpha), {0, 0}};
}

std::tuple<Interval, Interval, Interval>
RotationTransformation1D::dz(const Point<Interval> &point,
                             const vector<Interval> &params) const {
  assert(params.size() == 1);
  return {{0, 0}, {0, 0}, {1, 1}};
}

SpatialTransformation3D *RotationTransformation1D::getInverse() {
  HyperBox new_domain =
      HyperBox({{-this->domain[0].sup, -this->domain[0].inf}});
  return new RotationTransformation1D(new_domain);
}
