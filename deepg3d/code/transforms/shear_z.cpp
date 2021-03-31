#include "shear_z.h"

Point<double>
ShearZTransformation3D::transform(const Point<double> &point,
                                  const std::vector<double> &params) const {
  assert(params.size() == 2);
  double sx = params[0];
  double sy = params[1];
  return {sx * point.z + point.x, sy * point.z + point.y, point.z};
}

Point<Interval>
ShearZTransformation3D::transform(const Point<Interval> &point,
                                  const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval sx = params[0];
  Interval sy = params[1];
  return {sx * point.z + point.x, sy * point.z + point.y, point.z};
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
ShearZTransformation3D::gradTransform(const Point<Interval> &point,
                                      const vector<Interval> &params) const {
  assert(params.size() == 2);
  return {{point.z, {0, 0}}, {{0, 0}, point.z}, {{0, 0}, {0, 0}}};
}

std::tuple<Interval, Interval, Interval>
ShearZTransformation3D::dx(const Point<Interval> &point,
                           const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval sx = params[0];
  return {{1, 1}, {0, 0}, sx};
}

std::tuple<Interval, Interval, Interval>
ShearZTransformation3D::dy(const Point<Interval> &point,
                           const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval sy = params[1];
  return {{0, 0}, {1, 1}, sy};
}

std::tuple<Interval, Interval, Interval>
ShearZTransformation3D::dz(const Point<Interval> &point,
                           const vector<Interval> &params) const {
  assert(params.size() == 2);
  return {{0, 0}, {0, 0}, {1, 1}};
}

SpatialTransformation3D *ShearZTransformation3D::getInverse() {
  HyperBox new_domain =
      HyperBox({{-this->domain[0].sup, -this->domain[0].inf},
                {-this->domain[1].sup, -this->domain[1].inf}});
  return new ShearZTransformation3D(new_domain);
}
