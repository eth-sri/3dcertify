#include "shear_x.h"

Point<double>
ShearXTransformation3D::transform(const Point<double> &point,
                                  const std::vector<double> &params) const {
  assert(params.size() == 2);
  double sy = params[0];
  double sz = params[1];
  return {point.x, sy * point.x + point.y, sz * point.x + point.z};
}

Point<Interval>
ShearXTransformation3D::transform(const Point<Interval> &point,
                                  const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval sy = params[0];
  Interval sz = params[1];
  return {point.x, sy * point.x + point.y, sz * point.x + point.z};
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
ShearXTransformation3D::gradTransform(const Point<Interval> &point,
                                      const vector<Interval> &params) const {
  assert(params.size() == 2);
  return {{{0, 0}, {0, 0}}, {point.x, {0, 0}}, {{0, 0}, point.x}};
}

std::tuple<Interval, Interval, Interval>
ShearXTransformation3D::dx(const Point<Interval> &point,
                           const vector<Interval> &params) const {
  assert(params.size() == 2);
  return {{1, 1}, {0, 0}, {0, 0}};
}

std::tuple<Interval, Interval, Interval>
ShearXTransformation3D::dy(const Point<Interval> &point,
                           const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval sy = params[0];
  return {sy, {1, 1}, {0, 0}};
}

std::tuple<Interval, Interval, Interval>
ShearXTransformation3D::dz(const Point<Interval> &point,
                           const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval sz = params[1];
  return {sz, {0, 0}, {1, 1}};
}

SpatialTransformation3D *ShearXTransformation3D::getInverse() {
  HyperBox new_domain =
      HyperBox({{-this->domain[0].sup, -this->domain[0].inf},
                {-this->domain[1].sup, -this->domain[1].inf}});
  return new ShearXTransformation3D(new_domain);
}
