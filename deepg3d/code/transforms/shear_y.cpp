#include "shear_y.h"

Point<double>
ShearYTransformation3D::transform(const Point<double> &point,
                                  const std::vector<double> &params) const {
  assert(params.size() == 2);
  double sx = params[0];
  double sz = params[1];
  return {sx * point.y + point.x, point.y, sz * point.y + point.z};
}

Point<Interval>
ShearYTransformation3D::transform(const Point<Interval> &point,
                                  const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval sx = params[0];
  Interval sz = params[1];
  return {sx * point.y + point.x, point.y, sz * point.y + point.z};
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
ShearYTransformation3D::gradTransform(const Point<Interval> &point,
                                      const vector<Interval> &params) const {
  assert(params.size() == 2);
  return {{point.y, {0, 0}}, {{0, 0}, {0, 0}}, {{0, 0}, point.y}};
}

std::tuple<Interval, Interval, Interval>
ShearYTransformation3D::dx(const Point<Interval> &point,
                           const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval sx = params[0];
  return {{1, 1}, sx, {0, 0}};
}

std::tuple<Interval, Interval, Interval>
ShearYTransformation3D::dy(const Point<Interval> &point,
                           const vector<Interval> &params) const {
  assert(params.size() == 2);
  return {{0, 0}, {1, 1}, {0, 0}};
}

std::tuple<Interval, Interval, Interval>
ShearYTransformation3D::dz(const Point<Interval> &point,
                           const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval sz = params[1];
  return {{0, 0}, sz, {1, 1}};
}

SpatialTransformation3D *ShearYTransformation3D::getInverse() {
  HyperBox new_domain =
      HyperBox({{-this->domain[0].sup, -this->domain[0].inf},
                {-this->domain[1].sup, -this->domain[1].inf}});
  return new ShearYTransformation3D(new_domain);
}
