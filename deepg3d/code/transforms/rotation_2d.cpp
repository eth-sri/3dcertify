#include "rotation_2d.h"

Point<double>
RotationTransformation2D::transform(const Point<double> &point,
                                    const std::vector<double> &params) const {
  assert(params.size() == 2);
  double alpha = params[0];
  double gamma = params[1];
  return {cos(alpha) * point.x - sin(alpha) * cos(gamma) * point.y +
              sin(alpha) * sin(gamma) * point.z,
          sin(alpha) * point.x + cos(alpha) * cos(gamma) * point.y -
              cos(alpha) * sin(gamma) * point.z,
          sin(gamma) * point.y + cos(gamma) * point.z};
}

Point<Interval>
RotationTransformation2D::transform(const Point<Interval> &point,
                                    const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval alpha = params[0];
  Interval gamma = params[1];
  return {cos(alpha) * point.x - sin(alpha) * cos(gamma) * point.y +
              sin(alpha) * sin(gamma) * point.z,
          sin(alpha) * point.x + cos(alpha) * cos(gamma) * point.y -
              cos(alpha) * sin(gamma) * point.z,
          sin(gamma) * point.y + cos(gamma) * point.z};
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
RotationTransformation2D::gradTransform(const Point<Interval> &point,
                                        const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval alpha = params[0];
  Interval gamma = params[1];
  return {
      {-sin(alpha) * point.x - cos(alpha) * cos(gamma) * point.y +
           cos(alpha) * sin(gamma) * point.z,
       -sin(alpha) * -sin(gamma) * point.y + sin(alpha) * cos(gamma) * point.z},
      {cos(alpha) * point.x - sin(alpha) * cos(gamma) * point.y +
           sin(alpha) * sin(gamma) * point.z,
       cos(alpha) * -sin(gamma) * point.y - cos(alpha) * cos(gamma) * point.z},
      {{0., 0.}, cos(gamma) * point.y - sin(gamma) * point.z}};
}

std::tuple<Interval, Interval, Interval>
RotationTransformation2D::dx(const Point<Interval> &point,
                             const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval alpha = params[0];
  Interval gamma = params[1];
  return {cos(alpha), -sin(alpha) * cos(gamma), sin(alpha) * sin(gamma)};
}

std::tuple<Interval, Interval, Interval>
RotationTransformation2D::dy(const Point<Interval> &point,
                             const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval alpha = params[0];
  Interval gamma = params[1];
  return {sin(alpha), cos(alpha) * cos(gamma), -cos(alpha) * sin(gamma)};
}

std::tuple<Interval, Interval, Interval>
RotationTransformation2D::dz(const Point<Interval> &point,
                             const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval gamma = params[1];
  return {{0, 0}, sin(gamma), cos(gamma)};
}

SpatialTransformation3D *RotationTransformation2D::getInverse() {
  HyperBox new_domain =
      HyperBox({{-this->domain[0].sup, -this->domain[0].inf},
                {-this->domain[1].sup, -this->domain[1].inf}});
  return new RotationTransformation2D(new_domain);
}
