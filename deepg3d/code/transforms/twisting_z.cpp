#include "twisting_z.h"

Point<double>
TwistingZTransformation3D::transform(const Point<double> &point,
                                     const std::vector<double> &params) const {
  assert(params.size() == 1);
  double alpha = params[0];
  return {cos(alpha * point.z) * point.x - sin(alpha * point.z) * point.y,
          sin(alpha * point.z) * point.x + cos(alpha * point.z) * point.y,
          point.z};
}

Point<Interval>
TwistingZTransformation3D::transform(const Point<Interval> &point,
                                     const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval alpha = params[0];
  return {cos(alpha * point.z) * point.x - sin(alpha * point.z) * point.y,
          sin(alpha * point.z) * point.x + cos(alpha * point.z) * point.y,
          point.z};
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
TwistingZTransformation3D::gradTransform(const Point<Interval> &point,
                                         const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval alpha = params[0];
  return {{-point.z *
           (sin(alpha * point.z) * point.x + cos(alpha * point.z) * point.y)},
          {point.z *
           (cos(alpha * point.z) * point.x - sin(alpha * point.z) * point.y)},
          {{0, 0}}};
}

std::tuple<Interval, Interval, Interval>
TwistingZTransformation3D::dx(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval alpha = params[0];
  return {cos(alpha * point.z), -sin(alpha * point.z),
          -alpha * (sin(alpha * point.z) * point.x +
                    cos(alpha * point.z) * point.y)};
}

std::tuple<Interval, Interval, Interval>
TwistingZTransformation3D::dy(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 1);
  Interval alpha = params[0];
  return {sin(alpha * point.z), cos(alpha * point.z),
          alpha * (cos(alpha * point.z) * point.x -
                   sin(alpha * point.z) * point.y)};
}

std::tuple<Interval, Interval, Interval>
TwistingZTransformation3D::dz(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 1);
  return {{0, 0}, {0, 0}, {1, 1}};
}

SpatialTransformation3D *TwistingZTransformation3D::getInverse() {
  HyperBox new_domain =
      HyperBox({{-this->domain[0].sup, -this->domain[0].inf}});
  return new TwistingZTransformation3D(new_domain);
}
