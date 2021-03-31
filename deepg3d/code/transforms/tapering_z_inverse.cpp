#include "tapering_z_inverse.h"
#include "tapering_z.h"

Point<double> TaperingZInverseTransformation3D::transform(
    const Point<double> &point, const std::vector<double> &params) const {
  assert(params.size() == 2);
  double alpha = params[0];
  double beta = params[1];
  const auto tmp = 0.5 * alpha * alpha * point.z + beta * point.z + 1;
  return {point.x / tmp, point.y / tmp, point.z};
}

Point<Interval> TaperingZInverseTransformation3D::transform(
    const Point<Interval> &point, const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval alpha = params[0];
  Interval beta = params[1];
  const auto tmp = 0.5 * alpha * alpha * point.z + beta * point.z + 1;
  return {point.x / tmp, point.y / tmp, point.z};
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
TaperingZInverseTransformation3D::gradTransform(
    const Point<Interval> &point, const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval alpha = params[0];
  Interval beta = params[1];
  const auto tmp = 0.5 * alpha * alpha * point.z + beta * point.z + 1;
  const auto tmp_squared = tmp * tmp;
  return {{-alpha * point.z * point.x / tmp_squared,
           -point.z * point.x / tmp_squared},
          {-alpha * point.z * point.y / tmp_squared,
           -point.z * point.y / tmp_squared},
          {{0., 0.}, {0, 0}}};
}

std::tuple<Interval, Interval, Interval>
TaperingZInverseTransformation3D::dx(const Point<Interval> &point,
                                     const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval alpha = params[0];
  Interval beta = params[1];
  const auto tmp = 0.5 * alpha * alpha * point.z + beta * point.z + 1;
  return {
      1. / tmp, {0, 0}, -(0.5 * alpha * alpha + beta) * point.x / (tmp * tmp)};
}

std::tuple<Interval, Interval, Interval>
TaperingZInverseTransformation3D::dy(const Point<Interval> &point,
                                     const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval alpha = params[0];
  Interval beta = params[1];
  const auto tmp = 0.5 * alpha * alpha * point.z + beta * point.z + 1;
  return {
      {0, 0}, 1. / tmp, -(0.5 * alpha * alpha + beta) * point.y / (tmp * tmp)};
}

std::tuple<Interval, Interval, Interval>
TaperingZInverseTransformation3D::dz(const Point<Interval> &point,
                                     const vector<Interval> &params) const {
  assert(params.size() == 2);
  return {{0, 0}, {0, 0}, {1, 1}};
}

SpatialTransformation3D *TaperingZInverseTransformation3D::getInverse() {
  return new TaperingZTransformation3D(this->domain);
}
