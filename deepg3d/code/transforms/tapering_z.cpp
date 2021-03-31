#include "tapering_z.h"
#include "tapering_z_inverse.h"

Point<double>
TaperingZTransformation3D::transform(const Point<double> &point,
                                     const std::vector<double> &params) const {
  assert(params.size() == 2);
  double alpha = params[0];
  double beta = params[1];
  const auto tmp = 0.5 * alpha * alpha * point.z + beta * point.z + 1;
  return {tmp * point.x, tmp * point.y, point.z};
}

Point<Interval>
TaperingZTransformation3D::transform(const Point<Interval> &point,
                                     const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval alpha = params[0];
  Interval beta = params[1];
  const auto tmp = 0.5 * alpha * alpha * point.z + beta * point.z + 1;
  return {tmp * point.x, tmp * point.y, point.z};
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
TaperingZTransformation3D::gradTransform(const Point<Interval> &point,
                                         const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval alpha = params[0];
  return {{alpha * point.z * point.x, point.z * point.x},
          {alpha * point.z * point.y, point.z * point.y},
          {{0., 0.}, {0, 0}}};
}

std::tuple<Interval, Interval, Interval>
TaperingZTransformation3D::dx(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval alpha = params[0];
  Interval beta = params[1];
  return {0.5 * alpha * alpha * point.z + beta * point.z + 1,
          {0, 0},
          (0.5 * alpha * alpha + beta) * point.x};
}

std::tuple<Interval, Interval, Interval>
TaperingZTransformation3D::dy(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 2);
  Interval alpha = params[0];
  Interval beta = params[1];
  return {{0, 0},
          0.5 * alpha * alpha * point.z + beta * point.z + 1,
          (0.5 * alpha * alpha + beta) * point.y};
}

std::tuple<Interval, Interval, Interval>
TaperingZTransformation3D::dz(const Point<Interval> &point,
                              const vector<Interval> &params) const {
  assert(params.size() == 2);
  return {{0, 0}, {0, 0}, {1, 1}};
}

SpatialTransformation3D *TaperingZTransformation3D::getInverse() {
  return new TaperingZInverseTransformation3D(this->domain);
}
