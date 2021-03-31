#ifndef DEEPG_TRANSFORMATION_3D_H
#define DEEPG_TRANSFORMATION_3D_H

#include <utils/lipschitz.h>

class SpatialTransformation3D {
public:
  HyperBox domain;
  size_t dim;

  explicit SpatialTransformation3D(const HyperBox &domain) {
    this->domain = domain;
    this->dim = domain.dim;
  }

  virtual Point<double> transform(const Point<double> &point,
                                  const std::vector<double> &params) const = 0;
  virtual Point<Interval>
  transform(const Point<Interval> &point,
            const std::vector<Interval> &params) const = 0;
  virtual std::tuple<std::vector<Interval>, std::vector<Interval>,
                     std::vector<Interval>>
  gradTransform(const Point<Interval> &point,
                const std::vector<Interval> &params) const = 0;
  virtual std::tuple<Interval, Interval, Interval>
  dx(const Point<Interval> &point,
     const std::vector<Interval> &params) const = 0;
  virtual std::tuple<Interval, Interval, Interval>
  dy(const Point<Interval> &pixel,
     const std::vector<Interval> &params) const = 0;
  virtual std::tuple<Interval, Interval, Interval>
  dz(const Point<Interval> &pixel,
     const std::vector<Interval> &params) const = 0;
  virtual SpatialTransformation3D *getInverse() = 0;
};

#endif // DEEPG_TRANSFORMATION_3D_H
