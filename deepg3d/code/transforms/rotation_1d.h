#ifndef DEEPG_ROTATION_1D_H
#define DEEPG_ROTATION_1D_H

#include "transforms/transformation_3d.h"

class RotationTransformation1D : public SpatialTransformation3D {
public:
  explicit RotationTransformation1D(const HyperBox &domain)
      : SpatialTransformation3D(domain) {
    assert(domain.dim == 1);
  }

  Point<double> transform(const Point<double> &point,
                          const std::vector<double> &params) const override;
  Point<Interval> transform(const Point<Interval> &point,
                            const std::vector<Interval> &params) const override;
  std::tuple<std::vector<Interval>, std::vector<Interval>,
             std::vector<Interval>>
  gradTransform(const Point<Interval> &point,
                const std::vector<Interval> &params) const override;

  std::tuple<Interval, Interval, Interval>
  dx(const Point<Interval> &point,
     const std::vector<Interval> &params) const override;
  std::tuple<Interval, Interval, Interval>
  dy(const Point<Interval> &pixel,
     const std::vector<Interval> &params) const override;
  std::tuple<Interval, Interval, Interval>
  dz(const Point<Interval> &pixel,
     const std::vector<Interval> &params) const override;
  SpatialTransformation3D *getInverse() override;
};

#endif // DEEPG_ROTATION_1D_H
