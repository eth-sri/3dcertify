#ifndef DEEPG_COMPOSITION_3D_H
#define DEEPG_COMPOSITION_3D_H

#include "transforms/transformation_3d.h"

class CompositionTransform3D : public SpatialTransformation3D {
public:
  std::vector<SpatialTransformation3D *> transformations;

  explicit CompositionTransform3D(
      const std::vector<SpatialTransformation3D *> &transformations)
      : SpatialTransformation3D(HyperBox()) {
    this->transformations = transformations;
    for (SpatialTransformation3D *transformation : transformations) {
      for (Interval it : transformation->domain.it) {
        this->domain.it.push_back(it);
      }
    }
    this->dim = this->domain.dim = this->domain.it.size();
  }
  Point<double> transform(const Point<double> &point,
                          const std::vector<double> &params) const override;
  Point<Interval> transform(const Point<Interval> &point,
                            const std::vector<Interval> &params) const override;
  std::tuple<std::vector<Interval>, std::vector<Interval>,
             std::vector<Interval>>
  computeGrad(const Point<Interval> &point, const std::vector<Interval> &params,
              Interval &dxdx, Interval &dxdy, Interval &dxdz, Interval &dydx,
              Interval &dydy, Interval &dydz, Interval &dzdx, Interval &dzdy,
              Interval &dzdz) const;
  std::tuple<std::vector<Interval>, std::vector<Interval>,
             std::vector<Interval>>
  gradTransform(const Point<Interval> &point,
                const std::vector<Interval> &params) const override;
  std::tuple<Interval, Interval, Interval>
  dx(const Point<Interval> &point,
     const std::vector<Interval> &params) const override;
  std::tuple<Interval, Interval, Interval>
  dy(const Point<Interval> &point,
     const std::vector<Interval> &params) const override;
  std::tuple<Interval, Interval, Interval>
  dz(const Point<Interval> &point,
     const std::vector<Interval> &params) const override;
  SpatialTransformation3D *getInverse() override;
};

#endif // DEEPG_COMPOSITION_3D_H
