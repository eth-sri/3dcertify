#include "composition_3d.h"

Point<double>
CompositionTransform3D::transform(const Point<double> &point,
                                  const vector<double> &params) const {
  assert(params.size() == dim);
  size_t param_idx = 0;
  Point<double> transformedPoint = point;

  for (SpatialTransformation3D *transformation : this->transformations) {
    std::vector<double> currentParams(params.begin() + param_idx,
                                      params.begin() + param_idx +
                                          transformation->dim);
    param_idx += transformation->dim;
    transformedPoint =
        transformation->transform(transformedPoint, currentParams);
  }

  return transformedPoint;
}

Point<Interval>
CompositionTransform3D::transform(const Point<Interval> &point,
                                  const vector<Interval> &params) const {
  assert(params.size() == dim);
  size_t param_idx = 0;
  Point<Interval> transformedPoint = point;

  for (SpatialTransformation3D *transformation : this->transformations) {
    std::vector<Interval> currentParams(params.begin() + param_idx,
                                        params.begin() + param_idx +
                                            transformation->dim);
    param_idx += transformation->dim;
    transformedPoint =
        transformation->transform(transformedPoint, currentParams);
  }

  return transformedPoint;
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
CompositionTransform3D::computeGrad(const Point<Interval> &point,
                                    const std::vector<Interval> &params,
                                    Interval &dxdx, Interval &dxdy,
                                    Interval &dxdz, Interval &dydx,
                                    Interval &dydy, Interval &dydz,
                                    Interval &dzdx, Interval &dzdy,
                                    Interval &dzdz) const {

  std::vector<Point<Interval>> transformedPoints;
  std::vector<std::vector<Interval>> transformationParameters;

  size_t param_idx = 0;
  Point<Interval> transformedPoint = point;

  for (SpatialTransformation3D *transformation : this->transformations) {
    std::vector<Interval> currentParams(params.begin() + param_idx,
                                        params.begin() + param_idx +
                                            transformation->dim);
    param_idx += transformation->dim;
    transformedPoints.push_back(transformedPoint);
    transformationParameters.push_back(currentParams);
    transformedPoint =
        transformation->transform(transformedPoint, currentParams);
  }

  std::vector<Interval> retX, retY, retZ;

  for (int i = (int)this->transformations.size() - 1; i >= 0; --i) {
    std::vector<Interval> gradX, gradY, gradZ;
    std::tie(gradX, gradY, gradZ) = this->transformations[i]->gradTransform(
        transformedPoints[i], transformationParameters[i]);

    for (int idx = (int)gradX.size() - 1; idx >= 0; --idx) {
      retX.insert(retX.begin(),
                  dxdx * gradX[idx] + dxdy * gradY[idx] + dxdz * gradZ[idx]);
      retY.insert(retY.begin(),
                  dydx * gradX[idx] + dydy * gradY[idx] + dydz * gradZ[idx]);
      retZ.insert(retZ.begin(),
                  dzdx * gradX[idx] + dzdy * gradY[idx] + dzdz * gradZ[idx]);
    }

    Interval cdxdx, cdxdy, cdxdz, cdydx, cdydy, cdydz, cdzdx, cdzdy, cdzdz;
    std::tie(cdxdx, cdxdy, cdxdz) = this->transformations[i]->dx(
        transformedPoints[i], transformationParameters[i]);
    std::tie(cdydx, cdydy, cdydz) = this->transformations[i]->dy(
        transformedPoints[i], transformationParameters[i]);
    std::tie(cdzdx, cdzdy, cdzdz) = this->transformations[i]->dz(
        transformedPoints[i], transformationParameters[i]);

    auto tmp_dxdx = dxdx * cdxdx + dxdy * cdydx + dxdz * cdzdx;
    auto tmp_dxdy = dxdx * cdxdy + dxdy * cdydy + dxdz * cdzdy;
    auto tmp_dxdz = dxdx * cdxdz + dxdy * cdydz + dxdz * cdzdz;
    auto tmp_dydx = dydx * cdxdx + dydy * cdydx + dydz * cdzdx;
    auto tmp_dydy = dydx * cdxdy + dydy * cdydy + dydz * cdzdy;
    auto tmp_dydz = dydx * cdxdz + dydy * cdydz + dydz * cdzdz;
    auto tmp_dzdx = dzdx * cdxdx + dzdy * cdydx + dzdz * cdzdx;
    auto tmp_dzdy = dzdx * cdxdy + dzdy * cdydy + dzdz * cdzdy;
    auto tmp_dzdz = dzdx * cdxdz + dzdy * cdydz + dzdz * cdzdz;

    dxdx = tmp_dxdx;
    dxdy = tmp_dxdy;
    dxdz = tmp_dxdz;
    dydx = tmp_dydx;
    dydy = tmp_dydy;
    dydz = tmp_dydz;
    dzdx = tmp_dzdx;
    dzdy = tmp_dzdy;
    dzdz = tmp_dzdz;
  }

  return {retX, retY, retZ};
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
CompositionTransform3D::gradTransform(const Point<Interval> &point,
                                      const vector<Interval> &params) const {
  Interval dxdx(1, 1), dxdy(0, 0), dxdz(0, 0), dydx(0, 0), dydy(1, 1),
      dydz(0, 0), dzdx(0, 0), dzdy(0, 0), dzdz(1, 1);

  return computeGrad(point, params, dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx,
                     dzdy, dzdz);
}

std::tuple<Interval, Interval, Interval>
CompositionTransform3D::dx(const Point<Interval> &point,
                           const vector<Interval> &params) const {
  Interval dxdx(1, 1), dxdy(0, 0), dxdz(0, 0), dydx(0, 0), dydy(1, 1),
      dydz(0, 0), dzdx(0, 0), dzdy(0, 0), dzdz(1, 1);
  computeGrad(point, params, dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy,
              dzdz);
  return {dxdx, dxdy, dxdz};
}

std::tuple<Interval, Interval, Interval>
CompositionTransform3D::dy(const Point<Interval> &point,
                           const vector<Interval> &params) const {
  Interval dxdx(1, 1), dxdy(0, 0), dxdz(0, 0), dydx(0, 0), dydy(1, 1),
      dydz(0, 0), dzdx(0, 0), dzdy(0, 0), dzdz(1, 1);
  computeGrad(point, params, dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy,
              dzdz);
  return {dydx, dydy, dydz};
}

std::tuple<Interval, Interval, Interval>
CompositionTransform3D::dz(const Point<Interval> &point,
                           const vector<Interval> &params) const {
  Interval dxdx(1, 1), dxdy(0, 0), dxdz(0, 0), dydx(0, 0), dydy(1, 1),
      dydz(0, 0), dzdx(0, 0), dzdy(0, 0), dzdz(1, 1);
  computeGrad(point, params, dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy,
              dzdz);
  return {dzdx, dzdy, dzdz};
}

SpatialTransformation3D *CompositionTransform3D::getInverse() {
  std::vector<SpatialTransformation3D *> inverseTransformations;

  for (auto it = this->transformations.rbegin();
       it != this->transformations.rend(); ++it) {
    inverseTransformations.push_back((*it)->getInverse());
  }

  return new CompositionTransform3D(inverseTransformations);
}
