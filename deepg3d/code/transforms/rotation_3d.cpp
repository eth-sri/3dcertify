#include "rotation_3d.h"

Point<double>
RotationTransformation3D::transform(const Point<double> &point,
                                    const std::vector<double> &params) const {
  assert(params.size() == 3);
  double alpha = params[0];
  double beta = params[1];
  double gamma = params[2];
  return {(cos(alpha) * cos(beta)) * point.x +
              (cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma)) *
                  point.y +
              (cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)) *
                  point.z,
          (sin(alpha) * cos(beta)) * point.x +
              (sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma)) *
                  point.y +
              (sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma)) *
                  point.z,
          (-sin(beta)) * point.x + (cos(beta) * sin(gamma)) * point.y +
              (cos(beta) * cos(gamma)) * point.z

  };
}

Point<Interval>
RotationTransformation3D::transform(const Point<Interval> &point,
                                    const vector<Interval> &params) const {
  assert(params.size() == 3);
  Interval alpha = params[0];
  Interval beta = params[1];
  Interval gamma = params[2];
  return {(cos(alpha) * cos(beta)) * point.x +
              (cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma)) *
                  point.y +
              (cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)) *
                  point.z,
          (sin(alpha) * cos(beta)) * point.x +
              (sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma)) *
                  point.y +
              (sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma)) *
                  point.z,
          (-sin(beta)) * point.x + (cos(beta) * sin(gamma)) * point.y +
              (cos(beta) * cos(gamma)) * point.z};
}

std::tuple<std::vector<Interval>, std::vector<Interval>, std::vector<Interval>>
RotationTransformation3D::gradTransform(const Point<Interval> &point,
                                        const vector<Interval> &params) const {
  assert(params.size() == 3);
  Interval alpha = params[0];
  Interval beta = params[1];
  Interval gamma = params[2];
  return {
      {(-sin(alpha) * cos(beta)) * point.x +
           (-sin(alpha) * sin(beta) * sin(gamma) - cos(alpha) * cos(gamma)) *
               point.y +
           (-sin(alpha) * sin(beta) * cos(gamma) + cos(alpha) * sin(gamma)) *
               point.z,
       (cos(alpha) * -sin(beta)) * point.x +
           (cos(alpha) * cos(beta) * sin(gamma)) * point.y +
           (cos(alpha) * cos(beta) * cos(gamma)) * point.z,
       (cos(alpha) * sin(beta) * cos(gamma) - sin(alpha) * -sin(gamma)) *
               point.y +
           (cos(alpha) * sin(beta) * -sin(gamma) + sin(alpha) * cos(gamma)) *
               point.z},
      {(cos(alpha) * cos(beta)) * point.x +
           (cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma)) *
               point.y +
           (cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)) *
               point.z,
       (sin(alpha) * -sin(beta)) * point.x +
           (sin(alpha) * cos(beta) * sin(gamma)) * point.y +
           (sin(alpha) * cos(beta) * cos(gamma)) * point.z,
       (sin(alpha) * sin(beta) * cos(gamma) + cos(alpha) * -sin(gamma)) *
               point.y +
           (sin(alpha) * sin(beta) * -sin(gamma) - cos(alpha) * cos(gamma)) *
               point.z},
      {{0., 0.},
       (-cos(beta)) * point.x + (-sin(beta) * sin(gamma)) * point.y +
           (-sin(beta) * cos(gamma)) * point.z,
       (cos(beta) * cos(gamma)) * point.y +
           (cos(beta) * -sin(gamma)) * point.z}};
}

std::tuple<Interval, Interval, Interval>
RotationTransformation3D::dx(const Point<Interval> &point,
                             const vector<Interval> &params) const {
  assert(params.size() == 3);
  Interval alpha = params[0];
  Interval beta = params[1];
  Interval gamma = params[2];
  return {cos(alpha) * cos(beta),
          cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma),
          cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)};
}

std::tuple<Interval, Interval, Interval>
RotationTransformation3D::dy(const Point<Interval> &point,
                             const vector<Interval> &params) const {
  assert(params.size() == 3);
  Interval alpha = params[0];
  Interval beta = params[1];
  Interval gamma = params[2];
  return {sin(alpha) * cos(beta),
          sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
          sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma)};
}

std::tuple<Interval, Interval, Interval>
RotationTransformation3D::dz(const Point<Interval> &point,
                             const vector<Interval> &params) const {
  assert(params.size() == 3);
  Interval beta = params[1];
  Interval gamma = params[2];
  return {-sin(beta), cos(beta) * sin(gamma), cos(beta) * cos(gamma)};
}

SpatialTransformation3D *RotationTransformation3D::getInverse() {
  HyperBox new_domain =
      HyperBox({{-this->domain[0].sup, -this->domain[0].inf},
                {-this->domain[1].sup, -this->domain[1].inf},
                {-this->domain[2].sup, -this->domain[2].inf}});
  return new RotationTransformation3D(new_domain);
}
