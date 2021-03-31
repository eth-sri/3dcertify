#include "abstraction/abstraction.h"
#include "gurobi_c++.h"
#include "utils/constants.h"
#include "utils/polyhedra_optimizer.h"
#include "utils/utilities.h"
#include <chrono>
#include <future>
#include <utility>

LipschitzFunction getLipschitzFunction(
    const Image &img, const Pixel<double> &pixel,
    const HyperBox &combinedDomain,
    const SpatialTransformation &spatialTransformation,
    const PixelTransformation &pixelTransformation,
    const InterpolationTransformation &interpolationTransformation,
    bool lower) {
  /**
   * Returns LipschitzFunction representing composition of spatial
   * transformation and interpolation.
   *
   * @param img Image to be interpolated
   * @param pixel Pixel which is being transformed and interpolated
   * @param combinedDomain HyperBox domain which is concatenation of two
   * transformations domains
   * @param spatialTransformation SpatialTransformation which is being applied
   * @param pixelTransformation PixelTransformation applied to pixel values
   * @param interpolationTransformation InterpolationTransformation which is
   * used to perform interpolation
   */
  std::function<double(vector<double>)> minF =
      [lower, pixel, &img, &spatialTransformation, &pixelTransformation,
       &interpolationTransformation](vector<double> params) {
        vector<double> spatialTransformParams(
            params.begin(), params.begin() + spatialTransformation.dim);
        vector<double> pixelTransformParams(
            params.begin() + spatialTransformation.dim, params.end());

        Pixel<double> tPix =
            spatialTransformation.transform(pixel, spatialTransformParams);
        Interval ret = interpolationTransformation.transform(tPix, img, lower);
        return pixelTransformation.transform(lower ? ret.inf : ret.sup,
                                             pixelTransformParams);
      };

  Pixel<Interval> abstractPixel_interval = {
      {pixel.x, pixel.x}, {pixel.y, pixel.y}, pixel.channel};

  std::function<pair<bool, vector<Interval>>(HyperBox)> gradF_interval =
      [lower, abstractPixel_interval, &img, &spatialTransformation,
       &pixelTransformation, &interpolationTransformation](HyperBox hbox) {
        vector<Interval> spatialTransformParams(
            hbox.it.begin(), hbox.it.begin() + spatialTransformation.dim);
        vector<Interval> pixelTransformParams(
            hbox.it.begin() + spatialTransformation.dim, hbox.it.end());

        vector<Interval> dx, dy, dInt, ret;
        std::tie(dx, dy) = spatialTransformation.gradTransform(
            abstractPixel_interval, spatialTransformParams);
        Pixel<Interval> transformedPixel = spatialTransformation.transform(
            abstractPixel_interval, spatialTransformParams);

        bool differentiable;
        std::tie(differentiable, dInt) =
            interpolationTransformation.gradTransform(transformedPixel, img,
                                                      lower);

        Interval dPix;
        vector<Interval> gradPix;
        Interval interpolatedPixel =
            interpolationTransformation.transform(transformedPixel, img, lower);
        dPix = pixelTransformation.dp(interpolatedPixel, pixelTransformParams);
        gradPix = pixelTransformation.gradTransform(interpolatedPixel,
                                                    pixelTransformParams);

        for (size_t i = 0; i < spatialTransformation.dim; ++i) {
          ret.push_back((dInt[0] * dx[i] + dInt[1] * dy[i]) * dPix);
        }
        ret.insert(ret.end(), gradPix.begin(), gradPix.end());

        assert(ret.size() == hbox.dim);
        return make_pair(differentiable, ret);
      };

  return LipschitzFunction(minF, combinedDomain, gradF_interval);
}

Interval computePixelBox(
    int r, int c, int channel, const vector<HyperBox> &boxes, const Image &img,
    const SpatialTransformation &spatialTransformation,
    const PixelTransformation &pixelTransformation,
    const InterpolationTransformation &interpolationTransformation) {
  Pixel<double> concretePixel = img.getPixel(r, c, channel);
  Pixel<Interval> abstractPixel = {Interval(concretePixel.x, concretePixel.x),
                                   Interval(concretePixel.y, concretePixel.y),
                                   channel};
  Interval ret;
  for (const auto &hbox : boxes) {
    HyperBox hboxSpatial, hboxPixel;
    hbox.split(spatialTransformation.dim, hboxSpatial, hboxPixel);

    Pixel<Interval> newPixel =
        spatialTransformation.transform(abstractPixel, hboxSpatial.it);

    Interval minPixelValue =
        interpolationTransformation.transform(newPixel, img, true);
    Interval maxPixelValue =
        interpolationTransformation.transform(newPixel, img, false);

    Interval newPixelValue = minPixelValue.join(maxPixelValue);

    newPixelValue = pixelTransformation.transform(newPixelValue, hboxPixel.it);
    ret = ret.join(newPixelValue);
  }
  return ret;
}

Image abstractWithSimpleBox(
    const HyperBox &combinedDomain, const Image &img,
    const SpatialTransformation &spatialTransformation,
    const PixelTransformation &pixelTransformation,
    const InterpolationTransformation &interpolationTransformation,
    int insideSplits) {
  Image ret(img.nRows, img.nCols, img.nChannels);
  vector<vector<double>> splitPoints;
  vector<HyperBox> boxes = combinedDomain.split(insideSplits, splitPoints);

  vector<future<Interval>> v;
  vector<Interval> cv;

  for (int r = 0; r < img.nRows; ++r) {
    for (int c = 0; c < img.nCols; ++c) {
      for (int channel = 0; channel < img.nChannels; ++channel) {
        v.push_back(async(std::launch::async, computePixelBox, r, c, channel,
                          boxes, img, std::ref(spatialTransformation),
                          std::ref(pixelTransformation),
                          std::ref(interpolationTransformation)));
        if ((int)v.size() == Constants::NUM_THREADS) {
          for (auto &x : v) {
            cv.push_back(x.get());
          }
          v.clear();
        }
      }
    }
  }
  for (auto &x : v) {
    cv.push_back(x.get());
  }

  for (int r = 0; r < img.nRows; ++r) {
    for (int c = 0; c < img.nCols; ++c) {
      for (int channel = 0; channel < img.nChannels; ++channel) {
        ret.a[r][c][channel] =
            cv[r * img.nCols * img.nChannels + c * img.nChannels + channel];
      }
    }
  }

  return ret;
}

Point<Interval>
computePointBox(int p, const vector<HyperBox> &boxes,
                const PointCloud &pointCloud,
                const SpatialTransformation3D &spatialTransformation) {
  Point<Interval> abstractPoint = pointCloud.points[p];
  Point<Interval> transformedPoint((Interval()), Interval(), Interval());

  for (const auto &hbox : boxes) {
    Point<Interval> newPoint =
        spatialTransformation.transform(abstractPoint, hbox.it);

    transformedPoint.x = transformedPoint.x.join(newPoint.x);
    transformedPoint.y = transformedPoint.y.join(newPoint.y);
    transformedPoint.z = transformedPoint.z.join(newPoint.z);
  }

  return transformedPoint;
}

PointCloud abstractWithSimpleBox(const HyperBox &domain,
                                 const PointCloud &pointCloud,
                                 SpatialTransformation3D &spatialTransformation,
                                 size_t insideSplits) {
  PointCloud transformedPointCloud(pointCloud.nPoints);
  std::vector<vector<double>> splitPoints;
  std::vector<HyperBox> boxes = domain.split(insideSplits, splitPoints);

  std::vector<std::future<Point<Interval>>> futures;

  for (size_t p = 0; p < pointCloud.nPoints; ++p) {
    futures.push_back(std::async(std::launch::async, computePointBox, p, boxes,
                                 pointCloud, std::ref(spatialTransformation)));
    if ((int)futures.size() == Constants::NUM_THREADS) {
      for (auto &future : futures) {
        transformedPointCloud.points.push_back(future.get());
      }
      futures.clear();
    }
  }

  for (auto &future : futures) {
    transformedPointCloud.points.push_back(future.get());
  }

  return transformedPointCloud;
}

vector<Polyhedra> abstractWithCustomDP(
    const HyperBox &combinedDomain, const Image &img,
    const SpatialTransformation &spatialTransformation,
    const InterpolationTransformation &interpolationTransformation,
    const Image &transformedImage) {
  vector<Polyhedra> ret;
  std::default_random_engine generator;
  int numSamples = 1000;
  vector<PointD> samples = combinedDomain.sample(numSamples, generator);
  double fullAvgDistance = 0;

  for (int r = 0; r < img.nRows; ++r) {
    for (int c = 0; c < img.nCols; ++c) {
      for (int channel = 0; channel < img.nChannels; ++channel) {
        Pixel<double> concretePixel = img.getPixel(r, c, channel);
        Pixel<Interval> abstractPixel = {
            Interval(concretePixel.x, concretePixel.x),
            Interval(concretePixel.y, concretePixel.y), channel};
        Pixel<Interval> newPixel =
            spatialTransformation.transform(abstractPixel, combinedDomain.it);
        Polyhedra tmp = interpolationTransformation.transformCustom(
            newPixel, concretePixel, img, combinedDomain,
            transformedImage.a[r][c][channel]);

        double avgDistance = 0;
        for (const auto &sample : samples) {
          double evalLower = tmp.biasLower, evalUpper = tmp.biasUpper;
          for (size_t i = 0; i < combinedDomain.dim; ++i) {
            evalLower += tmp.wLower[i] * sample.x[i];
            evalUpper += tmp.wUpper[i] * sample.x[i];
          }
          assert(evalLower <= evalUpper);
          avgDistance += evalUpper - evalLower;
        }
        avgDistance /= numSamples;
        fullAvgDistance += avgDistance;

        ret.push_back(tmp);
      }
    }
  }
  fullAvgDistance /= img.nRows * img.nCols * img.nChannels;
  cout << "[CustomDP] Average distance between Polyhedra: " << fullAvgDistance
       << endl;
  return ret;
}

double checkBoundsWithSampling(int k, int degree,
                               std::default_random_engine generator,
                               const LipschitzFunction &lfLower,
                               const LipschitzFunction &lfUpper,
                               double biasLower, vector<double> wLower,
                               double biasUpper, vector<double> wUpper) {
  vector<PointD> samples = lfLower.domain.sample(k, generator);
  double avgLen = 0;
  for (auto sample : samples) {
    double evalFLow = lfLower.f(sample), evalFUpp = lfUpper.f(sample),
           evalLower = biasLower, evalUpper = biasUpper;
    for (size_t i = 0; i < lfLower.domain.dim; ++i) {
      for (int j = 0; j < degree; ++j) {
        evalLower += wLower[i * degree + j] * pow(sample.x[i], j + 1);
        evalUpper += wUpper[i * degree + j] * pow(sample.x[i], j + 1);
      }
    }

    assert(evalLower <= evalFLow + Constants::EPS &&
           evalFUpp <= evalUpper + Constants::EPS);
    avgLen += evalUpper - evalLower;
  }
  avgLen /= samples.size();
  return avgLen;
}

pair<double, Polyhedra> computePixelPolyhedra(
    const HyperBox &combinedDomain, int r, int c, int i, int degree, double eps,
    const GRBEnv &env, const Image &img,
    const SpatialTransformation &spatialTransformation,
    const PixelTransformation &pixelTransformation,
    const InterpolationTransformation &interpolationTransformation,
    // const Image& transformedImage,
    const Interval &tfInterval, Statistics &counter) {
  std::default_random_engine generator;
  Pixel<double> pixel = img.getPixel(r, c, i);
  auto fLower = getLipschitzFunction(img, pixel, combinedDomain,
                                     spatialTransformation, pixelTransformation,
                                     interpolationTransformation, true);
  auto fUpper = getLipschitzFunction(img, pixel, combinedDomain,
                                     spatialTransformation, pixelTransformation,
                                     interpolationTransformation, false);

  vector<double> wLower, wUpper;
  double biasLower, biasUpper;
  std::tie(wLower, biasLower) = findLower(
      env, fLower, generator, Constants::LP_SAMPLES, eps, degree, counter);
  std::tie(wUpper, biasUpper) = findUpper(
      env, fUpper, generator, Constants::LP_SAMPLES, eps, degree, counter);

  double avgLen = checkBoundsWithSampling(Constants::NUM_POLY_CHECK, degree,
                                          generator, fLower, fUpper, biasLower,
                                          wLower, biasUpper, wUpper);

  // If average length between polyhedra bounds is worse than interval bounds,
  // resort to interval
  if (avgLen > tfInterval.sup - tfInterval.inf) {
    biasLower = tfInterval.inf;
    biasUpper = tfInterval.sup;
    for (size_t j = 0; j < wLower.size(); ++j) {
      wLower[j] = 0;
      wUpper[j] = 0;
    }
    avgLen = biasUpper - biasLower;
  }
  return {avgLen, {tfInterval, wLower, biasLower, wUpper, biasUpper, degree}};
}

vector<Polyhedra> abstractWithPolyhedra(
    const HyperBox &combinedDomain, const GRBEnv &env, int degree, double eps,
    const Image &img, const SpatialTransformation &spatialTransformation,
    const PixelTransformation &pixelTransformation,
    const InterpolationTransformation &interpolationTransformation,
    const Image &transformedImage, Statistics &counter) {
  vector<Polyhedra> ret;
  vector<future<pair<double, Polyhedra>>> v;
  double mean = 0;

  for (int r = 0; r < img.nRows; ++r) {
    for (int c = 0; c < img.nCols; ++c) {
      for (int i = 0; i < img.nChannels; ++i) {
        counter.active_px = Pixel<int>(r, c, i);
        v.push_back(async(
            std::launch::async, computePixelPolyhedra, combinedDomain, r, c, i,
            degree, eps, std::ref(env), std::ref(img),
            std::ref(spatialTransformation), std::ref(pixelTransformation),
            std::ref(interpolationTransformation),
            std::ref(transformedImage.a[r][c][i]), std::ref(counter)));
        if ((int)v.size() == Constants::NUM_THREADS) {
          for (auto &x : v) {
            auto tmp = x.get();
            ret.push_back(tmp.second);
            mean += tmp.first;
          }
          v.clear();
        }
      }
    }
  }
  for (auto &x : v) {
    auto tmp = x.get();
    ret.push_back(tmp.second);
    mean += tmp.first;
  }
  mean /= img.nRows * img.nCols * img.nChannels;
  counter.updateAveragePolyhedra(mean);
  cout << "Average distance between polyhedra: " << mean << endl;
  return ret;
}

LipschitzFunction
getLipschitzFunction(const Point<double> &point, const HyperBox &domain,
                     const SpatialTransformation3D &spatialTransformation,
                     size_t coordinate) {
  std::function<double(std::vector<double>)> minF =
      [coordinate, point,
       &spatialTransformation](const std::vector<double> &params) {
        Point<double> transformedPoint =
            spatialTransformation.transform(point, params);

        switch (coordinate) {
        case 0:
          return transformedPoint.x;
        case 1:
          return transformedPoint.y;
        case 2:
          return transformedPoint.z;
        default:
          throw std::invalid_argument("coordinate out of range");
        }
      };

  Point<Interval> abstractPoint = {
      {point.x, point.x}, {point.y, point.y}, {point.z, point.z}};

  std::function<std::pair<bool, std::vector<Interval>>(HyperBox)> gradF =
      [coordinate, abstractPoint, &spatialTransformation](HyperBox hbox) {
        std::vector<Interval> params(
            hbox.it.begin(), hbox.it.begin() + spatialTransformation.dim);

        std::vector<Interval> dx, dy, dz, result;
        std::tie(dx, dy, dz) =
            spatialTransformation.gradTransform(abstractPoint, params);

        switch (coordinate) {
        case 0:
          result = dx;
          break;
        case 1:
          result = dy;
          break;
        case 2:
          result = dz;
          break;
        default:
          throw std::invalid_argument("coordinate out of range");
        }

        assert(result.size() == hbox.dim);
        return make_pair(true, result);
      };

  return LipschitzFunction(minF, domain, gradF);
}

std::pair<double, Polyhedra> computePointPolyhedra(
    const HyperBox &domain, size_t p, size_t coordinate, int degree, double eps,
    const GRBEnv &env, const PointCloud &pointCloud,
    const SpatialTransformation3D &spatialTransformation,
    const PointCloud &transformedPointCloud, Statistics &counter) {
  std::default_random_engine generator;

  Point<Interval> abstractPoint = pointCloud.points[p];
  assert(abstractPoint.x.inf == abstractPoint.x.sup);
  assert(abstractPoint.y.inf == abstractPoint.y.sup);
  assert(abstractPoint.z.inf == abstractPoint.z.sup);
  Point<double> point(abstractPoint.x.inf, abstractPoint.y.inf,
                      abstractPoint.z.sup);
  auto f =
      getLipschitzFunction(point, domain, spatialTransformation, coordinate);

  std::vector<double> wLower, wUpper;
  double biasLower, biasUpper;
  std::tie(wLower, biasLower) =
      findLower(env, f, generator, Constants::LP_SAMPLES, eps, degree, counter);
  std::tie(wUpper, biasUpper) =
      findUpper(env, f, generator, Constants::LP_SAMPLES, eps, degree, counter);

  double avgLen =
      checkBoundsWithSampling(Constants::NUM_POLY_CHECK, degree, generator, f,
                              f, biasLower, wLower, biasUpper, wUpper);

  Interval transformedInterval;
  switch (coordinate) {
  case 0:
    transformedInterval = transformedPointCloud.points[p].x;
    break;
  case 1:
    transformedInterval = transformedPointCloud.points[p].y;
    break;
  case 2:
    transformedInterval = transformedPointCloud.points[p].z;
    break;
  default:
    throw std::invalid_argument("coordinate out of range");
  }

  // If average length between polyhedra bounds is worse than interval bounds,
  // resort to interval
  if (avgLen > transformedInterval.length()) {
    biasLower = transformedInterval.inf;
    biasUpper = transformedInterval.sup;
    for (size_t j = 0; j < wLower.size(); ++j) {
      wLower[j] = 0;
      wUpper[j] = 0;
    }
    avgLen = biasUpper - biasLower;
  }
  return {avgLen,
          {transformedInterval, wLower, biasLower, wUpper, biasUpper, degree}};
}

std::vector<Polyhedra>
abstractWithPolyhedra(const HyperBox &domain, const GRBEnv &env, int degree,
                      double eps, const PointCloud &pointCloud,
                      const SpatialTransformation3D &spatialTransformation,
                      const PointCloud &transformedPointCloud,
                      Statistics &counter) {
  std::vector<Polyhedra> results;
  double mean = 0;

  std::vector<std::future<std::pair<double, Polyhedra>>> futures;

  for (size_t p = 0; p < pointCloud.nPoints; ++p) {
    for (size_t coordinate = 0; coordinate < 3; ++coordinate) {
      futures.push_back(
          std::async(std::launch::async, computePointPolyhedra, domain, p,
                     coordinate, degree, eps, std::ref(env),
                     std::ref(pointCloud), std::ref(spatialTransformation),
                     std::ref(transformedPointCloud), std::ref(counter)));
      if ((int)futures.size() == Constants::NUM_THREADS) {
        for (auto &future : futures) {
          auto tmp = future.get();
          mean += tmp.first;
          results.push_back(tmp.second);
        }
        futures.clear();
      }
    }
  }
  for (auto &future : futures) {
    auto tmp = future.get();
    mean += tmp.first;
    results.push_back(tmp.second);
  }

  mean /= 3. * pointCloud.nPoints;
  counter.updateAveragePolyhedra(mean);
  std::cout << "Average distance between polyhedra: " << mean << std::endl;

  return results;
}