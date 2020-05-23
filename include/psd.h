#ifndef PSD_H_
#define PSD_H_

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>


using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  
  Classifier(const string& model_file,const string& trained_file);

  ~Classifier();

  void Inference(const cv::Mat& img);

 private:

  void Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);

  void InitDeviceAndHostMemory();

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;

  uint8_t *device_in_seg_;
  uint8_t *host_in_seg_;
};

#endif
