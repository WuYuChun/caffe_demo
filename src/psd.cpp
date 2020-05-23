#include "psd.h"
#include "cuda_function.h"


//这里把输出的blob的名字都写死了
std::string pt_class_prob{"pt_class_prob"};
std::string pt_reg_pred_y{"pt_reg_pred_y"};
std::string pt_reg_pred_x{"pt_reg_pred_x"};
std::string pt_seg_pred{"pt_seg_pred"};
std::string lm_seg_pred{"lm_seg_pred"};



Classifier::Classifier(const string& model_file,const string& trained_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 5) << "Network should have exactly fine output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  CHECK(net_->has_blob(pt_class_prob) && net_->has_blob(pt_reg_pred_y) && net_->has_blob(pt_reg_pred_x) 
                       && net_->has_blob(pt_seg_pred) && net_->has_blob(lm_seg_pred));          

  // Blob<float>* output_layer = net_->output_blobs()[0];
  // std::cout << "----------------: " << output_layer->channels() << "  " << output_layer->count() << std::endl;
  // std::cout << "----------------:  " << output_layer->height() << std::endl;
  InitDeviceAndHostMemory();

}

Classifier::~Classifier(){
  cudaFree(device_in_seg_);
  device_in_seg_ = nullptr;

  free(host_in_seg_);
  host_in_seg_ = nullptr;
}


void Classifier::Inference(const cv::Mat& img ) {
   Predict(img);
}

void Classifier::Predict(const cv::Mat& img) {



  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,input_geometry_.height, input_geometry_.width);

  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);


  net_->Forward();

  //get output
  auto output_pt_class_prob = net_->blob_by_name(pt_class_prob);
  auto output_pt_reg_pred_y =  net_->blob_by_name(pt_reg_pred_y);
  auto output_pt_reg_pred_x = net_->blob_by_name(pt_reg_pred_x);
  auto output_pt_seg_pred = net_->blob_by_name(pt_seg_pred);
  auto output_lm_seg_pred = net_->blob_by_name(lm_seg_pred);

  //check output size
  //pt_class_prob: 4  130  80
  //pt_reg_pred_y: 4  130  80
  //pt_reg_pred_x: 4  130  80
  //pt_seg_pred:   3  130  80
  //lm_seg_pred:   10  130  80
  std::cout << "pt_class_prob: " << output_pt_class_prob->channels() << "  " 
                                 << output_pt_class_prob->height()   << "  "<< output_pt_class_prob->width() << std::endl;
  std::cout << "pt_reg_pred_y: " << output_pt_reg_pred_y->channels() << "  " 
                                 << output_pt_reg_pred_y->height()   << "  " << output_pt_reg_pred_y->width() << std::endl;
  std::cout << "pt_reg_pred_x: " << output_pt_reg_pred_x->channels() << "  " 
                                 << output_pt_reg_pred_x->height()   << "  " << output_pt_reg_pred_x->width() << std::endl;
  std::cout << "pt_seg_pred:   " << output_pt_seg_pred->channels()   << "  " 
                                 << output_pt_seg_pred->height()     << "  " << output_pt_seg_pred->width() << std::endl;
  std::cout << "lm_seg_pred:   " << output_lm_seg_pred->channels()   << "  " 
                                 << output_lm_seg_pred->height()      << "  " << output_lm_seg_pred->width() << std::endl;  


  //get output data
  const float *output_pt_class_prob_value = output_pt_class_prob->cpu_data();
  const float *output_pt_reg_pred_y_value = output_pt_reg_pred_y->cpu_data();
  const float *output_pt_reg_pred_x_value = output_pt_reg_pred_x->cpu_data();
  const float *utput_pt_seg_pred_value = output_pt_seg_pred->cpu_data();
  const float *output_lm_seg_pred_value = output_lm_seg_pred->cpu_data();

  float *device_pt_class_prob_value = output_pt_class_prob->mutable_gpu_data();

  ArgmaBilinearResizeGPU(device_pt_class_prob_value,1,130,80,4,device_in_seg_,1040,640);
  cudaMemcpy(host_in_seg_,device_in_seg_,sizeof(uint8_t)*1040*640,cudaMemcpyDeviceToHost);


  for (size_t i = 0; i < 10; i++){
    std::cout << output_lm_seg_pred_value[i] << "  ";
  }
  std::cout << std::endl;
  


  //进行后处理
  // binarize_seg(input_seg, height, width, findcc_input);



                                




  /* Copy the output layer to a std::vector */

  //Blob<float>* output_layer = net_->output_blobs()[0];
  // const float* begin = output_layer->cpu_data();
  // const float* end = begin + output_layer->channels();
  // return std::vector<float>(begin, end);
  
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  //cv::Mat sample_normalized;
  //cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

void Classifier::InitDeviceAndHostMemory(){
  cudaMalloc(&device_in_seg_,sizeof(uint8_t*)*1040*640);

  host_in_seg_ = (uint8_t *)malloc(sizeof(uint8_t)*1040*640);
}