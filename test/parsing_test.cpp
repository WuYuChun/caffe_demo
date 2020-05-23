#include "vgg16.h"

#include <iostream>

 std::vector<std::vector<int> > color_tab = { {0, 0, 0},
                                              {205, 92, 92},
                                              {250, 128, 128},
                                              {255, 160, 122},
                                              {255, 0, 0},
                                              {178, 34, 34},
                                              {139, 0, 0},
                                              {255, 192, 203},
                                              {255, 20, 147},
                                              {199, 21, 133},
                                              {255, 140, 0},
                                              {255, 215, 0},
                                              {255, 255, 0},
                                              {216, 191, 216},
                                              {138, 43, 226},
                                              {148, 0, 211},
                                              {139, 0, 139},
                                              {75, 0, 130},
                                              {72, 61, 139},
                                              {173, 255, 0},
                                              {0, 255, 0},
                                              {50, 205, 50},
                                              {0, 100, 0},
                                              {154, 205, 50},
                                              {128, 128, 0},
                                              {85, 107, 47},
                                              {0, 128, 128},
                                              {0, 255, 255},
                                              {175, 238, 238},
                                              {127, 255, 212},
                                              {70, 130, 180},
                                              {30, 144, 255}
 };


void applyColor(cv::Mat &ori_img, cv::Mat &color_img, std::vector<std::vector<int> > &color_map){
     int width = ori_img.cols;
     int height = ori_img.rows;
     for(int x = 0; x < width; ++x){
         for(int y = 0; y < height; ++y){
             uint8_t value = ori_img.at<uint8_t>(cv::Point(x,y));
             std::vector<int> color_value = color_map[value];
             color_img.at<cv::Vec3b>(cv::Point(x,y)) = cv::Vec3b(color_value[2],color_value[1],color_value[0]);
         }
     }
}


int main(int argc, char** argv) {

  // if (argc != 5) {
  //   std::cerr << "Usage: " << argv[0]
  //             << " deploy.prototxt network.caffemodel"
  //             << " labels.txt img.jpg" << std::endl;
  //   return 1;
  // }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  // string label_file   = argv[3];
  Classifier classifier(model_file, trained_file);

  // string file = argv[4];

  // std::cout << "---------- Prediction for "<< file << " ----------" << std::endl;

  //cv::Mat img = cv::imread(file, -1);
  //CHECK(!img.empty()) << "Unable to decode image " << file;

 float *input_data_bin = new float[1*3*720*1280];
   
  std::ifstream inF;
  inF.open(argv[3], std::ifstream::binary);
  inF.read(reinterpret_cast<char*>(input_data_bin), sizeof(float)*1*3*720*1280);
  inF.close();

  std::cout << "input data: " << std::endl;
  for (size_t i = 0; i < 12; i++){
    std::cout << input_data_bin[i] << std::endl;
  }
  

  std::vector<float> layer_output = classifier.Classify(input_data_bin);

  cv::Mat input_img(720, 1280, CV_32FC1, layer_output.data());

  cv::Mat conver_img;
  input_img.convertTo(conver_img,CV_8UC1);
         
  cv::Mat corlor_img = cv::Mat::zeros(720, 1280, CV_8UC3);
  applyColor(conver_img, corlor_img, color_tab);
  std::string img_save_file_name = "result.jpg";
  cv::imwrite(img_save_file_name, corlor_img);


  if(input_data_bin){
    delete []input_data_bin;
    input_data_bin = NULL;
  }

}