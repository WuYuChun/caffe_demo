



#include<iostream>
#include<fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/***
 * mnist文件的格式
 *  32bits int |  32bits int | 32bits int | 32bits int | uint8 pixse
 *  魔法数字    |   图像个数    | 高度28     | 宽度28
 *   
 ***/

//把大端数据转换为我们常用的小端数据  
uint32_t swap_endian(uint32_t val){
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}


int main(int argc ,char *argv[]){

    std::string mnist_images_file(argv[1]);
    std::ifstream mnist_images(mnist_images_file.c_str(),std::ios::in | std::ios::binary);
    if(!mnist_images.is_open()){
        std::cout << "cannot open file\n";
        return -1;
    }

    uint32_t magic;
    uint32_t num_items;
    uint32_t rows;
    uint32_t cols;

    //!读取魔数
    mnist_images.read(reinterpret_cast<char*>(&magic),4);
    mnist_images.read(reinterpret_cast<char*>(&num_items), 4);
    mnist_images.read(reinterpret_cast<char*>(&rows), 4);
	rows = swap_endian(rows);
	mnist_images.read(reinterpret_cast<char*>(&cols), 4);
	cols = swap_endian(cols);

    for (size_t i = 0; i < 2; i++){
        cv::Mat image(rows,cols,CV_8UC1,cv::Scalar(0));
        mnist_images.read(reinterpret_cast<char*>(image.data),rows*cols);
        std::string image_name = "mnist_image_" + std::to_string(i) + ".jpg";
        cv::imwrite(image_name,image);
    }

    return 0;
}