#ifndef SVM_REMOTE_CONTROL_H
#define SVM_REMOTE_CONTROL_H

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

typedef enum _Model{
    MODEL_32W,
    MODEL_49X,
    MODEL_55X,
    MODEL_65X
} Model;

bool svmDetectRemote(Mat &img, Model model);

#endif // SVM_REMOTE_CONTROL_H
