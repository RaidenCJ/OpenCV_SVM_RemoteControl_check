#ifndef CFACTORYSVM_H
#define CFACTORYSVM_H

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

class CFactorySvm
{
public:
    CFactorySvm();
    void SvmTrain(string &svmDataFile, string &outputFile);
    bool SvmDetect(string &svmMode, string &testPath, string &predict);
    bool HasRemoteControl(Mat &src);

private:
    int mImgWidht;
    int mImgHeight;
    Ptr<SVM> svm;
    void LoadMode();
    int GetMaxindex(vector<vector<Point> > &contours);
    void GetPic(Mat &src, Mat &roiImg);
    HOGDescriptor* GetHog();
};

#endif // CFACTORYSVM_H
