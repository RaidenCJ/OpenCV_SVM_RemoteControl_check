#include "svmRemoteControl.h"
#include "opencv/cv.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <vector>

#define SVM_MODE_55 "/home/zhgg/share/workspace/svm/55_svm_data.xml"
#define SVM_MODE_65 "/home/zhgg/share/workspace/svm/65_svm_data.xml"
#define SVM_MODE_TOTAL "/home/zhgg/share/workspace/svm/total_svm_data.xml"
#define RESIZE_WIDTH 64
#define RESIZE_HEIGHT 64
#define LIMIT_NUMBER 10


int GetMaxindex(vector<vector<Point> > &contours)
{
    double max_area=0.0;
    double area = 0.0;
    vector<Point> poly;
    int max_area_index = 0;
    for( size_t i = 0; i < contours.size(); i++ ){
        approxPolyDP( Mat(contours[i]), poly, 3, true );
        area = fabs(contourArea(poly,true));
        if(area > max_area){
            max_area = area;
            max_area_index = i;
        }
    }
    return max_area_index;
}

void GetPic(Mat &src, Mat &roiImg)
{
   
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int max_area_index = 0;

    Mat gray_img;
    Mat threshold_output;
    cvtColor( src, gray_img, COLOR_BGR2GRAY );
    threshold( gray_img, threshold_output, 150, 255, THRESH_BINARY );
    findContours( threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    Mat mask = Mat::zeros(threshold_output.size(),CV_8UC1);
    max_area_index = GetMaxindex(contours);
    drawContours(mask,contours,max_area_index,Scalar::all(255),-1);
    src.copyTo(roiImg,mask);
}

HOGDescriptor* GetHog()
{
    HOGDescriptor *hog = new HOGDescriptor(cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT), cv::Size((RESIZE_WIDTH/2), (RESIZE_HEIGHT/2)), cv::Size((RESIZE_WIDTH/8), (RESIZE_HEIGHT/8)), cv::Size((RESIZE_WIDTH/8), (RESIZE_HEIGHT/8)), 9);
    return hog;
}

bool svmDetectRemote(Mat &img, Model model)
{
    bool bFlag = false;
    static int true_number = 0;
    static int false_number = 0;
    //检测样本
    Ptr<SVM> svm;
    svm = SVM::load(SVM_MODE_TOTAL);
    #if 0
    if (MODEL_55X == model)
    {
        svm = SVM::load(SVM_MODE_55);
    }
    else if (MODEL_65X == model)
    {
        svm = SVM::load(SVM_MODE_65);
    }
    else
    {
        return bFlag;
    }
    #endif
    
    Mat resizeImg = Mat::zeros(RESIZE_WIDTH, RESIZE_HEIGHT, CV_8UC3);//需要分析的图片
    //检测窗口,块尺寸,块步长,cell尺寸,直方图bin个数9
    HOGDescriptor* hog = GetHog();
    
    // 将图片二值化，过滤边缘地方
    //Mat roiImg;
    //GetPic(test, roiImg);
    resize(img, resizeImg, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT), 0, 0, INTER_CUBIC);//要搞成同样的大小才可以检测到
    std::vector<float> descriptors;
    hog->compute(resizeImg, descriptors, cv::Size(RESIZE_WIDTH / 8, RESIZE_HEIGHT / 8));
    Mat SVMtrainMat =  Mat::zeros(1,descriptors.size(),CV_32FC1);

    int n = 0;
    for(vector<float>::iterator iter=descriptors.begin(); iter != descriptors.end(); iter++)
    {
        SVMtrainMat.at<float>(0,n) = *iter;
        n++;
    }

    int ret = svm->predict(SVMtrainMat);
    
    if (1 == ret)
    {
        if (true_number < LIMIT_NUMBER)
        {
            true_number++;
        }
        if (false_number < LIMIT_NUMBER || true_number >= LIMIT_NUMBER)
        {
            false_number = 0;
        }
        return true;
    }
    else
    {
        if (false_number < LIMIT_NUMBER)
        {
            false_number++;
        }
        if (true_number < LIMIT_NUMBER || false_number >= LIMIT_NUMBER)
        {
            true_number = 0;
        }
    }
    
    if (true_number >= LIMIT_NUMBER)
    {
        bFlag = true;
    }
    
    return bFlag;
}

