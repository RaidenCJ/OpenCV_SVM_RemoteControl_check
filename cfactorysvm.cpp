#include "cfactorysvm.h"
#include "opencv/cv.h"


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>


#include <fstream>
#include <vector>
#include <time.h>




CFactorySvm::CFactorySvm()
{
    mImgWidht = 64;
    mImgHeight = 64;
    LoadMode();
}

int CFactorySvm::GetMaxindex(vector<vector<Point> > &contours)
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

void CFactorySvm::GetPic(Mat &src, Mat &roiImg)
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

void CFactorySvm::SvmTrain(string &svmDataFile, string &outputFile)
{
    vector<string> img_path;
    vector<int> img_catg;

    int nLine = 0;
    unsigned long n;

    string tempstr;
    int tempi=-2;
    char strPath[128] = {0};
    string buf;

    ifstream svm_data(svmDataFile.c_str());
    while(svm_data)
    {
        if( getline( svm_data, buf ) )
        {
            nLine ++;
            tempstr = buf[buf.size()-1];
            tempi=atoi(tempstr.c_str());
            memset(strPath,0x00,sizeof(strPath));
            strncpy(strPath,buf.c_str(),buf.size() - 2);
            strPath[strlen(strPath)] = '\0';
            img_path.push_back( strPath );//图像路径
            img_catg.push_back(tempi);
        }
    }
    svm_data.close();

    Mat data_mat,label_mat;
    int nImgNum = nLine;            //读入样本数量

   //检测窗口,块尺寸,块步长,cell尺寸,直方图bin个数9

    cv::HOGDescriptor* hog = GetHog();

    Mat trainImg = Mat::zeros(mImgWidht, mImgHeight , CV_8UC3);//需要分析的图片
    for( string::size_type i = 0; i != img_path.size(); i++ )
    {
        cout<<"processing "<<img_path[i].c_str()<<endl;

        cv::Mat src = imread(img_path[i].c_str());
        if(src.empty()) {
            cout<<"image empty"<<endl;
            continue;
        }
        // 将图片二值化，过滤边缘地方
        // Mat roiImg;
        // GetPic(src, roiImg);

        cv::resize(src, trainImg, cv::Size(mImgWidht, mImgHeight), 0, 0, INTER_CUBIC);
        //HOG描述子向量
        std::vector<float> descriptors;
        //计算HOG描述子，检测窗口移动步长(8,8)
        hog->compute(trainImg, descriptors, cv::Size(mImgWidht / 8, mImgHeight / 8));

        //cout<<"HOG dims: "<<descriptors.size()<<endl;
        cout << "nImgNum is " << nImgNum << std::endl;
        cout << "descriptors.size() is " << descriptors.size() << std::endl;
        if (i==0)
        {
            data_mat = Mat::zeros( nImgNum, descriptors.size(), CV_32FC1 ); //根据输入图片大小进行分配空间
            label_mat = Mat::zeros( nImgNum, 1, CV_32SC1 );
        }
        n=0;
        for(vector<float>::iterator iter=descriptors.begin(); iter!=descriptors.end(); iter++)
        {
            data_mat.at<float>(i,n) = *iter;
            n++;
        }
        label_mat.at<int>(i, 0) =  img_catg[i];
        //cout<<" end processing "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;
    }
    std::cout << "processing finished" << std::endl;


    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));

    svm->train(data_mat, ROW_SAMPLE,  label_mat);
    std::cout << "training finished" << std::endl;
    svm->save(outputFile);
    std::cout << "save finished" << std::endl;
}

bool CFactorySvm::SvmDetect(string &svmMode, string &testPath, string &predict)
{
    //检测样本
    Ptr<SVM> svm = SVM::load(svmMode.c_str());
    vector<string> img_tst_path;
    vector<int> img_catg;
    //ifstream img_tst(imgPath.c_str());
    //string testPath = "/home/zhgg/workspace/rc_check/factory/svm_data_test.txt";
    //string testPath = "/home/zhgg/workspace/rc_check/factory/svm_data_train.txt";
    ifstream img_tst(testPath.c_str());
    char strTestImg[128];
    string buf;
    Mat trainImg = Mat::zeros(mImgWidht, mImgHeight, CV_8UC3);//需要分析的图片
    //检测窗口,块尺寸,块步长,cell尺寸,直方图bin个数9
    HOGDescriptor* hog = GetHog();
    string tempstr;
    int tempi=-2;
    while( img_tst )
    {
        if( getline( img_tst, buf ) )
        {
            tempstr = buf[buf.size()-1];
            tempi=atoi(tempstr.c_str());
            memset(strTestImg,0x00,sizeof(strTestImg));
            strncpy(strTestImg,buf.c_str(),buf.size() - 2);
            img_tst_path.push_back( strTestImg );
            img_catg.push_back(tempi);
        }
    }
    img_tst.close();

    char line[512];
    int n = 0;
    //string predict = "/home/zhgg/workspace/rc_check/factory/SVM_PREDICT.txt";
    ofstream predict_txt(predict.c_str());
    for( string::size_type j = 0; j != img_tst_path.size(); j++ )
    {
        cout<<"processing "<<img_tst_path[j].c_str()<<endl;
        cv::Mat test = imread(img_tst_path[j].c_str());//读入图像
        if(test.empty()) {
            cout<<"image empty"<<endl;
            continue;
        }
        // 将图片二值化，过滤边缘地方
        //Mat roiImg;
        //GetPic(test, roiImg);
        resize(test, trainImg, cv::Size(mImgWidht, mImgHeight), 0, 0, INTER_CUBIC);//要搞成同样的大小才可以检测到
        std::vector<float> descriptors;
        hog->compute(trainImg, descriptors, cv::Size(mImgWidht / 8, mImgHeight / 8));
        Mat SVMtrainMat =  Mat::zeros(1,descriptors.size(),CV_32FC1);

        n=0;
        for(vector<float>::iterator iter=descriptors.begin(); iter!=descriptors.end(); iter++)
        {
            SVMtrainMat.at<float>(0,n) = *iter;
            n++;
        }

        int ret = svm->predict(SVMtrainMat);

        std::sprintf( line, "%s %d %d\r\n", img_tst_path[j].c_str(), img_catg[j], ret );
        printf("%s %d\r\n", img_tst_path[j].c_str(), ret);
        predict_txt<<line;
    }
    predict_txt.close();

    return true;
}

HOGDescriptor* CFactorySvm::GetHog()
{
    HOGDescriptor *hog = new HOGDescriptor(cv::Size(mImgWidht, mImgHeight), cv::Size((mImgWidht/2), (mImgHeight/2)), cv::Size((mImgWidht/8), (mImgHeight/8)), cv::Size((mImgWidht/8), (mImgHeight/8)), 9);
    return hog;
}

bool CFactorySvm::HasRemoteControl(Mat &src)
{
    bool bFlag = false;

  //检测样本
    if (NULL == svm)
    {
        LoadMode();
    }
    Mat roiImg;
    GetPic(src, roiImg);
imshow("roiImg", roiImg);
waitKey();
    Mat testImg = Mat::zeros(mImgWidht, mImgHeight, CV_8UC3);//需要分析的图片
    //检测窗口,块尺寸,块步长,cell尺寸,直方图bin个数9
//cout << mImgWidht/4 << "___" << mImgHeight/4 << endl;
    HOGDescriptor* hog = GetHog();
    
    if(!roiImg.empty())
    {
        resize(roiImg, testImg, cv::Size(mImgWidht, mImgHeight), 0, 0, INTER_CUBIC);//要搞成同样的大小才可以检测到
        std::vector<float> descriptors;
        hog->compute(testImg, descriptors, cv::Size(8, 16));
        Mat SVMtrainMat =  Mat::zeros(1,descriptors.size(),CV_32FC1);

        int n=0;
        for(vector<float>::iterator iter=descriptors.begin(); iter!=descriptors.end(); iter++)
        {
            SVMtrainMat.at<float>(0,n) = *iter;
            n++;
        }

        int ret = svm->predict(SVMtrainMat);

        bFlag = (1 == ret);
    }
    
    return bFlag;
}

void CFactorySvm::LoadMode()
{
    string svmMode = "./svm_data.xml";
    fstream _file;
    _file.open(svmMode.c_str(),ios::in);
    if(_file)
    {
        svm = SVM::load(svmMode.c_str());
    }
}
