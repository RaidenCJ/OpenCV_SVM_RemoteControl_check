#include "cfactorysvm.h"
#include "remoteControl.h"
#include "string.h"
#include <time.h>
#include "svmRemoteControl.h"


int main( int argc, char** argv )
{
  CFactorySvm *mySvm = new CFactorySvm();
  string dataTrain = "/home/zhgg/share/workspace/svm/total_svm_data_train.txt";
  string dataMode = "/home/zhgg/share/workspace/svm/total_svm_data.xml";
  string testPath = "/home/zhgg/share/workspace/svm/total_svm_data_test.txt";
  string predict = "/home/zhgg/share/workspace/svm/SVM_PREDICT.txt";
  cv::Mat src;
  //cv::Mat src = imread("/home/zhgg/workspace/rc_check/factory/test/no_remote/crop-36144.jpg", 1);

  if (0 == strncmp("train", argv[1], strlen("train")))
  {
    mySvm->SvmTrain(dataTrain, dataMode);
  }
  else if (0 == strncmp("test", argv[1], strlen("test")))
  {
    mySvm->SvmDetect(dataMode, testPath, predict);
  }
  else if (0 == strncmp("remote", argv[1], strlen("remote")))
  { 
    if (mySvm->HasRemoteControl(src))
    {
        cout << "has remote!" << endl;
    }
    else
    {
        cout << "no remote!" << endl;
    }
  }
  else
  {
    //struct timespec start, end;
    //clock_gettime(CLOCK_REALTIME, &start); 
    //cout << "111111111111111111" << endl;
    src = imread(argv[1]);
    
    if (svmDetectRemote(src, MODEL_65X))
    {
        cout << "has remote!" << endl;
    }
    else
    {
        cout << "no remote!" << endl;
    }
    //clock_gettime(CLOCK_REALTIME, &end);
    //double cost = ((end.tv_sec - start.tv_sec) * 1000 * 1000 *1000 + (end.tv_nsec - start.tv_nsec))/(1000 *1000);
    //cout << "execute time is "<< cost << endl;
    //cout << "usage: ./demo train or ./demo test or ./demo remote" << endl;
    cvNamedWindow("rc");
    imshow("rc", src);
    waitKey(0);
  }

  if (mySvm)
  {
    delete mySvm;
  }
  return(0);
}








