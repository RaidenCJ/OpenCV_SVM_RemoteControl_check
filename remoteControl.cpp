#include "remoteControl.h"
#include "cfactorysvm.h"
#include <time.h>

bool HasRemoteControl(Mat src)
{
  struct timespec start, end;
  double cost;
  bool bFlag = false;
  

  CFactorySvm *mySvm = new CFactorySvm();
  clock_gettime(CLOCK_REALTIME, &start);  
  if (NULL != mySvm)
  {
    bFlag = mySvm->HasRemoteControl(src);
    delete mySvm;
  }
  clock_gettime(CLOCK_REALTIME, &end);
  cost = ((end.tv_sec - start.tv_sec) * 1000 * 1000 *1000 + (end.tv_nsec - start.tv_nsec))/(1000 *1000);
  cout << "execute time is "<< cost << endl;
  return bFlag;
}
