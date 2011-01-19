#include "classifyImages.h"
#include <boost/thread.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION < 103500
	#include <boost/thread/detail/lock.hpp>
#endif
#include <boost/thread/xtime.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <exception>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include "eigenbackground/src/Tracker.hh"
#include "eigenbackground/src/Helpers.hh"
//==============================================================================
classifyImages::classifyImages() {
}
//==============================================================================
classifyImages::~classifyImages() {
}
//==============================================================================
/** Regression SVM classification.
 */
void classifyImages::classifySVM(cv::Mat* trainData, cv::Mat* sample){
	cv::Mat* responses;
	CvSVMParams params = CvSVMParams();
	params.svm_type    = CvSVM::EPS_SVR; //or params.svm_type=CvSVM::NU_SVR
	params.kernel_type = 3;
	params.C           = 0;
	params.gamma       = 1;
	params.coef0       = 4;
	params.p           = 2;

	CvSVM svm;
	svm.train(trainData, responses, 0, 0, params);
	cout<<svm.predict(sample);
}

