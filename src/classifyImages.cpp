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
#include "featureDetector.h"
//==============================================================================
classifyImages::classifyImages(int argc, char **argv){
	cerr<<">>>>>>>>>>>>>>>>"<<argc<<endl;
	if(argc != 6){
		cerr<<"Usage: classifier <trainFolder> <testFolder> <bgTrain|bgModel> "<<\
			"<calib> <prior>"<<endl;
		exit(1);
	}else{
		char** argvTest = argv;
		for(unsigned i=0; i<argc; i++){
			cout<<(std::string)argv[i]<<endl;

			if(i==1){
				this->trainFolder = (std::string)argv[i];
				argvTest[i] = "";
			}else if(i==2){
				this->testFolder = (std::string)argv[i];
				argv[i] = "";
			}
		}
		argc--;

		for(unsigned i=0; i<argc; i++){
			cout<<(std::string)argv[i]<<endl;
			cout<<(std::string)argvTest[i]<<endl;
		}

		this->testFeatures = new featureDetector(argc,argv,false);
		this->trainFeatures = new featureDetector(argc,argvTest,false);
	}
}
//==============================================================================
classifyImages::~classifyImages(){
	this->trainData.release();
	this->testData.release();
}
//==============================================================================
/** Creates the training data/test data.
 */
void classifyImages::createData(std::vector<std::string> options){
	if(options[0] == "test"){
		this->testFeatures->run();
	}else{
		this->trainFeatures->run();
	}
}
//==============================================================================
/** Regression SVM classification.
 */
void classifyImages::classifySVM(){
	cv::Mat responses, var_idx, sample_idx;
	CvSVMParams params = CvSVMParams();
	params.svm_type    = CvSVM::EPS_SVR; //or params.svm_type=CvSVM::NU_SVR
	params.kernel_type = 3;
	params.C           = 0;
	params.gamma       = 1;
	params.coef0       = 4;
	params.p           = 2;

	CvSVM svm;
	this->createData(std::vector<std::string>());

	svm.train(this->trainData, responses, var_idx, sample_idx, params);
	cout<<svm.predict(this->testData, true);
}
//==============================================================================
/*int main(int argc, char **argv){
	classifyImages classfier(int argc, char **argv);
	//classifier.classifySVM();
}*/
