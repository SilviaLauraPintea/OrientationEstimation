/* classifyImages.cpp
 * Author: Silvia-Laura Pintea
 */
#include "classifyImages.h"
//==============================================================================
classifyImages::classifyImages(int argc, char **argv){
	if(argc != 8){
		cerr<<"Usage: classifier <bgTrain|bgModel> <calib> <prior> "<<\
			"<trainFolder> <annotationsTrain> <testFolder> <annotationsTest>"<<endl;
		exit(1);
	}else{
		for(unsigned i=0; i<argc; i++){
			if(i==4){
				this->trainFolder = (std::string)argv[i];
			}else if(i==5){
				this->annotationsTrain = (std::string)argv[i];
				argv[i] = "";
			}else if(i==6){
				this->testFolder = (std::string)argv[i];
				argv[i] = "";
			}else if(i==7){
				this->annotationsTest = (std::string)argv[i];
				argv[i] = "";
			}
		}
		argc -= 3;
		this->features = new featureDetector(argc,argv,false);
	}
}
//==============================================================================
classifyImages::~classifyImages(){
	this->trainData.release();
	this->testData.release();
	this->trainTargets.release();
	this->testTargets.release();
}
//==============================================================================
/** Creates the training data (according to the options), the labels and
 * trains the a \c GaussianProcess on the data.
 */
void classifyImages::trainGP(std::vector<std::string> options){
	this->features->init(this->trainFolder);
	this->features->run();

	this->trainData = cv::Mat::zeros(cv::Size(this->features->data[0].cols,\
						this->features->data.size()),cv::DataType<double>::type);
	for(std::size_t i=0; i<this->features->data.size(); i++){
		cv::Mat dummy = this->trainData.row(i);
		this->features->data[i].convertTo(this->features->data[i],cv::DataType<double>::type);
		cv::Mat(this->features->data[i]).copyTo(dummy);

		cv::imshow("kkt",this->trainData.row(i).reshape(0,100));
		cv::waitKey(0);

	}
	this->trainTargets = cv::Mat::zeros(cv::Size(2, this->trainData.rows),\
							this->trainData.type());

	std::vector<annotationsHandle::FULL_ANNOTATIONS> targetAnno;
	annotationsHandle::loadAnnotations(const_cast<char*>\
		(this->annotationsTrain.c_str()), targetAnno);
	for(std::size_t i=0; i<targetAnno.size(); i++){
		for(std::size_t j=0; j<targetAnno[i].annos.size(); j++){
			unsigned aSize = i*targetAnno[i].annos.size()+j;
			double angle = static_cast<double>\
				(targetAnno[i].annos[j].poses[annotationsHandle::ORIENTATION]);
			this->trainTargets.at<double>(aSize,0) = std::sin(angle*M_PI/180.0);
			this->trainTargets.at<double>(aSize,1) = std::cos(angle*M_PI/180.0);
		}
	}
	this->trainData.convertTo(this->trainData, cv::DataType<double>::type);
	this->trainTargets.convertTo(this->trainTargets, cv::DataType<double>::type);
	this->gpCos.train(this->trainData,this->trainTargets.col(0),\
		&gaussianProcess::expCovar, 0.1);
	this->gpSin.train(this->trainData,this->trainTargets.col(1),\
		&gaussianProcess::expCovar, 0.1);
}
//==============================================================================
/** Creates the test data and applies \c GaussianProcess prediction on the test
 * data.
 */
void classifyImages::predictGP(std::vector<std::string> options){
	this->features->init(this->testFolder);
	this->features->run();
	this->testData = cv::Mat::zeros(cv::Size(this->features->data[0].cols,\
					this->features->data.size()), cv::DataType<double>::type);
	for(std::size_t i=0; i<this->features->data.size(); i++){
		cv::Mat dummy = this->testData.row(i);
		this->features->data[i].convertTo(this->features->data[i],cv::DataType<double>::type);
		this->features->data[i].copyTo(dummy);

		cv::imshow("kkkkkkkt",this->testData.row(i).reshape(0,100));
		cv::waitKey(0);

	}

	std::vector<annotationsHandle::FULL_ANNOTATIONS> targetAnno;
	this->testTargets = cv::Mat::zeros(cv::Size(2, this->testData.rows),\
						this->testData.type());

	annotationsHandle::loadAnnotations(const_cast<char*>\
		(this->annotationsTest.c_str()),targetAnno);
	for(std::size_t i=0; i<targetAnno.size(); i++){
		for(std::size_t j=0; j<targetAnno[i].annos.size(); j++){
			unsigned aSize = i*targetAnno[i].annos.size()+j;
			double angle = static_cast<double>\
				(targetAnno[i].annos[j].poses[annotationsHandle::ORIENTATION]);
			this->testTargets.at<double>(aSize,0) = std::sin(angle*M_PI/180.0);
			this->testTargets.at<double>(aSize,1) = std::cos(angle*M_PI/180.0);
		}
	}
	this->testData.convertTo(this->testData, cv::DataType<double>::type);
	this->testTargets.convertTo(this->testTargets, cv::DataType<double>::type);

	for(unsigned i=0; i<this->testData.rows; i++){
		gaussianProcess::prediction prediCos;
		this->gpCos.predict(this->testData.row(i), prediCos);
		std::cout<<"label: ("<<this->testTargets.at<double>(i,0)<<","<<\
			") mean:("<<prediCos.mean[0]<<","<<prediCos.mean[1]<<\
			") variance:"<<prediCos.variance[0]<<std::endl;
		prediCos.mean.clear();
		prediCos.variance.clear();

		gaussianProcess::prediction prediSin;
		this->gpSin.predict(this->testData.row(i), prediSin);
		std::cout<<"label: ("<<this->testTargets.at<double>(i,1)<<\
			") mean:("<<prediSin.mean[0]<<","<<prediSin.mean[1]<<\
			") variance:"<<prediSin.variance[0]<<std::endl;
		prediSin.mean.clear();
		prediSin.variance.clear();
	}
}
//==============================================================================
int main(int argc, char **argv){
	classifyImages classi(argc, argv);
	classi.trainGP();
	classi.predictGP();
}

