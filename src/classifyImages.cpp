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
				this->trainFolder = static_cast<std::string>(argv[i]);
			}else if(i==5){
				this->annotationsTrain = static_cast<std::string>(argv[i]);
				argv[i]="";
			}else if(i==6){
				this->testFolder = static_cast<std::string>(argv[i]);
				argv[i]="";
			}else if(i==7){
				this->annotationsTest = static_cast<std::string>(argv[i]);
				argv[i]="";
			}
		}
		argc -= 3;
		this->features = new featureDetector(argc,argv,false);
	}
	this->feature=featureDetector::EDGES;
}
//==============================================================================
classifyImages::~classifyImages(){
	this->trainData.release();
	this->testData.release();
	this->trainTargets.release();
	this->testTargets.release();
}

//==============================================================================
/** Initialize the options for the Gaussian Process regression.
 */
void classifyImages::init(double theNoise, double theLength,\
gaussianProcess::kernelFunction theKFunction, featureDetector::FEATURE theFeature){
	this->noise     = theNoise;
	this->length    = theLength;
	this->kFunction = theKFunction;
	this->feature   = theFeature;
}
//==============================================================================
/** Creates the training data (according to the options), the labels and
 * trains the a \c GaussianProcess on the data.
 */
void classifyImages::trainGP(){
	this->features->setFeatureType(this->feature);
	this->features->init(this->trainFolder, this->annotationsTrain);
	this->features->run();
	this->trainData = cv::Mat::zeros(cv::Size(this->features->data[0].cols,\
						this->features->data.size()),cv::DataType<double>::type);
	this->trainTargets = cv::Mat::zeros(cv::Size(this->features->targets[0].cols,\
							this->features->targets.size()),cv::DataType<double>::type);

	std::cout<<"SIZE: "<<this->trainData.cols<<" "<<this->trainData.rows<<std::endl;
	std::cout<<"SIZE: "<<this->trainTargets.cols<<" "<<this->trainTargets.rows<<std::endl;

	// CONVERT FROM VECTOR OF CV::MAT TO MAT
	for(std::size_t i=0; i<this->features->data.size(); i++){
		cv::Mat dummy1 = this->trainData.row(i);
		cv::Mat(this->features->data[i]).copyTo(dummy1);

		cv::Mat dummy2 = this->trainTargets.row(i);
		cv::Mat(this->features->targets[i]).copyTo(dummy2);
	}

	normalizeMat(this->trainData);
	this->trainData.convertTo(this->trainData, cv::DataType<double>::type);
	this->trainTargets.convertTo(this->trainTargets, cv::DataType<double>::type);

	// TRAIN THE SIN AND COS SEPARETELY
	this->gpSin.train(this->trainData,this->trainTargets.col(0),\
		this->kFunction, this->noise, this->length);
	this->gpCos.train(this->trainData,this->trainTargets.col(1),\
		this->kFunction, this->noise, this->length);
}
//==============================================================================
/** Creates the test data and applies \c GaussianProcess prediction on the test
 * data.
 */
void classifyImages::predictGP(){
	this->features->setFeatureType(this->feature);
	this->features->init(this->testFolder, this->annotationsTest);
	this->features->run();
	this->testData = cv::Mat::zeros(cv::Size(this->features->data[0].cols,\
					this->features->data.size()), cv::DataType<double>::type);
	this->testTargets = cv::Mat::zeros(cv::Size(this->features->targets[0].cols,\
							this->features->targets.size()),cv::DataType<double>::type);

	// CONVERT FROM VECTOR OF CV::MAT TO MAT
	for(std::size_t i=0; i<this->features->data.size(); i++){
		cv::Mat dummy1 = this->testData.row(i);
		this->features->data[i].copyTo(dummy1);

		cv::Mat dummy2 = this->testTargets.row(i);
		cv::Mat(this->features->targets[i]).copyTo(dummy2);
	}

	normalizeMat(this->testData);
	this->testData.convertTo(this->testData, cv::DataType<double>::type);
	this->testTargets.convertTo(this->testTargets, cv::DataType<double>::type);

	// FOR EACH ROW IN THE TEST MATRIX PREDICT
	for(unsigned i=0; i<this->testData.rows; i++){
		gaussianProcess::prediction prediSin;
		this->gpSin.predict(this->testData.row(i), prediSin, this->length);
		std::cout<<"SIN label: "<<this->testTargets.at<double>(i,0)<<\
			" mean:"<<prediSin.mean[0]<<\
			" variance:"<<prediSin.variance[0]<<std::endl;
		prediSin.mean.clear();
		prediSin.variance.clear();

		gaussianProcess::prediction prediCos;
		this->gpCos.predict(this->testData.row(i), prediCos, this->length);
		std::cout<<"COS label: "<<this->testTargets.at<double>(i,1)<<","<<\
			" mean:"<<prediCos.mean[0]<<\
			" variance:"<<prediCos.variance[0]<<std::endl;
		prediCos.mean.clear();
		prediCos.variance.clear();
	}
}
//==============================================================================
int main(int argc, char **argv){
	classifyImages classi(argc, argv);
	classi.init(1e-3,100.0,&gaussianProcess::sqexp,featureDetector::BLOB);
	classi.trainGP();
	classi.predictGP();
}


