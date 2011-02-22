/* classifyImages.cpp
 * Author: Silvia-Laura Pintea
 */
#include "classifyImages.h"

//==============================================================================
classifyImages::classifyImages(int argc, char **argv){
	// DEAFAULT INITIALIZATION
	this->trainFolder      = "";
	this->testFolder       = "";
	this->annotationsTrain = "";
	this->annotationsTest  = "";
	this->noise            = 0.01;
	this->length           = 1.0;
	this->kFunction        = &gaussianProcess::sqexp;
	this->feature          = featureDetector::EDGES;
	this->features         = NULL;

	// READ THE COMMAND LINE ARGUMENTS
	if(argc != 8 && argc!=5){
		cerr<<"Usage: classifier <bgTrain|bgModel> <calib> <prior> "<<\
			"<trainFolder> <annotationsTrain> <testFolder> <annotationsTest>"<<endl;
		exit(1);
	}else if(argc==8){
		for(unsigned i=0; i<argc; i++){
			if(i==4){
				this->trainFolder = std::string(argv[i]);
			}else if(i==5){
				this->annotationsTrain = std::string(argv[i]);
				argv[i]="";
			}else if(i==6){
				this->testFolder = std::string(argv[i]);
				argv[i]="";
			}else if(i==7){
				this->annotationsTest = std::string(argv[i]);
				argv[i]="";
			}
		}
		argc -= 3;
		this->features = new featureDetector(argc,argv,false);
	}else if(argc==5){
		for(unsigned i=0; i<argc; i++){
			if(i==4){
				this->trainFolder = std::string(argv[i]);
			}
			this->annotationsTrain = "";
			this->testFolder       = "";
			this->annotationsTest  = "";
		}
		this->features = new featureDetector(argc,argv,false);
	}
}
//==============================================================================
classifyImages::~classifyImages(){
	if(this->features){
		delete this->features;
		this->features = NULL;
	}

	if(!this->trainData.empty()){
		this->trainData.release();
	}
	if(!this->testData.empty()){
		this->testData.release();
	}
	if(!this->trainTargets.empty()){
		this->trainTargets.release();
	}
	if(!this->testTargets.empty()){
		this->testTargets.release();
	}
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
		this->features->data[i].copyTo(dummy1);

		cv::Mat dummy2 = this->trainTargets.row(i);
		this->features->targets[i].copyTo(dummy2);
	}
	this->trainData.convertTo(this->trainData, cv::DataType<double>::type);
	this->trainTargets.convertTo(this->trainTargets, cv::DataType<double>::type);
	range1Mat(this->trainData);

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
void classifyImages::predictGP(cv::Mat &predictions){
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
		this->features->targets[i].copyTo(dummy2);
	}
	this->testData.convertTo(this->testData, cv::DataType<double>::type);
	this->testTargets.convertTo(this->testTargets, cv::DataType<double>::type);
	range1Mat(this->testData);

	// FOR EACH ROW IN THE TEST MATRIX PREDICT
	predictions = cv::Mat::zeros(this->testTargets.size(),\
					cv::DataType<double>::type);
	for(unsigned i=0; i<this->testData.rows; i++){
		gaussianProcess::prediction prediSin;
		this->gpSin.predict(this->testData.row(i), prediSin, this->length);
		gaussianProcess::prediction prediCos;
		this->gpCos.predict(this->testData.row(i), prediCos, this->length);
		predictions.at<double>(i,0) = prediSin.mean[0];
		predictions.at<double>(i,1) = prediCos.mean[0];
		prediSin.mean.clear();
		prediSin.variance.clear();
		prediCos.mean.clear();
		prediCos.variance.clear();
	}

	double error, accuracy;
	this->evaluate(predictions, error, accuracy);
}
//==============================================================================
/** Evaluate one prediction versus its target.
 */
void classifyImages::evaluate(cv::Mat predictions, double &error, double &accuracy,\
char choice){
	accuracy = 0.0;
	error    = 0.0;
	for(int x=0; x<this->testTargets.cols; x++){
		if(choice == 'O'){
			double targetAngle = std::atan2(this->testTargets.at<double>(x,0),\
								this->testTargets.at<double>(x,1));
			double prediAngle = std::atan2(predictions.at<double>(x,0),\
								predictions.at<double>(x,1));

			std::cout<<"target: "<<targetAngle*180.0/M_PI<<\
				" VS "<<prediAngle*180.0/M_PI<<std::endl;

			error += std::pow(std::cos(targetAngle)-std::cos(prediAngle),2)+\
					std::pow(std::sin(targetAngle)-std::sin(prediAngle),2);
		}else{
			error += std::abs(this->testTargets.at<double>(x,0)-\
						predictions.at<double>(x,0));
		}
	}
	error   /= this->testTargets.cols;
	accuracy = 1-error;
	std::cout<<"Error: "<<error<<" Accuracy: "<<accuracy<<std::endl;
}
//==============================================================================
/** Build dictionary for vector quantization.
 */
void classifyImages::buildDictionary(char* fileToStore, char* dataFile){
	// EXTRACT FEATURES
	this->features->setFeatureType(featureDetector::SIFT);
	this->features->init(dataFile, "");
	this->features->run();
	// DO K-means

	// STORE IT
}
//==============================================================================
int main(int argc, char **argv){
	classifyImages classi(argc, argv);
	classi.init(1e-3,100.0,&gaussianProcess::sqexp,featureDetector::EDGES);
	classi.trainGP();
	cv::Mat predictions;
	classi.predictGP(predictions);
}


