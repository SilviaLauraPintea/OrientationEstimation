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
				argv[i]=const_cast<char*>("");
			}else if(i==6){
				this->testFolder = std::string(argv[i]);
				argv[i]=const_cast<char*>("");
			}else if(i==7){
				this->annotationsTest = std::string(argv[i]);
				argv[i]=const_cast<char*>("");
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
gaussianProcess::kernelFunction theKFunction, featureDetector::FEATURE theFeature,\
char* fileSIFT, int colorSp){
	this->noise     = theNoise;
	this->length    = theLength;
	this->kFunction = theKFunction;
	this->feature   = theFeature;
	this->features->setSIFTDictionary(fileSIFT);
	this->features->colorspaceCode = colorSp;
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
	double sinCosAccuracy = 0.0, sinCosError    = 0.0;
	accuracy = 0.0;
	error    = 0.0;
	for(int y=0; y<this->testTargets.rows; y++){
		if(choice == 'O'){
			double targetAngle = std::atan2(this->testTargets.at<double>(y,0),\
								this->testTargets.at<double>(y,1));
			double prediAngle = std::atan2(predictions.at<double>(y,0),\
								predictions.at<double>(y,1));

			std::cout<<"target: "<<targetAngle*180.0/M_PI<<\
				" VS "<<prediAngle*180.0/M_PI<<std::endl;

			sinCosError += std::pow(std::cos(targetAngle)-std::cos(prediAngle),2)+\
					std::pow(std::sin(targetAngle)-std::sin(prediAngle),2);

			if(std::abs(targetAngle-prediAngle)>M_PI){
				error += std::pow((2*M_PI-std::abs(targetAngle-prediAngle))/M_PI,2);
			}else{
				error += std::pow(std::abs(targetAngle-prediAngle)/M_PI,2);
			}
		}else{
			sinCosError += std::abs(this->testTargets.at<double>(y,0)-\
						predictions.at<double>(y,0));
		}
	}
	sinCosError   /= this->testTargets.rows;
	sinCosAccuracy = 1-sinCosError;
	error         /= this->testTargets.rows;
	error          = std::sqrt(error);
	accuracy       = 1-error;

	std::cout<<"Sin-Cos Error: "<<sinCosError<<" Sin-Cos Accuracy: "<<\
		sinCosAccuracy<<std::endl;
	std::cout<<"RMS Error: "<<error<<" RMS Accuracy: "<<accuracy<<std::endl;
}
//==============================================================================
/** Build dictionary for vector quantization.
 */
void classifyImages::buildDictionary(char* fileToStore, char* dataFile){
	// EXTRACT THE SIFT FEATURES AND CONCATENATE THEM
	if(std::string(this->features->dictFileName).size()<2){
		this->features->setSIFTDictionary(fileToStore);
	}
	this->features->setFeatureType(featureDetector::SIFT_DICT);
	this->features->init(dataFile, "");
	this->features->run();
	int rows = 0;
	for(std::size_t i=0; i<this->features->data.size(); i++){
		rows += this->features->data[i].rows;
	}
	cv::Mat dictData = cv::Mat::zeros(cv::Size(this->features->data[0].cols,\
						rows), cv::DataType<double>::type);

	// CONVERT FROM VECTOR OF CV::MAT TO MAT
	int contor = 0;
	for(std::size_t i=0; i<this->features->data.size(); i++){
		cv::Mat dummy = dictData.rowRange(contor,contor+this->features->data[i].rows);
		this->features->data[i].copyTo(dummy);
		contor += this->features->data[i].rows;
	}

	// DO K-means IN ORDER TO RETRIEVE BACK THE CLUSTER MEANS
	cv::Mat labels = cv::Mat::zeros(cv::Size(1,dictData.rows),\
					cv::DataType<double>::type); //LABEL EACH SAMPLE ASSIGNMENT
	cv::Mat* centers = new cv::Mat(cv::Size(dictData.cols,this->features->noMeans),\
						cv::DataType<double>::type);
	dictData.convertTo(dictData, cv::DataType<float>::type);
	cv::kmeans(dictData,this->features->noMeans,labels,\
		cv::TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,2,1),\
		5,cv::KMEANS_RANDOM_CENTERS,centers);
	dictData.release();
	labels.release();

	// NORMALIZE THE CENTERS AND STORE THEM
	for(int y=0; y<centers->rows; y++){
		cv::Mat rowsI = centers->row(y);
		rowsI         = rowsI/cv::norm(rowsI);
	}

	// WRITE TO FILE THE MEANS
	cv::Mat matrix(*centers);
	mat2BinFile(matrix, this->features->dictFileName);
	centers->release();
	matrix.release();
	delete centers;
}
//==============================================================================
int main(int argc, char **argv){
  	classifyImages classi(argc, argv);
/*
	classi.buildDictionary();
*/

	classi.init(1e-3,100.0,&gaussianProcess::sqexp,featureDetector::IPOINTS);
	classi.trainGP();
	cv::Mat predictions;
	classi.predictGP(predictions);

}


