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
	this->readFromFolder   = true;
	this->noise            = 0.01;
	this->length           = 1.0;
	this->kFunction        = &gaussianProcess::sqexp;
	this->feature          = featureDetector::EDGES;
	this->features         = NULL;
	this->foldSize		   = 0;

	// READ THE COMMAND LINE ARGUMENTS
	if(argc != 8 && argc!=5 && argc!=6){
		cerr<<"Usage: classifier <bgTrain|bgModel> <calib> <prior> "<<\
			"<trainFolder> [<annotationsTrain> [<testFolder> <annotationsTest>]]"<<endl;
		exit(1);
	}else if(argc==8){ // TEST = FINAL EVALUATION
		for(unsigned i=0; i<argc; i++){
			if(i==4){
				this->trainFolder = std::string(argv[i]);
			}else if(i==5){
				this->annotationsTrain = std::string(argv[i]);
				argv[i]=const_cast<char*>(" ");
			}else if(i==6){
				this->testFolder = std::string(argv[i]);
				argv[i]=const_cast<char*>(" ");
			}else if(i==7){
				this->annotationsTest = std::string(argv[i]);
				argv[i]=const_cast<char*>(" ");
			}
		}
		argc -= 3;
		this->features = new featureDetector(argc,argv);
	}else if(argc==6){ // K-FOLD CROSS-VALIDATION
		for(unsigned i=0; i<argc; i++){
			if(i==4){
				this->trainFolder = std::string(argv[i]);
			}else if(i==5){
				this->annotationsTrain = std::string(argv[i]);
				argv[i]=const_cast<char*>(" ");
			}
		}
		argc -= 1;
		this->features = new featureDetector(argc,argv);
	}else if(argc==5){ // FOR BUILDING THE SIFT DICTIONARY
		for(unsigned i=0; i<argc; i++){
			if(i==4){
				this->trainFolder = std::string(argv[i]);
			}
			this->annotationsTrain = "";
			this->testFolder       = "";
			this->annotationsTest  = "";
		}
		this->features = new featureDetector(argc,argv);
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
char* fileSIFT, int colorSp, bool fromFolder){
	this->noise     = theNoise;
	this->length    = theLength;
	this->kFunction = theKFunction;
	this->feature   = theFeature;
	this->features->setSIFTDictionary(fileSIFT);
	this->features->colorspaceCode = colorSp;
	this->readFromFolder           = fromFolder;
}
//==============================================================================
/** Creates the training data (according to the options), the labels and
 * trains the a \c GaussianProcess on the data.
 */
void classifyImages::trainGP(annotationsHandle::POSE what){
	this->features->setFeatureType(this->feature);
	this->features->init(this->trainFolder, this->annotationsTrain,this->readFromFolder);
	this->features->run(this->readFromFolder);
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

	// TRAIN THE SIN AND COS SEPARETELY FOR LONGITUDE || LATITUDe
	if(what == annotationsHandle::LONGITUDE){
		this->gpSin.train(this->trainData,this->trainTargets.col(0),\
			this->kFunction, this->noise, this->length);
		this->gpCos.train(this->trainData,this->trainTargets.col(1),\
			this->kFunction, this->noise, this->length);
	}else if(what == annotationsHandle::LATITUDE){
		// TRAIN THE SIN AND COS SEPARETELY FOR LATITUDE
		this->gpSin.train(this->trainData,this->trainTargets.col(2),\
			this->kFunction, this->noise, this->length);
		this->gpCos.train(this->trainData,this->trainTargets.col(3),\
			this->kFunction, this->noise, this->length);
	}
}
//==============================================================================
/** Creates the test data and applies \c GaussianProcess prediction on the test
 * data.
 */
void classifyImages::predictGP(std::vector<gaussianProcess::prediction> &predictionsSin,\
std::vector<gaussianProcess::prediction> &predictionsCos,\
annotationsHandle::POSE what){
	this->features->setFeatureType(this->feature);
	this->features->init(this->testFolder, this->annotationsTest,this->readFromFolder);
	this->features->run(this->readFromFolder);
	this->testData = cv::Mat::zeros(cv::Size(this->features->data[0].cols,\
					this->features->data.size()), cv::DataType<double>::type);
	this->testTargets = cv::Mat::zeros(cv::Size(2,this->features->data.size()),\
					cv::DataType<double>::type);

	// CONVERT FROM VECTOR OF CV::MAT TO MAT
	for(std::size_t i=0; i<this->features->data.size(); i++){
		cv::Mat dummy1 = this->testData.row(i);
		this->features->data[i].copyTo(dummy1);

		cv::Mat dummy2;
		dummy2 = this->testTargets.row(i);
		if(what == annotationsHandle::LONGITUDE){
			this->features->targets[i].colRange(0,2).copyTo(dummy2);
		}else if(what == annotationsHandle::LATITUDE){
			this->features->targets[i].colRange(2,4).copyTo(dummy2);
		}
	}
	this->testData.convertTo(this->testData, cv::DataType<double>::type);
	this->testTargets.convertTo(this->testTargets, cv::DataType<double>::type);
	range1Mat(this->testData);

	// FOR EACH ROW IN THE TEST MATRIX PREDICT
	for(unsigned i=0; i<this->testData.rows; i++){
		gaussianProcess::prediction prediSin;
		this->gpSin.predict(this->testData.row(i), prediSin, this->length);
		gaussianProcess::prediction prediCos;
		this->gpCos.predict(this->testData.row(i), prediCos, this->length);
		predictionsSin.push_back(prediSin);
		predictionsCos.push_back(prediCos);
		prediSin.mean.clear();
		prediSin.variance.clear();
		prediCos.mean.clear();
		prediCos.variance.clear();
	}
}
//==============================================================================
/** Evaluate one prediction versus its target.
 */
void classifyImages::evaluate(std::vector<gaussianProcess::prediction>\
predictionsSin, std::vector<gaussianProcess::prediction> predictionsCos,\
double &error, double &accuracy, annotationsHandle::POSE what){
	double sinCosAccuracy = 0.0, sinCosError = 0.0;
	accuracy = 0.0;
	error    = 0.0;
	for(int y=0; y<this->testTargets.rows; y++){
		double targetAngle = std::atan2(std::sqrt(this->testTargets.at<double>(y,0)),\
								std::sqrt(this->testTargets.at<double>(y,1)));

		double prediAngle = this->optimizePrediction(predictionsCos[y],\
							predictionsSin[y]);

		std::cout<<"target: "<<targetAngle*180.0/M_PI<<\
			" VS "<<prediAngle*180.0/M_PI<<std::endl;

		sinCosError += std::pow(std::cos(targetAngle)-std::cos(prediAngle),2)+\
				std::pow(std::sin(targetAngle)-std::sin(prediAngle),2);

		if(std::abs(targetAngle-prediAngle)>M_PI){
			error += std::pow((2*M_PI-std::abs(targetAngle-prediAngle))/M_PI,2);
		}else{
			error += std::pow(std::abs(targetAngle-prediAngle)/M_PI,2);
		}
	}
	sinCosError   /= this->testTargets.rows;
	sinCosAccuracy = 1-sinCosError;
	error         /= this->testTargets.rows;
	error          = std::sqrt(error);
	accuracy       = 1-error;

	std::cout<<"sin-cos-error:"<<sinCosError<<" sin-cos-accuracy:"<<\
		sinCosAccuracy<<std::endl;
	std::cout<<"RMS-error: "<<error<<" RMS-accuracy:"<<accuracy<<std::endl;
}
//==============================================================================
/** Try to optimize the prediction of the angle considering the variance of sin
 * and cos.
 */
double classifyImages::optimizePrediction(gaussianProcess::prediction \
predictionsSin, gaussianProcess::prediction predictionsCos){
	double closeTo = std::atan2(std::sqrt(predictionsSin.mean[0]),\
						std::sqrt(predictionsCos.mean[0]));

	std::cout<<"close to: "<<closeTo<<std::endl;

	double betaS = 1.0/(predictionsSin.variance[0]);
	double betaC = 1.0/(predictionsCos.variance[0]);
	double x     = predictionsSin.mean[0];
	double y     = predictionsCos.mean[0];
	std::vector<double> alphas;
	if(betaS != betaC){
		std::cout<<"betaS="<<betaS<<" betaC="<<betaC<<" x="<<x<<" y="<<y<<std::endl;

		double b = -1.0*(betaS*x + betaC*y + betaS - betaC);
		double a = betaS - betaC;
		double c = betaS*x;

		std::cout<<"b="<<b<<" a="<<a<<" c="<<c<<std::endl;
		std::cout<<"alpha1: "<<((-b + std::sqrt(b*b - 4.0*a*c))/2.0*a)<<std::endl;
		std::cout<<"alpha2: "<<((-b - std::sqrt(b*b - 4.0*a*c))/2.0*a)<<std::endl;

		alphas.push_back((-b + std::sqrt(b*b - 4.0*a*c))/2.0*a);
		alphas.push_back((-b - std::sqrt(b*b - 4.0*a*c))/2.0*a);
	}else{
		std::cout<<"alpha1: "<<(betaS*x/(betaS*x + betaC*y))<<std::endl;
		alphas.push_back(betaS*x/(betaS*x + betaC*y));
	}
	double minDist = 2.0*M_PI, minAngle;
	for(unsigned i=0; i<alphas.size(); i++){
		if(alphas[i]>=0){
			double alpha1 = std::asin(std::sqrt(alphas[i]));
			double alpha2 = std::asin(-std::sqrt(alphas[i]));
			if(std::abs(alpha1-closeTo)<minDist){
				minDist  = std::abs(alpha1-closeTo);
				minAngle = alpha1;
			}
			if(std::abs(alpha2-closeTo)<minDist){
				minDist  = std::abs(alpha2-closeTo);
				minAngle = alpha2;
			}
		}
	}
	return minAngle;
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
	this->features->init(dataFile, "", this->readFromFolder);
	this->features->run(this->readFromFolder);
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
/** Does the cross-validation and computes the average error over all folds.
 */
void classifyImages::runCrossValidation(unsigned k, double theNoise,\
double theLength, gaussianProcess::kernelFunction theKFunction,\
featureDetector::FEATURE theFeature, char* fileSIFT, int colorSp, bool fromFolder){
	double finalErrorLong=0.0, finalAccuracyLong=0.0;
	double finalErrorLat=0.0, finalAccuracyLat=0.0;
	for(unsigned i=0; i<k; i++){
		// SPLIT TRAINING AND TESTING ACCORDING TO THE CURRENT FOLD
		this->crossValidation(k,i);

	  	//LONGITUDE TRAINING AND PREDICTING
		this->init(theNoise,theLength,theKFunction,theFeature,fileSIFT,colorSp,\
			fromFolder);
		this->trainGP(annotationsHandle::LONGITUDE);
		std::vector<gaussianProcess::prediction> predictionsSin;
		std::vector<gaussianProcess::prediction> predictionsCos;
		this->predictGP(predictionsSin,predictionsCos,annotationsHandle::LONGITUDE);
		double errorLong, accuracyLong;
		this->evaluate(predictionsSin, predictionsCos, errorLong, accuracyLong,\
			annotationsHandle::LONGITUDE);
		finalErrorLong += errorLong;
		finalAccuracyLong += accuracyLong;
		std::cout<<"!!!!!!!!!!!!LONGITUDE>>> sum-of-error:"<<finalErrorLong<<\
			" sum-of-accuracy:"<<finalAccuracyLong<<" k:"<<i<<std::endl;

	  	//LATITUDE TRAINING AND PREDICTING
		this->init(theNoise,theLength,theKFunction,theFeature,fileSIFT,colorSp,\
			fromFolder);
		this->trainGP(annotationsHandle::LATITUDE);
		this->predictGP(predictionsSin,predictionsCos,annotationsHandle::LATITUDE);
		double errorLat, accuracyLat;
		this->evaluate(predictionsSin, predictionsCos, errorLat, accuracyLat,\
			annotationsHandle::LATITUDE);
		finalErrorLat += errorLat;
		finalAccuracyLat += accuracyLat;
		std::cout<<"!!!!!!!!!!!!LATITUDE>>> sum-of-error:"<<finalErrorLat<<\
			" sum-of-accuracy:"<<finalAccuracyLat<<" k:"<<i<<std::endl;
	}
	finalErrorLong /= static_cast<double>(k);
	finalAccuracyLong /= static_cast<double>(k);
	std::cout<<"LONGITUDE>>> final-RMS-error:"<<finalErrorLong<<\
		" final-RMS-accuracy:"<<finalAccuracyLong<<std::endl;

	finalErrorLat /= static_cast<double>(k);
	finalAccuracyLat /= static_cast<double>(k);
	std::cout<<"LATITUDE>>> final-RMS-error:"<<finalErrorLat<<\
		" final-RMS-accuracy:"<<finalAccuracyLat<<std::endl;
}
//==============================================================================
/** Do k-fold cross-validation by splitting the training folder into training-set
 * and validation-set.
 */
void classifyImages::crossValidation(unsigned k, unsigned fold){
	// READ ALL IMAGES ONCE AND NOT THEY ARE SORTED
	if(this->imageList.empty()){
		this->imageList = readImages(this->trainFolder.c_str());
		this->foldSize  = this->imageList.size()/k;

		ifstream annoIn(this->annotationsTrain.c_str());
		if(annoIn.is_open()){
			while(annoIn.good()){
				std::string line;
				std::getline(annoIn, line);
				if(!line.empty()){
					this->annoList.push_back(line);
				}
				line.clear();
			}
			annoIn.close();
		}
		sort(this->annoList.begin(),this->annoList.end());

		if(this->annoList.size()!=this->imageList.size()){
			std::cerr<<"The number of images != The number of annotations!"<<\
				std::endl;
			exit(1);
		}
	}

	this->trainFolder      = "targets.txt";
	this->testFolder       = "ttargets.txt";
	this->annotationsTrain = "annoTargets.txt";
	this->annotationsTest  = "annoTtargets.txt";

	// WRITE THE IMAGE NAMES & ANNOTATIONS IN THE CORRESPONDING FILES
	ofstream testOut, trainOut, annoTest, annoTrain;
	testOut.open(this->testFolder.c_str(), ios::out);
	if(!testOut){
		errx(1,"Cannot open file %s", this->testFolder.c_str());
	}
	trainOut.open(this->trainFolder.c_str(), ios::out);
	if(!trainOut){
		errx(1,"Cannot open file %s", this->trainFolder.c_str());
	}

	annoTest.open(this->annotationsTest.c_str(), ios::out);
	if(!annoTest){
		errx(1,"Cannot open file %s", this->annotationsTest.c_str());
	}
	annoTrain.open(this->annotationsTrain.c_str(), ios::out);
	if(!annoTrain){
		errx(1,"Cannot open file %s", this->annotationsTrain.c_str());
	}

	for(unsigned i=0; i<this->imageList.size(); i++){
		if(i>=(this->foldSize*fold) && i<(this->foldSize*(fold+1))){
			testOut<<this->imageList[i]<<std::endl;
			annoTest<<this->annoList[i]<<std::endl;
		}else{
			trainOut<<this->imageList[i]<<std::endl;
			annoTrain<<this->annoList[i]<<std::endl;
		}
	}
	testOut.close();
	trainOut.close();
	annoTest.close();
	annoTrain.close();
}
//==============================================================================
/** Runs the final evaluation (test).
 */
void classifyImages::runTest(double theNoise, double theLength,\
gaussianProcess::kernelFunction theKFunction, featureDetector::FEATURE theFeature,\
char* fileSIFT, int colorSp, bool fromFolder){
  	//LONGITUDE TRAINING AND PREDICTING
	this->init(theNoise,theLength,theKFunction,theFeature,fileSIFT,colorSp,\
		fromFolder);
	this->trainGP(annotationsHandle::LONGITUDE);
	std::vector<gaussianProcess::prediction> predictionsSin;
	std::vector<gaussianProcess::prediction> predictionsCos;
	this->predictGP(predictionsSin,predictionsCos,annotationsHandle::LONGITUDE);
	double errorLong, accuracyLong;
	this->evaluate(predictionsSin, predictionsCos, errorLong, accuracyLong,\
		annotationsHandle::LONGITUDE);

  	//LATITUDE TRAINING AND PREDICTING
	this->init(theNoise,theLength,theKFunction,theFeature,fileSIFT,colorSp,\
		fromFolder);
	this->trainGP(annotationsHandle::LATITUDE);
	this->predictGP(predictionsSin,predictionsCos,annotationsHandle::LATITUDE);
	double errorLat, accuracyLat;
	this->evaluate(predictionsSin, predictionsCos, errorLat, accuracyLat,\
		annotationsHandle::LATITUDE);
}
//==============================================================================
int main(int argc, char **argv){
  	classifyImages classi(argc, argv);
/*
	// BUILD THE SIFT DICTIONARY
	classi.buildDictionary(const_cast<char*>("dictSIFT.bin"),\
		const_cast<char*>("test/sift/"));
*/

/*
	// PERFORMANCE EVALUATION
	classi.runTest(1e-3,100.0,&gaussianProcess::sqexp,featureDetector::SURF);
*/

  	// CROSS-VALIDATION
  	classi.runCrossValidation(2,1e-3,100.0,&gaussianProcess::sqexp,\
  		featureDetector::SURF);
}



