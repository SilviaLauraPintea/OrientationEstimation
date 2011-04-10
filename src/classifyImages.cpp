/* classifyImages.cpp
 * Author: Silvia-Laura Pintea
 */
#include "classifyImages.h"
//==============================================================================
classifyImages::classifyImages(int argc, char **argv, classifyImages::USES use){
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
	this->foldSize		   = 5;
	this->storeData 	   = true;
	this->modelName        = "";
	this->what             = use;

	// READ THE COMMAND LINE ARGUMENTS
	if(argc != 3 && argc != 4){
		cerr<<"Usage: classifier datasetFolder/ [testsetFolder/] textOfImageName\n"<<\
			"datasetFolder/ and testsetFolder -- contain: \n"<<\
			"\t train: 'annotated_train/'\n "<<\
			"\t train targets: 'annotated_train.txt'\n "<<\
			"\t [test: 'annotated_test/']\n "<<\
			"\t [test targets: 'annotated_test.txt']\n "<<\
			"\t calibration: 'CALIB_textOfImageName.txt'\n "<<\
			"\t prior: 'PRIOR_textOfImageName.txt'\n"<<\
			"\t background: 'BG_textOfImageName.bin'"<<\
			"\t [SIFT data: 'annotated_SIFT/']\n"<<\
			"\t [SIFT dictionary: 'SIFT_textOfImageName.bin']\n"<<std::endl;
		exit(1);
	}else{
		std::string datasetPath,imgString,testsetPath;
		if(argc == 3){
			datasetPath = std::string(argv[1]);
			imgString   = std::string(argv[2]);
		}else if(argc == 4){
			datasetPath = std::string(argv[1]);
			testsetPath = std::string(argv[2]);
			imgString   = std::string(argv[3]);
			if(testsetPath[testsetPath.size()-1]!='/'){
				testsetPath += '/';
			}
		}
		if(datasetPath[datasetPath.size()-1]!='/'){
			datasetPath += '/';
		}
		switch(this->what){
			case(classifyImages::TEST):
				if(argc != 4){
					std::cerr<<"4 Arguments are needed for the final test."<<std::endl;
					exit(1);
				}
				// IF WE WANT TO TEST THE FINAL CLASIFIER'S PERFORMANCE
				this->trainFolder      = datasetPath+"annotated_train/";
				this->annotationsTrain = datasetPath+"annotated_train.txt";
				this->testFolder       = testsetPath+"annotated_test/";
				this->annotationsTest  = testsetPath+"annotated_test.txt";
				break;
			case(classifyImages::EVALUATE):
				// IF WE WANT TO EVALUATE WITH CORSSVALIDATION
				this->trainFolder      = datasetPath+"annotated_train/";
				this->annotationsTrain = datasetPath+"annotated_train.txt";
				break;
			case(classifyImages::BUILD_DICTIONARY):
				// IF WE WANT TO BUILD SIFT DICTIONARY
				this->trainFolder = datasetPath+"annotated_SIFT/";
				break;
		}

		this->features = new featureDetector(argc,argv);
		this->features->setSIFTDictionary(const_cast<char*>((datasetPath+"SIFT_"+\
							imgString+".bin").c_str()));
		this->modelName = datasetPath+"models/";
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
int colorSp, bool fromFolder, bool store){
	this->noise     = theNoise;
	this->length    = theLength;
	this->kFunction = theKFunction;
	this->feature   = theFeature;
	this->features->colorspaceCode = colorSp;
	this->readFromFolder           = fromFolder;
	this->storeData                = store;
	switch(this->feature){
		case(featureDetector::IPOINTS):
			this->modelName += "IPOINTS/";
			break;
		case(featureDetector::EDGES):
			this->modelName += "EDGES/";
			break;
		case(featureDetector::SURF):
			this->modelName += "SURF/";
			break;
		case(featureDetector::GABOR):
			this->modelName += "GABOR/";
			break;
		case(featureDetector::SIFT):
			this->modelName += "SIFT/";
			break;
	}
}
//==============================================================================
/** Creates the training data (according to the options), the labels and
 * trains the a \c GaussianProcess on the data.
 */
void classifyImages::trainGP(annotationsHandle::POSE what){
	// WE ASSUME THAT IF WE DO NOT WANT TO STORE DATA THEN WE WANT TO LOAD DATA
	if(this->storeData){
		this->features->setFeatureType(this->feature);
		this->features->init(this->trainFolder,this->annotationsTrain,this->readFromFolder);
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

		//IF WE WANT TO STORE DATA, THEN WE STORE IT
		if(!this->modelName.empty()){
			mat2BinFile(this->trainData,const_cast<char*>((this->modelName+\
				"Data.bin").c_str()),false);
			mat2BinFile(this->trainTargets,const_cast<char*>((this->modelName+\
				"Labels.bin").c_str()),false);
		}
	}else if(!this->modelName.empty()){
		// WE JUST LOAD THE DATA AND TRAIN AND PREDICT
		binFile2mat(this->trainData,const_cast<char*>((this->modelName+\
			"Data.bin").c_str()));
		binFile2mat(this->trainTargets,const_cast<char*>((this->modelName+\
			"Labels.bin").c_str()));
	}
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
void classifyImages::predictGP(std::deque<gaussianProcess::prediction> &predictionsSin,\
std::deque<gaussianProcess::prediction> &predictionsCos,\
annotationsHandle::POSE what){
	// WE ASSUME THAT IF WE DO NOT WANT TO STORE DATA THEN WE WANT TO LOAD DATA
	if(this->storeData){
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

		//IF WE WANT TO STORE DATA, THEN WE STORE IT
		if(!this->modelName.empty()){
			mat2BinFile(this->testData,const_cast<char*>((this->modelName+\
				"Data.bin").c_str()),false);
			mat2BinFile(this->testTargets,const_cast<char*>((this->modelName+\
				"Labels.bin").c_str()),false);
		}
	}else if(!this->modelName.empty()){
		// WE JUST LOAD THE TEST DATA AND TEST
		binFile2mat(this->testData,const_cast<char*>((this->modelName+\
			"Data.bin").c_str()));
		binFile2mat(this->testTargets,const_cast<char*>((this->modelName+\
			"Labels.bin").c_str()));
	}

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
void classifyImages::evaluate(std::deque<gaussianProcess::prediction>\
predictionsSin, std::deque<gaussianProcess::prediction> predictionsCos,\
double &error,double &normError, annotationsHandle::POSE what){
	double normAccuracy = 0.0;
	error = 0.0; normError = 0.0;
	for(int y=0; y<this->testTargets.rows; y++){
		double targetAngle;
		targetAngle = std::atan2(this->testTargets.at<double>(y,0),\
						this->testTargets.at<double>(y,1));

		// GET THE PREDICTED ANGLE
		double prediAngle = this->optimizePrediction(predictionsCos[y],\
							predictionsSin[y]);
		std::cout<<"target: "<<(targetAngle*180.0/M_PI)<<\
			" VS "<<(prediAngle*180.0/M_PI)<<std::endl;

		if(std::abs(targetAngle-prediAngle)>M_PI){
			error += std::pow((2*M_PI-std::abs(targetAngle-prediAngle)),2);
			normError += std::pow((2*M_PI-std::abs(targetAngle-prediAngle))/M_PI,2);
		}else{
			error += std::pow(std::abs(targetAngle-prediAngle),2);
			normError += std::pow(std::abs(targetAngle-prediAngle)/M_PI,2);
		}
	}
	error        = std::sqrt(error/this->testTargets.rows);
	normError    = std::sqrt(normError/this->testTargets.rows);
	normAccuracy = 1-normError;

	std::cout<<"RMS-error normalized:"<<normError<<" RMS-accuracy normalized:"<<\
		normAccuracy<<std::endl;
	std::cout<<"RMS-error: "<<error<<std::endl;
}
//==============================================================================
/** Try to optimize the prediction of the angle considering the variance of sin
 * and cos.
 */
double classifyImages::optimizePrediction(gaussianProcess::prediction \
predictionsSin, gaussianProcess::prediction predictionsCos){
	double betaS = 1.0/(predictionsSin.variance[0]);
	double betaC = 1.0/(predictionsCos.variance[0]);
	double x     = predictionsSin.mean[0];
	double y     = predictionsCos.mean[0];

	if(betaS == betaC){
		return std::atan2(betaS*x,betaC*y);
	}else{
		return std::atan2(x,y);
	}
	/*
	double closeTo;
	closeTo = std::atan2(predictionsSin.mean[0],predictionsCos.mean[0]);
	std::deque<double> alphas;
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
	*/
}
//==============================================================================
/** Build dictionary for vector quantization.
 */
void classifyImages::buildDictionary(){
	// EXTRACT THE SIFT FEATURES AND CONCATENATE THEM
	this->features->setFeatureType(featureDetector::SIFT_DICT);
	this->features->init(this->trainFolder,std::string(),this->readFromFolder);
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
featureDetector::FEATURE theFeature,int colorSp, bool fromFolder,bool store){
	double finalErrorLong=0.0, finalNormErrorLong=0.0;
	double finalErrorLat=0.0, finalNormErrorLat=0.0;
	for(unsigned i=0; i<k; i++){
		// SPLIT TRAINING AND TESTING ACCORDING TO THE CURRENT FOLD
		this->crossValidation(k,i);

	  	//LONGITUDE TRAINING AND PREDICTING
		this->init(theNoise,theLength,theKFunction,theFeature,colorSp,\
			fromFolder,store);

		// ADD THE NUMBER OF THE CURRENT FOLD & THEN REMOVE IT BACK
		this->modelName += ("train/"+int2string(i));
		this->trainGP(annotationsHandle::LONGITUDE);
		this->modelName = this->modelName.substr(0,this->modelName.size()-7);

		// ADD THE NUMBER OF THE CURRENT FOLD & THEN REMOVE IT BACK
		this->modelName += ("eval/"+int2string(i));
		std::deque<gaussianProcess::prediction> predictionsSin;
		std::deque<gaussianProcess::prediction> predictionsCos;
		this->predictGP(predictionsSin,predictionsCos,annotationsHandle::LONGITUDE);
		this->modelName = this->modelName.substr(0,this->modelName.size()-6);

		double errorLong, normErrorLong;
		this->evaluate(predictionsSin, predictionsCos, errorLong, normErrorLong,\
			annotationsHandle::LONGITUDE);
		finalErrorLong += errorLong;
		finalNormErrorLong += normErrorLong;

	  	//LATITUDE TRAINING AND PREDICTING
		this->init(theNoise,theLength,theKFunction,theFeature,colorSp,\
			fromFolder,store);
		this->trainGP(annotationsHandle::LATITUDE);
		this->predictGP(predictionsSin,predictionsCos,annotationsHandle::LATITUDE);
		double errorLat, normErrorLat;
		this->evaluate(predictionsSin, predictionsCos, errorLat, normErrorLat,\
			annotationsHandle::LATITUDE);
		finalErrorLat += errorLat;
		finalNormErrorLat += normErrorLat;
	}
	finalErrorLong /= static_cast<double>(k);
	finalNormErrorLong /= static_cast<double>(k);
	std::cout<<"LONGITUDE>>> final-RMS-error:"<<finalErrorLong<<\
		" final-RMS-normalized-error:"<<finalNormErrorLong<<std::endl;

	finalErrorLat /= static_cast<double>(k);
	finalNormErrorLat /= static_cast<double>(k);
	std::cout<<"LATITUDE>>> final-RMS-error:"<<finalErrorLat<<\
		" final-RMS-normalized-error:"<<finalNormErrorLat<<std::endl;
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

	unsigned pos     = this->trainFolder.find_first_of("/\\");
	std::string root = this->trainFolder.substr(0,pos+1);
	this->trainFolder      = root+"trash/targets.txt";root+
	this->trainFolder      = root+"trash/targets.txt";
	this->testFolder       = root+"trash/ttargets.txt";
	this->annotationsTrain = root+"trash/annoTargets.txt";
	this->annotationsTest  = root+"trash/annoTtargets.txt";
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
int colorSp, bool fromFolder, bool store){
  	//LONGITUDE TRAINING AND PREDICTING
	this->init(theNoise,theLength,theKFunction,theFeature,colorSp,\
		fromFolder,store);

	// ADD THE NUMBER OF THE CURRENT FOLD & THEN REMOVE IT BACK
	this->modelName += ("model/");
	this->trainGP(annotationsHandle::LONGITUDE);
	this->modelName = this->modelName.substr(0,this->modelName.size()-5);

	this->modelName += ("test/");
	std::deque<gaussianProcess::prediction> predictionsSin;
	std::deque<gaussianProcess::prediction> predictionsCos;
	this->predictGP(predictionsSin,predictionsCos,annotationsHandle::LONGITUDE);
	this->modelName = this->modelName.substr(0,this->modelName.size()-4);

	double errorLong, normErrorLong;
	this->evaluate(predictionsSin, predictionsCos, errorLong, normErrorLong,\
		annotationsHandle::LONGITUDE);

  	//LATITUDE TRAINING AND PREDICTING
	this->init(theNoise,theLength,theKFunction,theFeature,colorSp,\
		fromFolder,store);
	this->trainGP(annotationsHandle::LATITUDE);
	this->predictGP(predictionsSin,predictionsCos,annotationsHandle::LATITUDE);
	double errorLat, normErrorLat;
	this->evaluate(predictionsSin, predictionsCos, errorLat, normErrorLat,\
		annotationsHandle::LATITUDE);
}
//==============================================================================
int main(int argc, char **argv){
  	classifyImages classi(argc, argv, classifyImages::EVALUATE);
/*
	// BUILD THE SIFT DICTIONARY
	classi.buildDictionary();

	// PERFORMANCE EVALUATION
	classi.runTest(1e-3,100.0,&gaussianProcess::sqexp,featureDetector::SURF);

  	// CROSS-VALIDATION
  	classi.runCrossValidation(5,1e-3,100.0,&gaussianProcess::sqexp,\
  		featureDetector::SURF);
*/
  	classi.runCrossValidation(2,1e-3,100.0,&gaussianProcess::sqexp,\
  		featureDetector::SURF,CV_BGR2Lab,false,true);
}



