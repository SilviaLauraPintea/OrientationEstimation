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
	if(argc != 3 && argc != 5){
		cerr<<"Usage: classifier datasetFolder/ textOfImageName [testsetFolder/ "<<\
			"textOfTestImageName] \n"<<\
			"datasetFolder/ and testsetFolder -- contain: \n"<<\
			"\t train: 'annotated_train/'\n "<<\
			"\t train targets: 'annotated_train.txt'\n "<<\
			"\t [test: 'annotated_test/']\n "<<\
			"\t [test targets: 'annotated_test.txt']\n "<<\
			"\t calibration: 'CALIB_textOfImageName.txt'\n "<<\
			"\t prior: 'PRIOR_textOfImageName.txt'\n"<<\
			"\t background: 'BG_textOfImageName.bin'\n"<<\
			"\t [SIFT data: 'annotated_SIFT/']\n"<<\
			"\t [SIFT dictionary: 'SIFT_textOfImageName.bin']\n"<<std::endl;
		exit(1);
	}else{
		this->trainDir       = std::string(argv[1]);
		this->trainImgString = std::string(argv[2]);
		if(this->trainDir[this->trainDir.size()-1]!='/'){
			this->trainDir += '/';
		}
		if(argc == 5){
			this->testDir = std::string(argv[3]);
			if(this->testDir[this->testDir.size()-1]!='/'){
				this->testDir += '/';
			}
			this->testImgString = std::string(argv[4]);
		}

		std::vector<std::string> files2check;
		switch(this->what){
			case(classifyImages::TEST):
				if(argc != 5){
					std::cerr<<"4 Arguments are needed for the final test: "<<\
						"classifier datasetFolder/ textOfImageName [testsetFolder/ "<<\
						"textOfImageNameTest]"<<std::endl;
					exit(1);
				}
				// IF WE WANT TO TEST THE FINAL CLASIFIER'S PERFORMANCE
				this->trainFolder      = this->trainDir+"annotated_train/";
				this->annotationsTrain = this->trainDir+"annotated_train.txt";
				this->testFolder       = this->testDir+"annotated_test/";
				this->annotationsTest  = this->testDir+"annotated_test.txt";
				files2check.push_back(this->trainFolder);
				files2check.push_back(this->annotationsTrain);
				files2check.push_back(this->testFolder);
				files2check.push_back(this->annotationsTest);
				break;
			case(classifyImages::EVALUATE):
				// IF WE WANT TO EVALUATE WITH CORSSVALIDATION
				this->trainFolder      = this->trainDir+"annotated_train/";
				this->annotationsTrain = this->trainDir+"annotated_train.txt";
				files2check.push_back(this->trainFolder);
				files2check.push_back(this->annotationsTrain);
				break;
			case(classifyImages::BUILD_DICTIONARY):
				// IF WE WANT TO BUILD SIFT DICTIONARY
				this->trainFolder = this->trainDir+"annotated_SIFT/";
				files2check.push_back(this->trainFolder);
				break;
		}
		for(std::size_t i=0; i<files2check.size();i++){
			if(!file_exists(files2check[i].c_str())){
				std::cerr<<"File/folder not found: "<<files2check[i]<<std::endl;
				exit(1);
			}
		}

		this->modelName = this->trainDir+"models/";
		file_exists(this->modelName.c_str(), true);
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
bool fromFolder, bool store){
	this->noise          = theNoise;
	this->length         = theLength;
	this->kFunction      = theKFunction;
	this->feature        = theFeature;
	this->readFromFolder = fromFolder;
	this->storeData      = store;
	switch(this->feature){
		case(featureDetector::IPOINTS):
			this->modelName = this->trainDir+"models/"+"IPOINTS/";
			file_exists(this->modelName.c_str(), true);
			break;
		case(featureDetector::EDGES):
			this->modelName = this->trainDir+"models/"+"EDGES/";
			file_exists(this->modelName.c_str(), true);
			break;
		case(featureDetector::SURF):
			this->modelName = this->trainDir+"models/"+"SURF/";
			file_exists(this->modelName.c_str(), true);
			break;
		case(featureDetector::GABOR):
			this->modelName = this->trainDir+"models/"+"GABOR/";
			file_exists(this->modelName.c_str(), true);
			break;
		case(featureDetector::SIFT):
			this->modelName = this->trainDir+"models/"+"SIFT/";
			file_exists(this->modelName.c_str(), true);
			break;
	}
}
//==============================================================================
/** Creates the training data (according to the options), the labels and
 * trains the a \c GaussianProcess on the data.
 */
void classifyImages::trainGP(annotationsHandle::POSE what){
	// WE ASSUME THAT IF WE DO NOT WANT TO STORE DATA THEN WE WANT TO LOAD DATA
	std::string modelNameData   = this->modelName+"Data.bin";
	std::string modelNameLabels = this->modelName+"Labels.bin";

	if(this->storeData || !file_exists(modelNameData.c_str()) || \
	!file_exists(modelNameLabels.c_str())){
		this->features->setFeatureType(this->feature);
		this->features->init(this->trainFolder,this->annotationsTrain,this->readFromFolder);
		this->features->run(this->readFromFolder);
		this->features->data.copyTo(this->trainData);
		this->features->targets.copyTo(this->trainTargets);

		std::cout<<"SIZE: "<<this->trainData.cols<<" "<<this->trainData.rows<<std::endl;
		std::cout<<"SIZE: "<<this->trainTargets.cols<<" "<<this->trainTargets.rows<<std::endl;

		this->trainData.convertTo(this->trainData, cv::DataType<double>::type);
		this->trainTargets.convertTo(this->trainTargets, cv::DataType<double>::type);
		range1Mat(this->trainData);

		//IF WE WANT TO STORE DATA, THEN WE STORE IT
		if(!this->modelName.empty()){
			file_exists(this->modelName.c_str(), true);
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
	std::string modelNameData   = this->modelName+"Data.bin";
	std::string modelNameLabels = this->modelName+"Labels.bin";

	if(this->storeData || !file_exists(modelNameData.c_str()) || \
	!file_exists(modelNameLabels.c_str())){
		this->features->setFeatureType(this->feature);
		this->features->init(this->testFolder, this->annotationsTest,this->readFromFolder);
		this->features->run(this->readFromFolder);
		this->features->data.copyTo(this->testData);

		// GET ONLY THE ANGLES YOU NEED
		if(what == annotationsHandle::LONGITUDE){
			cv::Mat dum = this->features->targets.colRange(0,2);
			dum.copyTo(this->testTargets);
			dum.release();
		}else if(what == annotationsHandle::LATITUDE){
			cv::Mat dum = this->features->targets.colRange(2,4);
			dum.copyTo(this->testTargets);
			dum.release();
		}
		this->testData.convertTo(this->testData, cv::DataType<double>::type);
		this->testTargets.convertTo(this->testTargets, cv::DataType<double>::type);
		range1Mat(this->testData);

		//IF WE WANT TO STORE DATA, THEN WE STORE IT
		if(!this->modelName.empty()){
			file_exists(this->modelName.c_str(), true);
			mat2BinFile(this->testData,const_cast<char*>((this->modelName+\
				"Data.bin").c_str()),false);
			mat2BinFile(this->testTargets,const_cast<char*>((this->modelName+\
				"Labels.bin").c_str()),false);
		}
	}else if(!this->modelName.empty()){
		// WE JUST LOAD THE TEST DATA AND TEST
		binFile2mat(this->testData,const_cast<char*>((this->modelName+\
			"Data.bin").c_str()));
		cv::Mat tmp;
		binFile2mat(tmp,const_cast<char*>((this->modelName+\
			"Labels.bin").c_str()));
		if(what == annotationsHandle::LONGITUDE){
			cv::Mat dum = tmp.colRange(0,2);
			dum.copyTo(this->testTargets);
			dum.release();
		}else if(what == annotationsHandle::LATITUDE){
			cv::Mat dum = tmp.colRange(2,4);
			dum.copyTo(this->testTargets);
			dum.release();
		}
		tmp.release();
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
double &error,double &normError,double &meanDiff,annotationsHandle::POSE what){
	double normAccuracy = 0.0;
	error = 0.0; normError = 0.0; meanDiff = 0.0;
	for(int y=0; y<this->testTargets.rows; y++){
		double targetAngle;
		targetAngle = std::atan2(this->testTargets.at<double>(y,0),\
						this->testTargets.at<double>(y,1));

		// GET THE PREDICTED ANGLE
		double prediAngle = this->optimizePrediction(predictionsCos[y],\
							predictionsSin[y]);
		std::cout<<"Target: "<<targetAngle<<"("<<(targetAngle*180.0/M_PI)<<\
			") VS "<<prediAngle<<"("<<(prediAngle*180.0/M_PI)<<")"<<std::endl;
		double absDiff = std::abs(targetAngle-prediAngle);
		if (absDiff > M_PI)
			absDiff = 2*M_PI - absDiff;
		std::cout<<"Difference: "<< absDiff <<std::endl;
		error     += std::pow(absDiff,2);
		normError += std::pow(absDiff/M_PI,2);
		meanDiff  += absDiff;
	}
	error        = std::sqrt(error)/this->testTargets.rows;
	normError    = std::sqrt(normError)/this->testTargets.rows;
	meanDiff     = meanDiff/this->testTargets.rows;
	normAccuracy = 1-normError;

	std::cout<<"RMS-error normalized: "<<normError<<std::endl;
	std::cout<<"RMS-accuracy normalized: "<<normAccuracy<<std::endl;
	std::cout<<"RMS-error: "<<error<<std::endl;
	std::cout<<"Avg-Degree-Difference: "<<meanDiff<<std::endl;
}
//==============================================================================
/** Try to optimize the prediction of the angle considering the variance of sin
 * and cos.
 */
double classifyImages::optimizePrediction(gaussianProcess::prediction \
predictionsSin, gaussianProcess::prediction predictionsCos){
	double x     = predictionsSin.mean[0];
	double y     = predictionsCos.mean[0];
	return std::atan2(x,y);

/*
	double betaS = 1.0/(predictionsSin.variance[0]);
	double betaC = 1.0/(predictionsCos.variance[0]);
	double x     = predictionsSin.mean[0];
	double y     = predictionsCos.mean[0];

	if(betaS == betaC){
		return std::atan2(betaS*x,betaC*y);
	}else{
		return std::atan2(x,y);
	}
*/
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
void classifyImages::buildDictionary(int colorSp){
	// SET THE CALIBRATION AND OTHER FEATURE SETTINGS
	this->resetFeatures(this->trainDir, this->trainImgString, colorSp);

	// EXTRACT THE SIFT FEATURES AND CONCATENATE THEM
	this->features->setFeatureType(featureDetector::SIFT_DICT);
	this->features->init(this->trainFolder,std::string(),this->readFromFolder);
	this->features->run(this->readFromFolder);
	cv::Mat dictData;
	this->features->data.copyTo(dictData);

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
	mat2BinFile(matrix, const_cast<char*>(this->features->dictFileName.c_str()));
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
	double finalMeanDiffLat=0.0, finalMeanDiffLong=0.0;

	// SET THE CALIBRATION ONLY ONCE (ALL IMAGES ARE READ FROM THE SAME DIR)
	this->resetFeatures(this->trainDir, this->trainImgString,colorSp);
	this->init(theNoise,theLength,theKFunction,theFeature,fromFolder,store);
	for(unsigned i=0; i<k; i++){
		std::cout<<"Round "<<i<<"___________________________________________"<<\
			"_____________________________________________________"<<std::endl;
		// SPLIT TRAINING AND TESTING ACCORDING TO THE CURRENT FOLD
		std::deque<gaussianProcess::prediction> predictionsSin;
		std::deque<gaussianProcess::prediction> predictionsCos;
		this->crossValidation(k,i);
		//______________________________________________________________________
	  	//LONGITUDE TRAINING AND PREDICTING
		std::cout<<"Longitude >>> "<<i<<"___________________________________"<<\
			"_____________________________________________________"<<std::endl;
		this->modelName += ("trainLong/"+int2string(i));
		this->trainGP(annotationsHandle::LONGITUDE);
		this->modelName = this->modelName.substr(0,this->modelName.size()-11);

		// PREDICT ON THE REST OF THE IMAGES
		this->modelName += ("evalLong/"+int2string(i));
		this->predictGP(predictionsSin,predictionsCos,annotationsHandle::LONGITUDE);
		this->modelName = this->modelName.substr(0,this->modelName.size()-10);
		double errorLong, normErrorLong, meanDiffLong;
		this->evaluate(predictionsSin, predictionsCos, errorLong, normErrorLong,\
				meanDiffLong, annotationsHandle::LONGITUDE);
		finalErrorLong += errorLong;
		finalNormErrorLong += normErrorLong;
		finalMeanDiffLong += meanDiffLong;

		//______________________________________________________________________
	  	//LATITUDE TRAINING AND PREDICTING
		/*
		std::cout<<"Latitude >>> "<<i<<"____________________________________"<<\
			"_____________________________________________________"<<std::endl;
		this->modelName += ("trainLat/"+int2string(i));
		this->trainGP(annotationsHandle::LATITUDE);
		this->modelName = this->modelName.substr(0,this->modelName.size()-10);

		// PREDICT ON THE REST OF THE IMAGES
		this->modelName += ("evalLat/"+int2string(i));
		this->predictGP(predictionsSin,predictionsCos,annotationsHandle::LATITUDE);
		this->modelName = this->modelName.substr(0,this->modelName.size()-9);
		double errorLat, normErrorLat, meanDiffLat;
		this->evaluate(predictionsSin, predictionsCos, errorLat, normErrorLat,\
			meanDiffLat, annotationsHandle::LATITUDE);
		finalErrorLat += errorLat;
		finalNormErrorLat += normErrorLat;
		finalMeanDiffLat += meanDiffLat;
		*/
	}
	finalErrorLong /= static_cast<double>(k);
	finalNormErrorLong /= static_cast<double>(k);
	finalMeanDiffLong /= static_cast<double>(k);
	std::cout<<"LONGITUDE>>> final-RMS-error:"<<finalErrorLong<<std::endl;
	std::cout<<"LONGITUDE>>> final-RMS-normalized-error:"<<finalNormErrorLong<<std::endl;
	std::cout<<"LONGITUDE>>> final-avg-difference:"<<finalMeanDiffLong<<std::endl;

	finalErrorLat /= static_cast<double>(k);
	finalNormErrorLat /= static_cast<double>(k);
	finalMeanDiffLat /= static_cast<double>(k);
	std::cout<<"LATITUDE>>> final-RMS-error:"<<finalErrorLat<<std::endl;
	std::cout<<"LATITUDE>>> final-RMS-normalized-error:"<<finalNormErrorLat<<std::endl;
	std::cout<<"LATITUDE>>> final-avg-difference:"<<finalMeanDiffLat<<std::endl;
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


	// DEFINE THE FOLDERS WERE THE TEMPORARY FILES NEED TO BE STORED
	unsigned pos       = this->trainFolder.find_first_of("/\\");
	std::string root   = this->trainFolder.substr(0,pos+1);
	std::string folder = root+"trash/";
	file_exists(folder.c_str(), true);
	this->trainFolder      = root+"trash/targets.txt";
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
/** Reset the features object when the training and testing might have different
 * calibration, background models...
 */
void classifyImages::resetFeatures(std::string dir,std::string imStr,int colorSp){
	if(this->features){
		delete this->features;
		this->features = NULL;
	}
	char** args = new char*[3];
	args[0] = const_cast<char*>("featureDetector");
	args[1] = const_cast<char*>(dir.c_str());
	args[2] = const_cast<char*>(imStr.c_str());
	this->features = new featureDetector(3,args,false,true);
	this->features->setSIFTDictionary(dir+"SIFT_"+imStr+".bin");
	this->features->colorspaceCode = colorSp;
	delete [] args;
}
//==============================================================================
/** Runs the final evaluation (test).
 */
void classifyImages::runTest(double theNoise, double theLength,\
gaussianProcess::kernelFunction theKFunction, featureDetector::FEATURE theFeature,\
int colorSp, bool fromFolder, bool store){
	std::deque<gaussianProcess::prediction> predictionsSin;
	std::deque<gaussianProcess::prediction> predictionsCos;
	this->init(theNoise,theLength,theKFunction,theFeature,fromFolder,store);

  	// LONGITUDE TRAINING AND PREDICTING
	std::cout<<"Longitude >>> ______________________________________________"<<\
		"_____________________________________________________"<<std::endl;
	// BEFORE TRAINING CAMERA CALIBRATION AND OTHER SETTINGS MIGHT NEED TO BE RESET
	this->resetFeatures(this->trainDir,this->trainImgString,colorSp);
	this->modelName += ("modelLong/");
	this->trainGP(annotationsHandle::LONGITUDE);
	this->modelName = this->modelName.substr(0,this->modelName.size()-10);

	// BEFORE TESTING CAMERA CALIBRATION AND OTHER SETTINGS MIGHT NEED TO BE RESET
	this->resetFeatures(this->testDir,this->testImgString,colorSp);
	this->modelName += ("testLong/");
	this->predictGP(predictionsSin,predictionsCos,annotationsHandle::LONGITUDE);
	this->modelName = this->modelName.substr(0,this->modelName.size()-9);
	double errorLong, normErrorLong, meanDiffLong;
	this->evaluate(predictionsSin, predictionsCos, errorLong, normErrorLong,\
			meanDiffLong,annotationsHandle::LONGITUDE);
	//__________________________________________________________________________
  	// LATITUDE TRAINING AND PREDICTING
	std::cout<<"Latitude >>> _______________________________________________"<<\
		"_____________________________________________________"<<std::endl;
	// BEFORE TRAINING CAMERA CALIBRATION AND OTHER SETTINGS MIGHT NEED TO BE RESET
	this->resetFeatures(this->trainDir,this->trainImgString,colorSp);
	this->modelName += ("modelLat/");
	this->trainGP(annotationsHandle::LATITUDE);
	this->modelName = this->modelName.substr(0,this->modelName.size()-9);

	// BEFORE TESTING CAMERA CALIBRATION AND OTHER SETTINGS MIGHT NEED TO BE RESET
	this->resetFeatures(this->testDir,this->testImgString,colorSp);
	this->modelName += ("testLat/");
	this->predictGP(predictionsSin,predictionsCos,annotationsHandle::LATITUDE);
	this->modelName = this->modelName.substr(0,this->modelName.size()-8);
	double errorLat, normErrorLat,meanDiffLat;
	this->evaluate(predictionsSin, predictionsCos, errorLat, normErrorLat,\
		meanDiffLat,annotationsHandle::LATITUDE);
}
//==============================================================================
int main(int argc, char **argv){
/*
	// test
	classifyImages classi(argc, argv, classifyImages::TEST);
 	classi.runTest(1e-3,100.0,&gaussianProcess::sqexp,\
 		featureDetector::SIFT,CV_BGR2Lab,true,true);
*/


/*
	// evaluate
	classifyImages classi(argc, argv, classifyImages::EVALUATE);
  	classi.runCrossValidation(5,1e-3,100.0,&gaussianProcess::sqexp,\
 		featureDetector::IPOINTS,CV_BGR2Lab,false,true);
  	classi.runCrossValidation(5,1e-3,100.0,&gaussianProcess::sqexp,\
 		featureDetector::EDGES,CV_BGR2Lab,false,true);
  	classi.runCrossValidation(5,1e-3,100.0,&gaussianProcess::sqexp,\
 		featureDetector::GABOR,CV_BGR2Lab,false,true);
  	classi.runCrossValidation(5,1e-3,100.0,&gaussianProcess::sqexp,\
 		featureDetector::SURF,CV_BGR2Lab,false,true);
  	classi.runCrossValidation(5,1e-3,100.0,&gaussianProcess::sqexp,\
 		featureDetector::SIFT,CV_BGR2Lab,false,true);
*/


	// BUILD THE SIFT DICTIONARY
  	classifyImages classi(argc, argv, classifyImages::BUILD_DICTIONARY);
	classi.buildDictionary(CV_BGR2Lab);

}



