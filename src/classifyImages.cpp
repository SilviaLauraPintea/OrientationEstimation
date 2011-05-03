/* classifyImages.cpp
 * Author: Silvia-Laura Pintea
 */
#include "classifyImages.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <err.h>
#include <exception>
#include "Auxiliary.h"
#include "eigenbackground/src/Tracker.hh"
#include "eigenbackground/src/Helpers.hh"
#include "eigenbackground/src/defines.hh"
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
	this->feature          = featureExtractor::EDGES;
	this->features         = NULL;
	this->foldSize		   = 5;
	this->storeData 	   = true;
	this->modelName        = "";
	this->what             = use;
	this->useGroundTruth   = false;

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
				this->annotationsTrain = this->trainDir+"annotated_SIFT.txt";
				files2check.push_back(this->annotationsTrain);
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
void classifyImages::init(float theNoise, float theLength,\
featureExtractor::FEATURE theFeature, gaussianProcess::kernelFunction theKFunction,\
bool fromFolder, bool store, bool toUseGT){
	this->noise          = theNoise;
	this->length         = theLength;
	this->kFunction      = theKFunction;
	this->feature        = theFeature;
	this->readFromFolder = fromFolder;
	this->storeData      = store;
	this->useGroundTruth = toUseGT;
	switch(this->feature){
		case(featureExtractor::IPOINTS):
			this->modelName = this->trainDir+"models/"+"IPOINTS/";
			file_exists(this->modelName.c_str(), true);
			break;
		case(featureExtractor::EDGES):
			this->useGroundTruth = false;
			this->modelName      = this->trainDir+"models/"+"EDGES/";
			file_exists(this->modelName.c_str(), true);
			break;
		case(featureExtractor::SURF):
			this->modelName = this->trainDir+"models/"+"SURF/";
			file_exists(this->modelName.c_str(), true);
			break;
		case(featureExtractor::GABOR):
			this->useGroundTruth = false;
			this->modelName = this->trainDir+"models/"+"GABOR/";
			file_exists(this->modelName.c_str(), true);
			break;
		case(featureExtractor::SIFT):
			this->modelName = this->trainDir+"models/"+"SIFT/";
			file_exists(this->modelName.c_str(), true);
			break;
		case(featureExtractor::PIXELS):
			this->modelName = this->trainDir+"models/"+"PIXELS/";
			file_exists(this->modelName.c_str(), true);
			break;
	}
}
//==============================================================================
/** Creates the training data (according to the options), the labels and
 * trains the a \c GaussianProcess on the data.
 */
void classifyImages::trainGP(annotationsHandle::POSE what){
	if(!this->trainData.empty()){
		this->trainData.release();
	}
	if(!this->trainTargets.empty()){
		this->trainTargets.release();
	}
	this->gpSin.init(this->kFunction);
	this->gpCos.init(this->kFunction);

	// WE ASSUME THAT IF WE DO NOT WANT TO STORE DATA THEN WE WANT TO LOAD DATA
	std::string modelNameData   = this->modelName+"Data.bin";
	std::string modelNameLabels = this->modelName+"Labels.bin";

	if(this->storeData || !file_exists(modelNameData.c_str()) || \
	!file_exists(modelNameLabels.c_str())){
		this->features->init(this->trainFolder,this->annotationsTrain,\
			this->feature,this->readFromFolder);
		this->features->start(this->readFromFolder,this->useGroundTruth);
		this->features->data.copyTo(this->trainData);
		this->features->targets.copyTo(this->trainTargets);

		std::cout<<"SIZE: "<<this->trainData.cols<<" "<<this->trainData.rows<<std::endl;
		std::cout<<"SIZE: "<<this->trainTargets.cols<<" "<<this->trainTargets.rows<<std::endl;

		this->trainData.convertTo(this->trainData, CV_32FC1);
		this->trainTargets.convertTo(this->trainTargets, CV_32FC1);

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
	if(!this->testData.empty()){
		this->testData.release();
	}
	if(!this->testTargets.empty()){
		this->testTargets.release();
	}

	// WE ASSUME THAT IF WE DO NOT WANT TO STORE DATA THEN WE WANT TO LOAD DATA
	std::string modelNameData   = this->modelName+"Data.bin";
	std::string modelNameLabels = this->modelName+"Labels.bin";

	if(this->storeData || !file_exists(modelNameData.c_str()) || \
	!file_exists(modelNameLabels.c_str())){
		this->features->init(this->testFolder, this->annotationsTest,\
			this->feature,this->readFromFolder);
		this->features->start(this->readFromFolder,this->useGroundTruth);
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
		this->testData.convertTo(this->testData, CV_32FC1);
		this->testTargets.convertTo(this->testTargets, CV_32FC1);

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
	for(int i=0; i<this->testData.rows; i++){
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
float &error,float &normError,float &meanDiff){
	error = 0.0; normError = 0.0; meanDiff = 0.0;
	unsigned ignored = 0;
	for(int y=0; y<this->testTargets.rows; y++){
		float targetAngle = std::atan2(this->testTargets.at<float>(y,0),\
								this->testTargets.at<float>(y,1));
		float prediAngle = this->optimizePrediction(predictionsSin[y],\
							predictionsCos[y]);
		angle0to360(targetAngle);
		angle0to360(prediAngle);

		std::cout<<"Target: "<<targetAngle<<"("<<(targetAngle*180.0/M_PI)<<\
			") VS "<<prediAngle<<"("<<(prediAngle*180.0/M_PI)<<")"<<std::endl;
		float absDiff = std::abs(targetAngle-prediAngle);
		if(absDiff > M_PI){
			absDiff = 2*M_PI - absDiff;
		}
		std::cout<<"Difference: "<< absDiff <<std::endl;
		error     += absDiff*absDiff;
		normError += (absDiff*absDiff)/(M_PI*M_PI);
		meanDiff  += absDiff;
	}

	std::cout<<"Number of people: "<<this->testTargets.rows-ignored<<std::endl;
	error     = std::sqrt(error/(this->testTargets.rows-ignored));
	normError = std::sqrt(normError/(this->testTargets.rows-ignored));
	meanDiff  = meanDiff/(this->testTargets.rows-ignored);

	std::cout<<"RMS-error normalized: "<<normError<<std::endl;
	std::cout<<"RMS-accuracy normalized: "<<(1-normError)<<std::endl;
	std::cout<<"RMS-error: "<<error<<std::endl;
	std::cout<<"Avg-Radians-Difference: "<<meanDiff<<std::endl;
}
//==============================================================================
/** Try to optimize the prediction of the angle considering the variance of sin
 * and cos.
 */
float classifyImages::optimizePrediction(gaussianProcess::prediction \
predictionsSin, gaussianProcess::prediction predictionsCos){
	float y = predictionsSin.mean[0];
	float x = predictionsCos.mean[0];
	return std::atan2(y,x);

/*
	float betaS = 1.0/(predictionsSin.variance[0]);
	float betaC = 1.0/(predictionsCos.variance[0]);
	float y     = predictionsSin.mean[0];
	float x     = predictionsCos.mean[0];

	if(betaS == betaC){
		return std::atan2(betaS*y,betaC*x);
	}else{
		return std::atan2(y,x);
	}
*/
	/*
	float closeTo;
	closeTo = std::atan2(predictionsSin.mean[0],predictionsCos.mean[0]);
	std::deque<float> alphas;
	if(betaS != betaC){
		std::cout<<"betaS="<<betaS<<" betaC="<<betaC<<" x="<<x<<" y="<<y<<std::endl;

		float b = -1.0*(betaS*x + betaC*y + betaS - betaC);
		float a = betaS - betaC;
		float c = betaS*x;

		std::cout<<"b="<<b<<" a="<<a<<" c="<<c<<std::endl;
		std::cout<<"alpha1: "<<((-b + std::sqrt(b*b - 4.0*a*c))/2.0*a)<<std::endl;
		std::cout<<"alpha2: "<<((-b - std::sqrt(b*b - 4.0*a*c))/2.0*a)<<std::endl;

		alphas.push_back((-b + std::sqrt(b*b - 4.0*a*c))/2.0*a);
		alphas.push_back((-b - std::sqrt(b*b - 4.0*a*c))/2.0*a);
	}else{
		std::cout<<"alpha1: "<<(betaS*x/(betaS*x + betaC*y))<<std::endl;
		alphas.push_back(betaS*x/(betaS*x + betaC*y));
	}
	float minDist = 2.0*M_PI, minAngle;
	for(unsigned i=0; i<alphas.size(); i++){
		if(alphas[i]>=0){
			float alpha1 = std::asin(std::sqrt(alphas[i]));
			float alpha2 = std::asin(-std::sqrt(alphas[i]));
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
void classifyImages::buildDictionary(int colorSp, bool toUseGT){
	// SET THE CALIBRATION AND OTHER FEATURE SETTINGS
	this->resetFeatures(this->trainDir, this->trainImgString, colorSp);

	// EXTRACT THE SIFT FEATURES AND CONCATENATE THEM
	this->features->init(this->trainFolder,this->annotationsTrain,\
		featureExtractor::SIFT_DICT,this->readFromFolder);
	this->features->start(this->readFromFolder, toUseGT);
	cv::Mat dictData;
	this->features->data.copyTo(dictData);

	// DO K-means IN ORDER TO RETRIEVE BACK THE CLUSTER MEANS
	cv::Mat labels = cv::Mat::zeros(cv::Size(1,dictData.rows),CV_32FC1);
	//LABEL EACH SAMPLE ASSIGNMENT
	cv::Mat* centers = new cv::Mat(cv::Size(dictData.cols,\
						this->features->extractor->readNoMeans()),CV_32FC1);
	dictData.convertTo(dictData, CV_32FC1);
	cv::kmeans(dictData,this->features->extractor->readNoMeans(),labels,\
		cv::TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,2,1),\
		5,cv::KMEANS_RANDOM_CENTERS,centers);
	dictData.release();
	labels.release();

	// WRITE TO FILE THE MEANS
	cv::Mat matrix(*centers);
	mat2BinFile(matrix, const_cast<char*>(this->features->extractor->readDictName().c_str()));
	centers->release();
	matrix.release();
	delete centers;
}
//==============================================================================
/** Does the cross-validation and computes the average error over all folds.
 */
float classifyImages::runCrossValidation(unsigned k, int colorSp, bool onTrain){
	this->readFromFolder = false;
	float finalErrorLong=0.0, finalNormErrorLong=0.0;
	float finalErrorLat=0.0, finalNormErrorLat=0.0;
	float finalMeanDiffLat=0.0, finalMeanDiffLong=0.0;

	// SET THE CALIBRATION ONLY ONCE (ALL IMAGES ARE READ FROM THE SAME DIR)
	this->resetFeatures(this->trainDir, this->trainImgString, colorSp);
	for(unsigned i=0; i<k; i++){
		std::cout<<"Round "<<i<<"___________________________________________"<<\
			"_____________________________________________________"<<std::endl;
		// SPLIT TRAINING AND TESTING ACCORDING TO THE CURRENT FOLD
		std::deque<gaussianProcess::prediction> predictionsSin;
		std::deque<gaussianProcess::prediction> predictionsCos;
		this->crossValidation(k,i,onTrain);
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
		float errorLong, normErrorLong, meanDiffLong;
		this->evaluate(predictionsSin, predictionsCos, errorLong, normErrorLong,\
				meanDiffLong);
		finalErrorLong += errorLong;
		finalNormErrorLong += normErrorLong;
		finalMeanDiffLong += meanDiffLong;
		predictionsSin.clear();
		predictionsCos.clear();

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
		float errorLat, normErrorLat, meanDiffLat;
		this->evaluate(predictionsSin, predictionsCos, errorLat, normErrorLat,\
			meanDiffLat);
		finalErrorLat += errorLat;
		finalNormErrorLat += normErrorLat;
		finalMeanDiffLat += meanDiffLat;
		predictionsSin.clear();
		predictionsCos.clear();
		*/
	}
	finalErrorLong /= static_cast<float>(k);
	finalNormErrorLong /= static_cast<float>(k);
	finalMeanDiffLong /= static_cast<float>(k);
	std::cout<<"LONGITUDE>>> final-RMS-error:"<<finalErrorLong<<std::endl;
	std::cout<<"LONGITUDE>>> final-RMS-normalized-error:"<<finalNormErrorLong<<std::endl;
	std::cout<<"LONGITUDE>>> final-avg-difference:"<<finalMeanDiffLong<<std::endl;

	finalErrorLat /= static_cast<float>(k);
	finalNormErrorLat /= static_cast<float>(k);
	finalMeanDiffLat /= static_cast<float>(k);
	std::cout<<"LATITUDE>>> final-RMS-error:"<<finalErrorLat<<std::endl;
	std::cout<<"LATITUDE>>> final-RMS-normalized-error:"<<finalNormErrorLat<<std::endl;
	std::cout<<"LATITUDE>>> final-avg-difference:"<<finalMeanDiffLat<<std::endl;
}
//==============================================================================
/** Do k-fold cross-validation by splitting the training folder into training-set
 * and validation-set.
 */
void classifyImages::crossValidation(unsigned k, unsigned fold, bool onTrain){
	// READ ALL IMAGES ONCE AND NOT THEY ARE SORTED
	if(this->imageList.empty()){
		this->imageList = readImages(this->trainFolder.c_str());
		this->foldSize  = this->imageList.size()/k;

		std::ifstream annoIn(this->annotationsTrain.c_str());
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
	this->annotationsTrain = root+"trash/annoTargets.txt";
	this->testFolder       = root+"trash/ttargets.txt";
	this->annotationsTest  = root+"trash/annoTtargets.txt";

	// WRITE THE IMAGE NAMES & ANNOTATIONS IN THE CORRESPONDING FILES
	std::ofstream testOut, trainOut, annoTest, annoTrain;
	testOut.open(this->testFolder.c_str(), std::ios::out);
	if(!testOut){
		errx(1,"Cannot open file %s", this->testFolder.c_str());
	}
	trainOut.open(this->trainFolder.c_str(), std::ios::out);
	if(!trainOut){
		errx(1,"Cannot open file %s", this->trainFolder.c_str());
	}

	annoTest.open(this->annotationsTest.c_str(), std::ios::out);
	if(!annoTest){
		errx(1,"Cannot open file %s", this->annotationsTest.c_str());
	}
	annoTrain.open(this->annotationsTrain.c_str(), std::ios::out);
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
	if(onTrain){
		this->testFolder      = root+"trash/targets.txt";
		this->annotationsTest = root+"trash/annoTargets.txt";
	}
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
	args[0] = const_cast<char*>("peopleDetector");
	args[1] = const_cast<char*>(dir.c_str());
	args[2] = const_cast<char*>(imStr.c_str());
	this->features = new peopleDetector(3,args,false,true);
	this->features->colorspaceCode = colorSp;
	delete [] args;
}
//==============================================================================
/** Runs the final evaluation (test).
 */
void classifyImages::runTest(int colorSp){
	std::deque<gaussianProcess::prediction> predictionsSin;
	std::deque<gaussianProcess::prediction> predictionsCos;

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
	float errorLong, normErrorLong, meanDiffLong;
	this->evaluate(predictionsSin, predictionsCos, errorLong, normErrorLong,\
			meanDiffLong);
	predictionsSin.clear();
	predictionsCos.clear();

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
	float errorLat, normErrorLat,meanDiffLat;
	this->evaluate(predictionsSin, predictionsCos, errorLat, normErrorLat,\
		meanDiffLat);
}
//==============================================================================
int main(int argc, char **argv){
/*
	// test
	classifyImages classi(argc, argv, classifyImages::TEST);
	classi.init(1e-3,70.0,featureExtractor::EDGES,&gaussianProcess::sqexp,\
		true, true, false);
 	classi.runTest(CV_BGR2Luv);
*/

  	// evaluate
 	classifyImages classi(argc, argv, classifyImages::EVALUATE);
	classi.init(0.1,50.0,featureExtractor::EDGES,&gaussianProcess::sqexp,\
			false, true, true);
	classi.runCrossValidation(5,CV_BGR2XYZ,false);


/*
  	std::ofstream train, test;
	train.open("train.txt", std::ios::out);
	test.open("test.txt", std::ios::out);
	classifyImages classi(argc, argv, classifyImages::EVALUATE);
	for(float v=1.8; v<5; v+=0.2){
		for(float l=65; l<125; l+=10){
			// evaluate
			classi.init(v,l,featureExtractor::EDGES,&gaussianProcess::sqexp,\
					false, true, true);
			float errorTrain = classi.runCrossValidation(5,CV_BGR2Luv,true);
			train<<v<<" "<<l<<" "<<errorTrain<<std::endl;

			classi.init(v,l,featureExtractor::EDGES,&gaussianProcess::sqexp,\
					false, true, true);
			float errorTest = classi.runCrossValidation(5,CV_BGR2Luv,false);
			test<<v<<" "<<l<<" "<<errorTest<<std::endl;
		}
	}
	train.close();
	test.close();
*/

/*
	// BUILD THE SIFT DICTIONARY
  	classifyImages classi(argc, argv, classifyImages::BUILD_DICTIONARY);
	classi.buildDictionary(CV_BGR2Luv, true);
*/
}



