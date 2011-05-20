/* classifyImages.cpp
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <err.h>
#include <exception>
#include "Auxiliary.h"
#include "eigenbackground/src/Tracker.hh"
#include "classifyImages.h"
//==============================================================================
classifyImages::classifyImages(int argc,char **argv,classifyImages::USES use){
	// DEAFAULT INITIALIZATION
	this->trainFolder      = "";
	this->testFolder       = "";
	this->annotationsTrain = "";
	this->annotationsTest  = "";
	this->noise            = 0.01;
	this->length           = 1.0;
	this->kFunction        = &gaussianProcess::sqexp;
	this->feature          = featureExtractor::EDGES;
	this->features         = NULL;
	this->foldSize		   = 5;
	this->modelName        = "";
	this->what             = use;
	this->useGroundTruth   = false;

	// INITIALIZE THE DATA MATRIX AND TARGETS MATRIX AND THE GAUSSIAN PROCESSES
	for(unsigned i=0;i<3;++i){
		this->trainData.push_back(cv::Mat());
		this->trainTargets.push_back(cv::Mat());
		this->testData.push_back(cv::Mat());
		this->testTargets.push_back(cv::Mat());

		this->gpSin.push_back(gaussianProcess());
		this->gpCos.push_back(gaussianProcess());
	}

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
		for(std::size_t i=0;i<files2check.size();++i){
			if(!Helpers::file_exists(files2check[i].c_str())){
				std::cerr<<"File/folder not found: "<<files2check[i]<<std::endl;
				exit(1);
			}
		}

		if(use == classifyImages::TEST){
			this->modelName = "data/TEST/";
		}else if(use == classifyImages::EVALUATE){
			this->modelName = "data/EVALUATE/";
		}
		Helpers::file_exists(this->modelName.c_str(),true);
	}
}
//==============================================================================
classifyImages::~classifyImages(){
	if(this->features){
		delete this->features;
		this->features = NULL;
	}

	// LOOP ONLY ONCE (ALL HAVE 3 CLASSES)
	for(std::size_t i=0;i<this->trainData.size();++i){
		if(!this->trainData[i].empty()){
			this->trainData[i].release();
		}
		if(!this->testData[i].empty()){
			this->testData[i].release();
		}
		if(!this->trainTargets[i].empty()){
			this->trainTargets[i].release();
		}
		if(!this->testTargets[i].empty()){
			this->testTargets[i].release();
		}
	}
	this->trainData.clear();
	this->trainTargets.clear();
	this->testData.clear();
	this->testTargets.clear();

	this->gpSin.clear();
	this->gpCos.clear();
}
//==============================================================================
/** Initialize the options for the Gaussian Process regression.
 */
void classifyImages::init(float theNoise,float theLength,\
featureExtractor::FEATURE theFeature,gaussianProcess::kernelFunction theKFunction,\
bool toUseGT){
	this->noise          = theNoise;
	this->length         = theLength;
	this->kFunction      = theKFunction;
	this->feature        = theFeature;
	this->useGroundTruth = toUseGT;
	switch(this->feature){
		case(featureExtractor::IPOINTS):
			this->modelName += "IPOINTS/";
			Helpers::file_exists(this->modelName.c_str(),true);
			break;
		case(featureExtractor::EDGES):
			this->useGroundTruth = false;
			this->modelName     += "EDGES/";
			Helpers::file_exists(this->modelName.c_str(),true);
			break;
		case(featureExtractor::SURF):
			this->modelName += "SURF/";
			Helpers::file_exists(this->modelName.c_str(),true);
			break;
		case(featureExtractor::GABOR):
			this->useGroundTruth = false;
			this->modelName += "GABOR/";
			Helpers::file_exists(this->modelName.c_str(),true);
			break;
		case(featureExtractor::SIFT):
			this->modelName += "SIFT/";
			Helpers::file_exists(this->modelName.c_str(),true);
			break;
		case(featureExtractor::PIXELS):
			this->modelName += "PIXELS/";
			Helpers::file_exists(this->modelName.c_str(),true);
			break;
		case(featureExtractor::HOG):
			this->modelName += "HOG/";
			Helpers::file_exists(this->modelName.c_str(),true);
			break;
	}
}
//==============================================================================
/** Concatenate the loaded data from the files to the currently computed data.
 */
void classifyImages::loadData(const cv::Mat &tmpData1,const cv::Mat &tmpTargets1,\
unsigned i){
	if(!this->trainData[i].empty()){
		this->trainData[i].release();
	}
	if(!this->trainTargets[i].empty()){
		this->trainTargets[i].release();
	}

	// LOAD THE DATA AND TARGET MATRIX FROM THE FILE IF IT'S THERE
	std::deque<std::string> names;
	names.push_back("CLOSE");names.push_back("MEDIUM");	names.push_back("FAR");

	cv::Mat tmpData2,tmpTargets2;
	std::string modelDataName    = this->modelName+names[i]+"/Data.bin";
	std::string modelTargetsName = this->modelName+names[i]+"/Labels.bin";
	if(Helpers::file_exists(modelDataName.c_str())){
		Auxiliary::binFile2mat(tmpData2,const_cast<char*>(modelDataName.c_str()));
	}else{
		tmpData2 = cv::Mat();
	}
	if(Helpers::file_exists(modelTargetsName.c_str())){
		Auxiliary::binFile2mat(tmpTargets2,const_cast<char*>(modelTargetsName.c_str()));
	}else{
		tmpTargets2 = cv::Mat();
	}

	// NOW COPY THEM TOGETHER INTO A FINAL MATRIX AND STORE IT.
	if(!tmpData1.empty() && !tmpData2.empty()){
		if(tmpData1.cols!=tmpData2.cols){
			std::cerr<<"The sizes of the stored data matrix and the newly generated "\
				<<"data matrix do not agree: "<<tmpData1.size()<<" VS. "<<\
				tmpData2.size()<<std::endl;
			exit(1);
		}
	}
	// IF ONE OF THE MATRIXES IS EMPTY USE THE OTHER ONE TO GET THE COLS
	cv::Mat dumData1,dumData2,dumTargets1,dumTargets2;
	unsigned colsData     = std::max(tmpData2.cols,tmpData1.cols);
	unsigned colsTargets  = std::max(tmpTargets2.cols,tmpTargets1.cols);
	this->trainData[i]    = cv::Mat::zeros(cv::Size(colsData,tmpData1.rows+\
		tmpData2.rows),CV_32FC1);
	this->trainTargets[i] = cv::Mat::zeros(cv::Size(colsTargets,tmpTargets1.rows+\
		tmpTargets2.rows),CV_32FC1);

	// COPY DATA1 AND TARGETS1 TO THE DATA MATRIX
	if(!tmpData1.empty() && !tmpTargets1.empty()){
		dumData1 = this->trainData[i].rowRange(0,tmpData1.rows);
		tmpData1.copyTo(dumData1);
		dumTargets1 = this->trainTargets[i].rowRange(0,tmpTargets1.rows);
		tmpTargets1.copyTo(dumTargets1);
	}

	// COPY DATA2 AND TARGETS2 TO THE DATA MATRIX
	if(!tmpData2.empty() && !tmpTargets2.empty()){
		dumData2 = this->trainData[i].rowRange(tmpData1.rows,tmpData1.rows+\
			tmpData2.rows);
		tmpData2.copyTo(dumData2);
		dumTargets2 = this->trainTargets[i].rowRange(tmpTargets1.rows,\
			tmpTargets1.rows+tmpTargets2.rows);
		tmpTargets2.copyTo(dumTargets2);
	}
	this->trainData[i].convertTo(this->trainData[i],CV_32FC1);
	this->trainTargets[i].convertTo(this->trainTargets[i],CV_32FC1);
	std::cout<<names[i]<<" SIZE: "<<this->trainData[i].size()<<"=("<<\
		tmpData1.size()<<"+"<<tmpData2.size()<<")"<<std::endl;
	std::cout<<names[i]<<" SIZE: "<<this->trainTargets.size()<<"=("<<\
		tmpTargets1.size()<<"+"<<tmpTargets2.size()<<")"<<std::endl;

	// RELEASE THE BILLION OF ALLOCATED MATRIXES
	dumData1.release();
	dumData2.release();
	dumTargets1.release();
	dumTargets2.release();
	tmpData2.release();
	tmpTargets2.release();
}
//==============================================================================
/** Creates the training data (according to the options),the labels and
 * trains the a \c GaussianProcess on the data.
 */
void classifyImages::trainGP(annotationsHandle::POSE what,bool fromFolder){
	// TRAIN ALL 3 CLASSES WITH DATA THAT WE HAVE,PREDICT ONLY ON THE GOOD CLASS
	this->features->init(this->trainFolder,this->annotationsTrain,this->feature,\
		fromFolder);
	this->features->start(fromFolder,this->useGroundTruth);

	std::deque<std::string> names;
	names.push_back("CLOSE");names.push_back("MEDIUM");	names.push_back("FAR");
	for(peopleDetector::CLASSES i=peopleDetector::CLOSE;i<=peopleDetector::FAR;++i){
		this->gpSin[i].init(this->kFunction);
		this->gpCos[i].init(this->kFunction);

		// IF WE CANNOT LOAD DATA,THEN WE BUILD IT
		std::string modelNameData   = this->modelName+"Data.bin";
		std::string modelNameLabels = this->modelName+"Labels.bin";

		cv::Mat tmpData,tmpTargets;
		this->features->data[i].copyTo(tmpData);
		this->features->targets[i].copyTo(tmpTargets);
		tmpData.convertTo(tmpData,CV_32FC1);
		tmpTargets.convertTo(tmpTargets,CV_32FC1);
		this->loadData(tmpData,tmpTargets,i);

		// CHECK TO SEE IF THERE IS ANY DATA IN THE CURRENT CLASS
		if(this->features->data[i].empty()) continue;

		// TRAIN THE SIN AND COS SEPARETELY FOR LONGITUDE || LATITUDe
		if(what == annotationsHandle::LONGITUDE){
			this->gpSin[i].train(this->trainData[i],this->trainTargets[i].col(0),\
				this->kFunction,this->noise,this->length);
			this->gpCos[i].train(this->trainData[i],this->trainTargets[i].col(1),\
				this->kFunction,this->noise,this->length);
		}else if(what == annotationsHandle::LATITUDE){
			// TRAIN THE SIN AND COS SEPARETELY FOR LATITUDE
			this->gpSin[i].train(this->trainData[i],this->trainTargets[i].col(2),\
				this->kFunction,this->noise,this->length);
			this->gpCos[i].train(this->trainData[i],this->trainTargets[i].col(3),\
				this->kFunction,this->noise,this->length);
		}
		tmpData.release();
		tmpTargets.release();
	}
}
//==============================================================================
/** Just build data matrix and store it;it can be called over multiple datasets
 * by adding the the new data rows at the end to the stored matrix.
 */
void classifyImages::buildDataMatrix(int colorSp){
	// SET THE CALIBRATION AND OTHER FEATURE SETTINGS
	this->resetFeatures(this->trainDir,this->trainImgString,colorSp);
	for(std::size_t i=0;i<this->trainData.size();++i){
		if(!this->trainData[i].empty()){
			this->trainData[i].release();
		}
		if(!this->trainTargets[i].empty()){
			this->trainTargets[i].release();
		}
	}

	this->features->init(this->trainFolder,this->annotationsTrain,\
		this->feature,true);
	this->features->start(true,this->useGroundTruth);
	std::deque<std::string> names;
	names.push_back("CLOSE");names.push_back("MEDIUM");	names.push_back("FAR");
	for(peopleDetector::CLASSES i=peopleDetector::CLOSE;i<=peopleDetector::FAR;++i){
		// CHECK TO SEE IF THE FOLDER IS ALREADY CREATED
		Helpers::file_exists((this->modelName+names[i]).c_str(),true);
		std::string modelNameData   = this->modelName+names[i]+"/Data.bin";
		std::string modelNameLabels = this->modelName+names[i]+"/Labels.bin";

		// LOAD THE DATA AND TARGET MATRIX FROM THE FILE IF IT'S THERE
		cv::Mat tmpData1,tmpTargets1,tmpData2,tmpTargets2;
		if(Helpers::file_exists(modelNameData.c_str())){
			Auxiliary::binFile2mat(tmpData1,const_cast<char*>(modelNameData.c_str()));
		}else{
			tmpData1 = cv::Mat();
		}
		if(Helpers::file_exists(modelNameLabels.c_str())){
			Auxiliary::binFile2mat(tmpTargets1,const_cast<char*>(modelNameLabels.c_str()));
		}else{
			tmpTargets1 = cv::Mat();
		}

		// NOW CREATE NEW DATA AND TARGETS AND ADD THEM TO THE OLD ONES
		this->features->data[i].copyTo(tmpData2);
		this->features->targets[i].copyTo(tmpTargets2);

		// NOW COPY THEM TOGETHER INTO A FINAL MATRIX AND STORE IT.
		if(!tmpData1.empty() && !tmpData2.empty()){
			if(tmpData1.cols!=tmpData2.cols){
				std::cerr<<"The sizes of the stored data matrix and the newly generated "\
					<<"data matrix do not agree: "<<tmpData1.size()<<" VS. "<<\
					tmpData2.size()<<std::endl;
				exit(1);
			}
		}
		cv::Mat dumData1,dumData2,dumTargets1,dumTargets2;
		this->trainData[i]    = cv::Mat::zeros(cv::Size(tmpData2.cols,tmpData1.rows+\
			tmpData2.rows),CV_32FC1);
		this->trainTargets[i] = cv::Mat::zeros(cv::Size(tmpTargets2.cols,\
			tmpTargets1.rows+tmpTargets2.rows),CV_32FC1);
		if(!tmpData1.empty()){
			dumData1 = this->trainData[i].rowRange(0,tmpData1.rows);
			tmpData1.copyTo(dumData1);
			dumTargets1 = this->trainTargets[i].rowRange(0,tmpTargets1.rows);
			tmpTargets1.copyTo(dumTargets1);
		}
		dumData2 = this->trainData[i].rowRange(tmpData1.rows,tmpData1.rows+tmpData2.rows);
		tmpData2.copyTo(dumData2);
		dumTargets2 = this->trainTargets[i].rowRange(tmpTargets1.rows,tmpTargets1.rows+\
			tmpTargets2.rows);
		tmpTargets2.copyTo(dumTargets2);

		// WRITE THE FINAL MATRIX TO THE FILES
		if(!this->trainData[i].empty()){
			Auxiliary::mat2BinFile(this->trainData[i],const_cast<char*>\
				(modelNameData.c_str()),false);
			Auxiliary::mat2BinFile(this->trainTargets[i],const_cast<char*>\
				(modelNameLabels.c_str()),false);
			std::cout<<"Data size: "<<this->trainData[i].size()<<std::endl;
			std::cout<<"Labels size: "<<this->trainTargets[i].size()<<std::endl;
			std::cout<<"Data stored to: "<<modelNameData<<" and "<<\
				modelNameLabels<<std::endl;
		}
		// RELEASE THE BILLION OF ALLOCATED MATRIXES
		dumData1.release();
		dumData2.release();
		dumTargets1.release();
		dumTargets2.release();
		tmpData1.release();
		tmpTargets1.release();
		tmpData2.release();
		tmpTargets2.release();
	}
}
//==============================================================================
/** Creates the test data and applies \c GaussianProcess prediction on the test
 * data.
 */
std::deque<std::deque<float> > classifyImages::predictGP\
(std::deque<gaussianProcess::prediction> &predictionsSin,\
std::deque<gaussianProcess::prediction> &predictionsCos,\
annotationsHandle::POSE what,bool fromFolder){
	for(std::size_t i=0;i<this->testData.size();++i){
		if(!this->testData[i].empty()){
			this->testData[i].release();
		}
		if(!this->testTargets[i].empty()){
			this->testTargets[i].release();
		}
	}
	this->features->init(this->testFolder,this->annotationsTest,\
		this->feature,fromFolder);
	this->features->start(fromFolder,this->useGroundTruth);

	// FOR TESTING WE ALWAYS BUILT THE DATA (THERE IS NOT SAVED MODEL)
	std::deque<std::string> names;
	names.push_back("CLOSE");names.push_back("MEDIUM");	names.push_back("FAR");
	std::deque<std::deque<float> > predictions;
	for(peopleDetector::CLASSES i=peopleDetector::CLOSE;i<=peopleDetector::FAR;++i){
		// CHECK TO SEE IF THERE IS ANY DATA IN THE CURRENT CLASS
		std::deque<float> oneClassPredictions;
		if(this->features->data[i].empty() || !this->gpSin[i].N || !this->gpCos[i].N){
			predictions.push_back(oneClassPredictions);
			continue;
		}
		this->features->data[i].copyTo(this->testData[i]);

		// GET ONLY THE ANGLES YOU NEED
		if(what == annotationsHandle::LONGITUDE){
			cv::Mat dum = this->features->targets[i].colRange(0,2);
			dum.copyTo(this->testTargets[i]);
			dum.release();
		}else if(what == annotationsHandle::LATITUDE){
			cv::Mat dum = this->features->targets[i].colRange(2,4);
			dum.copyTo(this->testTargets[i]);
			dum.release();
		}
		this->testData[i].convertTo(this->testData[i],CV_32FC1);
		this->testTargets[i].convertTo(this->testTargets[i],CV_32FC1);

		// FOR EACH ROW IN THE TEST MATRIX PREDICT
		for(int j=0;j<this->testData[i].rows;++j){
			gaussianProcess::prediction prediSin;
			this->gpSin[i].predict(this->testData[i].row(j),prediSin,this->length);
			gaussianProcess::prediction prediCos;
			this->gpCos[i].predict(this->testData[i].row(j),prediCos,this->length);
			predictionsSin.push_back(prediSin);
			predictionsCos.push_back(prediCos);
			oneClassPredictions.push_back(this->optimizePrediction(prediSin,prediCos));
			prediSin.mean.clear();
			prediSin.variance.clear();
			prediCos.mean.clear();
			prediCos.variance.clear();
		}
		predictions.push_back(oneClassPredictions);
	}
	return predictions;
}
//==============================================================================
/** Evaluate one prediction versus its target.
 */
void classifyImages::evaluate(const std::deque<std::deque<float> > &prediAngles,\
float &error,float &normError,float &meanDiff){
	error = 0.0;normError = 0.0;meanDiff = 0.0;
	unsigned noPeople = 0;
	std::deque<std::string> names;
	names.push_back("CLOSE");names.push_back("MEDIUM");	names.push_back("FAR");
	for(peopleDetector::CLASSES i=peopleDetector::CLOSE;i<=peopleDetector::FAR;++i){
		std::cout<<"Class "<<names[i]<<": "<<this->testTargets[i].size()<<\
			" people"<<std::endl;
		assert(this->testTargets[i].rows == prediAngles[i].size());
		for(int y=0;y<this->testTargets[i].rows;++y){
			float targetAngle = std::atan2(this->testTargets[i].at<float>(y,0),\
									this->testTargets[i].at<float>(y,1));
			Auxiliary::angle0to360(targetAngle);

			std::cout<<"Target: "<<targetAngle<<"("<<(targetAngle*180.0/M_PI)<<\
				") VS "<<prediAngles[i][y]<<"("<<(prediAngles[i][y]*180.0/M_PI)<<\
				")"<<std::endl;
			float absDiff = std::abs(targetAngle-prediAngles[i][y]);
			if(absDiff > M_PI){
				absDiff = 2*M_PI - absDiff;
			}
			std::cout<<"Difference: "<< absDiff <<std::endl;
			error     += absDiff*absDiff;
			normError += (absDiff*absDiff)/(M_PI*M_PI);
			meanDiff  += absDiff;
		}
		noPeople += this->testTargets[i].rows;
	}
	std::cout<<"Number of people: "<<noPeople<<std::endl;
	error     = std::sqrt(error/(noPeople));
	normError = std::sqrt(normError/(noPeople));
	meanDiff  = meanDiff/(noPeople);

	std::cout<<"RMS-error normalized: "<<normError<<std::endl;
	std::cout<<"RMS-accuracy normalized: "<<(1-normError)<<std::endl;
	std::cout<<"RMS-error: "<<error<<std::endl;
	std::cout<<"Avg-Radians-Difference: "<<meanDiff<<std::endl;
}
//==============================================================================
/** Try to optimize the prediction of the angle considering the variance of sin
 * and cos.
 */
float classifyImages::optimizePrediction(const gaussianProcess::prediction \
&predictionsSin,const gaussianProcess::prediction &predictionsCos){
	float y          = predictionsSin.mean[0];
	float x          = predictionsCos.mean[0];
	float prediction = std::atan2(y,x);
	Auxiliary::angle0to360(prediction);
	return prediction;
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
	float minDist = 2.0*M_PI,minAngle;
	for(unsigned i=0;i<alphas.size();++i){
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
void classifyImages::buildDictionary(int colorSp,bool toUseGT){
	// SET THE CALIBRATION AND OTHER FEATURE SETTINGS
	this->resetFeatures(this->trainDir,this->trainImgString,colorSp);

	// EXTRACT THE SIFT FEATURES AND CONCATENATE THEM
	this->features->init(this->trainFolder,this->annotationsTrain,\
		featureExtractor::SIFT_DICT,true);
	this->features->start(true,toUseGT);

	std::deque<std::string> names;
	names.push_back("CLOSE");names.push_back("MEDIUM");	names.push_back("FAR");
	for(peopleDetector::CLASSES i=peopleDetector::CLOSE;i<=peopleDetector::FAR;++i){
		if(this->features->data[i].empty()) continue;
		cv::Mat dictData;
		this->features->data[i].copyTo(dictData);
		this->features->extractor->setImageClass(static_cast<unsigned>(i));

		// DO K-means IN ORDER TO RETRIEVE BACK THE CLUSTER MEANS
		cv::Mat labels = cv::Mat::zeros(cv::Size(1,dictData.rows),CV_32FC1);

		//LABEL EACH SAMPLE ASSIGNMENT
		cv::Mat* centers = new cv::Mat(cv::Size(dictData.cols,\
			this->features->extractor->readNoMeans()),CV_32FC1);
		dictData.convertTo(dictData,CV_32FC1);
		cv::kmeans(dictData,this->features->extractor->readNoMeans(),labels,\
			cv::TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS,2,1),\
			5,cv::KMEANS_RANDOM_CENTERS,centers);
		dictData.release();
		labels.release();

		// WRITE TO FILE THE MEANS
		cv::Mat matrix(*centers);
		std::string dictName = this->features->extractor->readDictName();
		std::cout<<"Size("<<names[i]<<"): "<<this->features->data[i].size()<<\
			" stored in: "<<dictName<<std::endl;

		Auxiliary::mat2BinFile(matrix,const_cast<char*>(dictName.c_str()));
		centers->release();
		matrix.release();
		delete centers;
	}
}
//==============================================================================
/** Does the cross-validation and computes the average error over all folds.
 */
float classifyImages::runCrossValidation(unsigned k,annotationsHandle::POSE what,\
int colorSp,bool onTrain){
	float finalError=0.0,finalNormError=0.0,finalMeanDiff=0.0;

	// SET THE CALIBRATION ONLY ONCE (ALL IMAGES ARE READ FROM THE SAME DIR)
	this->resetFeatures(this->trainDir,this->trainImgString,colorSp);
	for(unsigned i=0;i<k;++i){
		std::cout<<"Round "<<i<<"___________________________________________"<<\
			"_____________________________________________________"<<std::endl;
		// SPLIT TRAINING AND TESTING ACCORDING TO THE CURRENT FOLD
		std::deque<gaussianProcess::prediction> predictionsSin;
		std::deque<gaussianProcess::prediction> predictionsCos;
		std::deque<std::deque<float> > predicted;
		this->crossValidation(k,i,onTrain);
		//______________________________________________________________________
		if(what == annotationsHandle::LONGITUDE){
			//LONGITUDE TRAINING AND PREDICTING
			std::cout<<"Longitude >>> "<<i<<"___________________________________"<<\
				"_____________________________________________________"<<std::endl;
			this->trainGP(annotationsHandle::LONGITUDE,false);

			// PREDICT ON THE REST OF THE IMAGES
			predicted = this->predictGP(predictionsSin,predictionsCos,\
				annotationsHandle::LONGITUDE,false);

			// EVALUATE PREDICITONS
			float errorLong,normErrorLong,meanDiffLong;
			this->evaluate(predicted,errorLong,normErrorLong,meanDiffLong);
			finalError     += errorLong;
			finalNormError += normErrorLong;
			finalMeanDiff  += meanDiffLong;
			predictionsSin.clear();
			predictionsCos.clear();
			predicted.clear();
		//______________________________________________________________________
		}else if(what == annotationsHandle::LATITUDE){
			//LATITUDE TRAINING AND PREDICTING
			std::cout<<"Latitude >>> "<<i<<"____________________________________"<<\
				"_____________________________________________________"<<std::endl;
			this->trainGP(annotationsHandle::LATITUDE,false);

			// PREDICT ON THE REST OF THE IMAGES
			predicted = this->predictGP(predictionsSin,predictionsCos,\
				annotationsHandle::LATITUDE,false);

			// EVALUATE PREDICITONS
			float errorLat,normErrorLat,meanDiffLat;
			this->evaluate(predicted,errorLat,normErrorLat,meanDiffLat);
			finalError     += errorLat;
			finalNormError += normErrorLat;
			finalMeanDiff  += meanDiffLat;
			predictionsSin.clear();
			predictionsCos.clear();
			predicted.clear();
		}
		sleep(6);
	}
	finalError     /= static_cast<float>(k);
	finalNormError /= static_cast<float>(k);
	finalMeanDiff  /= static_cast<float>(k);
	std::cout<<">>> final-RMS-error:"<<finalError<<std::endl;
	std::cout<<">>> final-RMS-normalized-error:"<<finalNormError<<std::endl;
	std::cout<<">>> final-avg-difference:"<<finalMeanDiff<<std::endl;
	return finalNormError;
}
//==============================================================================
/** Do k-fold cross-validation by splitting the training folder into training-set
 * and validation-set.
 */
void classifyImages::crossValidation(unsigned k,unsigned fold,bool onTrain){
	// READ ALL IMAGES ONCE AND NOT THEY ARE SORTED
	if(this->imageList.empty()){
		this->imageList = Helpers::readImages(this->trainFolder.c_str());
		this->foldSize  = this->imageList.size()/k;

		std::ifstream annoIn(this->annotationsTrain.c_str());
		if(annoIn.is_open()){
			while(annoIn.good()){
				std::string line;
				std::getline(annoIn,line);
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
	Helpers::file_exists(folder.c_str(),true);
	this->trainFolder      = root+"trash/targets.txt";
	this->annotationsTrain = root+"trash/annoTargets.txt";
	this->testFolder       = root+"trash/ttargets.txt";
	this->annotationsTest  = root+"trash/annoTtargets.txt";

	// WRITE THE IMAGE NAMES & ANNOTATIONS IN THE CORRESPONDING FILES
	std::ofstream testOut,trainOut,annoTest,annoTrain;
	testOut.open(this->testFolder.c_str(),std::ios::out);
	if(!testOut){
		errx(1,"Cannot open file %s",this->testFolder.c_str());
	}
	trainOut.open(this->trainFolder.c_str(),std::ios::out);
	if(!trainOut){
		errx(1,"Cannot open file %s",this->trainFolder.c_str());
	}

	annoTest.open(this->annotationsTest.c_str(),std::ios::out);
	if(!annoTest){
		errx(1,"Cannot open file %s",this->annotationsTest.c_str());
	}
	annoTrain.open(this->annotationsTrain.c_str(),std::ios::out);
	if(!annoTrain){
		errx(1,"Cannot open file %s",this->annotationsTrain.c_str());
	}
	for(unsigned i=0;i<this->imageList.size();++i){
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
 * calibration,background models...
 */
void classifyImages::resetFeatures(const std::string &dir,const std::string &imStr,\
int colorSp){
	if(this->features){
		delete this->features;
		this->features = NULL;
	}
	char** args = new char*[3];
	args[0] = const_cast<char*>("peopleDetector");
	args[1] = const_cast<char*>(dir.c_str());
	args[2] = const_cast<char*>(imStr.c_str());
	this->features = new peopleDetector(3,args,false,false,colorSp);
	delete [] args;
}
//==============================================================================
/** Runs the final evaluation (test).
 */
std::deque<std::deque<float> > classifyImages::runTest(int colorSp,\
annotationsHandle::POSE what,float &normError){
	// LONGITUDE TRAINING AND PREDICTING
	std::deque<gaussianProcess::prediction> predictionsSin;
	std::deque<gaussianProcess::prediction> predictionsCos;
	std::deque<std::deque<float> > predicted;
	if(what == annotationsHandle::LONGITUDE){
		std::cout<<"Longitude >>> ______________________________________________"<<\
			"_____________________________________________________"<<std::endl;
		// BEFORE TRAINING CAMERA CALIBRATION AND OTHER SETTINGS MIGHT NEED TO BE RESET
		this->resetFeatures(this->trainDir,this->trainImgString,colorSp);
		this->trainGP(annotationsHandle::LONGITUDE,true);

		// BEFORE TESTING CAMERA CALIBRATION AND OTHER SETTINGS MIGHT NEED TO BE RESET
		this->resetFeatures(this->testDir,this->testImgString,colorSp);
		predicted = this->predictGP(predictionsSin,predictionsCos,\
			annotationsHandle::LONGITUDE,true);

		// EVALUATE PREDICTIONS
		float errorLong,normErrorLong,meanDiffLong;
		this->evaluate(predicted,errorLong,normErrorLong,meanDiffLong);
		normError = normErrorLong;
	}else if(what == annotationsHandle::LATITUDE){
		//__________________________________________________________________________
		// LATITUDE TRAINING AND PREDICTING
		std::cout<<"Latitude >>> _______________________________________________"<<\
			"_____________________________________________________"<<std::endl;
		// BEFORE TRAINING CAMERA CALIBRATION AND OTHER SETTINGS MIGHT NEED TO BE RESET
		this->resetFeatures(this->trainDir,this->trainImgString,colorSp);
		this->trainGP(annotationsHandle::LATITUDE,true);

		// BEFORE TESTING CAMERA CALIBRATION AND OTHER SETTINGS MIGHT NEED TO BE RESET
		this->resetFeatures(this->testDir,this->testImgString,colorSp);
		predicted = this->predictGP(predictionsSin,predictionsCos,\
			annotationsHandle::LATITUDE,true);

		// EVALUATE PREDICTIONS
		float errorLat,normErrorLat,meanDiffLat;
		this->evaluate(predicted,errorLat,normErrorLat,meanDiffLat);
		normError = normErrorLat;
	}
	return predicted;
}
//==============================================================================
/** Get the minimum and maximum angle given the motion vector.
 */
void classifyImages::getAngleLimits(unsigned classNo,unsigned predNo,\
float &angleMin,float &angleMax){
	if(this->features->dataMotionVectors[classNo][predNo] == -1.0){
		angleMax = 2*M_PI;
		angleMin = 0.0;
	}else{
		angleMin = this->features->dataMotionVectors[classNo][predNo]-M_PI/2.0;
		Auxiliary::angle0to360(angleMin);
		angleMax = this->features->dataMotionVectors[classNo][predNo]+M_PI/2.0;
		Auxiliary::angle0to360(angleMax);
		if(angleMin>angleMax){
			float aux = angleMin;
			angleMin  = angleMax;
			angleMax  = aux;
		}
	}
}
//==============================================================================
/** Combine the output of multiple classifiers (only on testing,no multiple
 * predictions).
 */
void multipleClassifier(int colorSp,annotationsHandle::POSE what,\
classifyImages &classi,float noise,float length,gaussianProcess::kernelFunction \
kernel,bool useGT){
	classi.init(noise,length,featureExtractor::IPOINTS,kernel,useGT);

	std::deque<std::deque<std::deque<float> > > predictions;
	std::deque<std::deque<float> > tmpPrediction;
	float dummy = 0;
	switch(classi.feature){
		case(featureExtractor::IPOINTS):
			classi.feature = featureExtractor::IPOINTS;
			tmpPrediction = classi.runTest(colorSp,what,dummy);
			predictions.push_back(tmpPrediction);
			tmpPrediction.clear();
/*
		case(featureExtractor::EDGES):
			classi.feature = featureExtractor::EDGES;
			tmpPrediction  = classi.runTest(colorSp,what,dummy);
			predictions.push_back(tmpPrediction);
			tmpPrediction.clear();
		case(featureExtractor::SURF):
			classi.feature = featureExtractor::SURF;
			tmpPrediction  = classi.runTest(colorSp,what,dummy);
			predictions.push_back(tmpPrediction);
			tmpPrediction.clear();
		case(featureExtractor::GABOR):
			classi.feature = featureExtractor::GABOR;
			tmpPrediction  = classi.runTest(colorSp,what,dummy);
			predictions.push_back(tmpPrediction);
			tmpPrediction.clear();
		case(featureExtractor::SIFT):
			classi.feature = featureExtractor::SIFT;
			tmpPrediction  = classi.runTest(colorSp,what,dummy);
			predictions.push_back(tmpPrediction);
			tmpPrediction.clear();
		case(featureExtractor::PIXELS):
			classi.feature = featureExtractor::PIXELS;
			tmpPrediction  = classi.runTest(colorSp,what,dummy);
			predictions.push_back(tmpPrediction);
			tmpPrediction.clear();
		case(featureExtractor::HOG):
			classi.feature = featureExtractor::HOG;
			tmpPrediction  = classi.runTest(colorSp,what,dummy);
			predictions.push_back(tmpPrediction);
			tmpPrediction.clear();
*/
	}

	// HOW TO COMBINE THEM?BIN VOTES EVERY 20DEGREES AND AVERAGE THE WINNING BIN
	std::deque<std::deque<float> > finalPreds;
	for(std::size_t n=0;n<predictions[0].size();++n){// CLASSES
		std::deque<float> preFinalPreds;
		for(std::size_t o=0;o<predictions[0][n].size();++o){// PREDICTIONS
			// FOR EACH PREDICTION BIN THE VOTES AND FIND THE "WINNING" ANGLE
			std::deque<unsigned> votes(18,0);
			std::deque<float> bins(18,0.0);
			unsigned winningLabel   = 0;
			unsigned winningNoVotes = 0;
			float angleMin,angleMax;
			classi.getAngleLimits(n,o,angleMin,angleMax);
			for(std::size_t m=0;m<predictions.size();++m){ // OVER FEATURES
				// CHECK IF THE PREDICTIONS ARE IN THE GOOD RANGE
				if(predictions[m][n][o]>=angleMin || predictions[m][n][o]<angleMax){
					unsigned degrees = static_cast<unsigned>(predictions[m][n][o]*\
						180.0/M_PI);
					unsigned label   = degrees/20;
					++votes[label];
					bins[label] += predictions[m][n][o];
					if(votes[label]>winningNoVotes){
						winningNoVotes = votes[label];
						winningLabel   = label;
					}
				}
			}
			// IF NOT PREDICTION WAS WITHIN THE LIMITS:
			float guess;
			if(winningNoVotes==0){
				guess = classi.features->dataMotionVectors[n][o];
			}else{
				guess = bins[winningLabel]/static_cast<float>(votes[winningLabel]);
			}
			// STORE THE FINAL PREDICTIONS
			preFinalPreds.push_back(guess);
			votes.clear();
			bins.clear();
		}
		finalPreds.push_back(preFinalPreds);
	}

	// FINALLY EVALUATE THE FINAL PREDICTIONS
	std::cout<<"FINAL EVALUATION________________________________________________"<<\
		"________________________"<<std::endl;
	float error = 0.0,normError=0.0,meanDiff=0.0;
	classi.evaluate(finalPreds,error,normError,meanDiff);
}
//==============================================================================
/** Run over multiple settings of the parameters to find the best ones.
 */
void parameterSetting(const std::string &errorsOnTrain,const std::string &errorsOnTest,\
classifyImages &classi,int argc,char** argv,featureExtractor::FEATURE feat,\
int colorSp,bool useGt,annotationsHandle::POSE what,\
gaussianProcess::kernelFunction kernel){
  	std::ofstream train,test;
	train.open(errorsOnTrain.c_str(),std::ios::out);
	test.open(errorsOnTest.c_str(),std::ios::out);
	for(float v=0.1;v<5;v+=0.1){
		for(float l=1;l<125;l+=1){
			classi.init(v,l,feat,kernel,useGt);
			float errorTrain;
			classi.runTest(colorSp,what,errorTrain);
			train<<v<<" "<<l<<" "<<errorTrain<<std::endl;
			classi.init(v,l,feat,kernel,useGt);
			float errorTest;
			classi.runTest(colorSp,what,errorTest);
			test<<v<<" "<<l<<" "<<errorTest<<std::endl;
		}
	}
	train.close();
	test.close();
}
//==============================================================================
int main(int argc,char **argv){
/*
	// test
	float normError = 0.0f;
	classifyImages classi(argc,argv,classifyImages::TEST);
	classi.init(0.85,85.0,featureExtractor::HOG,&gaussianProcess::sqexp,true);
 	classi.runTest(-1,annotationsHandle::LONGITUDE,normError);
*/
	//--------------------------------------------------------------------------
/*
	// build data matrix
 	classifyImages classi(argc,argv,classifyImages::EVALUATE);
	classi.init(0.85,85.0,featureExtractor::HOG,&gaussianProcess::sqexp,true);
	classi.buildDataMatrix();
*/
	//--------------------------------------------------------------------------

	// evaluate
 	classifyImages classi(argc,argv,classifyImages::EVALUATE);
	classi.init(0.85,85.0,featureExtractor::HOG,&gaussianProcess::sqexp,false);
	classi.runCrossValidation(5,annotationsHandle::LONGITUDE,-1,false);

	//--------------------------------------------------------------------------
/*
	// BUILD THE SIFT DICTIONARY
  	classifyImages classi(argc,argv,classifyImages::BUILD_DICTIONARY);
	classi.buildDictionary(-1,true);
*/
	//--------------------------------------------------------------------------
/*
	// find parmeteres
	classifyImages classi(argc,argv,classifyImages::TEST);
	parameterSetting("train.txt","text.txt",classi,argc,argv,featureExtractor::HOG,\
		-1,true,annotationsHandle::LONGITUDE,&gaussianProcess::sqexp);
*/
	//--------------------------------------------------------------------------
/*
	// multiple classifiers
	classifyImages classi(argc,argv,classifyImages::TEST);
	multipleClassifier(-1,annotationsHandle::LONGITUDE,classi,0.85,\
		85,&gaussianProcess::sqexp,false);
*/
}



