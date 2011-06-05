/* ClassifyImages.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#ifndef CLASSIFYIMAGES_H_
#define CLASSIFYIMAGES_H_
#include <tr1/memory>
#include "eigenbackground/src/Helpers.hh"
#include "AnnotationsHandle.h"
#include "GaussianProcess.h"
#include "PeopleDetector.h"
/** Class used for classifying the training data.
 */
class ClassifyImages {
	public:
		//======================================================================
		/** All available uses of this class.
		 */
		enum CLASSIFIER {GAUSSIAN_PROCESS,NEURAL_NETWORK};
		/** All available uses of this class.
		 */
		enum USES {EVALUATE,BUILD_DICTIONARY,TEST,BUILD_DATA};
		/** Constructor & destructor of the class.
		 */
		ClassifyImages(int argc,char **argv,ClassifyImages::USES use=\
			ClassifyImages::EVALUATE,ClassifyImages::CLASSIFIER classi=\
			ClassifyImages::GAUSSIAN_PROCESS);
		virtual ~ClassifyImages();

		/** Build dictionary for vector quantization.
		 */
		void buildDictionary(int colorSp=-1,bool toUseGT=true);
		/** Trains on the training data using the indicated classifier.
		 */
		void train(AnnotationsHandle::POSE what,bool fromFolder);
		/** Creates the training data (according to the options),the labels and
		 * trains the a \c GaussianProcess on the data.
		 */
		void trainGP(AnnotationsHandle::POSE what,int i);
		/** Creates the training data (according to the options),the labels and
		 * trains the a \c Neural Network on the data.
		 */
		void trainNN(AnnotationsHandle::POSE what,int i);
		/** Creates the test data and applies \c GaussianProcess prediction on the test
		 * data.
		 */
		std::deque<float> predictGP(int i);
		/** Creates the test data and applies \c Neural Network prediction on the test
		 * data.
		 */
		std::deque<float> predictNN(int i);
		/** Initialize the options for the Gaussian Process regression.
		 */
		void init(float theNoise,float theLength,\
			const std::deque<FeatureExtractor::FEATURE> &theFeature,\
			GaussianProcess::kernelFunction theKFunction=&GaussianProcess::sqexp,\
			bool toUseGT=false);
		/** Check if the classifier was initialized.
		 */
		bool isClassiInit(int i);
		/** Evaluate one prediction versus its target.
		 */
		void evaluate(const std::deque<std::deque<float> > &prediAngles,\
			float &error,float &normError,float &meanDiff);

		/** Do k-fold cross-validation by splitting the training folder into
		 * training-set and validation-set.
		 */
		void crossValidation(unsigned k,unsigned fold,bool onTrain=false);

		/** Does the cross-validation and computes the average error over all folds.
		 */
		float runCrossValidation(unsigned k,AnnotationsHandle::POSE what,\
			int colorSp=-1,bool onTrain=false,FeatureExtractor::FEATUREPART part=\
			FeatureExtractor::WHOLE);
		/** Runs the final evaluation (test).
		 */
		std::deque<std::deque<float> > runTest(int colorSp,\
			AnnotationsHandle::POSE what,float &normError,\
			FeatureExtractor::FEATUREPART part);
		/** Try to optimize the prediction of the angle considering the variance
		 * of sin and cos.
		 */
		float optimizePrediction(const GaussianProcess::prediction &predictionsSin,\
			const GaussianProcess::prediction &predictionsCos);
		/** Reset the features object when the training and testing might have different
		 * calibration,background models...
		 */
		void resetFeatures(const std::string &dir,const std::string &imStr,\
			int colorSp,FeatureExtractor::FEATUREPART part=FeatureExtractor::WHOLE);
		/** Just build data matrix and store it;it can be called over multiple
		 * datasets by adding the the new data rows at the end to the stored
		 * matrix.
		 */
		void buildDataMatrix(int colorSp=-1,FeatureExtractor::FEATUREPART part=\
			FeatureExtractor::WHOLE);
		/** Concatenate the loaded data from the files to the currently computed data.
		 */
		void loadData(const cv::Mat &tmpData1,const cv::Mat &tmpTargets1,\
			unsigned i,cv::Mat &outData,cv::Mat &outTargets);
		/** Run over multiple settings of the parameters to find the best ones.
		 */
		friend void parameterSetting(const std::string &errorsOnTrain,\
			const std::string &errorsOnTest,ClassifyImages &classi,int argc,\
			char** argv,const std::deque<FeatureExtractor::FEATURE> &feat,\
			int colorSp,bool useGt,AnnotationsHandle::POSE what,\
			GaussianProcess::kernelFunction kernel);
		/** Combine the output of multiple classifiers (only on testing,no multiple
		 * predictions).
		 */
		friend void multipleClassifier(int colorSp,AnnotationsHandle::POSE what,\
			ClassifyImages &classi,float noise,float length,\
			GaussianProcess::kernelFunction kernel,bool useGT,\
			FeatureExtractor::FEATUREPART part);
		/** Get the minimum and maximum angle given the motion vector.
		 */
		void getAngleLimits(unsigned classNo,unsigned predNo,float &angleMin,\
			float &angleMax);

		/** Applies PCA on top of a data-row to reduce its dimensionality.
		 */
		cv::Mat reduceDimensionality(const cv::Mat &data,bool train,\
			int nEigens=0,int reshapeRows=0);
		/** Read and load the training/testing data.
		 */
		void getData(std::string trainFld,std::string annoFld,bool fromFolder);
		/** Predicts on the test data.
		 */
		std::deque<std::deque<float> > predict(AnnotationsHandle::POSE what,\
			bool fromFolder);

		//======================================================================
	private:
		/** @var nnCos_
		 * An instance of the class cv_ANN_MLP.
		 */
		std::deque<CvANN_MLP> nnCos_;
		/** @var nnSin_
		 * An instance of the class cv_ANN_MLP.
		 */
		std::deque<CvANN_MLP> nnSin_;
		/** @var features_
		 * An instance of \c PeopleDetector class.
		 */
		std::tr1::shared_ptr<PeopleDetector> features_;
		/** @var trainData_
		 * The training data matrix.
		 */
		std::vector<cv::Mat> trainData_;
		/** @var testData_
		 * The test data matrix.
		 */
		std::vector<cv::Mat> testData_;
		/** @var trainFolder_
		 * The folder containing the training images.
		 */
		std::string trainFolder_;
		/** @var testFolder_
		 * The folder containing the test images.
		 */
		std::string testFolder_;
		/** @var annotationsTrain_
		 * The file contains the annotations for the training images.
		 */
		std::string annotationsTrain_;
		/** @var annotationsTest_
		 * The file contains the annotations for the test images.
		 */
		std::string annotationsTest_;
		/** @var trainTargets_
		 * The column matrix containing the train annotation data (targets).
		 */
		std::vector<cv::Mat> trainTargets_;
		/** @var testTargets_
		 * The column matrix containing the test annotation data (targets).
		 */
		std::vector<cv::Mat> testTargets_;
		/** @var gpCos_
		 * An instance of the class GaussianProcess.
		 */
		std::deque<GaussianProcess> gpCos_;
		/** @var gpSin_
		 * An instance of the class GaussianProcess.
		 */
		std::deque<GaussianProcess> gpSin_;
		/** @var noise_
		 * The noise level of the data.
		 */
		float noise_;
		/** @var length_
		 * The length in the Gaussian Process.
		 */
		float length_;
		/** @var kFunction_
		 * The kernel function in the Gaussian Process.
		 */
		GaussianProcess::kernelFunction kFunction_;
		/** @var feature_
		 * Feature to be extracted.
		 */
		std::deque<FeatureExtractor::FEATURE> feature_;
		/** @var readFromFolder_
		 * If the images are read from folder or from a file with image names.
		 */
		bool readFromFolder_;
		/** @var imageList_
		 * All images are stored in this list for cross-validation.
		 */
		std::deque<std::string> imageList_;
		/** @var annoList_
		 * All annotations for all images are stored in this list for cross-validation.
		 */
		std::deque<std::string> annoList_;
		/** @var foldSize_
		 * The size of one fold in cross-validation.
		 */
		unsigned foldSize_;
		/** @var modelName_
		 * The name of the model the be loaded/saved.
		 */
		std::string modelName_;
		/** @var what_
		 * What should the class be used for.
		 */
		ClassifyImages::USES what_;
		/** @var testDir_
		 * Directory in which to look for the test images & other files.
		 */
		std::string testDir_;
		/** @var testImgString_
		 * The letters in the image names for the test data.
		 */
		std::string testImgString_;
		/** @var trainDir_
		 * Directory in which to look for the train images & other files.
		 */
		std::string trainDir_;
		/** @var trainImgString_
		 * The letters in the image names for the train data.
		 */
		std::string trainImgString_;
		/** @var useGroundTruth_
		 * Use the annotations' positions or use the tracker.
		 */
		bool useGroundTruth_;
		/** @var dimRed_
		 * True if dimensionality reduction should be used.
		 */
		bool dimRed_;
		/** @var pca_
		 * An instance of the class cv::PCA
		 */
		std::tr1::shared_ptr<cv::PCA> pca_;
		/** @ plot_
		 * To display images or not.
		 */
		bool plot_;
		/** @var clasifier_
		 * The classifier to be used indicates the train function that needs to
		 * be called.
		 */
		ClassifyImages::CLASSIFIER clasifier_;
		//======================================================================
	private:
		DISALLOW_COPY_AND_ASSIGN(ClassifyImages);
};
#endif /* CLASSIFYIMAGES_H_ */
