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
		/** Options for the main
		 */
		enum TORUN {run_build_data,run_test,run_evaluate,run_build_dictionary,\
			run_build_pca,run_find_params,run_multiple_class};
		/** All available uses of this class.
		 */
		enum CLASSIFIER {GAUSSIAN_PROCESS,NEURAL_NETWORK,K_NEAREST_NEIGHBORS,\
			DIST2PCA};
		/** All available uses of this class.
		 */
		enum USES {EVALUATE,BUILD_DICTIONARY,TEST,BUILD_DATA};
		/** Constructor & destructor of the class.
		 */
		ClassifyImages(int argc,char **argv,ClassifyImages::USES use=\
			ClassifyImages::EVALUATE,ClassifyImages::CLASSIFIER classi=\
			ClassifyImages::GAUSSIAN_PROCESS);
		virtual ~ClassifyImages();

		ClassifyImages::USES what();
		/** Build dictionary for vector quantization.
		 */
		void buildDictionary(int colorSp=-1,bool toUseGT=true);
		/** Trains on the training data using the indicated classifier.
		 */
		void train(AnnotationsHandle::POSE what,bool fromFolder,bool justLoad=true);
		/** Creates the training data (according to the options),the labels and
		 * trains the a \c GaussianProcess on the data.
		 */
		void trainGP(AnnotationsHandle::POSE what,int i);
		/** Creates the training data (according to the options),the labels and
		 * trains the a \c Neural Network on the data.
		 */
		void trainNN(int i,bool together=false);
		/** Creates the test data and applies \c GaussianProcess prediction on the test
		 * data.
		 */
		cv::Point2f predictGP(cv::Mat &testRow,int i);
		/** Creates the test data and applies \c Neural Network prediction on the test
		 * data.
		 */
		cv::Point2f predictNN(cv::Mat &testRow,AnnotationsHandle::POSE what,\
			int i,bool together=false);
		/** Initialize the options for the Gaussian Process regression.
		 */
		void init(float theNoise,float theLengthSin,float theLengthCos,\
			const std::deque<FeatureExtractor::FEATURE> &theFeature,\
			GaussianProcess::kernelFunction theKFunction=&GaussianProcess::sqexp,\
			bool toUseGT=false);
		/** Check if the classifier was initialized.
		 */
		bool isClassiInit(int i);
		/** Evaluate one prediction versus its target.
		 */
		void evaluate(const std::deque<std::deque<cv::Point2f> > &prediAngles,\
			float &error,float &normError,float &meanDiff,cv::Mat &bins);

		/** Do k-fold cross-validation by splitting the training folder into
		 * training-set and validation-set.
		 */
		void crossValidation(unsigned k,unsigned fold,bool onTrain=false,\
			bool rndm = false);
		/** Does the cross-validation and computes the average error over all folds.
		 */
		float runCrossValidation(unsigned k,AnnotationsHandle::POSE what,\
			int colorSp=-1,bool onTrain=false,FeatureExtractor::FEATUREPART part=\
			FeatureExtractor::WHOLE);
		/** Runs the final evaluation (test).
		 */
		std::deque<std::deque<cv::Point2f> > runTest(int colorSp,\
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
			GaussianProcess::kernelFunction kernel,unsigned folds=0,\
			FeatureExtractor::FEATUREPART part=FeatureExtractor::TOP);
		/** Combine the output of multiple classifiers (only on testing,no multiple
		 * predictions).
		 */
		friend void multipleClassifier(int colorSp,AnnotationsHandle::POSE what,\
			ClassifyImages &classi,float noise,float lengthSin,float lengthCos,\
			GaussianProcess::kernelFunction kernel,bool useGT,\
			FeatureExtractor::FEATUREPART part);
		/** Get the minimum and maximum angle given the motion vector.
		 */
		void getAngleLimits(unsigned classNo,unsigned predNo,float &angleMin,\
			float &angleMax);

		/** Applies PCA on top of a data-row to reduce its dimensionality.
		 */
		cv::Mat reduceDimensionality(const cv::Mat &data,int i,bool train,\
			int nEigens=0,int reshapeRows=0);
		/** Read and load the training/testing data.
		 */
		void getData(std::string &trainFld,std::string &annoFld,bool fromFolder,\
			bool test,bool justLoad=false);
		/** Starts the threading such that each test row is generated and
		 * predicted in real time.
		 */
		std::deque<std::deque<cv::Point2f> > predict(AnnotationsHandle::POSE what,\
			bool fromFolder);
		/** Predicts on the test data.
		 */
		std::deque<cv::Point2f> doPredict(std::tr1::shared_ptr\
			<PeopleDetector::DataRow> dataRow,AnnotationsHandle::POSE what,\
			bool fromFolder);
		/** Try to optimize the prediction of the angle considering the variance of
		 * sin^2 and cos^2.
		 */
		float optimizeSin2Cos2Prediction(const GaussianProcess::prediction \
			&predictionsSin,const GaussianProcess::prediction &predictionsCos);
		/** Creates the training data (according to the options),the labels and
		 * trains the a kNN on the data.
		 */
		void trainKNN(AnnotationsHandle::POSE what,int i);
		/** Creates the test data and applies \c kNN prediction on the test data.
		 */
		cv::Point2f predictKNN(cv::Mat &testRow,int i);
		/** Creates the training data (according to the options),the labels and
		 * builds the eigen-orientations.
		 */
		void trainDist2PCA(AnnotationsHandle::POSE what,int i,unsigned bins=0,\
			unsigned dimensions=1);
		/** Creates the test data and applies computes the distances to the stored
		 * eigen-orientations.
		 */
		cv::Point2f predictDist2PCA(cv::Mat &testRow,\
			AnnotationsHandle::POSE what,int i);
		/** Backproject each image on the 4 models, compute distances and return.
		 */
		cv::Mat getPCAModel(const cv::Mat &data,int i,unsigned bins);
		/** Build a class model for each one of the 4 classes.
		 */
		void buildPCAModels(int colorSp,FeatureExtractor::FEATUREPART part);
		/** Reorder a vector to a given ordering.
		 */
		static void reorderDeque(const std::vector<unsigned> &order,\
			std::deque<std::string> &input);
		/** Set the use of the classifier */
		void setWhat(int argc,char **argv,ClassifyImages::USES use);
		/** Implements all the file/folder actions.
		 */
		void doWhat(int argc,char **argv);
		//======================================================================
	private:
		/** @var nn_
		 * An instance of the class cv_ANN_MLP.
		 */
		std::deque<CvANN_MLP> nn_;
		/** @var nn_
		 * An instance of the class cv_ANN_MLP.
		 */
		std::deque<CvANN_MLP> nnCos_;
		/** @var nn_
		 * An instance of the class cv_ANN_MLP.
		 */
		std::deque<CvANN_MLP> nnSin_;
		/** @var sinKNN_
		 * An instance of the class CvKNearest.
		 */
		std::deque<CvKNearest> sinKNN_;
		/** @var cosKNN_
		 * An instance of the class CvKNearest.
		 */
		std::deque<CvKNearest> cosKNN_;
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
		/** @var lengthSin_
		 * The length in the Gaussian Process.
		 */
		float lengthSin_;
		/** @var lengthCos_
		 * The length in the Gaussian Process.
		 */
		float lengthCos_;
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
		/** @var tmpModelName_
		 * The temporary name of the model the be loaded/saved.
		 */
		std::string tmpModelName_;
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
		 * A vector of pointers to instance of the class cv::PCA
		 */
		std::vector<std::tr1::shared_ptr<cv::PCA> > pca_;
		/** @var classiPca_
		 * An instance of the class cv::PCA
		 */
		std::deque<std::deque<std::tr1::shared_ptr<cv::PCA> > > classiPca_;
		/** @ plot_
		 * To display images or not.
		 */
		bool plot_;
		/** @var clasifier_
		 * The classifier to be used indicates the train function that needs to
		 * be called.
		 */
		ClassifyImages::CLASSIFIER clasifier_;
		/** @var dimPCA_
		 * The number of components to be kept when using PCA.
		 */
		unsigned dimPCA_;
		/** @var withFlip_
		 * If the images should be flipped or not.
		 */
		bool withFlip_;
		/** @var usePCAModel_
		 * A PCA model is build for 4 classes for each data feature
		 */
		bool usePCAModel_;
		/** @var trainMean_
		 * The mean vector of the training dataset.
		 */
		std::vector<cv::Mat> trainMean_;
		/** @var trainVar_
		 * The variance of the training dataset.
		 */
		std::vector<cv::Mat> trainVar_;
		//======================================================================
	private:
		DISALLOW_COPY_AND_ASSIGN(ClassifyImages);
};
#endif /* CLASSIFYIMAGES_H_ */
