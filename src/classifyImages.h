/* classifyImages.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code, but please retain the above copyright notice.
 */
#ifndef CLASSIFYIMAGES_H_
#define CLASSIFYIMAGES_H_
#include <opencv2/opencv.hpp>
#include "annotationsHandle.h"
#include "gaussianProcess.h"
#include "peopleDetector.h"
/** Class used for classifying the training data.
 */
class classifyImages {
	public:
		//======================================================================
		/** All available uses of this class.
		 */
		enum USES {EVALUATE, BUILD_DICTIONARY, TEST};
		/** Constructor & destructor of the class.
		 */
		classifyImages(int argc, char **argv, classifyImages::USES use = \
			classifyImages::EVALUATE);
		virtual ~classifyImages();

		/** Build dictionary for vector quantization.
		 */
		void buildDictionary(int colorSp = CV_BGR2Lab, bool toUseGT = true);

		/** Creates the training data (according to the options), the labels and
		 * trains the a \c GaussianProcess on the data.
		 */
		void trainGP(annotationsHandle::POSE what);

		/** Creates the test data and applies \c GaussianProcess prediction on the test
		 * data.
		 */
		void predictGP(std::deque<gaussianProcess::prediction> &predictionsSin,\
			std::deque<gaussianProcess::prediction> &predictionsCos,\
			annotationsHandle::POSE what);

		/** Initialize the options for the Gaussian Process regression.
		 */
		void init(float theNoise, float theLength, featureExtractor::FEATURE \
			theFeature, gaussianProcess::kernelFunction theKFunction = \
			&gaussianProcess::sqexp, bool fromFolder=true, bool store=true,\
			bool toUseGT = false);

		/** Evaluate one prediction versus its target.
		 */
		void evaluate(std::deque<gaussianProcess::prediction> predictionsSin,\
			std::deque<gaussianProcess::prediction> predictionsCos,\
			float &error, float &normError, float &meanDiff);

		/** Do k-fold cross-validation by splitting the training folder into
		 * training-set and validation-set.
		 */
		void crossValidation(unsigned k, unsigned fold, bool onTrain = false);

		/** Does the cross-validation and computes the average error over all folds.
		 */
		float runCrossValidation(unsigned k, int colorSp = CV_BGR2Lab,\
			bool onTrain = false);

		/** Runs the final evaluation (test).
		 */
		void runTest(int colorSp = CV_BGR2Lab);

		/** Try to optimize the prediction of the angle considering the variance of sin
		 * and cos.
		 */
		float optimizePrediction(gaussianProcess::prediction predictionsSin,\
			gaussianProcess::prediction predictionsCos);
		/** Reset the features object when the training and testing might have different
		 * calibration, background models...
		 */
		void resetFeatures(std::string dir, std::string imStr, int colorSp);
		//======================================================================
	protected:
		/** @var features
		 * An instance of \c peopleDetector class.
		 */
		peopleDetector *features;

		/** @var trainData
		 * The training data matrix.
		 */
		cv::Mat trainData;

		/** @var testData
		 * The test data matrix.
		 */
		cv::Mat testData;

		/** @var trainFolder
		 * The folder containing the training images.
		 */
		std::string trainFolder;

		/** @var testFolder
		 * The folder containing the test images.
		 */
		std::string testFolder;

		/** @var annotationsTrain
		 * The file contains the annotations for the training images.
		 */
		std::string annotationsTrain;

		/** @var annotationsTest
		 * The file contains the annotations for the test images.
		 */
		std::string annotationsTest;

		/** @var trainTargets
		 * The column matrix containing the train annotation data (targets).
		 */
		cv::Mat trainTargets;

		/** @var testTargets
		 * The column matrix containing the test annotation data (targets).
		 */
		cv::Mat testTargets;

		/** @var gaussianProcess
		 * An instance of the class gaussianProcess.
		 */
		gaussianProcess gpCos;

		/** @var gaussianProcess
		 * An instance of the class gaussianProcess.
		 */
		gaussianProcess gpSin;

		/** @var noise
		 * The noise level of the data.
		 */
		float noise;

		/** @var length
		 * The length in the Gaussian Process.
		 */
		float length;

		/** @var kFunction
		 * The kernel function in the Gaussian Process.
		 */
		gaussianProcess::kernelFunction kFunction;

		/** @var feature
		 * Feature to be extracted.
		 */
		featureExtractor::FEATURE feature;

		/** @var readFromFolder
		 * If the images are read from folder or from a file with image names.
		 */
		bool readFromFolder;

		/** @var imageList
		 * All images are stored in this list for cross-validation.
		 */
		std::deque<std::string> imageList;

		/** @var annoList
		 * All annotations for all images are stored in this list for cross-validation.
		 */
		std::deque<std::string> annoList;

		/** @var foldSize
		 * The size of one fold in cross-validation.
		 */
		unsigned foldSize;

		/** @var storeData
		 * If data is stored locally or not.
		 */
		bool storeData;

		/** @var modelName
		 * The name of the model the be loaded/saved.
		 */
		std::string modelName;

		/** @var what
		 * What should the class be used for.
		 */
		classifyImages::USES what;

		/** @var testDir
		 * Directory in which to look for the test images & other files.
		 */
		std::string testDir;
		/** @var testImgString
		 * The letters in the image names for the test data.
		 */
		std::string testImgString;
		/** @var trainDir
		 * Directory in which to look for the train images & other files.
		 */
		std::string trainDir;
		/** @var trainImgString
		 * The letters in the image names for the train data.
		 */
		std::string trainImgString;
		/** @var useGroundTruth
		 * Use the annotations' positions or use the tracker.
		 */
		bool useGroundTruth;
};

#endif /* CLASSIFYIMAGES_H_ */
