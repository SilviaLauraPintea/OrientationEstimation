/* classifyImages.h
 * Author: Silvia-Laura Pintea
 */
#ifndef CLASSIFYIMAGES_H_
#define CLASSIFYIMAGES_H_
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <exception>
#include <opencv2/opencv.hpp>
#include "eigenbackground/src/Tracker.hh"
#include "eigenbackground/src/Helpers.hh"
#include "eigenbackground/src/defines.hh"
#include "featureDetector.h"
#include "gaussianProcess.h"

/** Class used for classifying the training data.
 */
class classifyImages {
	protected:
		/** @var features
		 * An instance of \c featureDetector class.
		 */
		featureDetector *features;

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
		double noise;

		/** @var length
		 * The length in the Gaussian Process.
		 */
		double length;

		/** @var kFunction
		 * The kernel function in the Gaussian Process.
		 */
		gaussianProcess::kernelFunction kFunction;

		/** @var feature
		 * Feature to be extracted.
		 */
		featureDetector::FEATURE feature;

		/** @var readFromFolder
		 * If the images are read from folder or from a file with image names.
		 */
		bool readFromFolder;

		/** @var imageList
		 * All images are stored in this list for cross-validation.
		 */
		std::vector<std::string> imageList;

		/** @var annoList
		 * All annotations for all images are stored in this list for cross-validation.
		 */
		std::vector<std::string> annoList;

		/** @var foldSize
		 * The size of one fold in cross-validation.
		 */
		unsigned foldSize;
		//======================================================================
	public:
		classifyImages(int argc, char **argv);
		virtual ~classifyImages();

		/** Build dictionary for vector quantization.
		 */
		void buildDictionary(char* fileToStore = const_cast<char*>("dictSIFT.bin"),\
			char* dataFile=const_cast<char*>("test/sift/"));

		/** Creates the training data (according to the options), the labels and
		 * trains the a \c GaussianProcess on the data.
		 */
		void trainGP(annotationsHandle::POSE what);

		/** Creates the test data and applies \c GaussianProcess prediction on the test
		 * data.
		 */
		void predictGP(std::vector<gaussianProcess::prediction> &predictionsSin,\
			std::vector<gaussianProcess::prediction> &predictionsCos,\
			annotationsHandle::POSE what);

		/** Initialize the options for the Gaussian Process regression.
		 */
		void init(double theNoise, double theLength, gaussianProcess::kernelFunction\
			theKFunction, featureDetector::FEATURE theFeature, char* fileSIFT =\
			const_cast<char*>("dictSIFT.bin"), int colorSp = CV_BGR2Lab,\
			bool fromFolder=true);

		/** Evaluate one prediction versus its target.
		 */
		void evaluate(std::vector<gaussianProcess::prediction> predictionsSin,\
			std::vector<gaussianProcess::prediction> predictionsCos,\
			double &error, double &accuracy,  annotationsHandle::POSE what);

		/** Do k-fold cross-validation by splitting the training folder into
		 * training-set and validation-set.
		 */
		void crossValidation(unsigned k, unsigned fold);

		/** Does the cross-validation and computes the average error over all folds.
		 */
		void runCrossValidation(unsigned k, double theNoise, double theLength,\
			gaussianProcess::kernelFunction theKFunction, featureDetector::FEATURE\
			theFeature, char* fileSIFT = const_cast<char*>("dictSIFT.bin"),\
			int colorSp = CV_BGR2Lab, bool fromFolder=false);

		/** Runs the final evaluation (test).
		 */
		void runTest(double theNoise, double theLength,\
		gaussianProcess::kernelFunction theKFunction, featureDetector::FEATURE\
		theFeature, char* fileSIFT = const_cast<char*>("dictSIFT.bin"),\
		int colorSp = CV_BGR2Lab, bool fromFolder=false);

		/** Try to optimize the prediction of the angle considering the variance of sin
		 * and cos.
		 */
		double optimizePrediction(gaussianProcess::prediction predictionsSin,\
			gaussianProcess::prediction predictionsCos);
};

#endif /* CLASSIFYIMAGES_H_ */
