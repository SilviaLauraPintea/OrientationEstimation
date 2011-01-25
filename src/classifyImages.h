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
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include "eigenbackground/src/Tracker.hh"
#include "eigenbackground/src/Helpers.hh"
#include "featureDetector.h"

/** Class used for classifying the training data.
 */
class classifyImages {
	protected:
		/** @var testFeatures
		 * An instance of \c featureDetector class.
		 */
		featureDetector *testFeatures;

		/** @var trainFeatures
		 * An instance of \c featureDetector class.
		 */
		featureDetector *trainFeatures;

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
		//======================================================================
	public:
		classifyImages(int argc, char **argv);
		virtual ~classifyImages();

		/** Creates the training data/test data.
		 */
		void createData(std::vector<std::string> options);

		/** Regression SVM classification.
		 */
		void classifySVM();
};

#endif /* CLASSIFYIMAGES_H_ */
