#ifndef CLASSIFYIMAGES_H_
#define CLASSIFYIMAGES_H_
#include <boost/thread.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION < 103500
	#include <boost/thread/detail/lock.hpp>
#endif
#include <boost/thread/xtime.hpp>

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

class classifyImages {
	public:
		featureDetector features;
		//======================================================================
		classifyImages();
		virtual ~classifyImages();

		/** Regression SVM classification.
		 */
		void classifySVM(cv::Mat* trainData, cv::Mat* sample);
};

#endif /* CLASSIFYIMAGES_H_ */
