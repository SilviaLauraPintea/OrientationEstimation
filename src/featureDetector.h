#ifndef FESTUREDETECTOR_H_
#define FESTUREDETECTOR_H_
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
#include <opencv/cv.h>
#include <exception>
#include <opencv/highgui.h>
#include "eigenbackground/src/Tracker.hh"
#include "eigenbackground/src/Helpers.hh"

/** Class used for detecting useful features in the images that can be later
 * used for training and classifying.*/

class featureDetector:public Tracker{
	public:
		//======================================================================
		/** Class constructor.
		 */
		featureDetector():Tracker(10, true, true){}

		/** Class destructor.
		 */
		~featureDetector(){}

		/** Function that gets the ROI corresponding to a head of a person in
		 * an image.
		 */
		cv::Mat getHeadROI(vector<unsigned> existing);

		/** Start the running in a parallel thread an instance of the tracker.
		 */
		virtual bool doFindPerson(unsigned imgNum, IplImage *src,\
			const vnl_vector<FLOAT> &imgVec, vnl_vector<FLOAT> &bgVec,\
			const FLOAT logBGProb,const vnl_vector<FLOAT> &logSumPixelBGProb);

		/** Simple "menu" for skipping to the next image or quitting the processing
		 */
		bool imageProcessingMenu();
		//======================================================================
	protected:

};
#endif /* FESTUREDETECTOR_H_ */
