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
		/** structure containing images of the size of the detected people
		 */
		struct people {
			cv::Point absoluteLoc;
			cv::Point relativeLoc;
			cv::Mat_<cv::Vec3b> pixels;
		};
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
		void getHeadROI(std::vector<featureDetector::people> allPeople);

		/** Start the running in a parallel thread an instance of the tracker.
		 */
		virtual bool doFindPerson(unsigned imgNum, IplImage *src,\
			const vnl_vector<FLOAT> &imgVec, vnl_vector<FLOAT> &bgVec,\
			const FLOAT logBGProb,const vnl_vector<FLOAT> &logSumPixelBGProb);

		/** Simple "menu" for skipping to the next image or quitting the processing
		 */
		bool imageProcessingMenu();

		/** Creates the \c Gabor filter with the given parameters and returns the \c wavelet.
		 */
		cv::Mat createGabor(float params[]);

		/** Convolves an image with a computed \c Gabor filter.
		 */
		cv::Mat convolveImage(cv::Point winCenter, cv::Mat image, float params[]);

		/** Get the foreground pixels corresponding to each person
		 */
		void getAllForegroundPixels(std::vector<featureDetector::people> &allPeople,\
			std::vector<unsigned> existing, IplImage *bg, double threshold);
		/** Gets the distance to the given template from a given pixel location.
		 */
		double getDistToTemplate(int pixelX,int pixelY,std::vector<CvPoint> templ);
		/** Checks to see if a given pixel is inside of a template.
		 */
		bool isInTemplate(unsigned pixelX, unsigned pixelY, std::vector<CvPoint> templ);
		/** Shows a ROI in a given image.
		 */
		void showROI(cv::Mat image, cv::Point top_left, cv::Size ROI_size);
		//======================================================================
	protected:

};
#endif /* FESTUREDETECTOR_H_ */
