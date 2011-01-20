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
#include <exception>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "eigenbackground/src/Tracker.hh"
#include "eigenbackground/src/Helpers.hh"

/** Class used for detecting useful features in the images that can be later
 * used for training and classifying.
 */
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
		/** Default Class constructor.
		 */
		featureDetector(int argc,char** argv):Tracker(argc, argv, 10, true, true){
			this->plotTracks = true;
		}
		featureDetector(int argc,char** argv,bool plot):Tracker(argc, argv, 10, \
		true, true){
			this->plotTracks = false;
		}

		/** Class destructor.
		 */
		virtual ~featureDetector(){};

		/** Function that gets the ROI corresponding to a head/feet of a person in
		 * an image.
		 */
		void upperLowerROI(featureDetector::people someone, double variance,\
		cv::Mat &upperRoi, cv::Mat &lowerRoi);

		/** Start the running in a parallel thread an instance of the tracker.
		 */
		bool doFindPerson(unsigned imgNum, IplImage *src,\
			const vnl_vector<FLOAT> &imgVec, vnl_vector<FLOAT> &bgVec,\
			const FLOAT logBGProb,const vnl_vector<FLOAT> &logSumPixelBGProb);

		/** Simple "menu" for skipping to the next image or quitting the processing
		 */
		bool imageProcessingMenu();

		/** Creates a symmetrical Gaussian kernel.
		 */
		void gaussianKernel(cv::Mat &gauss, cv::Size size, double sigma,cv::Point offset);

		/** Hough transform for circle detection.
		 */
		void ellipseDetection(cv::Mat img);

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

		/** Get perpendicular to a line given by 2 points A, B in point C.
		 */
		void getLinePerpendicular(cv::Point A, cv::Point B, cv::Point C, \
			double &m, double &b);

		/** Checks to see if a point is on the same side of a line like another given point.
		 */
		bool sameSubplane(cv::Point test,cv::Point point, double m, double b);

		/** Gets strong corner points in an image.
		 */
		void getCornerPoints(std::vector<cv::Point2f> &corners, cv::Mat image);

		/** Gets the edges in an image.
		 */
		void getEdges(cv::Mat_<uchar> edges, cv::Mat image);

		/** SURF descriptors (Speeded Up Robust Features).
		 */
		void getSURF(std::vector<float>& descriptors, cv::Mat image);

		/** Blob detector in RGB color space.
		 */
		void blobDetector(std::vector<std::vector<cv::Point> >& msers, cv::Mat image);

		/** Just displaying an image a bit larger to see it better.
		 */
		void showZoomedImage(cv::Mat image, std::string title="zoomed");

		/** Head detection by fitting ellipses.
		 */
		void skinEllipses(cv::RotatedRect &box, cv::Mat img);

		/** Head detection by fitting ellipses (if templateCenter is relative to the img
		 * the offset needs to be used).
		 */
		void skinEllipses(cv::RotatedRect &finalBox, cv::Mat img, cv::Point \
		templateCenter, cv::Point offset=cv::Point(0,0), double minHeadSize=20,\
		double maxHeadSize=40);

		/** Creates the \c Gabor filter with the given parameters and returns the \c wavelet.
		 */
		void getGabor(cv::Mat &response, cv::Mat image, float params[]=\
		{3,0.4,2,M_PI/4,4,20});
		//======================================================================
	protected:
		bool plotTracks;

};
#endif /* FESTUREDETECTOR_H_ */
