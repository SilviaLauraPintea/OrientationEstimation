/* featureDetector.h
 * Author: Silvia-Laura Pintea
 */
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
//#include <opencv2/core/mat.hpp>
#include "eigenbackground/src/Tracker.hh"
#include "eigenbackground/src/Helpers.hh"
#include "annotationsHandle.h"

/** Class used for detecting useful features in the images that can be later
 * used for training and classifying.
 */
class featureDetector:public Tracker{
	public:
		/** Structure containing images of the size of the detected people.
		 */
		struct people {
			cv::Point absoluteLoc;
			cv::Point relativeLoc;
			std::vector<unsigned> borders;
			cv::Mat_<cv::Vec3b> pixels;
		};

		/** All available feature types.
		 */
		enum FEATURE {BLOB, ELLIPSE, CORNER, EDGES, GABOR, SURF};
		//======================================================================
		featureDetector(int argc,char** argv):Tracker(argc, argv, 10, true, true){
			this->plotTracks      = true;
			this->featureType     = EDGES;
		}

		featureDetector(int argc,char** argv,bool plot):Tracker(argc, argv, 10, \
		true, true){
			this->plotTracks      = plot;
			this->featureType     = EDGES;
		}

		virtual ~featureDetector(){
			this->producer = NULL;
			for(size_t i=0; i<this->data.size(); i++){
				this->data[i].release();
			}
			for(size_t i=0; i<this->targets.size(); i++){
				this->targets[i].release();
			}
			this->targets.clear();
			this->data.clear();

			// CLEAR THE ANNOTATIONS
			for(std::size_t i=0; i<this->targetAnno.size(); i++){
				for(std::size_t j=0; j<this->targetAnno[i].annos.size(); j++){
					this->targetAnno[i].annos[j].poses.clear();
				}
				this->targetAnno[i].annos.clear();
			}
			this->targetAnno.clear();
		}

		/** Function that gets the ROI corresponding to a head/feet of a person in
		 * an image.
		 */
		void upperLowerROI(featureDetector::people someone, double variance,\
		cv::Mat &upperRoi, cv::Mat &lowerRoi);

		/** Overwrites the \c doFindPeople function from the \c Tracker class
		 * to make it work with the feature extraction.
		 */
		bool doFindPerson(unsigned imgNum, IplImage *src,\
			const vnl_vector<FLOAT> &imgVec, vnl_vector<FLOAT> &bgVec,\
			const FLOAT logBGProb,const vnl_vector<FLOAT> &logSumPixelBGProb);

		/** Simple "menu" for skipping to the next image or quitting the processing.
		 */
		bool imageProcessingMenu();

		/** Creates a symmetrical Gaussian kernel.
		 */
		void gaussianKernel(cv::Mat &gauss, cv::Size size, double sigma,cv::Point offset);

		/** Get the foreground pixels corresponding to each person.
		 */
		void allForegroundPixels(std::vector<featureDetector::people> &allPeople,\
			std::vector<unsigned> existing, IplImage *bg, double threshold);

		/** Gets the distance to the given template from a given pixel location.
		 */
		double getDistToTemplate(int pixelX,int pixelY,std::vector<CvPoint> templ);

		/** Checks to see if a given pixel is inside a template.
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
		/** Gets strong corner points in an image.
		 */
		void getCornerPoints(cv::Mat &feature, cv::Mat image, std::vector<unsigned> \
			borders, cv::Mat thresholded);

		/** Gets the edges in an image.
		 */
		void getEdges(cv::Mat &feature, cv::Mat image, std::vector<unsigned> \
			borders, unsigned reshape=1, cv::Mat thresholded=cv::Mat());

		/** SURF descriptors (Speeded Up Robust Features).
		 */
		void getSURF(cv::Mat &feature, cv::Mat image,\
			std::vector<unsigned> borders, cv::Mat thresholded);

		/** Blob detector in RGB color space.
		 */
		void blobDetector(cv::Mat &feature, cv::Mat image, std::vector<unsigned> \
			borders, cv::Mat thresholded);

		/** Just displaying an image a bit larger to visualize it better.
		 */
		void showZoomedImage(cv::Mat image, std::string title="zoomed");

		/** Head detection by fitting ellipses (if \i templateCenter is relative to
		 * the \i img the offset needs to be used).
		 */
		void skinEllipses(cv::RotatedRect &finalBox, cv::Mat img, cv::Point \
		templateCenter, cv::Point offset=cv::Point(0,0), double minHeadSize=20,\
		double maxHeadSize=40);

		/** Convolves an image with a Gabor filter with the given parameters and
		 * returns the response image.
		 */
		void getGabor(cv::Mat &response, cv::Mat image, float *params = NULL);


		/** Set what kind of features to extract.
		 */
		void setFeatureType(FEATURE type);

		/** Creates on data row in the final data matrix by getting the feature
		 * descriptors.
		 */
		void extractDataRow(std::vector<unsigned> existing, IplImage *bg);

		/** For each row added in the data matrix (each person detected for which we
		 * have extracted some features) find the corresponding label.
		 */
		void fixLabels(std::vector<cv::Point> feetPos, string imageName,\
		unsigned index);

		/** Returns the size of a window around a template centered in a given point.
		 */
		void templateWindow(cv::Size imgSize, int &minX, int &maxX,\
		int &minY, int &maxY, std::vector<CvPoint> &templ, unsigned tplBorder = 60);

		/** Initializes the parameters of the tracker.
		 */
		void init(std::string dataFolder, std::string theAnnotationsFile);

		/** Checks to see if an annotation can be assigned to a detection.
		 */
		bool canBeAssigned(unsigned l,std::vector<double> &minDistances,\
		unsigned k,double distance, std::vector<int> &assignment);
		//======================================================================
	public:
		/** @var plotTracks
		 * If it is true it displays the tracks of the people in the images.
		 */
		bool plotTracks;

		/** @var featureType
		 * Can have one of the values indicating the feature type.
		 */
		FEATURE featureType;

		/** @var data
		 * The training data obtained from the feature descriptors.
		 */
		std::vector<cv::Mat_<double> > data;

		/** @var data
		 * The targets/labels of the data.
		 */
		std::vector<cv::Mat_<double> > targets;

		/** @var annotations
		 * Loaded annotations for the read images.
		 */
		std::vector<annotationsHandle::FULL_ANNOTATIONS> targetAnno;

		/** @var lastIndex
		 * The previous size of the data matrix before adding new detections.
		 */
		unsigned lastIndex;
};
#endif /* FESTUREDETECTOR_H_ */
