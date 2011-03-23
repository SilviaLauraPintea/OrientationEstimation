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
#include <algorithm>
#include <fstream>
#include <string>
#include <cmath>
#include <exception>
#include <opencv2/opencv.hpp>
#include "eigenbackground/src/Tracker.hh"
#include "eigenbackground/src/Helpers.hh"
#include "eigenbackground/src/defines.hh"
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
			people(){
				this->absoluteLoc = cv::Point(0,0);
				this->relativeLoc = cv::Point(0,0);
			}
			~people(){
				if(!this->borders.empty()){
					this->borders.clear();
				}
				if(!this->pixels.empty()){
					this->pixels.release();
				}
			}
		};

		/** All available feature types.
		 */
		enum FEATURE {IPOINTS, EDGES, SIFT_DICT, SURF, SIFT, GABOR};

		/** Structure for storing keypoints and descriptors.
		 */
		struct keyDescr {
			cv::KeyPoint keys;
			std::vector<float> descr;
			~keyDescr(){
				if(!this->descr.empty()){
					this->descr.clear();
				}
			}
		};
		//======================================================================
		featureDetector(int argc,char** argv):Tracker(argc, argv, 10, true, true){
			this->plotTracks     = false;
			this->featureType    = EDGES;
			this->lastIndex      = 0;
			this->producer       = NULL;
			this->dictFileName   = const_cast<char*>("dictSIFT.bin");
			this->noMeans        = 10;
			this->meanSize       = 128;
			this->colorspaceCode = CV_BGR2Lab;
			this->featurePart    = 't';
		}

		virtual ~featureDetector(){
			if(this->producer){
				delete this->producer;
				this->producer = NULL;
			}
			// CLEAR DATA AND TARGETS
			if(!this->targets.empty()){
				for(std::size_t i=0; i<this->targets.size(); i++){
					this->targets[i].release();
				}
				this->targets.clear();
			}
			if(!this->data.empty()){
				for(std::size_t i=0; i<this->data.size(); i++){
					this->data[i].release();
				}
				this->data.clear();
			}

			// CLEAR THE ANNOTATIONS
			if(!this->targetAnno.empty()){
				this->targetAnno.clear();
			}
		}

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

		/** Gets the edges in an image.
		 */
		void getEdges(cv::Point center, cv::Mat &feature, cv::Mat image,\
			unsigned reshape=1, cv::Mat thresholded=cv::Mat());

		/** SURF descriptors (Speeded Up Robust Features).
		 */
		void getSURF(cv::Mat &feature, cv::Mat image, int minX, int minY,\
			std::vector<CvPoint> templ, cv::Point center);

		/** Compute the features from the SIFT descriptors by doing vector
		 * quantization.
		 */
		void getSIFT(cv::Mat &feature, cv::Mat image, int minX, int minY,\
			std::vector<CvPoint> templ, cv::Point center);

		/** SIFT descriptors (Scale Invariant Feature Transform).
		 */
		void extractSIFT(cv::Mat &feature, cv::Mat image, int minX, int minY,\
			std::vector<CvPoint> templ, cv::Point center);

		/** Creates a "histogram" of interest points + number of blobs.
		 */
		void interestPointsGrid(cv::Mat &feature, cv::Mat image,\
			std::vector<CvPoint> templ, int minX, int minY, cv::Point center);

		/** Just displaying an image a bit larger to visualize it better.
		 */
		void showZoomedImage(cv::Mat image, std::string title="zoomed");

		/** Head detection by fitting ellipses (if \i templateCenter is relative to
		 * the \i img the offset needs to be used).
		 */
		void skinEllipses(cv::RotatedRect &finalBox, cv::Mat img, cv::Point \
		templateCenter, cv::Point offset=cv::Point(0,0), double minHeadSize=20,\
		double maxHeadSize=40);

		/** Creates a gabor with the parameters given by the parameter vector.
		 */
		cv::Mat createGabor(double *params = NULL);

		/** Convolves an image with a Gabor filter with the given parameters and
		 * returns the response image.
		 */
		void getGabor(cv::Mat &feature,cv::Mat image,cv::Mat thresholded,\
			cv::Point center);

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
		void fixLabels(std::vector<cv::Point> feetPos);

		/** Returns the size of a window around a template centered in a given point.
		 */
		void templateWindow(cv::Size imgSize, int &minX, int &maxX,\
		int &minY, int &maxY, std::vector<CvPoint> &templ, int tplBorder = 100);

		/** Initializes the parameters of the tracker.
		 */
		void init(std::string dataFolder, std::string theAnnotationsFile);

		/** Checks to see if an annotation can be assigned to a detection.
		 */
		bool canBeAssigned(unsigned l,std::vector<double> &minDistances,\
		unsigned k,double distance, std::vector<int> &assignment);

		/** Fixes the angle to be relative to the camera position with respect to the
		 * detected position.
		 */
		double fixAngle(cv::Point feetLocation, cv::Point cameraLocation,\
			double angle);

		/** Compares SURF 2 descriptors and returns the boolean value of their comparison.
		 */
		static bool compareDescriptors(const featureDetector::keyDescr k1,\
			const featureDetector::keyDescr k2);

		/** Rotate matrix wrt to the camera location.
		 */
		cv::Mat rotateWrtCamera(cv::Point feetLocation, cv::Point cameraLocation,\
			cv::Mat toRotate, cv::Point &borders);

		/** Set the name of the file where the SIFT dictionary is stored.
		 */
		void setSIFTDictionary(char* fileSIFT);

		/** Rotate the points corresponding to the template wrt to the camera location.
		 */
		std::vector<CvPoint> rotateTemplWrtCamera(cv::Point feetLocation,\
			cv::Point cameraLocation, std::vector<CvPoint> templ,\
			cv::Point rotBorders, cv::Point2f rotCenter);

		/** Get template extremities (if needed, considering some borders --
		 * relative to the ROI).
		 */
		void templateExtremes(std::vector<CvPoint> templ, double\
			&minTmplX, double &maxTmplX, double &minTmplY, double &maxTmplY,\
			int minX = 0, int minY = 0);

		/** If only a part needs to be used to extract the features then the threshold
		 * and the template need to be changed.
		 */
		void onlyPart(cv::Mat &thresholded, std::vector<CvPoint> &templ,\
			double offsetX, double offsetY);

		/** Computes the motion vector for the current image given the tracks so far.
		 */
		double motionVector(cv::Point center);
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
		std::vector<cv::Mat> data;

		/** @var data
		 * The targets/labels of the data.
		 */
		std::vector<cv::Mat> targets;

		/** @var annotations
		 * Loaded annotations for the read images.
		 */
		std::vector<annotationsHandle::FULL_ANNOTATIONS> targetAnno;

		/** @var lastIndex
		 * The previous size of the data matrix before adding new detections.
		 */
		unsigned lastIndex;

		/** @var dictionarySIFT
		 * The SIFT dictionary used for vector quantization.
		 */
		cv::Mat dictionarySIFT;

		/** @var dictionarySIFT
		 * The SIFT dictionary used for vector quantization.
		 */
		char* dictFileName;

		/** @var noMeans
		 * The number of means used for kmeans.
		 */
		unsigned noMeans;

		/** @var meanSize
		 * The meanSize (128 for regular SIFT features) .
		 */
		unsigned meanSize;

		/** @var colorspaceCode
		 * The colorspace code to be used before extracting the features.
		 */
		int colorspaceCode;

		/** @var featurePart
		 * Indicates if the part from the image to be used (feet, head, or both).
		 */
		char featurePart;
};
#endif /* FESTUREDETECTOR_H_ */
