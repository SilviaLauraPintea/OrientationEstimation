/* peopleDetector.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#ifndef PEOPLEDETECTOR_H_
#define PEOPLEDETECTOR_H_
#include <opencv2/opencv.hpp>
#include <vnl/vnl_vector.h>
#include "eigenbackground/src/defines.hh"
#include "eigenbackground/src/Tracker.hh"
#include "featureExtractor.h"
#include "annotationsHandle.h"
/** Class used for detecting useful features in the images that can be later
 * used for training and classifying.
 */
class peopleDetector:public Tracker{
	public:
		peopleDetector(int argc,char** argv,bool extract=false,bool buildBg=\
			false,int colorSp=-1);
		virtual ~peopleDetector();
		/** Classes/groups (wrt the camera) in which to store the image data.
		 */
		enum CLASSES {CLOSE,MEDIUM,FAR};
		/** What values can be used for the feature part to be extracted.
		 */
		enum FEATUREPART {TOP,BOTTOM,WHOLE};
		/** Overwrites the \c doFindPeople function from the \c Tracker class
		 * to make it work with the feature extraction.
		 */
		bool doFindPerson(unsigned imgNum,IplImage *src,\
			const vnl_vector<FLOAT> &imgVec,vnl_vector<FLOAT> &bgVec,\
			const FLOAT logBGProb,const vnl_vector<FLOAT> &logSumPixelBGProb);
		/** Simple "menu" for skipping to the next image or quitting the processing.
		 */
		bool imageProcessingMenu();
		/** Get the foreground pixels corresponding to each person.
		 */
		void allForegroundPixels(std::deque<featureExtractor::people> &allPeople,\
			const std::deque<unsigned> &existing,const IplImage *bg,\
			float threshold);
		/** Gets the distance to the given template from a given pixel location.
		 */
		float getDistToTemplate(const int pixelX,const int pixelY,\
			const std::vector<cv::Point2f> &templ);
		/** Creates on data row in the final data matrix by getting the feature
		 * descriptors.
		 */
		void extractDataRow(std::deque<unsigned> &existing,const IplImage *oldBg);
		/** For each row added in the data matrix (each person detected for which we
		 * have extracted some features) find the corresponding label.
		 */
		void fixLabels(std::deque<unsigned> &existing);
		/** Returns the size of a window around a template centered in a given point.
		 */
		void templateWindow(const cv::Size &imgSize,int &minX,int &maxX,\
		int &minY,int &maxY,const featureExtractor::templ &aTempl);
		/** Initializes the parameters of the tracker.
		 */
		void init(const std::string &dataFolder,const std::string &theAnnotationsFile,\
			const featureExtractor::FEATURE feat,const bool readFromFolder = true);
		/** Checks to see if an annotation can be assigned to a detection.
		 */
		bool canBeAssigned(unsigned l,std::deque<float> &minDistances,\
			unsigned k,float distance,std::deque<int> &assignment);
		/** Fixes the angle to be relative to the camera position with respect to the
		 * detected position.
		 */
		float fixAngle(const cv::Point2f &feetLocation,\
			const cv::Point2f &cameraLocation,float angle);
		/** Get template extremities (if needed,considering some borders --
		 * relative to the ROI).
		 */
		void templateExtremes(const std::vector<cv::Point2f> &templ,\
			std::deque<float> &extremes,int minX = 0,int minY = 0);
		/** If only a part needs to be used to extract the features then the threshold
		 * and the template need to be changed.
		 */
		void templatePart(cv::Mat &thresholded,int k,float offsetX,float offsetY);
		/** Computes the motion vector for the current image given the tracks so far.
		 */
		float motionVector(const cv::Point2f &head,const cv::Point2f &center,\
			bool &moved);
		/** Compute the dominant direction of the SIFT or SURF features.
		 */
		float opticalFlow(cv::Mat &currentImg,cv::Mat &nextImg,\
			const std::vector<cv::Point2f> &keyPts,const cv::Point2f &head,\
			const cv::Point2f &center,bool maxOrAvg);
		/** Keeps only the largest blob from the thresholded image.
		 */
		void keepLargestBlob(cv::Mat &thresh,const cv::Point2f &center,\
			float tmplArea);
		/** Reads the locations at which there are people in the current frame (for the
		 * case in which we do not want to use the tracker or build a bgModel).
		 */
		void readLocations(std::deque<unsigned> &locations);
		/** Starts running something (either the tracker or just mimics it).
		 */
		void start(bool readFromFolder,bool useGT);
		/** Adds a templates to the vector of templates at detected positions.
		 */
		void add2Templates(const std::deque<unsigned> &existing);
		/** Assigns pixels to templates based on proximity.
		 */
		void pixels2Templates(int maxX,int minX,int maxY,int minY,int k,\
			const cv::Mat &thresh,float tmplHeight,cv::Mat &colorRoi);
		/** Return rotation angle given the head and feet position.
		 */
		float rotationAngle(const cv::Point2f &headLocation,\
			const cv::Point2f &feetLocation);
		/** Fixes the existing/detected locations of people and updates the tracks and
		 * creates the bordered image.
		 */
		void fixLocationsTracksBorderes(std::deque<unsigned> &existing);
		/** Find the class in which we can store the current image (the data is
		 * split in 3 classes depending on the position of the person wrt camera).
		 */
		peopleDetector::CLASSES findImageClass(const cv::Point2f &feet,\
			const cv::Point2f &head);
		/** Initialize the inverse value of the color space used in feature extraction.
		 */
		void initInvColoprSp();
		//======================================================================
	public:
		/** @var print
		 * To print some feature values or not.
		 */
		bool print;
		/** @var plot
		 * If it is true it displays the tracks of the people in the images.
		 */
		bool plot;
		/** @var data
		 * The training data obtained from the feature descriptors (for 3
		 * positions wrt the camera).
		 */
		std::vector<cv::Mat> data;
		/** @var targets
		 * The targets/labels of the data (for 3 positions wrt the camera).
		 */
		std::vector<cv::Mat> targets;
		/** @var annotations
		 * Loaded annotations for the read images.
		 */
		std::deque<annotationsHandle::FULL_ANNOTATIONS> targetAnno;
		/** @var colorspaceCode
		 * The colorspace code to be used before extracting the features.
		 */
		int colorspaceCode;
		/** @var invColorspaceCode
		 * The code to be used to convert an image to gray.
		 */
		int invColorspaceCode;
		/** @var featurePart
		 * Indicates if the part from the image to be used (feet,head,or both).
		 */
		peopleDetector::FEATUREPART featurePart;
		/** @var tracking
		 * If the data is sequential motion information can be used.
		 */
		unsigned tracking;
		/** @var entireNext
		 * The the previous image.
		 */
		cv::Mat entireNext;
		/** @var onlyExtract
		 * If only the features need to be extracted or the data.
		 */
		bool onlyExtract;
		/** @var useGroundTruth
		 * Use ground truth to detect the people instead.
		 */
		bool useGroundTruth;
		/** @var extractor
		 * An instance of the class featureExtractor.
		 */
		featureExtractor *extractor;
		/** @var datasetPath
		 * The path to the dataset to be used.
		 */
		std::string datasetPath;
		/** @var imageString
		 * The string that appears in the name of the images.
		 */
		std::string imageString;
		/** @var imageString
		 * The string that appears in the name of the images.
		 */
		std::vector<featureExtractor::templ> templates;
		/** @var dataMotionVectors
		 * The motion vectors for all the images in the data matrix
		 */
		std::deque<std::deque<float> > dataMotionVectors;
		/** The minimum/maximum template size for each class as a vector of points
		 */
		std::vector<cv::Point2f> classesRange;
		//======================================================================
	private:
		DISALLOW_COPY_AND_ASSIGN(peopleDetector);
};
//==============================================================================
/** Define a post-fix increment operator for the enum \c POSE.
 */
void operator++(peopleDetector::CLASSES &refClass);
//==============================================================================
#endif /* PEOPLEDETECTOR_H_ */
