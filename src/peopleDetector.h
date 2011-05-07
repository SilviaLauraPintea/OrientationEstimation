/* peopleDetector.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code, but please retain the above copyright notice.
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
		peopleDetector(int argc,char** argv, bool extract=false, bool buildBg=\
			false);
		virtual ~peopleDetector();
		/** What values can be used for the feature part to be extracted.
		 */
		enum FEATUREPART {TOP, BOTTOM, WHOLE};
		/** Overwrites the \c doFindPeople function from the \c Tracker class
		 * to make it work with the feature extraction.
		 */
		bool doFindPerson(unsigned imgNum, IplImage *src,\
			const vnl_vector<FLOAT> &imgVec, vnl_vector<FLOAT> &bgVec,\
			const FLOAT logBGProb,const vnl_vector<FLOAT> &logSumPixelBGProb);
		/** Simple "menu" for skipping to the next image or quitting the processing.
		 */
		bool imageProcessingMenu();
		/** Get the foreground pixels corresponding to each person.
		 */
		void allForegroundPixels(std::deque<featureExtractor::people> &allPeople,\
			std::deque<unsigned> existing, IplImage *bg, float threshold);
		/** Gets the distance to the given template from a given pixel location.
		 */
		float getDistToTemplate(int pixelX,int pixelY,std::vector<cv::Point2f> templ);
		/** Creates on data row in the final data matrix by getting the feature
		 * descriptors.
		 */
		void extractDataRow(std::deque<unsigned> &existing, IplImage *bg);
		/** For each row added in the data matrix (each person detected for which we
		 * have extracted some features) find the corresponding label.
		 */
		std::deque<unsigned> fixLabels(std::deque<unsigned> existing);
		/** Returns the size of a window around a template centered in a given point.
		 */
		void templateWindow(IplImage *img,cv::Size imgSize,int &minX,int &maxX,\
			int &minY,int &maxY,featureExtractor::templ aTempl,int tplBorder=150);
		/** Initializes the parameters of the tracker.
		 */
		void init(std::string dataFolder, std::string theAnnotationsFile,\
			featureExtractor::FEATURE feat, bool readFromFolder = true);
		/** Checks to see if an annotation can be assigned to a detection.
		 */
		bool canBeAssigned(unsigned l,std::deque<float> &minDistances,\
			unsigned k,float distance, std::deque<int> &assignment);
		/** Fixes the angle to be relative to the camera position with respect to the
		 * detected position.
		 */
		float fixAngle(cv::Point2f feetLocation, cv::Point2f cameraLocation,\
			float angle);
		/** Get template extremities (if needed, considering some borders --
		 * relative to the ROI).
		 */
		std::deque<float> templateExtremes(std::vector<cv::Point2f> templ,\
			int minX = 0, int minY = 0);
		/** If only a part needs to be used to extract the features then the threshold
		 * and the template need to be changed.
		 */
		void templatePart(cv::Mat &thresholded,int k,float offsetX,float offsetY);
		/** Computes the motion vector for the current image given the tracks so far.
		 */
		float motionVector(cv::Point2f head, cv::Point2f center);
		/** Compute the dominant direction of the SIFT or SURF features.
		 */
		float opticalFlow(cv::Mat currentImg, cv::Mat nextImg,\
			std::vector<cv::Point2f> keyPts,cv::Point2f head, cv::Point2f center,\
			bool maxOrAvg);
		/** Keeps only the largest blob from the thresholded image.
		 */
		void keepLargestBlob(cv::Mat &thresh, cv::Point2f center,\
			float tmplArea);
		/** Reads the locations at which there are people in the current frame (for the
		 * case in which we do not want to use the tracker or build a bgModel).
		 */
		std::deque<unsigned> readLocations();
		/** Starts running something (either the tracker or just mimics it).
		 */
		void start(bool readFromFolder, bool useGT);
		/** Adds a templates to the vector of templates at detected positions.
		 */
		void add2Templates(std::deque<unsigned> existing);
		/** Assigns pixels to templates based on proximity.
		 */
		void pixels2Templates(int maxX,int minX,int maxY,int minY, int k,\
			cv::Mat thresh, cv::Mat &colorRoi, float tmplHeight);
		/** Gets the location of the head given the feet location.
		 */
		cv::Point2f headLocation(cv::Point2f center);
		/** Return rotation angle given the head and feet position.
		 */
		float rotationAngle(cv::Point2f headLocation,cv::Point2f feetLocation);
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
		 * The training data obtained from the feature descriptors.
		 */
		cv::Mat data;
		/** @var data
		 * The targets/labels of the data.
		 */
		cv::Mat targets;
		/** @var annotations
		 * Loaded annotations for the read images.
		 */
		std::deque<annotationsHandle::FULL_ANNOTATIONS> targetAnno;
		/** @var lastIndex
		 * The previous size of the data matrix before adding new detections.
		 */
		unsigned lastIndex;
		/** @var colorspaceCode
		 * The colorspace code to be used before extracting the features.
		 */
		int colorspaceCode;
		/** @var featurePart
		 * Indicates if the part from the image to be used (feet, head, or both).
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
		//======================================================================
	private:
		DISALLOW_COPY_AND_ASSIGN(peopleDetector);
};
#endif /* PEOPLEDETECTOR_H_ */
