/* PeopleDetector.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#ifndef PEOPLEDETECTOR_H_
#define PEOPLEDETECTOR_H_
#include <tr1/memory>
#include "eigenbackground/src/Helpers.hh"
#include "eigenbackground/src/Tracker.hh"
#include "FeatureExtractor.h"
#include "AnnotationsHandle.h"
/** Class used for detecting useful features in the images that can be later
 * used for training and classifying.
 */
class PeopleDetector:public Tracker{
	public:
		/** Structure to store the existing/detected locations.
		 */
		struct DataRow{
			std::string imgName_;
			cv::Point2f location_;
			unsigned int groupNo_;
			cv::Mat testRow_;
			cv::Mat testTarg_;
			DataRow(const cv::Point2f &exi,unsigned int grNo,std::string name,\
			const cv::Mat &row,const cv::Mat &targ):imgName_(name),location_(exi),\
			groupNo_(grNo){
				row.copyTo(this->testRow_);
				targ.copyTo(this->testTarg_);
			}
			virtual ~DataRow(){
				this->testRow_.release();
				this->testTarg_.release();
			}
			DataRow(const DataRow &exi){
				this->location_ = exi.location_;
				this->groupNo_  = exi.groupNo_;
				this->imgName_  = exi.imgName_;
				exi.testRow_.copyTo(this->testRow_);
				exi.testTarg_.copyTo(this->testTarg_);
			}
			DataRow& operator=(const DataRow &exi){
				if(this == &exi) return *this;
				this->location_ = exi.location_;
				this->groupNo_  = exi.groupNo_;
				this->imgName_  = exi.imgName_;
				this->testRow_.release();
				this->testTarg_.release();
				exi.testRow_.copyTo(this->testRow_);
				exi.testTarg_.copyTo(this->testTarg_);
				return *this;
			}
		};

		/** Structure to store the existing/detected locations.
		 */
		struct Existing{
			cv::Point2f location_;
			unsigned int groupNo_;
			Existing(const cv::Point2f &exi=cv::Point2f(0,0),unsigned int grNo=0):\
				location_(exi),groupNo_(grNo){}
			virtual ~Existing(){}
			Existing(const Existing &exi){
				this->location_ = exi.location_;
				this->groupNo_  = exi.groupNo_;
			}
			Existing& operator=(const Existing &exi){
				if(this == &exi) return *this;
				this->location_ = exi.location_;
				this->groupNo_  = exi.groupNo_;
				return *this;
			}
		};
		/** Classes/groups (wrt the camera) in which to store the image data.
		 */
		enum CLASSES {CLOSE,MEDIUM,FAR};
		PeopleDetector(int argc,char** argv,bool extract=false,bool buildBg=\
			false,int colorSp=-1,FeatureExtractor::FEATUREPART part=\
			FeatureExtractor::WHOLE,bool flip=true);
		virtual ~PeopleDetector();
		/** Overwrites the \c doFindPeople function from the \c Tracker class
		 * to make it work with the feature extraction.
		 */
		virtual bool doFindPerson(unsigned imgNum,IplImage *src,\
			const vnl_vector<float> &imgVec,vnl_vector<float> &bgVec,\
			const float logBGProb,const vnl_vector<float> &logSumPixelBGProb);
		/** Simple "menu" for skipping to the next image or quitting the processing.
		 */
		bool imageProcessingMenu();
		/** Get the foreground pixels corresponding to each person.
		 */
		void allForegroundPixels(std::deque<FeatureExtractor::people> &allPeople,\
			const IplImage *bg,float threshold);
		/** Gets the distance to the given template from a given pixel location.
		 */
		float getDistToTemplate(const int pixelX,const int pixelY,\
			const std::vector<cv::Point2f> &templ);
		/** Creates on data row in the final data matrix by getting the feature
		 * descriptors.
		 */
		void extractDataRow(const IplImage *oldBg,bool flip,\
			const std::deque<unsigned> &existing=std::deque<unsigned>(),\
			float threshVal=50.0);
//			float threshVal=25.0);
//			float threshVal=60.0);
		/** For each row added in the data matrix (each person detected for which we
		 * have extracted some features) find the corresponding label.
		 */
		void fixLabels(const std::deque<unsigned> &existing,bool flip);
		/** Returns the size of a window around a template centered in a given point.
		 */
		void templateWindow(const cv::Size &imgSize,int &minX,int &maxX,\
		int &minY,int &maxY,const FeatureExtractor::templ &aTempl);
		/** Initializes the parameters of the tracker.
		 */
		void init(const std::string &dataFolder,const std::string &theAnnotationsFile,\
			const std::deque<FeatureExtractor::FEATURE> &feat,bool test,\
			bool readFromFolder = true);
		/** Checks to see if an annotation can be assigned to a detection.
		 */
		bool canBeAssigned(unsigned l,std::deque<float> &minDistances,\
			unsigned k,float distance,std::deque<int> &assignment);
		/** Fixes the angle to be relative to the camera position with respect to the
		 * detected position.
		 */
		float fixAngle(const cv::Point2f &feetLocation,\
			const cv::Point2f &cameraLocation,float angle,bool flip);
		/** Un-does the rotation with respect to the camera.
		 */
		float unfixAngle(const cv::Point2f &headLocation,\
			const cv::Point2f &feetLocation,float angle);
		/** Get template extremities (if needed,considering some borders --
		 * relative to the ROI).
		 */
		void templateExtremes(const std::vector<cv::Point2f> &templ,\
			std::deque<float> &extremes,int minX = 0,int minY = 0);
		/** If only a part needs to be used to extract the features then the threshold
		 * and the template need to be changed.
		 */
		void templatePart(int k,FeatureExtractor::people &person);
		/** Computes the motion vector for the current image given the tracks so far.
		 */
		float motionVector(const cv::Point2f &head,const cv::Point2f &center,\
			bool flip,bool &moved);
		/** Compute the dominant direction of the SIFT or SURF features.
		 */
		float opticalFlow(cv::Mat &currentImg,cv::Mat &nextImg,\
			const std::vector<cv::Point2f> &keyPts,const cv::Point2f &head,\
			const cv::Point2f &center,bool maxOrAvg,bool flip);
		/** Keeps only the largest blob from the thresholded image.
		 */
		void keepLargestBlob(cv::Mat &thresh,const cv::Point2f &center,\
			float tmplArea);
		/** Reads the locations at which there are people in the current frame (for the
		 * case in which we do not want to use the tracker or build a bgModel).
		 */
		void readLocations(bool flip);
		/** Starts running something (either the tracker or just mimics it).
		 */
		void start(bool readFromFolder,bool useGT);
		/** Adds a templates to the vector of templates at detected positions.
		 */
		void add2Templates();
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
		void fixLocationsTracksBorderes(const std::deque<unsigned> &existing,\
			bool flip);
		/** Initialize the inverse value of the color space used in feature extraction.
		 */
		void initInvColoprSp();
		/** Find the class in which we can store the current image (the data is split in
		 * 3 classes depending on the position of the person wrt camera).
		 */
		PeopleDetector::CLASSES findImageClass(const cv::Point2f &feet,\
			const cv::Point2f &head,bool oneClass=true);
		/** Get distance wrt the camera in the image.
		 */
		float distanceWRTcamera(const cv::Point2f &feet);
		/** Applies PCA on top of a data-row to reduce its dimensionality.
		 */
		cv::Mat reduceDimensionality(const cv::Mat &data,int nEigens=0,\
			int reshapeRows=0);
		/** Extracts a circle around the predicted/annotated head positon.
		 */
		void extractHeadArea(int i,FeatureExtractor::people &person);
		std::vector<cv::Mat> data();
		std::vector<cv::Mat> targets();
		std::deque<std::deque<float> > dataMotionVectors();
		std::tr1::shared_ptr<FeatureExtractor> extractor();
		void setFlip(bool flip);
		/** Draws the target orientation and the predicted orientation on the image.
		 */
		void drawPredictions(const cv::Point2f &pred,std::tr1::shared_ptr\
			<PeopleDetector::DataRow> dataRow);
		/** Returns the last element in the data vector.
		 */
		std::tr1::shared_ptr<PeopleDetector::DataRow> popDataRow();
		/** Returns the data info size.
		 */
		unsigned dataInfoSize();
		//======================================================================
		/** @var dataMutex_
		 * Used to check if the data is produced or not.
		 */
		static boost::mutex dataMutex_;
		/** @var dataIsProduced_
		 * It is true if the data is produced.
		 */
		static bool dataIsProduced_;
	private:
		/** @var print_
		 * To print some feature values or not.
		 */
		bool print_;
		/** @var plot_
		 * If it is true it displays the tracks of the people in the images.
		 */
		bool plot_;
		/** @var data_
		 * The training data obtained from the feature descriptors (for 3
		 * positions wrt the camera).
		 */
		std::vector<cv::Mat> data_;
		/** @var targets_
		 * The targets/labels of the data (for 3 positions wrt the camera).
		 */
		std::vector<cv::Mat> targets_;
		/** @var targetAnno_
		 * Loaded annotations for the read images.
		 */
		std::deque<AnnotationsHandle::FULL_ANNOTATIONS> targetAnno_;
		/** @var colorspaceCode_
		 * The colorspace code to be used before extracting the features.
		 */
		int colorspaceCode_;
		/** @var invColorspaceCode_
		 * The code to be used to convert an image to gray.
		 */
		int invColorspaceCode_;
		/** @var featurePart_
		 * Indicates if the part from the image to be used (feet,head,or both).
		 */
		FeatureExtractor::FEATUREPART featurePart_;
		/** @var tracking_
		 * If the data is sequential motion information can be used.
		 */
		unsigned tracking_;
		/** @var entireNext_
		 * The the previous image.
		 */
		cv::Mat entireNext_;
		/** @var onlyExtract_
		 * If only the features need to be extracted or the data.
		 */
		bool onlyExtract_;
		/** @var useGroundTruth_
		 * Use ground truth to detect the people instead.
		 */
		bool useGroundTruth_;
		/** @var extractor_
		 * An instance of the class FeatureExtractor.
		 */
		std::tr1::shared_ptr<FeatureExtractor> extractor_;
		/** @var datasetPath_
		 * The path to the dataset to be used.
		 */
		std::string datasetPath_;
		/** @var imageString_
		 * The string that appears in the name of the images.
		 */
		std::string imageString_;
		/** @var templates_
		 * The vector of templates for each image.
		 */
		std::vector<FeatureExtractor::templ> templates_;
		/** @var dataMotionVectors_
		 * The motion vectors for all the images in the data matrix
		 */
		std::deque<std::deque<float> > dataMotionVectors_;
		/** @var classesRange_
		 * The minimum/maximum template size for each class as a vector of points
		 */
		std::vector<cv::Point2f> classesRange_;
		/** @var existing_
		 * Stores the annotated/detected locations of the people and the
		 * corresponding class.
		 */
		std::vector<PeopleDetector::Existing> existing_;
		/** @var flip_
		 * To flip the images or not.
		 */
		bool flip_;
		/** @var dataInfo_;
		 * Information regarding each row in the data matrix.
		 */
		std::deque<PeopleDetector::DataRow> dataInfo_;
		/** @var isTest_;
		 * True if the data is the test set.
		 */
		bool isTest_;
		//======================================================================
	private:
		DISALLOW_COPY_AND_ASSIGN(PeopleDetector);
};
//==============================================================================
/** Define a post-fix increment operator for the enum \c POSE.
 */
void operator++(PeopleDetector::CLASSES &refClass);
//==============================================================================
#endif /* PEOPLEDETECTOR_H_ */
