/* FeatureExtractor.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#ifndef FEATUREEXTRACTOR_H_
#define FEATUREEXTRACTOR_H_
#include <deque>
#include "eigenbackground/src/Helpers.hh"
/** Extracts the actual features from the images and stores them in data matrix.
 */
class FeatureExtractor {
	public:
		FeatureExtractor();
		virtual ~FeatureExtractor();
		/** Structure for storing keypoints and descriptors.
		 */
		struct keyDescr {
			public:
				cv::KeyPoint keys_;
				std::deque<float> descr_;
				keyDescr(){};
				virtual ~keyDescr(){
					if(!this->descr_.empty()){
						this->descr_.clear();
					}
				}
				keyDescr(const keyDescr &kdescr){
					this->keys_  = kdescr.keys_;
					this->descr_ = kdescr.descr_;
				}
				keyDescr& operator=(const keyDescr &kdescr){
					if(this == &kdescr) return *this;
					this->keys_  = kdescr.keys_;
					this->descr_ = kdescr.descr_;
					return *this;
				}
		};
		/** Structure containing images of the size of the detected people.
		 */
		struct people{
			public:
				cv::Point2f absoluteLoc_;
				cv::Point2f relativeLoc_;
				std::deque<unsigned> borders_;
				cv::Mat pixels_;
				cv::Mat thresh_;
				people(){
					this->absoluteLoc_ = cv::Point2f(0,0);
					this->relativeLoc_ = cv::Point2f(0,0);
					this->thresh_      = cv::Mat();
					this->pixels_      = cv::Mat();
				}
				virtual ~people(){
					if(!this->borders_.empty()){
						this->borders_.clear();
					}
					if(!this->pixels_.empty()){
						this->pixels_.release();
					}
					if(!this->thresh_.empty()){
						this->thresh_.release();
					}
				}
				people(const people &person){
					this->absoluteLoc_ = person.absoluteLoc_;
					this->relativeLoc_ = person.relativeLoc_;
					this->borders_     = person.borders_;
					if(!this->pixels_.empty()){
						this->pixels_.release();
					}
					if(!this->thresh_.empty()){
						this->thresh_.release();
					}
					person.pixels_.copyTo(this->pixels_);
					person.thresh_.copyTo(this->thresh_);
				}
				people& operator=(const people &person){
					if(this == &person) return *this;
					this->absoluteLoc_ = person.absoluteLoc_;
					this->relativeLoc_ = person.relativeLoc_;
					this->borders_     = person.borders_;
					if(!this->pixels_.empty()){
						this->pixels_.release();
					}
					if(!this->thresh_.empty()){
						this->thresh_.release();
					}
					person.pixels_.copyTo(this->pixels_);
					person.thresh_.copyTo(this->thresh_);
					return *this;
				}
		};
		/** Structure to store templates so they don't get recomputed all the time.
		 */
		struct templ{
			public:
				cv::Point2f center_;
				cv::Point2f head_;
				std::deque<float> extremes_;
				std::vector<cv::Point2f> points_;
				templ(cv::Point theCenter){
					this->center_ = theCenter;
				}
				virtual ~templ(){
					this->extremes_.clear();
					this->points_.clear();
				}
				templ(const templ &aTempl){
					this->center_   = aTempl.center_;
					this->head_     = aTempl.head_;
					this->extremes_ = aTempl.extremes_;
					this->points_   = aTempl.points_;
				}
				templ& operator=(const templ &aTempl){
					if(this == &aTempl) return *this;
					this->center_   = aTempl.center_;
					this->head_     = aTempl.head_;
					this->extremes_ = aTempl.extremes_;
					this->points_   = aTempl.points_;
					return *this;
				}
		};
		/** What values can be used for the feature part to be extracted.
		 */
		enum FEATUREPART {TOP,BOTTOM,WHOLE,HEAD};
		/** All available feature types.
		 */
		enum FEATURE {EDGES,GABOR,HOG,IPOINTS,RAW_PIXELS,SIFT,SIFT_DICT,SURF,\
			TEMPL_MATCHES};
		/** What needs to be rotated.
		 */
		enum ROTATE {MATRIX,TEMPLATE,KEYS};
		/** Initializes the class elements.
		 */
		void init(const std::deque<FeatureExtractor::FEATURE> &fType,\
			const std::string &featFile,int colorSp,int invColorSp,\
			FeatureExtractor::FEATUREPART part);
		/** Resets the variables to the default values.
		 */
		void reset();
		/** Initializes the settings for the SIFT dictionary.
		 */
		void initSIFT(const std::string &dictName,unsigned means=500,unsigned size=128);
		/** Creates a data matrix for each image and stores it locally.
		 */
		void extractFeatures(cv::Mat &image,const std::string &sourceName);
		/** Extract the interest points in a gird and returns them.
		 */
		cv::Mat extractPointsGrid(cv::Mat &image);
		/** Extract edges from the whole image.
		 */
		cv::Mat extractEdges(cv::Mat &image);
		/** Convolves the whole image with some Gabors wavelets and then stores the
		 * results.
		 */
		cv::Mat extractGabor(cv::Mat &image);
		/** Extracts SIFT features from the image and stores them in a matrix.
		 */
		cv::Mat extractSIFT(cv::Mat &image,const std::vector<cv::Point2f> &templ,\
			const cv::Rect &roi);
		/** Extracts all the surf descriptors from the whole image and writes them in a
		 * matrix.
		 */
		cv::Mat extractSURF(cv::Mat &image);
		/** Gets the plain pixels corresponding to the upper part of the body.
		 */
		cv::Mat getTemplMatches(const FeatureExtractor::people &person,\
			const FeatureExtractor::templ &aTempl,const cv::Rect &roi);
		/** Gets the HOG descriptors over an image.
		 */
		cv::Mat getHOG(const FeatureExtractor::people &person,\
			const FeatureExtractor::templ &aTempl,const cv::Rect &roi);
		/** Gets the edges in an image.
		 */
		cv::Mat getEdges(cv::Mat &feature,const cv::Mat &thresholded,\
			const cv::Rect &roi,const FeatureExtractor::templ &aTempl,float rotAngle,\
			bool contours = false);
		/** SURF descriptors (Speeded Up Robust Features).
		 */
		cv::Mat getSURF(cv::Mat &feature,const std::vector<cv::Point2f> &templ,\
			const cv::Rect &roi,const cv::Mat &test,std::vector<cv::Point2f> &indices);
		/** Compute the features from the SIFT descriptors by doing vector
		 * quantization.
		 */
		cv::Mat getSIFT(const cv::Mat &feature,const std::vector<cv::Point2f> &templ,\
			const cv::Rect &roi,const cv::Mat &test,std::vector<cv::Point2f> &indices);
		/** Creates a "histogram" of interest points + number of blobs.
		 */
		cv::Mat getPointsGrid(const cv::Mat &feature,const cv::Rect &roi,\
			const FeatureExtractor::templ &aTempl,const cv::Mat &test);
		/** Convolves an image with a Gabor filter with the given parameters and
		 * returns the response image.
		 */
		cv::Mat getGabor(cv::Mat &feature,const cv::Mat &thresholded,\
			const cv::Rect &roi,const cv::Size &foregrSize,\
			const FeatureExtractor::templ &aTempl,float rotAngle,int aheight);
		/** Gets the raw pixels corresponding to body of the person +/- background pixels.
		 */
		cv::Mat getRawPixels(const FeatureExtractor::people &person,\
			const FeatureExtractor::templ &aTempl,const cv::Rect &roi,\
			bool vChannel=true);
		/** Creates a gabor with the parameters given by the parameter vector.
		 */
		void createGabor(cv::Mat &gabor,float *params = NULL);
		/** Returns the row corresponding to the indicated feature type.
		 */
		cv::Mat getDataRow(int imageRows,const FeatureExtractor::templ &aTempl,\
			const cv::Rect &roi,const FeatureExtractor::people &person,\
			const std::string &imgName,cv::Point2f &absRotCenter,\
			cv::Point2f &rotBorders,float rotAngle,std::vector<cv::Point2f> &keys);
		/** Compares SURF 2 descriptors and returns the boolean value of their comparison.
		 */
		static bool compareDescriptors(const FeatureExtractor::keyDescr &k1,\
			const FeatureExtractor::keyDescr &k2);
		/** Checks to see if a given pixel is inside a template.
		 */
		static bool isInTemplate(unsigned pixelX,unsigned pixelY,\
			const std::vector<cv::Point2f> &templ);
		/** Rotate a matrix/a template/keypoints wrt to the camera location.
		 */
		void rotate2Zero(float rotAngle,FeatureExtractor::ROTATE what,\
			const cv::Rect roi,cv::Point2f &rotCenter,cv::Point2f &rotBorders,\
			std::vector<cv::Point2f> &pts,cv::Mat &toRotate);
		/**Return number of means.
		 */
		unsigned readNoMeans();
		/**Return name of the SIFT dictionary.
		 */
		std::string readDictName();
		/** Sets the image class and resets the dictionary name.
		 */
		unsigned setImageClass(unsigned aClass);
		/** Find the extremities of the thresholded image.
		 */
		void getThresholdBorderes(int &minX,int &maxX,int &minY,\
			int &maxY,const cv::Mat &thresh);
		/** Cut the image around the template or bg bordered depending on which
		 * is used and resize to a common size.
		 */
		cv::Mat cutAndResizeImage(const cv::Rect &roiCut,const cv::Mat &img);
		/** Find if a feature type is in the vector of features.
		 */
		static bool isFeatureIn(std::deque<FeatureExtractor::FEATURE> feats,\
			FeatureExtractor::FEATURE feat);
	//==========================================================================
	private:
		/** @var isInit_
 		 * It is true if the class was already initialized.
 		 */
		bool isInit_;
		/** @var data_
 		 * The matrix in which the features are stored.
 		 */
		cv::Mat data_;
		/** @var featureType_
		 * Indicates the type of the feature to be used.
		 */
		std::deque<FeatureExtractor::FEATURE> featureType_;
		/** @var dictFilename_
		 * The SIFT dictionary used for vector quantization.
		 */
		std::string dictFilename_;
		/** @var noMeans_
		 * The number of means used for kmeans.
		 */
		unsigned noMeans_;
		/** @var meanSize_
		 * The meanSize (128 for regular SIFT features) .
		 */
		unsigned meanSize_;
		/** @var featureFile_
		 * The folder were the features are stored for each image.
		 */
		std::string featureFile_;
		/** @var print_
		 * Print some values out or not.
		 */
		bool print_;
		/** @var plot_
		 * Plot some features or not.
		 */
		bool plot_;
		/** @var dictionarySIFT_
		 * The SIFT dictionary loaded as from the file where is stored.
		 */
		cv::Mat dictionarySIFT_;
		/** @var imageClass_
		 * The class of the image corresponding to its position wrt the camera.
		 */
		std::string imageClass_;
		/** @var invColorspaceCode_
		 * The code to be used to convert an image to gray.
		 */
		int invColorspaceCode_;
		/** @var colorspaceCode_
		 * The code to be used to convert an image to gray.
		 */
		int colorspaceCode_;
		/** @var bodyPart_
		 * It is true if only the upper/lower part of the body is used for training.
		 */
		FeatureExtractor::FEATUREPART bodyPart_;
		//======================================================================
	private:
		DISALLOW_COPY_AND_ASSIGN(FeatureExtractor);
};
//==============================================================================
//==============================================================================
/** Define a post-fix increment operator for the enum \c FEATURE.
 */
void operator++(FeatureExtractor::FEATURE &feature);
//==============================================================================
#endif /* FEATUREEXTRACTOR_H_ */
