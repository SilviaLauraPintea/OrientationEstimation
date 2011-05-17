/* featureExtractor.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#ifndef FEATUREEXTRACTOR_H_
#define FEATUREEXTRACTOR_H_
#include <opencv2/opencv.hpp>
#include <deque>
/** Extracts the actual features from the images and stores them in data matrix.
 */
class featureExtractor {
	public:
		featureExtractor();
		virtual ~featureExtractor();
		/** Structure for storing keypoints and descriptors.
		 */
		struct keyDescr {
			public:
				cv::KeyPoint keys;
				std::deque<float> descr;
				keyDescr(){};
				virtual ~keyDescr(){
					if(!this->descr.empty()){
						this->descr.clear();
					}
				}
				keyDescr(const keyDescr &kdescr){
					this->keys  = kdescr.keys;
					this->descr = kdescr.descr;
				}
				keyDescr& operator=(const keyDescr &kdescr){
					if(this == &kdescr) return *this;
					this->keys  = kdescr.keys;
					this->descr = kdescr.descr;
					return *this;
				}
		};
		/** Structure containing images of the size of the detected people.
		 */
		struct people{
			public:
				cv::Point2f absoluteLoc;
				cv::Point2f relativeLoc;
				std::deque<unsigned> borders;
				cv::Mat_<cv::Vec3b> pixels;
				people(){
					this->absoluteLoc = cv::Point2f(0,0);
					this->relativeLoc = cv::Point2f(0,0);
				}
				virtual ~people(){
					if(!this->borders.empty()){
						this->borders.clear();
					}
					if(!this->pixels.empty()){
						this->pixels.release();
					}
				}
				people(const people &person){
					this->absoluteLoc = person.absoluteLoc;
					this->relativeLoc = person.relativeLoc;
					this->borders     = person.borders;
					if(!this->pixels.empty()){
						this->pixels.release();
					}
					person.pixels.copyTo(this->pixels);
				}
				people& operator=(const people &person){
					if(this == &person) return *this;
					this->absoluteLoc = person.absoluteLoc;
					this->relativeLoc = person.relativeLoc;
					this->borders     = person.borders;
					if(!this->pixels.empty()){
						this->pixels.release();
					}
					person.pixels.copyTo(this->pixels);
					return *this;
				}
		};
		/** Structure to store templates so they don't get recomputed all the time.
		 */
		struct templ{
			public:
				cv::Point2f center;
				cv::Point2f head;
				std::deque<float> extremes;
				std::vector<cv::Point2f> points;
				templ(cv::Point theCenter){
					this->center = theCenter;
				}
				virtual ~templ(){
					this->extremes.clear();
					this->points.clear();
				}
				templ(const templ &aTempl){
					this->center   = aTempl.center;
					this->head     = aTempl.head;
					this->extremes = aTempl.extremes;
					this->points   = aTempl.points;
				}
				templ& operator=(const templ &aTempl){
					if(this == &aTempl) return *this;
					this->center   = aTempl.center;
					this->head     = aTempl.head;
					this->extremes = aTempl.extremes;
					this->points   = aTempl.points;
					return *this;
				}
		};
		/** All available feature types.
		 */
		enum FEATURE {IPOINTS,EDGES,SIFT_DICT,SURF,SIFT,GABOR,PIXELS,\
			HOG};
		/** What needs to be rotated.
		 */
		enum ROTATE {MATRIX,TEMPLATE,KEYS};
		/** Initializes the class elements.
		 */
		void init(featureExtractor::FEATURE fType,const std::string &featFile,\
			int colorSp,int invColorSp);
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
		cv::Mat getPixels(const cv::Mat &image,const featureExtractor::templ &aTempl,\
			const cv::Rect &roi);
		/** Gets the HOG descriptors over an image.
		 */
		cv::Mat getHOG(const cv::Mat &pixels,const featureExtractor::templ &aTempl,\
			const cv::Rect &roi);
		/** Gets the edges in an image.
		 */
		cv::Mat getEdges(cv::Mat &feature,const cv::Mat &thresholded,\
			const cv::Rect &roi,const featureExtractor::templ &aTempl,\
			float rotAngle);
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
			const featureExtractor::templ &aTempl,const cv::Mat &test);
		/** Convolves an image with a Gabor filter with the given parameters and
		 * returns the response image.
		 */
		cv::Mat getGabor(cv::Mat &feature,const cv::Mat &thresholded,\
			const cv::Rect &roi,const cv::Size &foregrSize,float rotAngle,\
			int aheight);
		/** Creates a gabor with the parameters given by the parameter vector.
		 */
		void createGabor(cv::Mat &gabor,float *params = NULL);
		/** Returns the row corresponding to the indicated feature type.
		 */
		cv::Mat getDataRow(cv::Mat &image,const featureExtractor::templ &aTempl,\
			const cv::Rect &roi,const featureExtractor::people &person,\
			const cv::Mat &thresholded,const std::string &imgName,\
			cv::Point2f &absRotCenter,cv::Point2f &rotBorders,float rotAngle,\
			cv::vector<cv::Point2f> &keys);
		/** Compares SURF 2 descriptors and returns the boolean value of their comparison.
		 */
		static bool compareDescriptors(const featureExtractor::keyDescr &k1,\
			const featureExtractor::keyDescr &k2);
		/** Checks to see if a given pixel is inside a template.
		 */
		static bool isInTemplate(unsigned pixelX,unsigned pixelY,\
			const std::vector<cv::Point2f> &templ);
		/** Rotate a matrix/a template/keypoints wrt to the camera location.
		 */
		void rotate2Zero(float rotAngle,featureExtractor::ROTATE what,\
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
	//==========================================================================
	private:
		/** @var isInit
 		 * It is true if the class was already initialized.
 		 */
		bool isInit;
		/** @var data
 		 * The matrix in which the features are stored.
 		 */
		cv::Mat data;
		/** @var featureType
		 * Indicates the type of the feature to be used.
		 */
		featureExtractor::FEATURE featureType;
		/** @var dictionarySIFT
		 * The SIFT dictionary used for vector quantization.
		 */
		std::string dictFilename;
		/** @var noMeans
		 * The number of means used for kmeans.
		 */
		unsigned noMeans;
		/** @var meanSize
		 * The meanSize (128 for regular SIFT features) .
		 */
		unsigned meanSize;
		/** @var featureFile
		 * The folder were the features are stored for each image.
		 */
		std::string featureFile;
		/** @var print
		 * Print some values out or not.
		 */
		bool print;
		/** @var plot
		 * Plot some features or not.
		 */
		bool plot;
		/** @var dictionarySIFT
		 * The SIFT dictionary loaded as from the file where is stored.
		 */
		cv::Mat dictionarySIFT;
		/** @var imageClass
		 * The class of the image corresponding to its position wrt the camera.
		 */
		std::string imageClass;
		/** @var invColorspaceCode
		 * The code to be used to convert an image to gray.
		 */
		int invColorspaceCode;
		/** @var colorspaceCode
		 * The code to be used to convert an image to gray.
		 */
		int colorspaceCode;
		//======================================================================
	private:
		DISALLOW_COPY_AND_ASSIGN(featureExtractor);
};

#endif /* FEATUREEXTRACTOR_H_ */
