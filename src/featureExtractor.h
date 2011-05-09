/* featureExtractor.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code, but please retain the above copyright notice.
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
				~keyDescr(){
					if(!this->descr.empty()){
						this->descr.clear();
					}
				}
				keyDescr(const keyDescr &kdescr){
					this->keys  = kdescr.keys;
					this->descr = kdescr.descr;
				}
				void operator=(const keyDescr &kdescr){
					this->keys  = kdescr.keys;
					this->descr = kdescr.descr;
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
				~people(){
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
				void operator=(const people &person){
					this->absoluteLoc = person.absoluteLoc;
					this->relativeLoc = person.relativeLoc;
					this->borders     = person.borders;
					if(!this->pixels.empty()){
						this->pixels.release();
					}
					person.pixels.copyTo(this->pixels);
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
				~templ(){
					this->extremes.clear();
					this->points.clear();
				}
				templ(const templ &aTempl){
					this->center   = aTempl.center;
					this->head     = aTempl.head;
					this->extremes = aTempl.extremes;
					this->points   = aTempl.points;
				}
				void operator=(const templ &aTempl){
					this->center   = aTempl.center;
					this->head     = aTempl.head;
					this->extremes = aTempl.extremes;
					this->points   = aTempl.points;
				}
		};
		/** All available feature types.
		 */
		enum FEATURE {IPOINTS, EDGES, SIFT_DICT, SURF, SIFT, GABOR, PIXELS,\
			HOG};
		/** What needs to be rotated.
		 */
		enum ROTATE {MATRIX, TEMPLATE, KEYS};
		/** Initializes the class elements.
		 */
		void init(featureExtractor::FEATURE fType, std::string featFile);
		/** Resets the variables to the default values.
		 */
		void reset();
		/** Creates a data matrix for each image and stores it locally.
		 */
		/** Initializes the settings for the SIFT dictionary.
		 */
		void initSIFT(std::string dictName, unsigned means=500, unsigned size=128);
		/** Creates a data matrix for each image and stores it locally.
		 */
		void extractFeatures(cv::Mat image,std::string sourceName,int colorspaceCode);
		/** Extract the interest points in a gird and returns them.
		 */
		cv::Mat extractPointsGrid(cv::Mat image);
		/** Extract edges from the whole image.
		 */
		cv::Mat extractEdges(cv::Mat image);
		/** Convolves the whole image with some Gabors wavelets and then stores the
		 * results.
		 */
		cv::Mat extractGabor(cv::Mat image);
		/** Extracts SIFT features from the image and stores them in a matrix.
		 */
		cv::Mat extractSIFT(cv::Mat image, std::vector<cv::Point2f> templ =\
			std::vector<cv::Point2f>(), cv::Rect roi = cv::Rect());
		/** Extracts all the surf descriptors from the whole image and writes them in a
		 * matrix.
		 */
		cv::Mat extractSURF(cv::Mat image);
		/** Gets the plain pixels corresponding to the upper part of the body.
		 */
		cv::Mat getPixels(cv::Mat image,featureExtractor::templ aTempl,\
			cv::Rect roi);
		/** Gets the HOG descriptors over an image.
		 */
		cv::Mat getHOG(cv::Mat pixels,featureExtractor::templ aTempl,cv::Rect roi);
		/** Gets the edges in an image.
		 */
		cv::Mat getEdges(cv::Mat feature, cv::Mat thresholded, cv::Rect roi,\
			featureExtractor::templ aTempl, float rotAngle);
		/** SURF descriptors (Speeded Up Robust Features).
		 */
		cv::Mat getSURF(cv::Mat feature, std::vector<cv::Point2f> templ,\
			std::vector<cv::Point2f> &indices, cv::Rect roi, cv::Mat test=cv::Mat());
		/** Compute the features from the SIFT descriptors by doing vector
		 * quantization.
		 */
		cv::Mat getSIFT(cv::Mat feature,std::vector<cv::Point2f> templ,\
			std::vector<cv::Point2f> &indices, cv::Rect roi, cv::Mat test=cv::Mat());
		/** Creates a "histogram" of interest points + number of blobs.
		 */
		cv::Mat getPointsGrid(cv::Mat feature, cv::Rect roi,\
			featureExtractor::templ aTempl, cv::Mat test=cv::Mat());
		/** Convolves an image with a Gabor filter with the given parameters and
		 * returns the response image.
		 */
		cv::Mat getGabor(cv::Mat feature, cv::Mat thresholded, cv::Rect roi,\
			cv::Size foregrSize, float rotAngle, int aheight);
		/** Creates a gabor with the parameters given by the parameter vector.
		 */
		cv::Mat createGabor(float *params = NULL);
		/** Returns the row corresponding to the indicated feature type.
		 */
		cv::Mat getDataRow(cv::Mat image,featureExtractor::templ aTempl, cv::Rect roi,\
		featureExtractor::people person, cv::Mat thresholded,cv::vector<cv::Point2f> &keys,\
		std::string imgName, cv::Point2f absRotCenter, cv::Point2f rotBorders,float rotAngle);
		/** Compares SURF 2 descriptors and returns the boolean value of their comparison.
		 */
		static bool compareDescriptors(const featureExtractor::keyDescr k1,\
			const featureExtractor::keyDescr k2);
		/** Checks to see if a given pixel is inside a template.
		 */
		static bool isInTemplate(unsigned pixelX, unsigned pixelY,\
			std::vector<cv::Point2f> templ);
		/** Rotate a matrix/a template/keypoints wrt to the camera location.
		 */
		cv::Mat rotate2Zero(float rotAngle, cv::Mat toRotate, cv::Point2f \
			&rotBorders, cv::Point2f rotCenter,featureExtractor::ROTATE what,\
			std::vector<cv::Point2f> &pts);
		/**Return number of means.
		 */
		unsigned readNoMeans();
		/**Return name of the SIFT dictionary.
		 */
		std::string readDictName();
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
		//======================================================================
	private:
		DISALLOW_COPY_AND_ASSIGN(featureExtractor);
};

#endif /* FEATUREEXTRACTOR_H_ */
