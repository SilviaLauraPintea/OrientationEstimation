/* featureExtractor.h
 * Author: Silvia-Laura Pintea
 */

#ifndef FEATUREEXTRACTOR_H_
#define FEATUREEXTRACTOR_H_
#include <opencv2/opencv.hpp>

/** Extracts the actual features from the images and stores them in data matrix.
 */
class featureExtractor {
	public:
		featureExtractor();
		virtual ~featureExtractor();
		/** Structure for storing keypoints and descriptors.
		 */
		struct keyDescr {
			cv::KeyPoint keys;
			std::deque<float> descr;
			~keyDescr(){
				if(!this->descr.empty()){
					this->descr.clear();
				}
			}
		};
		/** All available feature types.
		 */
		enum FEATURE {IPOINTS, EDGES, SIFT_DICT, SURF, SIFT, GABOR, HOG};
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
		void extractFeatures();
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
		/** Extract some HOG descriptors out of an image.
		 */
		cv::Mat extractHOG(cv::Mat image);
		/** Gets the edges in an image.
		 */
		cv::Mat getEdges(cv::Mat feature, cv::Mat thresholded, cv::Rect roi,\
			cv::Point2f head, cv::Point2f center);
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
		cv::Mat featureExtractor::getPointsGrid(cv::Mat feature, cv::Rect roi,\
			peopleDetector::templ aTempl, cv::Mat test=cv::Mat());
		/** Convolves an image with a Gabor filter with the given parameters and
		 * returns the response image.
		 */
		cv::Mat getGabor(cv::Mat feature, cv::Mat thresholded, cv::Rect roi,\
			cv::Point2f center, cv::Point2f head, cv::Size foregrSize);
		/** Creates a gabor with the parameters given by the parameter vector.
		 */
		cv::Mat createGabor(double *params = NULL);
		/** Returns the row corresponding to the indicated feature type.
		 */
		cv::Mat getDataRow(peopleDetector::templ aTempl, cv::Rect roi,\
		peopleDetector::people person, cv::Mat thresholded,cv::vector<cv::Point2f> &keys,\
		std::string imgName, cv::Point2f absRotCenter, cv::Point2f rotBorders);
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
};

#endif /* FEATUREEXTRACTOR_H_ */
