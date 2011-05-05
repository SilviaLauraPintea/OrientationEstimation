/* annotationsHandle.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code, but please retain the above copyright notice.
 */
#ifndef ANNOTATIONSHANDLE_H_
#define ANNOTATIONSHANDLE_H_
using namespace std;
#include <deque>
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <vnl/vnl_vector.h>
#include "eigenbackground/src/defines.hh"

/** Class for annotating both positions and poses of the people in the images.
 */
class annotationsHandle {
	public:
		/** All considered poses.
		 */
		enum POSE {SITTING, STANDING, BENDING, LONGITUDE, LATITUDE};
		/** A structure that stores a single annotation for a specific person.
 		 */
		struct ANNOTATION {
			short int id;
			cv::Point2f location;
			std::deque<unsigned int> poses;
			ANNOTATION(){
				this->id    = 0;
				this->poses = std::deque<unsigned int>(5,0);
			}
			~ANNOTATION(){
				if(!this->poses.empty()){
					this->poses.clear();
				}
			}
		};
		/** Structure containing a vector of annotations for each image.
		 */
		struct FULL_ANNOTATIONS {
			std::string imgFile;
			std::deque<annotationsHandle::ANNOTATION> annos;
			FULL_ANNOTATIONS(){
				this->imgFile = "";
			}
			~FULL_ANNOTATIONS(){
				if(!this->annos.empty()){
					this->annos.clear();
				}
			}
		};
		/** Shows which id from the old annotations is assigned to which id from
		 * the new annotations based on what minimal distance.
		 */
		struct ASSIGNED {
			short int id;
			short int to;
			float dist;
			ASSIGNED(){
				this->id   = 0;
				this->to   = 0;
				this->dist = 0.0;
			}
		};
		//======================================================================
		annotationsHandle(){
			image     = NULL;
			choice    = ' ';
			withPoses = false;
			poseSize  = 5;
		};

		virtual ~annotationsHandle(){
			//free annotations
			annotations.clear();
			if(image){
				cvReleaseImage(&image);
				image = NULL;
			}
		};

		/** Mouse handler for annotating people's positions and poses.
		 */
		static void mouseHandlerAnn(int event, int x, int y, int \
			flags, void *param);

		/** Draws the "menu" of possible poses for the current position.
		 */
		static void showMenu(cv::Point2f center);

		/** Plots the hull indicated by the parameter \c hull on the given image.
		 */
		static void plotHull(IplImage *img, std::vector<cv::Point2f> &hull);

		/** Starts the annotation of the images. The parameters that need to be indicated
		 * are:
		 * \li step       -- every "step"^th image is opened for annotation
		 * \li usedImages -- the folder where the annotated images are moved
		 * \li imgIndex   -- the image index from which to start
		 * \li argv[1]    -- name of directory containing the images
		 * \li argv[2]    -- the file contains the calibration data of the camera
		 * \li argv[3]    -- the file in which the annotation data needs to be stored
		 */
		static int runAnn(int argc, char **argv, unsigned step,  std::string \
			usedImages, int imgIndex=-1);

		/** The "on change" handler for the track-bars.
		 */
		static void trackbar_callback(int position,void *param);

		/** A function that starts a new thread which handles the track-bar event.
		 */
		static void trackBarHandleFct(int position,void *param);

		/** Load annotations from file.
		 */
		static void loadAnnotations(char* filename,\
			std::deque<annotationsHandle::FULL_ANNOTATIONS> &loadedAnno);

		/** Computes the average distance from the predicted location and the
		 * annotated one, the number of unpredicted people in each image and
		 * the differences in the pose estimation.
		 */
		static void annoDifferences(std::deque<annotationsHandle::FULL_ANNOTATIONS>\
			&train, std::deque<annotationsHandle::FULL_ANNOTATIONS> &test,\
			float &avgDist, float &Ndiff, float ssdLongDiff, float ssdLatDiff,\
			float poseDiff);

		/** Correlate annotations' from locations in \c annoOld to locations in
		 * \c annoNew through IDs.
		 */
		static void correltateLocs(std::deque<annotationsHandle::ANNOTATION> &annoOld,\
			std::deque<annotationsHandle::ANNOTATION> &annoNew,\
			std::deque<annotationsHandle::ASSIGNED> &idAssignedTo);

		/** Checks to see if a location can be assigned to a specific ID given the
		 * new distance.
		 */
		static bool canBeAssigned(std::deque<annotationsHandle::ASSIGNED> &idAssignedTo, short int id, \
			float newDist, short int to);

		/** Displays the complete annotations for all images.
		 */
		static void displayFullAnns(std::deque<annotationsHandle::FULL_ANNOTATIONS>\
			&fullAnns);

		/** Starts the annotation of the images. The parameters that need to be
		 * indicated are:
		 *
		 * \li argv[1] -- train file with the correct annotations;
		 * \li argv[2] -- test file with predicted annotations;
		 */
		static int runEvaluation(int argc, char **argv);

		/** Shows how the selected orientation looks on the image.
		 */
		static void drawOrientation(cv::Point2f center, unsigned int orient,\
			annotationsHandle::POSE pose);

		/** Shows how the selected orientation looks on the image.
		 */
		static void drawLatitude(cv::Point2f head, cv::Point2f feet,\
			unsigned int orient, annotationsHandle::POSE pose);

		static cv::Mat rotateWrtCamera(cv::Point2f headLocation,\
			cv::Point2f feetLocation, cv::Mat toRotate, cv::Point2f &borders);

		/** Writes a given FULL_ANNOTATIONS structure into a given file.
		 */
		static void writeAnnoToFile(std::deque<annotationsHandle::FULL_ANNOTATIONS>\
			fullAnno, std::string fileName);

		/** Initializes all the values of the class variables.
		 */
		static void init();
		//======================================================================
	protected:
		/** @var image
		 * The currently processed image.
		 */
		static IplImage *image;
		/** @var image
		 * An instance of the structure \c ANNOTATIONS storing the annotations
		 * for each image.
		 */
		static std::deque<annotationsHandle::ANNOTATION> annotations;
		/** @var choice
		 * Indicates if the pose was defined for the current frame.
		 */
		static char choice;
		/** @var trackbarMutex
		 * A mutex for controlling the access to the annotations.
		 */
		static boost::mutex trackbarMutex;
		/** @var poseSize
		 * The number of elements in the POSE enum.
		 */
		static unsigned poseSize;

		/** @var withPoses
		 * With poses or just orientation.
		 */
		static bool withPoses;

		/** @var poseNames
		 * The strings corresponding to the names of the poses
		 */
		static std::deque<std::string> poseNames;
};

#endif /* ANNOTATIONSHANDLE_H_ */
