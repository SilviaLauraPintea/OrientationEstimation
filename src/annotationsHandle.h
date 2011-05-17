/* annotationsHandle.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#ifndef ANNOTATIONSHANDLE_H_
#define ANNOTATIONSHANDLE_H_
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
		enum POSE {SITTING,STANDING,BENDING,LONGITUDE,LATITUDE};
		/** A structure that stores a single annotation for a specific person.
 		 */
		struct ANNOTATION {
			public:
				short int id;
				cv::Point2f location;
				std::deque<unsigned int> poses;
				ANNOTATION(){
					this->id    = 0;
					this->poses = std::deque<unsigned int>(5,0);
				}
				virtual ~ANNOTATION(){
					if(!this->poses.empty()){
						this->poses.clear();
					}
				}
				ANNOTATION(const ANNOTATION &anno){
					this->id       = anno.id;
					this->location = anno.location;
					this->poses    = anno.poses;
				}
				ANNOTATION& operator=(const ANNOTATION &anno){
					if(this == &anno) return *this;
					this->id       = anno.id;
					this->location = anno.location;
					this->poses    = anno.poses;
					return *this;
				}
		};
		/** Structure containing a vector of annotations for each image.
		 */
		struct FULL_ANNOTATIONS {
			public:
				std::string imgFile;
				std::deque<annotationsHandle::ANNOTATION> annos;
				FULL_ANNOTATIONS(){
					this->imgFile = "";
				}
				virtual ~FULL_ANNOTATIONS(){
					if(!this->annos.empty()){
						this->annos.clear();
					}
				}
				FULL_ANNOTATIONS(const FULL_ANNOTATIONS &fanno){
					this->imgFile = fanno.imgFile;
					this->annos   = fanno.annos;
				}
				FULL_ANNOTATIONS& operator=(const FULL_ANNOTATIONS &fanno){
					if(this == &fanno) return *this;
					this->imgFile = fanno.imgFile;
					this->annos   = fanno.annos;
					return *this;
				}
		};
		/** Shows which id from the old annotations is assigned to which id from
		 * the new annotations based on what minimal distance.
		 */
		struct ASSIGNED {
			public:
				short int id;
				short int to;
				float dist;
				ASSIGNED(){
					this->id   = 0;
					this->to   = 0;
					this->dist = 0.0;
				}
				virtual ~ASSIGNED(){};
				ASSIGNED(const ASSIGNED &assig){
					this->id   = assig.id;
					this->to   = assig.to;
					this->dist = assig.dist;
				}
				ASSIGNED& operator=(const ASSIGNED &assig){
					if(this == &assig) return *this;
					this->id   = assig.id;
					this->to   = assig.to;
					this->dist = assig.dist;
					return *this;
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
		static void mouseHandlerAnn(int event,int x,int y,int \
			flags,void *param);

		/** Draws the "menu" of possible poses for the current position.
		 */
		static void showMenu(const cv::Point2f &center);

		/** Plots the hull indicated by the parameter \c hull on the given image.
		 */
		static void plotHull(IplImage *img,std::vector<cv::Point2f> &hull);

		/** Starts the annotation of the images. The parameters that need to be indicated
		 * are:
		 * \li step       -- every "step"^th image is opened for annotation
		 * \li usedImages -- the folder where the annotated images are moved
		 * \li imgIndex   -- the image index from which to start
		 * \li argv[1]    -- name of directory containing the images
		 * \li argv[2]    -- the file contains the calibration data of the camera
		 * \li argv[3]    -- the file in which the annotation data needs to be stored
		 */
		static int runAnn(int argc,char **argv,unsigned step, const std::string \
			&usedImages,int imgIndex=-1);

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
		 * annotated one,the number of unpredicted people in each image and
		 * the differences in the pose estimation.
		 */
		static void annoDifferences(std::deque<annotationsHandle::FULL_ANNOTATIONS>\
			&train,std::deque<annotationsHandle::FULL_ANNOTATIONS> &test,\
			float &avgDist,float &Ndiff,float ssdLongDiff,float ssdLatDiff,\
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
		static bool canBeAssigned(std::deque<annotationsHandle::ASSIGNED> &idAssignedTo,\
			short int id,float newDist,short int to);

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
		static int runEvaluation(int argc,char **argv);

		/** Shows how the selected orientation looks on the image.
		 */
		static void drawOrientation(const cv::Point2f &center,unsigned int orient,\
			annotationsHandle::POSE pose);

		/** Shows how the selected orientation looks on the image.
		 */
		static void drawLatitude(const cv::Point2f &head,const cv::Point2f &feet,\
			unsigned int orient,annotationsHandle::POSE pose);

		static cv::Mat rotateWrtCamera(const cv::Point2f &headLocation,\
			const cv::Point2f &feetLocation,const cv::Mat &toRotate,cv::Point2f &borders);

		/** Writes a given FULL_ANNOTATIONS structure into a given file.
		 */
		static void writeAnnoToFile(const std::deque<annotationsHandle::FULL_ANNOTATIONS>\
			&fullAnno,const std::string &fileName);

		/** Initializes all the values of the class variables.
		 */
		static void init();
		/** Check calibration: shows how the projection grows depending on the location
		 * of the point.
		 */
		static void checkCalibration(int argc,char **argv);
		/** Starts the annotation of the images on the artificial data (labels in the
		 * image name).
		 */
		static int runAnnArtificial(int argc,char **argv,unsigned step,\
			const std::string &usedImages,int imgIndex,int imoffset,unsigned set);
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
		//======================================================================
	private:
		DISALLOW_COPY_AND_ASSIGN(annotationsHandle);
};

#endif /* ANNOTATIONSHANDLE_H_ */
