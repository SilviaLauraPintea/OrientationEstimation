/* AnnotationsHandle.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#ifndef ANNOTATIONSHANDLE_H_
#define ANNOTATIONSHANDLE_H_
#include <boost/thread.hpp>
#include <tr1/memory>
#include "eigenbackground/src/Helpers.hh"

/** Class for annotating both positions and poses of the people in the images.
 */
class AnnotationsHandle {
	public:
		/** All considered poses.
		 */
		enum POSE {SITTING,STANDING,BENDING,LONGITUDE,LATITUDE};
		/** A structure that stores a single annotation for a specific person.
 		 */
		struct ANNOTATION {
			public:
				short int id_;
				cv::Point2f location_;
				std::deque<unsigned int> poses_;
				ANNOTATION(){
					this->id_    = 0;
					this->poses_ = std::deque<unsigned int>(5,0);
				}
				virtual ~ANNOTATION(){
					if(!this->poses_.empty()){
						this->poses_.clear();
					}
				}
				ANNOTATION(const ANNOTATION &anno){
					this->id_       = anno.id_;
					this->location_ = anno.location_;
					this->poses_    = anno.poses_;
				}
				ANNOTATION& operator=(const ANNOTATION &anno){
					if(this == &anno) return *this;
					this->id_       = anno.id_;
					this->location_ = anno.location_;
					this->poses_    = anno.poses_;
					return *this;
				}
		};
		/** Structure containing a vector of annotations for each image.
		 */
		struct FULL_ANNOTATIONS {
			public:
				std::string imgFile_;
				std::deque<AnnotationsHandle::ANNOTATION> annos_;
				FULL_ANNOTATIONS(){
					this->imgFile_ = "";
				}
				virtual ~FULL_ANNOTATIONS(){
					if(!this->annos_.empty()){
						this->annos_.clear();
					}
				}
				FULL_ANNOTATIONS(const FULL_ANNOTATIONS &fanno){
					this->imgFile_ = fanno.imgFile_;
					this->annos_   = fanno.annos_;
				}
				FULL_ANNOTATIONS& operator=(const FULL_ANNOTATIONS &fanno){
					if(this == &fanno) return *this;
					this->imgFile_ = fanno.imgFile_;
					this->annos_   = fanno.annos_;
					return *this;
				}
		};
		/** Shows which id from the old annotations is assigned to which id from
		 * the new annotations based on what minimal distance.
		 */
		struct ASSIGNED {
			public:
				short int id_;
				short int to_;
				float dist_;
				ASSIGNED(){
					this->id_   = 0;
					this->to_   = 0;
					this->dist_ = 0.0;
				}
				virtual ~ASSIGNED(){};
				ASSIGNED(const ASSIGNED &assig){
					this->id_   = assig.id_;
					this->to_   = assig.to_;
					this->dist_ = assig.dist_;
				}
				ASSIGNED& operator=(const ASSIGNED &assig){
					if(this == &assig) return *this;
					this->id_   = assig.id_;
					this->to_   = assig.to_;
					this->dist_ = assig.dist_;
					return *this;
				}
		};
		//======================================================================
		AnnotationsHandle(){
			choice_    = ' ';
			withPoses_ = false;
			poseSize_  = 5;
		};
		virtual ~AnnotationsHandle(){
			//free annotations
			annotations_.clear();
			if(this->image_){
				image_.reset();
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
		/** Starts the annotation process for the images.
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
			std::deque<AnnotationsHandle::FULL_ANNOTATIONS> &loadedAnno);
		/** Computes the average distance from the predicted location and the
		 * annotated one,the number of unpredicted people in each image and
		 * the differences in the pose estimation.
		 */
		static void annoDifferences(std::deque<AnnotationsHandle::FULL_ANNOTATIONS>\
			&train,std::deque<AnnotationsHandle::FULL_ANNOTATIONS> &test,\
			float &avgDist,float &Ndiff,float ssdLongDiff,float ssdLatDiff,\
			float poseDiff);
		/** Correlate annotations' from locations in \c annoOld to locations in
		 * \c annoNew through IDs.
		 */
		static void correltateLocs(std::deque<AnnotationsHandle::ANNOTATION> &annoOld,\
			std::deque<AnnotationsHandle::ANNOTATION> &annoNew,\
			std::deque<AnnotationsHandle::ASSIGNED> &idAssignedTo);
		/** Checks to see if a location can be assigned to a specific ID given the
		 * new distance.
		 */
		static bool canBeAssigned(std::deque<AnnotationsHandle::ASSIGNED> &idAssignedTo,\
			short int id,float newDist,short int to);
		/** Displays the complete annotations for all images.
		 */
		static void displayFullAnns(std::deque<AnnotationsHandle::FULL_ANNOTATIONS>\
			&fullAnns);
		/** Evaluates the annotation of the images.
		 */
		static int runEvaluation(int argc,char **argv);
		/** Shows how the selected orientation looks on the image.
		 */
		static void drawOrientation(const cv::Point2f &center,unsigned int orient,\
			AnnotationsHandle::POSE pose);
		/** Shows how the selected orientation looks on the image.
		 */
		static void drawLatitude(const cv::Point2f &head,const cv::Point2f &feet,\
			unsigned int orient,AnnotationsHandle::POSE pose);
		static cv::Mat rotateWrtCamera(const cv::Point2f &headLocation,\
			const cv::Point2f &feetLocation,const cv::Mat &toRotate,cv::Point2f &borders);
		/** Writes a given FULL_ANNOTATIONS structure into a given file.
		 */
		static void writeAnnoToFile(const std::deque<AnnotationsHandle::FULL_ANNOTATIONS>\
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
			const std::string &usedImages,int imgIndex,int imoffset,unsigned lati,\
			int setoffset);
		//======================================================================
	private:
		/** @var image_
		 * The currently processed image.
		 */
		static std::tr1::shared_ptr<IplImage> image_;
		/** @var annotations_
		 * An instance of the structure \c ANNOTATIONS storing the annotations
		 * for each image.
		 */
		static std::deque<AnnotationsHandle::ANNOTATION> annotations_;
		/** @var choice_
		 * Indicates if the pose was defined for the current frame.
		 */
		static char choice_;
		/** @var trackbarMutex_
		 * A mutex for controlling the access to the annotations.
		 */
		static boost::mutex trackbarMutex_;
		/** @var poseSize_
		 * The number of elements in the POSE enum.
		 */
		static unsigned poseSize_;
		/** @var withPoses_
		 * With poses or just orientation.
		 */
		static bool withPoses_;
		/** @var poseNames_
		 * The strings corresponding to the names of the poses
		 */
		static std::deque<std::string> poseNames_;
		//======================================================================
	private:
		DISALLOW_COPY_AND_ASSIGN(AnnotationsHandle);
};

#endif /* ANNOTATIONSHANDLE_H_ */
