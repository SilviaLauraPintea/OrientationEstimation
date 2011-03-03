/* annotationsHandle.h
 * Author: Silvia-Laura Pintea
 */
#ifndef ANNOTATIONSHANDLE_H_
#define ANNOTATIONSHANDLE_H_
#include <iostream>
#include <exception>
#include <cmath>
#include <fstream>
#include <string>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <boost/version.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#if BOOST_VERSION < 103500
	#include <boost/thread/detail/lock.hpp>
#endif
#include <boost/thread/xtime.hpp>
#include "eigenbackground/src/Annotate.hh"
#include "eigenbackground/src/Helpers.hh"
#include "eigenbackground/src/defines.hh"
#include "Auxiliary.h"
using namespace std;

/** Class for annotating both positions and poses of the people in the images.
 */
class annotationsHandle {
	public:
		/** All considered poses.
		 */
		enum POSE {SITTING, STANDING, BENDING, ORIENTATION};

		/** A structure that stores a single annotation for a specific person.
 		 */
		struct ANNOTATION {
			short int id;
			cv::Point location;
			std::vector<unsigned int> poses;
			ANNOTATION(){
				this->id    = 0;
				this->poses = std::vector<unsigned int>(4,0);
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
			string imgFile;
			std::vector<annotationsHandle::ANNOTATION> annos;
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
			double dist;
			ASSIGNED(){
				this->id   = 0;
				this->to   = 0;
				this->dist = 0.0;
			}
		};
		//======================================================================
		annotationsHandle(){
			image  = NULL;
			choice = ' ';
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
		static void showMenu(cv::Point center);

		/** Plots the hull indicated by the parameter \c hull on the given image.
		 */
		static void plotHull(IplImage *img, std::vector<CvPoint> &hull);

		/** Starts the annotation of the images. The parameters that need to be
		 * indicated are:
		 *
		 * \li argv[1] -- name of directory containing the images
		 * \li argv[2] -- the file contains the calibration data of the camera
		 * \li argv[3] -- the file in which the annotation data needs to be stored
		 */
		static int runAnn(int argc, char **argv, unsigned step = 100);

		/** The "on change" handler for the track-bars.
		 */
		static void trackbar_callback(int position,void *param);

		/** A function that starts a new thread which handles the track-bar event.
		 */
		static void trackBarHandleFct(int position,void *param);

		/** Load annotations from file.
		 */
		static void loadAnnotations(char* filename,\
			std::vector<annotationsHandle::FULL_ANNOTATIONS> &loadedAnno);

		/** Computes the average distance from the predicted location and the
		 * annotated one, the number of unpredicted people in each image and
		 * the differences in the pose estimation.
		 */
		static void annoDifferences(std::vector<annotationsHandle::FULL_ANNOTATIONS>\
			&train, std::vector<annotationsHandle::FULL_ANNOTATIONS> &test,\
			double &avgDist, double &Ndiff, double avgOrientDiff, double poseDiff);

		/** Correlate annotations' from locations in \c annoOld to locations in
		 * \c annoNew through IDs.
		 */
		static void correltateLocs(std::vector<annotationsHandle::ANNOTATION> &annoOld,\
			std::vector<annotationsHandle::ANNOTATION> &annoNew,\
			std::vector<annotationsHandle::ASSIGNED> &idAssignedTo);

		/** Checks to see if a location can be assigned to a specific ID given the
		 * new distance.
		 */
		static bool canBeAssigned(std::vector<annotationsHandle::ASSIGNED> &idAssignedTo, short int id, \
			double newDist, short int to);

		/** Displays the complete annotations for all images.
		 */
		static void displayFullAnns(std::vector<annotationsHandle::FULL_ANNOTATIONS>\
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
		static void drawOrientation(cv::Point center, unsigned int orient);

		/** Writes a given FULL_ANNOTATIONS structure into a given file.
		 */
		void writeAnnoToFile(std::vector<annotationsHandle::FULL_ANNOTATIONS>\
			fullAnno, std::string fileName);
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
		static std::vector<annotationsHandle::ANNOTATION> annotations;
		/** @var choice
		 * Indicates if the pose was defined for the current frame.
		 */
		static char choice;
		/** @var trackbarMutex
		 * A mutex for controlling the access to the annotations.
		 */
		static boost::mutex trackbarMutex;
};

#endif /* ANNOTATIONSHANDLE_H_ */
