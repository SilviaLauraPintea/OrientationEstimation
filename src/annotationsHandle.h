#ifndef ANNOTATIONSHANDLE_H_
#define ANNOTATIONSHANDLE_H_
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "eigenbackground/src/annotatepos.hh"
#include "eigenbackground/src/Helpers.hh"
#include <boost/thread.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION < 103500
	#include <boost/thread/detail/lock.hpp>
#endif
#include <boost/thread/xtime.hpp>
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
			unsigned int id;
			cv::Point location;
			vector<unsigned int> poses;
		};

		/** Structure containing a vector of annotations for each image.
		 */
		struct FULL_ANNOTATIONS {
			string imgFile;
			vector<ANNOTATION> annos;
		};

		struct ASSIGNED {
			unsigned int id;
			unsigned int to;
			double dist;
		};

		annotationsHandle(){};

		virtual ~annotationsHandle(){};

		/** Mouse handler for annotating people's positions and poses.
		 */
		static void mouseHandlerAnn(int event, int x, int y, int \
			flags, void *param);

		/** Draws the "menu" of possible poses for the current position.
		 */
		static void showMenu(cv::Point center);

		/** Starts the annotation of the images.
		 */
		static int runAnn(int argc, char **argv);

		/** The "on change" handler for the track-bars.
		 */
		static void trackbar_callback(int position,void *param);

		/** A function that starts a new thread which handles the track-bar event.
		 */
		static void trackBarHandleFct(int position,void *param);
		/** Load annotations from file.
		 */
		static void loadAnnotations(char* filename, vector<FULL_ANNOTATIONS> &loadedAnno);
		/** Computes the average distance from the predicted location and the
		 * annotated one, the number of unpredicted people in each image and
		 * the differences in the pose estimation.
		 */
		static void annoDifferences(vector<FULL_ANNOTATIONS> &train, \
			vector<FULL_ANNOTATIONS> &test, double &avgDist, double &Ndiff, \
			double avgOrientDiff, double poseDiff);
		/** Correlate annotations' from locations in \c annoOld to locations in
		 * \c annoNew through IDs.
		 */
		static void correltateLocs(vector<ANNOTATION> &annoOld, vector<ANNOTATION> \
			&annoNew,vector<ASSIGNED> &idAssignedTo,vector<bool> &assignedSoFar);
		/** Checks to see if a location can be assigned to a specific ID given the
		 * new distance.
		 */
		static bool canBeAssigned(vector<ASSIGNED> &idAssignedTo, unsigned int id, \
			double newDist, unsigned int to, vector<bool> &assignedSoFar, \
			bool &isHere);
	private:
		/** @var image
		 * The currently processed image.
		 */
		static IplImage *image;
		/** @var image
		 * An instance of the structure \c ANNOTATIONS storing the annotations
		 * for each image.
		 */
		static vector<ANNOTATION> annotations;
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
