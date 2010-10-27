#ifndef ANNOTATEPOSPOSE_H_
#define ANNOTATEPOSPOSE_H_
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
class annotatePosPose {
	public:
		/** All considered poses.
		 */
		enum POSE {SITTING, STANDING, BENDING, ORIENTATION};

		/** A structure that stores annotations.
 		 */
		struct ANNOTATION {
			cv::Point location;
			vector<unsigned int> poses;
		};

		annotatePosPose(){};

		virtual ~annotatePosPose(){};

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

#endif /* ANNOTATEPOSPOSE_H_ */
