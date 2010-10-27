#include "annotatePosPose.h"
#include <iostream>
#include <opencv/cv.h>
#include <exception>
#include <opencv/highgui.h>
#include <boost/thread.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION < 103500
	#include <boost/thread/detail/lock.hpp>
#endif
#include <boost/thread/xtime.hpp>
#include "eigenbackground/src/annotatepos.hh"
#include "eigenbackground/src/Helpers.hh"
using namespace std;

/** Define a post-fix increment operator for the enum \c POSE.
 */
annotatePosPose::POSE operator++(annotatePosPose::POSE &refPose, int){
	annotatePosPose::POSE oldPose = refPose;
	refPose                       = (annotatePosPose::POSE)(refPose + 1);
	return oldPose;
}

/** Mouse handler for annotating people's positions and poses.
 */
void annotatePosPose::mouseHandlerAnn(int event, int x, int y, int flags, void *param){
	cv::Point pt = cv::Point(x,y);
	y            = pt.y;
	x            = pt.x;
	static bool down = false;
	switch (event){
		case CV_EVENT_LBUTTONDOWN:
			if(choice == 'c'){
				down = true;
				cout<<"Left button down at >>> ("<<x<<","<<y<<")"<<endl;
				AnnotatePos::plotAreaTmp(image,x,y);
			}
			break;
		case CV_EVENT_MOUSEMOVE:
			if(down){
				AnnotatePos::plotAreaTmp(image,x,y);
			}
			break;
		case CV_EVENT_LBUTTONUP:
			if(choice == 'c'){
				cout<<"Left button up at >>> ("<<x<<","<<y<<")"<<endl;
				choice = ' ';
				down = false;
				ANNOTATION temp;
				temp.location = pt;
				annotations.push_back(temp);
				for(unsigned i=0;i!=annotations.size(); ++i){
					AnnotatePos::plotArea(image, annotations[i].location.x, \
						annotations[i].location.y);
				}
				showMenu(pt);
			}
			break;
	}
}

/** Draws the "menu" of possible poses for the current position.
 */
void annotatePosPose::showMenu(cv::Point center){
	int pose0 = 0;
	int pose1 = 0;
	int pose2 = 0;
	int pose3 = 0;
	cv::namedWindow("Poses",CV_WINDOW_AUTOSIZE);
	cv::imshow("Poses", cv::Mat(1, 300, CV_8UC1, cv::Scalar(255,255,255)));
	for(POSE p=SITTING; p<=ORIENTATION;p++){
		void *param = (void *) new unsigned int(p);
		switch(p){
			case SITTING:
				cv::createTrackbar("Sitting","Poses", &pose0, 1, \
					trackbar_callback, param);
				break;
			case STANDING:
				cv::createTrackbar("Standing","Poses", &pose1, 1, \
					trackbar_callback, param);
				break;
			case BENDING:
				cv::createTrackbar("Bending","Poses", &pose2, 1, \
					trackbar_callback, param);
				break;
			case ORIENTATION:
				cv::createTrackbar("Orientation", "Poses", &pose3, 360, \
					trackbar_callback, param);
				break;
			default:
				//do nothing
				break;
		}
	}
	cout<<"Press 'c' once the annotation for poses is done."<<endl;
	while(choice != 'c'){
		choice = (char)(cv::waitKey(0));
	}
	cvDestroyWindow("Poses");
}

/** A function that starts a new thread which handles the track-bar event.
 */
void annotatePosPose::trackBarHandleFct(int position,void *param){
	//cout<< "lock" << *(unsigned int *)(param) << endl;
	trackbarMutex.lock();
	unsigned int *ii    = (unsigned int *)(param);
	ANNOTATION lastAnno = annotations.back();
	annotations.pop_back();
	if(lastAnno.poses.size()==0){
		for(unsigned int i=0; i<4; i++){
			lastAnno.poses.push_back(0);
		}
	}
	try{
		lastAnno.poses.at(*ii) = position;
	}catch (std::exception &e){
		cout<<"Exception "<<e.what()<<endl;
		exit(1);
	}
	annotations.push_back(lastAnno);
	trackbarMutex.unlock();
	if((POSE)(*ii) == SITTING){
		cv::setTrackbarPos("Standing", "Poses", (1-position));
	} else if((POSE)(*ii) == STANDING){
		cv::setTrackbarPos("Sitting", "Poses", (1-position));
	} else if((POSE)(*ii) == ORIENTATION){
		if(position % 10 != 0){
			position = (position / 10) * 10;
			cv::setTrackbarPos("Orientation", "Poses", position);
		}
	}
	//cout<< "unlock" << *(unsigned int *)(param) << endl;
}
/** The "on change" handler for the track-bars.
 */
void annotatePosPose::trackbar_callback(int position,void *param){
	boost::thread *trackbarHandle;
	trackbarHandle = new boost::thread(&annotatePosPose::trackBarHandleFct,\
		position, param);
	//trackBarHandleFct(position, param);
	trackbarHandle->join();
}

/** Starts the annotation of the images.
 */
int annotatePosPose::runAnn(int argc, char **argv){
	choice = 'c';
	if(argc != 4){
		cerr<<"usage: ./annotatepos <img_list.txt> <calib.xml> <annotation.txt>\n"<< \
		"<img_list.txt>    => the file contains the list of image names (relative paths)\n"<< \
		"<calib.xml>       => the file contains the calibration data of the camera\n"<< \
		"<annotations.txt> => the file in which the annotation data needs to be stored\n"<<endl;
		exit(-1);
	} else {
		cout<<"Help info:\n"<< \
		"> press 'q' to quite before all the images are annotated;\n"<< \
		"> press 's' to save the annotations for the current image and go to the next one;\n"<<endl;
	}
	unsigned index = 0;
	vector<string> imgs;
	listImages(argv[1],imgs);
	IplImage *src = cvLoadImage(imgs[index].c_str());
	loadCalibration(argv[2]);

	// set the handler of the mouse events to the method: <<mouseHandler>>
	image = src;
	cv::namedWindow("image");
	cvSetMouseCallback("image", mouseHandlerAnn, NULL);
	cv::imshow("image", src);

	// used to write the output stream to a file given in <<argv[3]>>
	ofstream
	ofs(argv[3]);
	if(!ofs){
		errx(1,"Cannot open file %s", argv[3]);
	}

	/* while 'q' was not pressed, annotate images and store the info in
	 * the annotation file */
	int key = 0;
	while((char)key != 'q' && index<imgs.size()) {
		key = cv::waitKey(0);
		/* if the pressed key is 's' stores the annotated positions
		 * for the current image */
		if((char)key == 's'){
			ofs<<imgs[index].substr(imgs[index].rfind("/")+1)<<" ";
			for(unsigned i=0; i!=annotations.size();++i){
				ofs <<"("<<annotations[i].location.x<<","\
					<<annotations[i].location.y<<")";
				for(unsigned j=0;j<annotations[i].poses.size();j++){
					switch((POSE)j){
						case SITTING:
							ofs<<"<sitting:"<<annotations[i].poses[j]<<">";
							break;
						case STANDING:
							ofs<<"<standing:"<<annotations[i].poses[j]<<">";
							break;
						case BENDING:
							ofs<<"<bending:"<<annotations[i].poses[j]<<">";
							break;
						case ORIENTATION:
							ofs<<"<orientation:"<<annotations[i].poses[j]<<">";
							break;
						default:
							cout<<"Unknown pose ;)";
							break;
					}
				}
			}
			ofs<<endl;
			cout<<"Annotations for image: "<<\
				imgs[index].substr(imgs[index].rfind("/")+1)\
				<<" were successfully saved!"<<endl;
			annotations.clear();
			cvReleaseImage(&src);

			// load the next image or break if it is the last one
			index++;
			if(index==imgs.size()){
				break;
			}
			src = cvLoadImage(imgs[index].c_str());
			image = src;
			cv::imshow("image", src);
		}else if(isalpha(key)){
			cout<<"key pressed >>> "<<(char)key<<"["<<key<<"]"<<endl;
		}
	}
	cout<<"Thank you for your time ;)!"<<endl;
	cvReleaseImage(&src);
	return 0;
}
char annotatePosPose::choice;
boost::mutex annotatePosPose::trackbarMutex;
IplImage *annotatePosPose::image;
vector<annotatePosPose::ANNOTATION> annotatePosPose::annotations;

int main(int argc, char **argv){
	annotatePosPose::runAnn(argc,argv);
}
