#include "featureDetector.h"
#include <boost/thread.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION < 103500
	#include <boost/thread/detail/lock.hpp>
#endif
#include <boost/thread/xtime.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <opencv/cv.h>
#include <exception>
#include <opencv/highgui.h>
#include "eigenbackground/src/Tracker.hh"
#include "eigenbackground/src/Helpers.hh"
unsigned MIN_TRACKLEN = 20;

//==============================================================================
/** Function that gets the ROI corresponding to a head of a person in
 * an image.
 */
cv::Mat featureDetector::getHeadROI(std::vector<unsigned> existing){
	for(unsigned i=0; i<existing.size(); i++){
		cv::Point center = this->cvPoint(existing[i]);
		std::vector<CvPoint> templ;
		genTemplate2(center, persHeight, camHeight, templ);

		cv::Mat tmpImage(this->current->img);

		unsigned wi = templ[14].x-templ[13].x;
		unsigned hi = templ[13].y-templ[12].y;
		//cv::Size someSize = cv::Size(wi,hi);
		//cv::Range xRange = cv::Range((int)templ[13].x, (int)templ[14].x);
		//cv::Range yRange = cv::Range((int)templ[12].y, (int)templ[13].y);
		//cv::Mat roi(xRange,yRange,tmpImage(someSize, center);
		cv::Mat roi(tmpImage, cv::Rect(center.x, center.y, wi, hi));
		cv::imshow("head", roi);
		roi.release();
		break;
	}

	return cv::Mat(cv::Size(300,1),8,1);
}
//==============================================================================
/** Overwrite the \c doFindPeople function from the \c Tracker class to make it
 * work with the feature extraction.
 */
bool featureDetector::doFindPerson(unsigned imgNum, IplImage *src,\
const vnl_vector<FLOAT> &imgVec, vnl_vector<FLOAT> &bgVec,\
const FLOAT logBGProb,const vnl_vector<FLOAT> &logSumPixelBGProb){
	//1) START THE TIMER & INITIALIZE THE PROBABLITIES, THE VECTOR OF POSITIONS
	stic();	std::vector<FLOAT> marginal;
	FLOAT lNone = logBGProb + this->logNumPrior[0], lSum = -INFINITY;
	std::vector<unsigned> existing;
	std::vector< vnl_vector<FLOAT> > logPosProb;
	std::vector<scanline_t> mask;
	marginal.push_back(lNone);
	logPosProb.push_back(vnl_vector<FLOAT>());

	//2) SCAN FOR POSSIBLE LOCATION OF PEOPLE GIVEN THE EXISTING ONES
	this->scanRest(existing,mask,scanres,logSumPixelBGProb,logPosProb, \
		marginal,lNone);

	//3) UPDATE THE PROBABILITIES
	for(unsigned i=0; i!=marginal.size(); ++i){
		lSum = log_sum_exp(lSum,marginal[i]);
	}

	//4) UPDATE THE MOST LIKELY NUMBER OF PEOPLE AND THE MARGINAL PROBABILITY
	unsigned mlnp = 0; // most likely number of people
	FLOAT mlprob  = -INFINITY;
	for(unsigned i=0; i!=marginal.size(); ++i) {
		if (marginal[i] > mlprob) {
			mlnp   = i;
			mlprob = marginal[i];
		}
	}

	//5) UPDATE THE TRACKS (WHERE ALL THE INFORMATION IS STORED FOR ALL IMAGES)
	this->updateTracks(imgNum, existing);

	//6) PLOT TRACKS FOR THE FIRST IMAGE
	for(unsigned i=0; i<tracks.size(); ++i){
		if(this->tracks[i].imgID.size() > MIN_TRACKLEN){
			if(imgNum-this->tracks[i].imgID.back() < 2){
				this->plotTrack(src,tracks[i],i,(unsigned)1);
			}
		}
	}

	//7') GET THE HEAD
	this->getHeadROI(existing);

	//7) SHOW THE FOREGROUND POSSIBLE LOCATIONS AND PLOT THE TEMPLATES
	IplImage *bg = vec2img((imgVec-bgVec).apply(fabs));
	this->plotHull(bg, this->priorHull);
	cerr<<"no. of detected people: "<<existing.size()<<endl;
	for(unsigned i=0; i!=existing.size(); ++i){
		cv::Point pt = this->cvPoint(existing[i]);
		plotTemplate2(bg,pt,persHeight,camHeight,CV_RGB(255,255,255));
		plotScanLines(this->current->img,mask,CV_RGB(0,255,0),0.3);
	}
	cvShowImage("bg", bg);
	cvReleaseImage(&bg);
	cvShowImage("image",src);

	//8) WHILE NOT q WAS PRESSED PROCESS THE NEXT IMAGES
	return this->imageProcessingMenu();
}
//==============================================================================
bool featureDetector::imageProcessingMenu(){
	cout<<" To quite press 'q'.\n To pause press 'p'.\n To skip 10 "<<\
		"images press 's'.\n To skip 100 images press 'f'.\n To go back 10 "<<\
		"images  press 'b'.\n To go back 100 images press 'r'.\n";
	int k = (char)cvWaitKey(this->waitTime);
	cout<<"Press 'n' to go to the next frame"<<endl;

	while((char)k != 'n' && (char)k != 'q'){
		switch(k){
			case 'p':
				this->waitTime = 20-this->waitTime;
				break;
			case 's':
				this->producer->forward(10);
				break;
			case 'f':
				this->producer->forward(100);
				break;
			case 'b':
				this->producer->backward(10);
				break;
			case 'r':
				this->producer->backward(100);
				break;
			default:
				break;
		}
		k = (char)cvWaitKey(this->waitTime);
	}
	if(k == 'q'){return false;}
	return true;
}
//==============================================================================
int main(int argc, char **argv){
	featureDetector feature;
	feature.run(argc,argv);
}
