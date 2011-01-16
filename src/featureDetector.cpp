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
#include <math.h>
#include <opencv/cv.h>
#include <exception>
#include <opencv/highgui.h>
#include "eigenbackground/src/Tracker.hh"
#include "eigenbackground/src/Helpers.hh"
unsigned MIN_TRACKLEN = 20;

//==============================================================================
/** Shows a ROI in a given image.
 */
void featureDetector::showROI(cv::Mat image, cv::Point top_left, cv::Size ROI_size){
	cv::Mat aRoi(image.clone(), cv::Rect(top_left,ROI_size));
	cv::imshow("ROI", aRoi);
	cv::waitKey(0);
	cvDestroyWindow("ROI");
	aRoi.release();
}
//==============================================================================
/** Gets the distance to the given template from a given pixel location.
 */
double featureDetector::getDistToTemplate(int pixelX, int pixelY, \
std::vector<CvPoint> templ){
	vector<CvPoint> hull;
	convexHull(templ, hull);
	double minDist = -1;
	unsigned i=0, j=1;
	while(i<hull.size() && j<hull.size()-1){
		double midX = (hull[i].x + hull[j].x)/2;
		double midY = (hull[i].y + hull[j].y)/2;
		double dist = std::sqrt((midX - pixelX)*(midX - pixelX) + \
						(midY - pixelY)*(midY - pixelY));
		if(minDist == -1 || minDist>dist){
			minDist = dist;
		}
		i++; j++;
	}
	return minDist;
}
//==============================================================================
/** Checks to see if a given pixel is inside of a template.
 */
bool featureDetector::isInTemplate(unsigned pixelX,unsigned pixelY,vector<CvPoint> templ){
	vector<CvPoint> hull;
	convexHull(templ, hull);
	vector<scanline_t> lines;
	getScanLines(hull, lines);
	for(vector<scanline_t>::const_iterator i=lines.begin();i!=lines.end();++i){
		if(i->line==pixelY && i->start<=pixelX && i->end>=pixelX){
			return true;
		};
	}
	return false;
}
//==============================================================================
/** Get the foreground pixels corresponding to each person
 */
void featureDetector::getAllForegroundPixels(vector<unsigned> existing, \
IplImage *bg, double threshold){
	// INITIALIZING STUFF
	cv::Mat thsh(cvCloneImage(bg));
	cv::Mat thrsh(thsh.rows, thsh.cols, CV_8UC1);
	cv::cvtColor(thsh, thrsh, CV_BGR2GRAY);
	cv::threshold(thrsh, thrsh, threshold, 255, cv::THRESH_BINARY);
	cv::Mat foregr(cvCloneImage(this->current->img));

	std::vector<featureDetector::people> allPeople(existing.size());
	std::vector<unsigned> tmpminX(existing.size(),thrsh.cols);
	std::vector<unsigned> tmpmaxX(existing.size(),0);
	std::vector<unsigned> tmpminY(existing.size(),thrsh.rows);
	std::vector<unsigned> tmpmaxY(existing.size(),0);

	// FOR EACH EXISTING TEMPLATE LOOK ON AN AREA OF 100 PIXELS AROUND IT
	cout<<"number of templates: "<<existing.size()<<endl;
	for(unsigned k=0; k<existing.size();k++){
		cv::Point center      = this->cvPoint(existing[k]);
		allPeople[k].location = center;
		std::vector<CvPoint> templ;
		genTemplate2(center, persHeight, camHeight, templ);

		//------------------------------------------
		//IplImage *test = new IplImage(foregr);
		//plotTemplate2(test, center, persHeight, camHeight, cvScalar(0,0,255));
		//------------------------------------------

		unsigned minY=thrsh.rows, maxY=0, minX=thrsh.cols, maxX=0;

		// GET THE MIN/MAX SIZE OF THE TEMPLATE
		for(unsigned i=0; i<templ.size(); i++){
			if(minX>templ[i].x){minX = templ[i].x;}
			if(maxX<templ[i].x){maxX = templ[i].x;}
			if(minY>templ[i].y){minY = templ[i].y;}
			if(maxY<templ[i].y){maxY = templ[i].y;}
		}
		minY = std::max((int)minY-100,0);
		maxY = std::min(thrsh.rows,(int)maxY+100);
		minX = std::max((int)minX-100,0);
		maxX = std::min(thrsh.cols,(int)maxX+100);

		unsigned width   = maxX-minX;
		unsigned height  = maxY-minY;
		cv::Mat colorRoi = cv::Mat(foregr.clone(),cv::Rect(cv::Point(minX,minY),\
							cv::Size(width,height)));

		// LOOP OVER THE AREA OF OUR TEMPLATE AND THERESHOLD ONLY THOSE PIXELS
		for(unsigned x=0; x<maxX-minX; x++){
			for(unsigned y=0; y<maxY-minY; y++){
				if((int)(thrsh.at<uchar>((int)(y+minY),(int)(x+minX)))>0){
					// IF THE PIXEL IS NOT INSIDE OF THE TEMPLATE
					if(!this->isInTemplate((x+minX),(y+minY),templ)){
						double minDist = thrsh.rows*thrsh.cols;
						unsigned label = -1;
						for(unsigned l=0; l<existing.size(); l++){
							cv::Point aCenter = this->cvPoint(existing[l]);
							std::vector<CvPoint> aTempl;
							genTemplate2(aCenter,persHeight,camHeight,aTempl);
							if(k!=l && this->isInTemplate((x+minX),(y+minY),aTempl)){
								minDist = 0;
								label   = l;
								break;
							}else{
								double dist = this->getDistToTemplate((int)(x+minX),\
												(int)(y+minY),aTempl);
								if(minDist>dist){
									minDist = dist;
									label   = l;
								}
							}
						}
						// IF THE PIXEL HAS A DIFFERENT LABEL THEN THE CURR TEMPL
						if(label != k || minDist>=persHeight/2){
							colorRoi.at<cv::Vec3b>((int)y,(int)x) = cv::Vec3b(0,0,0);
						// IF IS ASSIGNED TO CURR TEMPL, UPDATE THE BORDER
						}else{
							if(tmpminX[label]>x){tmpminX[label] = x;}
							if(tmpmaxX[label]<x){tmpmaxX[label] = x;}
							if(tmpminY[label]>y){tmpminY[label] = y;}
							if(tmpmaxY[label]<y){tmpmaxY[label] = y;}
						}
						// IF IS ASSIGNED TO CURR TEMPL, UPDATE THE BORDER
					}else{
						if(tmpminX[k]>x){tmpminX[k] = x;}
						if(tmpmaxX[k]<x){tmpmaxX[k] = x;}
						if(tmpminY[k]>y){tmpminY[k] = y;}
						if(tmpmaxY[k]<y){tmpmaxY[k] = y;}
					}
				// IF THE PIXEL VALUE IS BELOW THE THRESHOLD
				}else{
					colorRoi.at<cv::Vec3b>((int)y,(int)x) = cv::Vec3b(0,0,0);
				}
			}
		}
		if(tmpminX[k]==thrsh.cols){tmpminX[k]=0;}
		if(tmpmaxX[k]==0){tmpmaxX[k]=maxX-minX;}
		if(tmpminY[k]==thrsh.rows){tmpminY[k]=0;}
		if(tmpmaxY[k]==0){tmpmaxY[k]=maxY-minY;}
		width  = tmpmaxX[k]-tmpminX[k];
		height = tmpmaxY[k]-tmpminY[k];

		allPeople[k].pixels = cv::Mat(colorRoi.clone(),\
			cv::Rect(cv::Point(tmpminX[k],tmpminY[k]),cv::Size(width,height)));
		cv::imshow("people",allPeople[k].pixels);
		cv::waitKey(0);
		colorRoi.release();
	}
	thrsh.release();
	foregr.release();
}
//==============================================================================
/** Creates the \c Gabor filter with the given parameters and returns the \c wavelet.
 */
cv::Mat featureDetector::createGabor(float params[]){
	// params[0] -- sigma: (3, 68)
	// params[1] -- gamma: (0.2, 1)
	// params[2] -- dimension: (1, 10)
	// params[3] -- theta: (0, 180) or (-90, 90)
	// params[4] -- lambda: (2, 256)
	// params[5] -- psi: (0, 180)

	float sigmaX = params[0];
	float sigmaY = params[0]/params[1];
	float xMax   = std::max(std::abs(params[2]*sigmaX*std::cos(params[3])), \
							std::abs(params[2]*sigmaY*std::sin(params[3])));
	xMax         = std::ceil(std::max((float)1.0, xMax));
	float yMax   = std::max(std::abs(params[2]*sigmaX*std::cos(params[3])), \
							std::abs(params[2]*sigmaY*std::sin(params[3])));
	yMax         = std::ceil(std::max((float)1.0, yMax));
	float xMin   = -xMax;
	float yMin   = -yMax;

	cv::Mat gabor = cv::Mat::zeros((int)(xMax-xMin),(int)(yMax-yMin),CV_32F);
	for(int x=(int)xMin; x<xMax; x++){
		for(int y=(int)yMin; y<yMax; y++){
			float xPrime = x*std::cos(params[3])+y*std::sin(params[3]);
			float yPrime = -x*std::sin(params[3])+y*std::cos(params[3]);
			gabor.at<float>((int)(x+xMax),(int)(y+yMax)) = \
				std::exp(-0.5*((xPrime*xPrime)/(sigmaX*sigmaX)+\
				(yPrime*yPrime)/(sigmaY*sigmaY)))*\
				std::cos(2.0 * M_PI/params[4]*xPrime*params[5]);
		}
	}
	cv::imshow("wavelet",gabor);
	cvDestroyWindow("wavelet");
	return gabor;
}
//==============================================================================
/** Convolves an image with a computed \c Gabor filter.
 */
cv::Mat featureDetector::convolveImage(cv::Point winCenter, cv::Mat image, \
float params[]){
	cv::Mat gabor = this->createGabor(params);

	cv::Mat edges(image.rows, image.cols, CV_8UC1);
	image.convertTo(edges,CV_8UC1,0,1);
	cv::cvtColor(image,edges,CV_RGB2GRAY,0);

	cv::Canny(edges, edges, 10.0, 100.0, 3, false);
	cv::imshow("edges",edges);

	cv::filter2D(edges,edges,-1,gabor,cv::Point(-1,-1),0,cv::BORDER_REPLICATE);
	gabor.release();
	cv::imshow("gabor",edges);
	cv::waitKey();

	cvDestroyWindow("edges");
	cvDestroyWindow("gabor");
	edges.release();
	return image;
}
//==============================================================================
/** Function that gets the ROI corresponding to a head of a person in
 * an image.
 */
void featureDetector::getHeadROI(std::vector<unsigned> existing){
	for(unsigned i=0; i<existing.size(); i++){
		cv::Point center = this->cvPoint(existing[i]);
		std::vector<CvPoint> templ;
		genTemplate2(center, persHeight, camHeight, templ);

		cv::Mat tmpImage(this->current->img);
		unsigned wi         = templ[14].x-templ[13].x;
		unsigned hi         = templ[13].y-templ[12].y;

		unsigned variance = 20;
		templ[12].x -= variance;
		templ[12].y -= variance;
		wi          += variance;
		hi          += variance;

		if((templ[12].x + wi)<tmpImage.rows && (templ[12].y + hi)<tmpImage.cols){
			cv::Mat aRoi(tmpImage.clone(), cv::Rect(templ[12],cv::Size(wi, hi)));
			cv::imshow("head", aRoi);
			cv::Point winCenter(templ[12].x+wi/2,templ[12].y+hi/2);
			float params[]    = {10.0, 1.0, 2.0, M_PI, 2.0, 0.1};
			cv::Mat convolved = this->convolveImage(winCenter,aRoi,params);
			aRoi.release();
		}
		cv::waitKey();
		tmpImage.release();
	}
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

	IplImage *bg = vec2img((imgVec-bgVec).apply(fabs));
	//7') GET ALL PIXELS CORRESPONDING TO PPL AND THEN EXTRACT HEADS AND FEET
	this->getAllForegroundPixels(existing, bg, 12.0);
	//this->getHeadROI(existing);

	//7) SHOW THE FOREGROUND POSSIBLE LOCATIONS AND PLOT THE TEMPLATES
	this->plotHull(bg, this->priorHull);
	cerr<<"no. of detected people: "<<existing.size()<<endl;
	for(unsigned i=0; i!=existing.size(); ++i){
		cv::Point pt = this->cvPoint(existing[i]);
		plotTemplate2(bg,pt,persHeight,camHeight,CV_RGB(255,255,255));
		plotScanLines(this->current->img,mask,CV_RGB(0,255,0),0.3);
		cv::Mat tmp = cv::Mat(this->current->img);
		char buffer[50];
		sprintf(buffer,"%u",i);
		cv::putText(tmp,(string)buffer,pt,3,3.0,cv::Scalar(0,0,255),1,8,false);
	}
	cvShowImage("bg", bg);
	cvShowImage("image",src);

	//8) WHILE NOT q WAS PRESSED PROCESS THE NEXT IMAGES
	return this->imageProcessingMenu();
	cvReleaseImage(&bg);
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
