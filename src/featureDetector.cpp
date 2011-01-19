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
/** Get perpendicular to a line given by 2 points A, B in point C.
 */
void featureDetector::getLinePerpendicular(cv::Point A, cv::Point B, cv::Point C, \
double &m, double &b){
	double slope = (double)(B.y - A.y)/(double)(B.x - A.x);
	m = -1.0/slope;
	b = C.y - m * C.x;
}
//==============================================================================
/** Checks to see if a point is on the same side of a line like another given point.
 */
bool featureDetector::sameSubplane(cv::Point test,cv::Point point, double m, double b){
	if(isnan(m)){
		return (point.x*test.x)>=0.0;
	}else if(m == 0){
		return (point.y*test.y)>=0.0;
	}else{
		return (m*point.x+b-point.y)*(m*test.x+b-test.y)>=0.0;
	}
}
//==============================================================================
/** Creates a symmetrical Gaussian kernel.
 */
void featureDetector::gaussianKernel(cv::Mat &gauss, cv::Size size, double sigma,\
cv::Point offset){
	int xmin = -1 * size.width/2, xmax = size.width/2;
	int ymin = -1 * size.height/2, ymax = size.height/2;

	gauss = cv::Mat::zeros(size.height,size.width,CV_32FC1);
	for(int x=xmin; x<xmax; x++){
		for(int y=ymin; y<ymax; y++){
			gauss.at<float>(y+ymax,x+xmax) = \
				std::exp(-0.5*((x+offset.x)*(x+offset.x)+(y+offset.y)*(y+offset.y))/(sigma*sigma));
		}
	}
}
//==============================================================================
/** Head detection by fitting ellipses.
 */
void featureDetector::ellipseDetection(cv::Mat img){
	cv::Mat large;
	cv::resize(img,large,cv::Size(0,0),5,5,cv::INTER_CUBIC);

	// TRANSFORM FROM BGR TO HSV TO BE ABLE TO DETECT MORE SKIN-LIKE PIXELS
	cv::Mat hsv;
	cv::cvtColor(large, hsv, CV_BGR2HSV);

	// THRESHOLD THE HSV TO KEEP THE HIGH-GREEN PIXELS => brown+green+skin
	for(int x=0; x<hsv.cols; x++){
		for(int y=0; y<hsv.rows; y++){
			if(((hsv.at<cv::Vec3b>(y,x)[0]<255 && hsv.at<cv::Vec3b>(y,x)[0]>50) && \
				(hsv.at<cv::Vec3b>(y,x)[1]<200 && hsv.at<cv::Vec3b>(y,x)[1]>0) && \
				(hsv.at<cv::Vec3b>(y,x)[2]<200 && hsv.at<cv::Vec3b>(y,x)[2]>0))){
				large.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,0);
			}
		}
	}
	cv::imshow("hsv",hsv);
	cv::imshow("large",large);

	/*
	cv::Mat gray;
	cv::cvtColor(hsv, gray, CV_BGR2GRAY);

	// EORDE AND DILATE WITH A GAUSSIAN KERNEL
	cv::Mat gauss;
	this->gaussianKernel(gauss,cv::Size(10,10),5.0,cv::Point(0,0));
	cv::erode(gray, gray, gauss, cv::Point(-1,-1), 2, cv::BORDER_CONSTANT,\
		cv::morphologyDefaultBorderValue());

	cv::dilate(gray,gray,gauss,cv::Point(-1, -1),4,cv::BORDER_CONSTANT,\
		cv::morphologyDefaultBorderValue());
	cv::equalizeHist(gray,gray);
	gauss.release();
	cv::imshow("eroded_dilated", gray);

	//RETAIN ONLY THE PIXELS THAT ARE INSIDE OF A BLOB
	for(int x=0; x<hsv.cols; x++){
		for(int y=0; y<hsv.rows; y++){
			if((int)gray.at<uchar>(y,x)<100){
				hsv.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,0);
			}
		}
	}
	cv::imshow("hsv",hsv);
	*/

	cv::Mat newGray, edges, corners;
	cv::cvtColor(large,newGray,CV_BGR2GRAY);
	cv::medianBlur(newGray,newGray,3);
	cv::Canny(newGray,edges,60,30,3,true);
	cv::imshow("new gray", edges);

	std::vector<cv::Point2f> cors;
	cv::goodFeaturesToTrack(newGray,cors,20,0.001,large.rows/4);
	for(std::size_t i = 0; i <cors.size(); i++){
		cv::circle(large,cv::Point(cors[i].x,cors[i].y),1,cv::Scalar(0,0,255),1,8,0);
	}
	cv::imshow("features", large);


	cv::Sobel(large,large,large.depth(),1,1,3,1,0,cv::BORDER_DEFAULT);
	cv::imshow("gradient", large);

	/*
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(newGray,contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
	//cv::Mat cimage = cv::Mat::zeros(newGray.size(),CV_8UC3);
	cv::drawContours(newGray, contours, -1, cv::Scalar::all(255), 1, 8);
	for(std::size_t i = 0; i < contours.size(); i++){
		size_t count = contours[i].size();
		if( count < 6 ) continue;
		cv::Mat pointsf;
		cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
		cv::RotatedRect box = fitEllipse(pointsf);
		box.angle = -box.angle;
		if(std::max(box.size.width, box.size.height)>\
		std::min(box.size.width, box.size.height)*30) continue;
		//cv::drawContours(cimage, contours, -1, cv::Scalar::all(255), 1, 8);
		cv::ellipse(newGray, box, cv::Scalar(0,0,255), 1, CV_AA);
	}
	cv::imshow("result", newGray);
	*/
}
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
	std::vector<CvPoint> hull;
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
bool featureDetector::isInTemplate(unsigned pixelX,unsigned pixelY,std::vector<CvPoint> templ){
	std::vector<CvPoint> hull;
	convexHull(templ, hull);
	std::vector<scanline_t> lines;
	getScanLines(hull, lines);
	for(std::vector<scanline_t>::const_iterator i=lines.begin();i!=lines.end();++i){
		if(i->line==pixelY && i->start<=pixelX && i->end>=pixelX){
			return true;
		};
	}
	return false;
}
//==============================================================================
/** Get the foreground pixels corresponding to each person
 */
void featureDetector::getAllForegroundPixels(std::vector<featureDetector::people> &allPeople,\
std::vector<unsigned> existing, IplImage *bg, double threshold){
	// INITIALIZING STUFF
	cv::Mat thsh(cvCloneImage(bg));
	cv::Mat thrsh(thsh.rows, thsh.cols, CV_8UC1);
	cv::cvtColor(thsh, thrsh, CV_BGR2GRAY);
	cv::threshold(thrsh, thrsh, threshold, 255, cv::THRESH_BINARY);
	cv::Mat foregr(cvCloneImage(this->current->img));

	std::vector<unsigned> tmpminX(existing.size(),thrsh.cols);
	std::vector<unsigned> tmpmaxX(existing.size(),0);
	std::vector<unsigned> tmpminY(existing.size(),thrsh.rows);
	std::vector<unsigned> tmpmaxY(existing.size(),0);

	// FOR EACH EXISTING TEMPLATE LOOK ON AN AREA OF 100 PIXELS AROUND IT
	cout<<"number of templates: "<<existing.size()<<endl;
	for(unsigned k=0; k<existing.size();k++){
		cv::Point center         = this->cvPoint(existing[k]);
		allPeople[k].absoluteLoc = center;
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
						if(label != k || minDist>=persHeight/5){
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
		allPeople[k].relativeLoc = cv::Point(center.x - (int)(minX + tmpminX[k]),\
										center.y - (int)(minY + tmpminY[k]));
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
void featureDetector::createGabor(cv::Mat &gabor, float params[]){
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

	gabor = cv::Mat::zeros((int)(xMax-xMin),(int)(yMax-yMin),CV_32F);
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
}
//==============================================================================
/** Convolves an image with a computed \c Gabor filter.
 */
cv::Mat featureDetector::convolveImage(cv::Point winCenter, cv::Mat image, \
float params[]){
	/*
	cv::cvtColor(large, gray, CV_BGR2GRAY);

	// NEEDS TO BE SMOOTHED TO REMOVE THE FALSE CIRCLES
	cv::GaussianBlur(gray,gray,cv::Size(9,9),2,2);
	cv::equalizeHist(gray,gray);
	std::vector<cv::Vec3f> circles;

	cv::Sobel(gray,gray,gray.depth(),1,1,1,1,0,cv::BORDER_DEFAULT);
	cv::imshow("laplace", gray);

	cv::imshow("pre-filtered",gray);

	// CONVOLVE WITH SOME GAUSSIAN FILTER OR SMTH
	cv::Mat gauss1, gauss2, gauss;
	this->gaussianKernel(gauss1,cv::Size(20,20),10.0,cv::Point(0,0));
	this->gaussianKernel(gauss2,cv::Size(20,20),5.0,cv::Point(-10,0));
	gauss = gauss1 - gauss2;
	cv::imshow("gauss",gauss);

	cv::filter2D(gray,gray,-1,gauss,cv::Point(-1,-1),0,cv::BORDER_REPLICATE);
	cv::imshow("filtered",gray);

	cv::HoughCircles(gray,circles,CV_HOUGH_GRADIENT,2,gray.rows/4,70,10,0,0);
	//cv::HoughCircles(gray,circles,CV_HOUGH_GRADIENT,2,gray.rows/2,100,100,0,0);
	for(std::size_t i=0; i<circles.size(); i++){
		 cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		 int radius = cvRound(circles[i][2]);

		 // draw the circle center
		 cv::circle(large,center,3,cv::Scalar(0,255,0),-1,8,0);
		 // draw the circle outline
		 cv::circle(large,center,radius,cv::Scalar(0,0,255),3,8,0);
	}
	cv::namedWindow("circles",1);
	cv::imshow("circles", large);

	cv::Sobel(gray,gray,gray.depth(),1,1,1,1,0,cv::BORDER_DEFAULT);
	cv::imshow("laplace", gray);

	 *
	cv::Mat ellipse(1,200,CV_32FC2);
	cv::fitEllipse(ellipse);
	cv::Point* pts = new cv::Point[200];
	for(unsigned i=0; i<ellipse.cols;i++){
		pts[i].x = ellipse.at<cv::Vec2f>(1,i)[0];
		pts[i].y = ellipse.at<cv::Vec2f>(1,i)[1];

		cout<<pts[i].y<<" "<<pts[i].x<<endl;
	}
	cv::fillConvexPoly(large,pts,200,cv::Scalar(200,10,10),8,0);
	cv::imshow("circles", large);

	*/

	cv::Mat gabor;
	this->createGabor(gabor,params);

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
void featureDetector::getHeadROI(featureDetector::people someone,
double variance){
	double offsetX = someone.absoluteLoc.x - someone.relativeLoc.x;
	double offsetY = someone.absoluteLoc.y - someone.relativeLoc.y;

	std::vector<CvPoint> templ;
	genTemplate2(someone.absoluteLoc, persHeight, camHeight, templ);

	templ[12].x -= offsetX; templ[12].y -= offsetY;
	templ[14].x -= offsetX; templ[14].y -= offsetY;
	templ[0].x  -= offsetX; templ[0].y  -= offsetY;
	templ[2].x  -= offsetX; templ[2].y  -= offsetY;
	cv::Point A((templ[14].x+templ[12].x)/2,(templ[14].y+templ[12].y)/2);
	cv::Point B((templ[2].x+templ[0].x)/2,(templ[2].y+templ[0].y)/2);

	double m,b;
	this->getLinePerpendicular(A,B,cv::Point((A.x+B.x)/2,(A.y+B.y)/2),m,b);
	cv::Mat upperRoi(someone.pixels.clone());
	cv::Mat lowerRoi(someone.pixels.clone());
	for(int x=0; x<someone.pixels.cols; x++){
		for(int y=0; y<someone.pixels.rows; y++){
			if(!this->sameSubplane(cv::Point(x,y),A,m,b)){
				upperRoi.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,0);
			}else{
				lowerRoi.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,0);
			}
		}
	}

	cv::imshow("upper part", upperRoi);
	this->ellipseDetection(upperRoi);

	cv::imshow("lower part", lowerRoi);
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
	std::vector<featureDetector::people> allPeople(existing.size());
	this->getAllForegroundPixels(allPeople, existing, bg, 7.0);
	this->getHeadROI(allPeople[0],0);
	//this->ellipseDetection(allPeople[0].pixels);

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
