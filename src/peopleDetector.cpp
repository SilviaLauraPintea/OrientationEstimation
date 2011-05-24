/* peopleDetector.cpp
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#include <boost/thread.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION < 103500
	#include <boost/thread/detail/lock.hpp>
#endif
#include <boost/thread/xtime.hpp>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <cmath>
#include <err.h>
#include <exception>
#include "Auxiliary.h"
#include "cmn/FastMath.hh"
#include "peopleDetector.h"
//==============================================================================
/** Define a post-fix increment operator for the enum \c POSE.
 */
void operator++(peopleDetector::CLASSES &refClass){
	refClass = peopleDetector::CLASSES(refClass+1);
}
//======================================================================
peopleDetector::peopleDetector(int argc,char** argv,bool extract,\
bool buildBg,int colorSp,peopleDetector::FEATUREPART part):Tracker\
(argc,argv,20,buildBg,true,200){
	if(argc == 3){
		std::string dataPath  = std::string(argv[1]);
		std::string imgString = std::string(argv[2]);
		if(dataPath[dataPath.size()-1]!='/'){
			dataPath += "/";
		}
		this->plot              = false;
		this->print             = true;
		this->useGroundTruth    = true;
		this->producer          = NULL;
		this->borderedIpl       = NULL;
		this->colorspaceCode    = colorSp;
		this->featurePart       = part;
		this->tracking 	        = 0;
		this->onlyExtract       = extract;
		this->datasetPath       = dataPath;
		this->imageString       = imgString;
		this->extractor         = new featureExtractor();
		this->initInvColoprSp();
	}else{
		std::cerr<<"Wrong number of arguments: command datasetPath/"<<\
			" imageString"<<std::endl;
		exit(1);
	}

	// INITIALIZE THE DATA MATRIX AND TARGETS MATRIX
	for(unsigned i=0;i<3;++i){
		this->data.push_back(cv::Mat());
		this->targets.push_back(cv::Mat());
		this->dataMotionVectors.push_back(std::deque<float>());
	}
}
//==============================================================================
peopleDetector::~peopleDetector(){
	delete this->extractor;
	if(this->producer){
		delete this->producer;
		this->producer = NULL;
	}
	for(std::size_t i=0;i<this->data.size();++i){
		if(!this->data[i].empty()){
			this->data[i].release();
		}
		if(!this->targets[i].empty()){
			this->targets[i].release();
		}
	}
	if(!this->targetAnno.empty()){
		this->targetAnno.clear();
	}
	if(!this->entireNext.empty()){
		this->entireNext.release();
	}
	this->templates.clear();
	if(this->borderedIpl){
		cvReleaseImage(&this->borderedIpl);
		this->borderedIpl = NULL;
	}
	this->dataMotionVectors.clear();
	this->existing.clear();
	this->classesRange.clear();
}
//==============================================================================
/** Initialize the inverse value of the color space used in feature extraction.
 */
void peopleDetector::initInvColoprSp(){
	switch(this->colorspaceCode){
		case(CV_BGR2XYZ):
			this->invColorspaceCode = CV_XYZ2BGR;
			break;
		case(CV_BGR2YCrCb):
			this->invColorspaceCode = CV_YCrCb2BGR;
			break;
		case(CV_BGR2HSV):
			this->invColorspaceCode = CV_HSV2BGR;
			break;
		case(CV_BGR2HLS):
			this->invColorspaceCode = CV_HLS2BGR;
			break;
		case(CV_BGR2Lab):
			this->invColorspaceCode = CV_Lab2BGR;
			break;
		case(CV_BGR2Luv):
			this->invColorspaceCode = CV_Luv2BGR;
			break;
	}
}
//==============================================================================
/** Checks the image name (used to find the corresponding labels for each image).
 */
struct compareImg{
	public:
		std::string imgName;
		compareImg(std::string image){
			std::deque<std::string> parts = Helpers::splitLine\
				(const_cast<char*>(image.c_str()),'/');
			this->imgName = parts[parts.size()-1];
		}
		virtual ~compareImg(){};
		bool operator()(annotationsHandle::FULL_ANNOTATIONS anno)const{
			return (anno.imgFile == this->imgName);
		}
		compareImg(const compareImg &comp){
			this->imgName = comp.imgName;
		}
		compareImg& operator=(const compareImg &comp){
			if(this == &comp) return *this;
			this->imgName = comp.imgName;
			return *this;
		}
};
//==============================================================================
/** Initializes the parameters of the tracker.
 */
void peopleDetector::init(const std::string &dataFolder,\
const std::string &theAnnotationsFile,const featureExtractor::FEATURE feat,\
const bool readFromFolder){
	this->dataMotionVectors.clear();
	this->existing.clear();
	this->classesRange.clear();
	this->initProducer(readFromFolder,const_cast<char*>(dataFolder.c_str()));
	if(this->borderedIpl){
		cvReleaseImage(&this->borderedIpl);
		this->borderedIpl = NULL;
	}
	if(!this->entireNext.empty()){
		this->entireNext.release();
	}
	for(std::size_t i=0;i<this->data.size();++i){
		if(!this->targets[i].empty()){
			this->targets[i].release();
		}
		if(!this->data[i].empty()){
			this->data[i].release();
		}
	}
	if(!this->targetAnno.empty()){
		this->targetAnno.clear();
	}
	// LOAD THE DESIRED ANNOTATIONS FROM THE FILE
	if(!theAnnotationsFile.empty() && this->targetAnno.empty()){
		annotationsHandle::loadAnnotations(const_cast<char*>\
			(theAnnotationsFile.c_str()),this->targetAnno);
	}

	if(this->onlyExtract){
		assert(feat!=featureExtractor::HOG && feat!=featureExtractor::PIXELS);
	}
	this->extractor->init(feat,this->datasetPath+"features/",this->colorspaceCode,\
		this->invColorspaceCode,(this->featurePart!=peopleDetector::WHOLE));
	if(feat==featureExtractor::SIFT_DICT){
		this->tracking = false;
	}
	this->templates.clear();

	for(unsigned i=0;i<3;++i){
		this->data.push_back(cv::Mat());
		this->targets.push_back(cv::Mat());
		this->dataMotionVectors.push_back(std::deque<float>());
	}
}
//==============================================================================
/** Get template extremities (if needed,considering some borders --
 * relative to the ROI).
 */
void peopleDetector::templateExtremes(const std::vector<cv::Point2f> &templ,\
std::deque<float> &extremes,int minX,int minY){
	if(extremes.empty()){
		extremes = std::deque<float>(4,0.0);
	}
	extremes[0] = std::max(0.0f,templ[0].x-minX); // MINX
	extremes[1] = std::max(0.0f,templ[0].x-minX); // MAXX
	extremes[2] = std::max(0.0f,templ[0].y-minY); // MINY
	extremes[3] = std::max(0.0f,templ[0].y-minY); // MAXY
	for(std::size_t i=0;i<templ.size();++i){
		if(extremes[0]>=templ[i].x-minX) extremes[0] = templ[i].x - minX;
		if(extremes[1]<=templ[i].x-minX) extremes[1] = templ[i].x - minX;
		if(extremes[2]>=templ[i].y-minY) extremes[2] = templ[i].y - minY;
		if(extremes[3]<=templ[i].y-minY) extremes[3] = templ[i].y - minY;
	}
}
//==============================================================================
/** Gets the distance to the given template from a given pixel location.
 */
float peopleDetector::getDistToTemplate(const int pixelX,const int pixelY,\
const std::vector<cv::Point2f> &templ){
	std::vector<cv::Point2f> hull;
	Helpers::convexHull(templ,hull);
	float minDist=-1;
	unsigned i=0,j=1;
	while(i<hull.size() && j<hull.size()-1){
		float midX  = (hull[i].x + hull[j].x)/2;
		float midY  = (hull[i].y + hull[j].y)/2;
		float aDist = std::sqrt((midX - pixelX)*(midX - pixelX) + \
						(midY - pixelY)*(midY - pixelY));
		if(minDist == -1 || minDist>aDist){
			minDist = aDist;
		}
		++i;++j;
	}
	return minDist;
}
//==============================================================================
/** Returns the size of a window around a template centered in a given point.
 */
void peopleDetector::templateWindow(const cv::Size &imgSize,int &minX,int &maxX,\
int &minY,int &maxY,const featureExtractor::templ &aTempl){
	// TRY TO ADD BORDERS TO MAKE IT 100
	minX = aTempl.extremes[0];
	maxX = aTempl.extremes[1];
	minY = aTempl.extremes[2];
	maxY = aTempl.extremes[3];
	int diffX = (this->border - (maxX-minX))/2;
	int diffY = (this->border - (maxY-minY))/2;

	// WILL ALWAYS FIT BECAUSE WE ADDED THE BORDER
	minY = std::max(minY-diffY,0);
	maxY = std::min(maxY+diffY,imgSize.height);
	minX = std::max(minX-diffX,0);
	maxX = std::min(maxX+diffX,imgSize.width);

	// MAKE SUTE WE ARE NOT MISSING A PIXEL OR SO
	if(maxX-minX!=this->border){
		int diffX2 = this->border-(maxX-minX);
		if(minX>diffX2){
			minX -= diffX2;
		}else if(maxX<imgSize.width-diffX2){
			maxX += diffX2;
		}
	}
	if(maxY-minY!=this->border){
		int diffY2 = this->border-(maxY-minY);
		if(minY>diffY2){
			minY -= diffY2;
		}else if(maxY<imgSize.height-diffY2){
			maxY += diffY2;
		}
	}
}
//==============================================================================
/** Assigns pixels to templates based on proximity.
 */
void peopleDetector::pixels2Templates(int maxX,int minX,int maxY,int minY,\
int k,const cv::Mat &thresh,float tmplHeight,cv::Mat &colorRoi){
	// LOOP OVER THE AREA OF OUR TEMPLATE AND THERESHOLD ONLY THOSE PIXELS
	int x1Border = std::max(minX,static_cast<int>(this->templates[k].extremes[0]-\
		tmplHeight/2/2));
	int x2Border = std::min(maxX,static_cast<int>(this->templates[k].extremes[1]+\
		tmplHeight/2/2));
	int y1Border = std::max(minY,static_cast<int>(this->templates[k].extremes[2]-\
		tmplHeight/2/2));
	int y2Border = std::min(maxY,static_cast<int>(this->templates[k].extremes[3]+\
		tmplHeight/2/2));

	cv::Rect roi(x1Border-minX,y1Border-minY,x2Border-x1Border,y2Border-y1Border);
	cv::Mat threshROI(thresh.clone(),cv::Rect(minX,minY,maxX-minX,maxY-minY));
	cv::Mat mask = cv::Mat::zeros(threshROI.size(),threshROI.type());
	cv::Mat maskROI(mask,roi);
	maskROI = cv::Scalar(255,255,255);
	cv::Mat threshPart;
	threshROI.copyTo(threshPart,mask);

	cv::Mat tmpROI = cv::Mat::zeros(colorRoi.size(),colorRoi.type());
	colorRoi.copyTo(tmpROI,threshPart);
	tmpROI.copyTo(colorRoi);
	tmpROI.release();
	mask.release();
	threshROI.release();
	maskROI.release();
	if(this->templates.size()>1){
		for(int x=x1Border-minX;x<x2Border-minX;++x){
			for(int y=y1Border-minY;y<y2Border-minY;++y){
				if(static_cast<int>(threshPart.at<uchar>(y,x))==0){continue;}
				if(!featureExtractor::isInTemplate((x+minX),(y+minY),\
				this->templates[k].points)){
					cv::Point2f mid((this->templates[k].center.x+\
						this->templates[k].head.x)/2.0,(this->templates[k].center.y+\
						this->templates[k].head.y)/2.0);
					float minDist = Helpers::dist(cv::Point2f(x+minX,y+minY),mid);
					int label     = k;
					for(int l=0;l<this->templates.size();++l){
						if(l==k){continue;}
						float tempDist = Helpers::dist(this->templates[k].center,\
							this->templates[l].center);
						if(tempDist >= tmplHeight*1.5){continue;}

						// IF IT IS IN ANOTHER TEMPLATE THEN IGNORE THE PIXEL
						if(featureExtractor::isInTemplate((x+minX),(y+minY),\
						this->templates[l].points)){
							minDist = 0;label = l;
							break;

						// ELSE COMPUTE THE DISTANCE FROM THE PIXEL TO THE TEMPLATE
						}else{
							cv::Point2f middle((this->templates[l].center.x+\
								this->templates[l].head.x)/2.0,\
								(this->templates[l].center.y+\
								this->templates[l].head.y)/2.0);
							float ptDist = Helpers::dist(cv::Point2f(x+minX,y+minY),\
								middle);
							if(minDist>ptDist){
								minDist = ptDist;label = l;
							}
						}
					}
					// IF THE PIXEL HAS A DIFFERENT LABEL THEN THE CURR TEMPL
					if(label != k){
						colorRoi.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,0);
					}
				}
			}
		}
	}
	threshPart.release();
}
//==============================================================================
/** Adds a templates to the vector of templates at detected positions.
 */
void peopleDetector::add2Templates(){
	this->templates.clear();
	for(unsigned i=0;i<this->existing.size();++i){
		featureExtractor::templ aTempl(this->existing[i].location);
		Helpers::genTemplate2(aTempl.center,Helpers::persHeight,Helpers::camHeight,\
			aTempl.points,this->border/2,this->border/2);
		aTempl.center = cv::Point2f((aTempl.points[0].x+aTempl.points[2].x)/2,\
			(aTempl.points[0].y+aTempl.points[2].y)/2);
		aTempl.head = cv::Point2f((aTempl.points[12].x+aTempl.points[14].x)/2,\
			(aTempl.points[12].y+aTempl.points[14].y)/2);
		this->templateExtremes(aTempl.points,aTempl.extremes);
		this->templates.push_back(aTempl);
		aTempl.extremes.clear();
		aTempl.points.clear();
	}
}
//==============================================================================
/** Get the foreground pixels corresponding to each person.
 */
void peopleDetector::allForegroundPixels(std::deque<featureExtractor::people>\
&allPeople,const IplImage *bg,float threshold){
	// INITIALIZING STUFF
	cv::Mat thsh,thrsh;
	if(bg){
		thsh  = cv::Mat(bg);
		cv::cvtColor(thsh,thsh,TO_IMG_FMT);
		cv::cvtColor(thsh,thrsh,CV_BGR2GRAY);
		cv::threshold(thrsh,thrsh,threshold,255,cv::THRESH_BINARY);
	}

	cv::Mat foregr(this->borderedIpl);
	// FOR EACH EXISTING TEMPLATE LOOK ON AN AREA OF 100 PIXELS AROUND IT
	for(unsigned k=0;k<this->existing.size();++k){
		allPeople[k].absoluteLoc = this->existing[k].location;
		float tmplHeight = dist(this->templates[k].head,this->templates[k].center);
		float tmplArea   = tmplHeight*Helpers::dist(this->templates[k].points[0],\
			this->templates[k].points[1]);

		// GET THE 100X100 WINDOW ON THE TEMPLATE
		int minY=foregr.rows,maxY=0,minX=foregr.cols,maxX=0;
		this->templateWindow(cv::Size(foregr.cols,foregr.rows),minX,maxX,\
			minY,maxY,this->templates[k]);
		int awidth  = maxX-minX;
		int aheight = maxY-minY;
		cv::Mat colorRoi = cv::Mat(foregr.clone(),cv::Rect(cv::Point2f(minX,minY),\
			cv::Size(awidth,aheight)));

		//IF THERE IS NO BACKGROUND THEN JUST COPY THE ROI
		cv::Mat thrshRoi;
		if(!bg){
			colorRoi.copyTo(allPeople[k].pixels);
		}else{
			// FOR MULTIPLE DISCONNECTED BLOBS KEEP THE CLOSEST TO CENTER
			this->pixels2Templates(maxX,minX,maxY,minY,k,thrsh,tmplHeight,colorRoi);
			if(this->colorspaceCode!=-1){
				cv::cvtColor(colorRoi,colorRoi,this->invColorspaceCode);
			}
			cv::cvtColor(colorRoi,thrshRoi,CV_BGR2GRAY);
			cv::Point templateMid = cv::Point2f((this->templates[k].head.x+\
				this->templates[k].center.x)/2 - minX,(this->templates[k].head.y+\
				this->templates[k].center.y)/2 - minY);
			this->keepLargestBlob(thrshRoi,templateMid,tmplArea);
			colorRoi.copyTo(allPeople[k].pixels,thrshRoi);
		}
		// SAVE IT IN THE STRUCTURE OF FOREGOUND IMAGES
		allPeople[k].relativeLoc = cv::Point2f(this->existing[k].location.x-(minX),\
			this->existing[k].location.y-(minY));
		allPeople[k].borders.assign(4,0);
		allPeople[k].borders[0] = minX;
		allPeople[k].borders[1] = maxX;
		allPeople[k].borders[2] = minY;
		allPeople[k].borders[3] = maxY;
		if(this->plot){
			cv::imshow("people",allPeople[k].pixels);
			if(bg){
				cv::imshow("threshold",thrshRoi);
			}
			cv::waitKey(0);
		}
		colorRoi.release();
		thrshRoi.release();
	}
	thrsh.release();
}
//==============================================================================
/** Keeps only the largest blob from the thresholded image.
 */
void peopleDetector::keepLargestBlob(cv::Mat &thresh,const cv::Point2f &center,\
float tmplArea){
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(thresh,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
	cv::drawContours(thresh,contours,-1,cv::Scalar(255,255,255),CV_FILLED);
	std::cout<<"Number of contours: "<<contours.size()<<std::endl;
	if(contours.size() == 1){
		return;
	}
	int contourIdx =-1;
	float minDist = thresh.cols*thresh.rows;
	for(size_t i=0;i<contours.size();++i){
		unsigned minX=contours[i][0].x,maxX=contours[i][0].x,minY=contours[i][0].y,\
			maxY=contours[i][0].y;
		for(size_t j=1;j<contours[i].size();++j){
			if(minX>=contours[i][j].x){minX = contours[i][j].x;}
			if(maxX<contours[i][j].x){maxX = contours[i][j].x;}
			if(minY>=contours[i][j].y){minY = contours[i][j].y;}
			if(maxY<contours[i][j].y){maxY = contours[i][j].y;}
		}
		float ptDist = dist(center,cv::Point2f((maxX+minX)/2,(maxY+minY)/2));
		float area   = (maxX-minX)*(maxY-minY);
		if(ptDist<minDist && area>=tmplArea/2){
			contourIdx = i;
			minDist    = ptDist;
		}
	}
	if(contourIdx!=-1){
		contours[contourIdx].clear();
		contours.erase(contours.begin()+contourIdx);
		cv::drawContours(thresh,contours,-1,cv::Scalar(0,0,0),CV_FILLED);
	}
}
//==============================================================================
/** Fixes the existing/detected locations of people and updates the tracks and
 * creates the bordered image.
 */
void peopleDetector::fixLocationsTracksBorderes(const std::deque<unsigned> &exi){
	//1) FIRST GET THE CORRECT LOCATIONS ON THE ORIGINAL IMAGE
	if(this->useGroundTruth){
		this->readLocations();
	}else{
		if(this->targetAnno.empty()){
			std::cerr<<"Annotations not loaded! "<<std::endl;
			std::abort();
		}
		this->fixLabels(exi);
	}

	//2) THEN CREATE THE BORDERED IMAGE
	cvCopyMakeBorder(this->current->img,this->borderedIpl,cv::Point\
		(this->border/2,this->border/2),IPL_BORDER_REPLICATE,cvScalarAll(0));

	//3) CREATE TEMPLATES
	this->add2Templates();

	//4) FIX THE LOCATIONS TO CORRESPOND TO THE BORDERED IMAGE:
	for(size_t i=0;i<this->existing.size();++i){
		this->existing[i].location.x += this->border/2;
		this->existing[i].location.y += this->border/2;
	}

	//5) UPDATE THE TRACKS (ON THE MODIFIED DETECTED LOCATIONS)
	this->updateTracks(this->current->index,exi,this->current->img->width);

	if(this->plot){
		IplImage *tmpSrc;
		tmpSrc = cvCloneImage(this->current->img);
		for(unsigned i=0;i<tracks.size();++i){
			if(this->tracks[i].imgID.size() > MIN_TRACKLEN){
				if(this->current->index-this->tracks[i].imgID.back()<2){
					this->plotTrack(tmpSrc,tracks[i],i,(unsigned)1);
				}
			}
		}
		cvShowImage("tracks",tmpSrc);
		cvWaitKey(0);
		cvReleaseImage(&tmpSrc);
	}
}
//==============================================================================
/** Creates on data row in the final data matrix by getting the feature
 * descriptors.
 */
void peopleDetector::extractDataRow(const IplImage *oldBg,\
const std::deque<unsigned> &exi,float threshVal){
	this->fixLocationsTracksBorderes(exi);

	// ADD A BORDER AROUND THE BACKGROUND TO HAVE THE TEMPLATES CENTERED
	IplImage *bg = NULL;
	if(oldBg){
		bg = cvCreateImage(cvSize(oldBg->width+this->border,oldBg->height+\
			this->border),oldBg->depth,oldBg->nChannels);
		cvCopyMakeBorder(oldBg,bg,cv::Point(this->border/2,this->border/2),\
			IPL_BORDER_REPLICATE,cvScalarAll(0));
	}

	// PROCESS THE REST OF THE IMAGES
	cv::Mat image(this->borderedIpl);
	if(this->colorspaceCode!=-1){
		cv::cvtColor(image,image,this->colorspaceCode);
	}
	if(this->tracking && this->current->index+1<this->producer->filelist.size()){
		this->entireNext = cv::imread(this->producer->filelist\
							[this->current->index+1].c_str());
		if(this->colorspaceCode!=-1){
			cv::cvtColor(this->entireNext,this->entireNext,this->colorspaceCode);
		}
	}

	// REDUCE THE IMAGE TO ONLY THE INTERESTING AREA
	std::deque<featureExtractor::people> allPeople(this->existing.size(),\
		featureExtractor::people());
	this->allForegroundPixels(allPeople,bg,threshVal);

	// GET ONLY THE IMAGE NAME OUT THE CURRENT IMAGE'S NAME
	unsigned pos1       = (this->current->sourceName).find_last_of("/\\");
	std::string imgName = (this->current->sourceName).substr(pos1+1);
	unsigned pos2       = imgName.find_last_of(".");
	imgName             = imgName.substr(0,pos2);

	// FOR EACH LOCATION IN THE IMAGE EXTRACT FEATURES,FILTER THEM AND RESHAPE
	std::deque<std::string> names;
	names.push_back("CLOSE");names.push_back("MEDIUM");	names.push_back("FAR");
	for(std::size_t i=0;i<this->existing.size();++i){
		// DEFINE THE IMAGE ROI OF THE SAME SIZE AS THE FOREGROUND
		cv::Rect roi(allPeople[i].borders[0],allPeople[i].borders[2],\
			allPeople[i].borders[1]-allPeople[i].borders[0],\
			allPeople[i].borders[3]-allPeople[i].borders[2]);

		// COMPUTE THE ROTATION ANGLE ONCE AND FOR ALL
		float rotAngle = this->rotationAngle(this->templates[i].head,\
						this->templates[i].center);

		// ROTATE THE FOREGROUND PIXELS,THREHSOLD AND THE TEMPLATE
		cv::Mat thresholded;
		cv::Point2f rotBorders,absRotCenter;
		std::vector<cv::Point2f> keys;
		this->extractor->rotate2Zero(rotAngle,featureExtractor::MATRIX,roi,\
			absRotCenter,rotBorders,keys,allPeople[i].pixels);
		absRotCenter = cv::Point2f(allPeople[i].pixels.cols/2.0+allPeople[i].borders[0],\
			allPeople[i].pixels.rows/2.0+allPeople[i].borders[2]);
		cv::Mat aDummy;
		this->extractor->rotate2Zero(rotAngle,featureExtractor::TEMPLATE,roi,\
			absRotCenter,rotBorders,this->templates[i].points,aDummy);

		// RESET THE POSITION OF THE HEAD IN THE ROATED TEMPLATE
		this->templates[i].center = cv::Point2f((this->templates[i].points[0].x+\
			this->templates[i].points[2].x)/2,(this->templates[i].points[0].y+\
			this->templates[i].points[2].y)/2);
		this->templates[i].head = cv::Point2f((this->templates[i].points[12].x+\
			this->templates[i].points[14].x)/2,(this->templates[i].points[12].y+\
			this->templates[i].points[14].y)/2);
		this->templateExtremes(this->templates[i].points,this->templates[i].extremes);

		// IF WE CAN THRESHOLD THE IMAGE USING THE BACKGROUND MODEL
		if(bg){
			cv::inRange(allPeople[i].pixels,cv::Scalar(1,1,1),cv::Scalar(255,225,225),\
				thresholded);
			cv::dilate(thresholded,thresholded,cv::Mat(),cv::Point(-1,-1),2);
		}

		// IF THE PART TO BE CONSIDERED IS ONLY FEET OR ONLY HEAD

		if(this->featurePart!=peopleDetector::WHOLE && this->existing[i].groupNo!=\
		peopleDetector::CLOSE){
			this->templatePart(thresholded,i,allPeople[i].borders[0],\
				allPeople[i].borders[2]);
		}

		// ANF FINALLY EXTRACT THE DATA ROW
		this->extractor->setImageClass(this->existing[i].groupNo);
		cv::Mat dataRow = this->extractor->getDataRow(this->borderedIpl->width,\
			this->templates[i],roi,allPeople[i],thresholded,imgName,absRotCenter,\
			rotBorders,rotAngle,keys);
		dataRow.at<float>(0,dataRow.cols-3) = this->distanceWRTcamera\
			(this->templates[i].center);

		// CHECK THE DIRECTION OF THE MOVEMENT
		bool moved;
		float direction = this->motionVector(this->templates[i].head,\
			this->templates[i].center,moved);
		if(!moved){direction = -1.0;}
		this->dataMotionVectors[this->existing[i].groupNo].push_back(direction);

		if(this->tracking && !this->entireNext.empty()){
			dataRow.at<float>(0,dataRow.cols-1) = direction;
			cv::Mat nextImg(this->entireNext,roi);
			this->extractor->rotate2Zero(rotAngle,featureExtractor::MATRIX,roi,\
				absRotCenter,rotBorders,keys,nextImg);
			dataRow.at<float>(0,dataRow.cols-2) = \
				this->opticalFlow(allPeople[i].pixels,nextImg,keys,\
				this->templates[i].head,this->templates[i].center,false);
			nextImg.release();
		}
		dataRow.convertTo(dataRow,CV_32FC1);
		Auxiliary::mean0Variance1(dataRow);

		// STORE THE EXTRACTED ROW INTO THE DATA MATRIX
		if(this->data[this->existing[i].groupNo].empty()){
			dataRow.copyTo(this->data[this->existing[i].groupNo]);
		}else{
			this->data[this->existing[i].groupNo].push_back(dataRow);
		}
		if(!thresholded.empty()){thresholded.release();}
		dataRow.release();

		float label = std::atan2(this->targets[this->existing[i].groupNo].\
			at<float>(this->data[this->existing[i].groupNo].rows-1,0),\
			this->targets[this->existing[i].groupNo].\
			at<float>(this->data[this->existing[i].groupNo].rows-1,1));
		Auxiliary::angle0to360(label);
		std::cout<<names[this->existing[i].groupNo]<<":"<<\
			this->templates[i].center<<">>>"<<"LONGITUDE>>>"<<(label*180.0/M_PI)<<\
			std::endl<<"-----------------------------------------------"<<std::endl;
	}
	if(bg){
		cvReleaseImage(&bg);
	}
}
//==============================================================================
/** Compute the dominant direction of the SIFT or SURF features.
 */
float peopleDetector::opticalFlow(cv::Mat &currentImg,cv::Mat &nextImg,\
const std::vector<cv::Point2f> &keyPts,const cv::Point2f &head,\
const cv::Point2f &center,bool maxOrAvg){
	// GET THE OPTICAL FLOW MATRIX FROM THE FEATURES
	float direction = -1;
	cv::Mat currGray,nextGray;
	if(this->colorspaceCode!=-1){
		cv::cvtColor(currentImg,currentImg,this->invColorspaceCode);
		cv::cvtColor(nextImg,nextImg,this->invColorspaceCode);
	}
	cv::cvtColor(currentImg,currGray,CV_RGB2GRAY);
	cv::cvtColor(nextImg,nextGray,CV_RGB2GRAY);
	std::vector<cv::Point2f> flow;
	std::vector<uchar> status;
	std::vector<float> error;
	cv::calcOpticalFlowPyrLK(currGray,nextGray,keyPts,flow,status,error,\
		cv::Size(15,15),3,cv::TermCriteria(cv::TermCriteria::COUNT+\
		cv::TermCriteria::EPS,30,0.01),0.5,0);

	//GET THE DIRECTION OF THE MAXIMUM OPTICAL FLOW
	float flowX = 0.0,flowY = 0.0,resFlowX = 0.0,resFlowY = 0.0,magni = 0.0;
	for(std::size_t i=0;i<flow.size();++i){
		float ix,iy,fx,fy;
		ix = keyPts[i].x;iy = keyPts[i].y;fx = flow[i].x;fy = flow[i].y;

		// IF WE WANT THE MAXIMUM OPTICAL FLOW
		if(maxOrAvg){
			float newMagni = std::sqrt((fx-ix)*(fx-ix) + (fy-iy)*(fy-iy));
			if(newMagni>magni){
				magni = newMagni;flowX = ix;flowY = iy;resFlowX = fx;resFlowY = fy;
			}
		// ELSE IF WE WANT THE AVERAGE OPTICAL FLOW
		}else{
			flowX += ix;flowY += iy;resFlowX += fx;resFlowY += fy;
		}
		if(this->plot){
			cv::line(currentImg,cv::Point2f(ix,iy),cv::Point2f(fx,fy),\
				cv::Scalar(0,0,255),1,8,0);
			cv::circle(currentImg,cv::Point2f(fx,fy),2,cv::Scalar(0,200,0),1,8,0);
		}
	}
	//	IF WE WANT THE AVERAGE OPTICAL FLOW
	if(!maxOrAvg){
		flowX /= keyPts.size();flowY /= keyPts.size();
		resFlowX /= keyPts.size();resFlowY /= keyPts.size();
	}

	if(this->plot){
		cv::line(currentImg,cv::Point2f(flowX,flowY),cv::Point2f(resFlowX,resFlowY),\
			cv::Scalar(255,0,0),1,8,0);
		cv::circle(currentImg,cv::Point2f(resFlowX,resFlowY),2,cv::Scalar(0,200,0),\
			1,8,0);
		cv::imshow("optical",currentImg);
		cv::waitKey(0);
	}

	// DIRECTION OF THE AVERAGE FLOW
	direction = std::atan2(resFlowY - flowY,resFlowX - flowX);
	std::cout<<"Flow direction: "<<direction*180/M_PI<<std::endl;
	currGray.release();
	nextGray.release();
	direction = this->fixAngle(head,center,direction);
	return direction;
}
//==============================================================================
/** Checks to see if an annotation can be assigned to a detection.
 */
bool peopleDetector::canBeAssigned(unsigned l,std::deque<float> &minDistances,\
unsigned k,float distance,std::deque<int> &assignment){
	unsigned isThere = 1;
	while(isThere){
		isThere = 0;
		for(std::size_t i=0;i<assignment.size();++i){
			// IF THERE IS ANOTHER ASSIGNMENT FOR K WITH A LARGER DIST
			if(assignment[i] == k && i!=l && minDistances[i]>distance){
				assignment[i]   = -1;
				minDistances[i] = (float)INFINITY;
				isThere         = 1;
				break;
			// IF THERE IS ANOTHER ASSIGNMENT FOR K WITH A SMALLER DIST
			}else if(assignment[i] == k && i!=l && minDistances[i]<distance){
				return false;
			}
		}
	}
	return true;
}
//==============================================================================
/** Fixes the angle to be relative to the camera position with respect to the
 * detected position.
 */
float peopleDetector::fixAngle(const cv::Point2f &headLocation,\
const cv::Point2f &feetLocation,float angle){
	float camAngle = -std::atan2((headLocation.y-feetLocation.y),\
		(headLocation.x-feetLocation.x));
	camAngle = M_PI/2.0-camAngle;
	float newAngle = angle+camAngle;
	Auxiliary::angle0to360(newAngle);
	return newAngle;
}
//==============================================================================
/** For each row added in the data matrix (each person detected for which we
 * have extracted some features) find the corresponding label.
 */
void peopleDetector::fixLabels(const std::deque<unsigned> &existing){
	// FIND	THE INDEX FOR THE CURRENT IMAGE
	std::deque<annotationsHandle::FULL_ANNOTATIONS>::iterator index = \
		std::find_if (this->targetAnno.begin(),this->targetAnno.end(),\
		compareImg(this->current->sourceName));
	if(index == this->targetAnno.end()){
		std::cerr<<"The image: "<<this->current->sourceName<<\
			" was not annotated"<<std::endl;
		exit(1);
	}

	//GET THE VALID DETECTED LOCATIONS
	std::vector<cv::Point2f> points;
	for(std::size_t i=0;i<existing.size();++i){
		cv::Point2f center = this->cvPoint(existing[i],this->current->img->width);
		points.push_back(center);
	}

	// LOOP OVER ALL ANNOTATIONS FOR THE CURRENT IMAGE AND FIND THE CLOSEST ONES
	std::deque<int> assignments((*index).annos.size(),-1);
	std::deque<float> minDistances((*index).annos.size(),(float)INFINITY);

	std::vector<cv::Point2f> allTrueHeads((*index).annos.size(),cv::Point2f());
	std::deque<bool> allTrueInside((*index).annos.size(),true);
	std::deque<float> allTrueWidth((*index).annos.size(),0.0f);
	unsigned canAssign = 1;
	while(canAssign){
		canAssign = 0;
		for(std::size_t l=0;l<(*index).annos.size();++l){
			// IF WE ARE LOOPING THE FIRST TIME THEN STORE THE TEMPLATE POINTS
			if(!allTrueWidth[l]){
				std::vector<cv::Point2f> aTempl;
				allTrueInside[l] = Helpers::genTemplate2((*index).annos[l].location,\
					Helpers::persHeight,Helpers::camHeight,aTempl);
				allTrueHeads[l] = cv::Point2f((aTempl[12].x+aTempl[14].x)/2,\
					(aTempl[12].y+aTempl[14].y)/2);
				allTrueWidth[l] = dist(aTempl[0],aTempl[1]);
			}

			// EACH ANNOTATION NEEDS TO BE ASSIGNED TO THE CLOSEST DETECTION
			float distance     = (float)INFINITY;
			unsigned annoIndex = -1;
			for(std::size_t k=0;k<points.size();++k){
				float dstnc = dist((*index).annos[l].location,points[k]);
				if(distance>dstnc && this->canBeAssigned(l,minDistances,k,dstnc,\
				assignments) && dstnc<2*allTrueWidth[l]){
					distance  = dstnc;
					annoIndex = k;
				}
			}
			assignments[l]  = annoIndex;
			minDistances[l] = distance;
		}
	}

	// DELETE DETECTED LOCATIONS THAT ARE NOT LABELLED ARE IGNORED
	for(std::size_t k=0;k<points.size();++k){
		// SEARCH FOR A DETECTED POSITION IN ASSIGNMENTS
		std::deque<int>::iterator targetPos = \
			std::find(assignments.begin(),assignments.end(),k);

		// IF THE POSITION IS FOUND READ THE LABEL AND SAVE IT
		if(targetPos != assignments.end()){
			int position = targetPos-assignments.begin();

			// IF THE BORDERS OF THE TEMPLATE ARE OUTSIDE THE IMAGE THEN IGNORE IT
			if(!allTrueInside[position]){
				continue;
			}
			cv::Point2f feet = (*index).annos[position].location;
			cv::Point2f head = allTrueHeads[position];

			// SAVE THE TARGET LABEL
			cv::Mat tmp = cv::Mat::zeros(1,4,CV_32FC1);
			// READ THE TARGET ANGLE FOR LONGITUDINAL ANGLE
			float angle = static_cast<float>((*index).annos[position].\
							poses[annotationsHandle::LONGITUDE]);
			angle = angle*M_PI/180.0;
			angle = this->fixAngle(head,feet,angle);
			tmp.at<float>(0,0) = std::sin(angle);
			tmp.at<float>(0,1) = std::cos(angle);

			// READ THE TARGET ANGLE FOR LATITUDINAL ANGLE
			angle = static_cast<float>((*index).annos[position].\
					poses[annotationsHandle::LATITUDE]);
			angle = angle*M_PI/180.0;
			tmp.at<float>(0,2) = std::sin(angle);
			tmp.at<float>(0,3) = std::cos(angle);

			// STORE THE LABELS IN THE TARGETS ON THE RIGHT POSITION
			peopleDetector::CLASSES groupNo=this->findImageClass(feet,head);
			peopleDetector::Existing tmpExisting(feet,static_cast<unsigned>(groupNo));
			this->existing.push_back(tmpExisting);
			if(this->targets[groupNo].empty()){
				tmp.copyTo(this->targets[groupNo]);
			}else{
				this->targets[groupNo].push_back(tmp);
			}
			tmp.release();
		}
	}
}
//==============================================================================
/** Overwrites the \c doFindPeople function from the \c Tracker class to make it
 * work with the feature extraction.
 */
bool peopleDetector::doFindPerson(unsigned imgNum,IplImage *src,\
const vnl_vector<FLOAT> &imgVec,vnl_vector<FLOAT> &bgVec,\
const FLOAT logBGProb,const vnl_vector<FLOAT> &logSumPixelBGProb){
	std::cout<<this->current->index<<") Image... "<<this->current->sourceName<<std::endl;

	//1) START THE TIMER & INITIALIZE THE PROBABLITIES,THE VECTOR OF POSITIONS
	stic();	std::deque<FLOAT> marginal;
	FLOAT lNone = logBGProb + this->logNumPrior[0],lSum = -INFINITY;
	std::deque<unsigned> existing;
	std::deque< vnl_vector<FLOAT> > logPosProb;
	std::deque<Helpers::scanline_t> mask;
	marginal.push_back(lNone);
	logPosProb.push_back(vnl_vector<FLOAT>());

	//2) SCAN FOR POSSIBLE LOCATION OF PEOPLE GIVEN THE EXISTING ONES
	this->scanRest(existing,mask,Helpers::scanres,logSumPixelBGProb,logPosProb,\
		marginal,lNone);

	//3) UPDATE THE PROBABILITIES
	for(unsigned i=0;i!=marginal.size();++i){
		lSum = log_sum_exp(lSum,marginal[i]);
	}

	//4) UPDATE THE MOST LIKELY NUMBER OF PEOPLE AND THE MARGINAL PROBABILITY
	unsigned mlnp = 0;// most likely number of people
	FLOAT mlprob  = -INFINITY;
	for(unsigned i=0;i!=marginal.size();++i) {
		if (marginal[i] > mlprob) {
			mlnp   = i;
			mlprob = marginal[i];
		}
	}

	//5) DILATE A BIT THE BACKGROUND SO THE BACKGROUND NOISE GOES NICELY
	IplImage *bg = Helpers::vec2img((imgVec-bgVec).apply(fabs));
	cvSmooth(bg,bg,CV_MEDIAN,7,7);
	cvErode(bg,bg,NULL,2);
	cvDilate(bg,bg,NULL,2);

	//7) SHOW THE FOREGROUND POSSIBLE LOCATIONS AND PLOT THE TEMPLATES
	cerr<<"no. of detected people: "<<existing.size()<<endl;
	if(this->plot){
		IplImage *tmpBg,*tmpSrc;
		tmpBg  = cvCloneImage(bg);
		tmpSrc = cvCloneImage(src);
		// PLOT HULL
		this->plotHull(bg,this->priorHull);

		// PLOT DETECTIONS
		for(unsigned i=0;i!=existing.size();++i){
			cv::Point2f pt = this->cvPoint(existing[i],this->current->img->width);
			std::vector<cv::Point2f> points;
			Helpers::plotTemplate2(tmpBg,pt,Helpers::persHeight,Helpers::camHeight,\
				CV_RGB(255,255,255),points);
			Helpers::plotScanLines(tmpSrc,mask,CV_RGB(0,255,0),0.3);
		}
		cvShowImage("bg",tmpBg);
		cvShowImage("image",tmpSrc);
		cvWaitKey(0);
		cvReleaseImage(&tmpBg);
		cvReleaseImage(&tmpSrc);
	}

	//10) EXTRACT FEATURES FOR THE CURRENTLY DETECTED LOCATIONS
	cout<<"Number of templates: "<<existing.size()<<endl;
	this->extractDataRow(bg,existing);
	cout<<"Number of templates afterwards: "<<existing.size()<<endl;

	//11) WHILE NOT q WAS PRESSED PROCESS THE NEXT IMAGES
	cvReleaseImage(&bg);
	return this->imageProcessingMenu();
}
//==============================================================================
/** Simple "menu" for skipping to the next image or quitting the processing.
 */
bool peopleDetector::imageProcessingMenu(){
	if(this->plot){
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
	}
	return true;
}
//==============================================================================
/** If only a part needs to be used to extract the features then the threshold
 * and the template need to be changed.
 */
void peopleDetector::templatePart(cv::Mat &thresholded,int k,float offsetX,\
float offsetY){
	// CHANGE THRESHOLD
	int minX,maxX,minY,maxY;
	if(!thresholded.empty()){
		this->extractor->getThresholdBorderes(minX,maxX,minY,maxY,thresholded);
		for(unsigned i=0;i<this->templates[k].points.size();++i){
			if(this->templates[k].points[i].x-offsetX>maxX){
				this->templates[k].points[i].x = maxX+offsetX;
			}
			if(this->templates[k].points[i].x-offsetX<minX){
				this->templates[k].points[i].x = minX+offsetX;
			}
			if(this->templates[k].points[i].y-offsetY>maxY){
				this->templates[k].points[i].y = maxY+offsetY;
			}
			if(this->templates[k].points[i].y-offsetY<minY){
				this->templates[k].points[i].y = minY+offsetY;
			}
		}
	}else{
		minX = this->templates[k].extremes[0]-offsetX;
		maxX = this->templates[k].extremes[1]-offsetX;
		minY = this->templates[k].extremes[2]-offsetY;
		maxY = this->templates[k].extremes[3]-offsetY;
	}

	float percent   = 0.50;
	float middleTop = (minY+maxY)*percent;
	float middleBot = (minY+maxY)*percent;
	if(!thresholded.empty()){
		for(int x=0;x<thresholded.cols;++x){
			for(int y=0;y<thresholded.rows;++y){
				if(y>middleTop && this->featurePart==peopleDetector::TOP){
					thresholded.at<uchar>(y,x) = 0;
				}else if(y<middleBot && this->featurePart==peopleDetector::BOTTOM){
					thresholded.at<uchar>(y,x) = 0;
				}
			}
		}
	}

	// CHANGE TEMPLATE
	for(unsigned i=0;i<this->templates[k].points.size();++i){
		if(this->featurePart == peopleDetector::TOP &&\
		(this->templates[k].points[i].y-offsetY)>middleTop){
			this->templates[k].points[i].y = middleTop+offsetY;
		}else if(this->featurePart==peopleDetector::BOTTOM &&\
		(this->templates[k].points[i].y-offsetY)<middleBot){
			this->templates[k].points[i].y = middleBot+offsetY;
		}
	}
	this->templateExtremes(this->templates[k].points,this->templates[k].extremes);

	if(this->plot){
		std::vector<cv::Point2f> tmpTempl = this->templates[k].points;
		for(unsigned i=0;i<tmpTempl.size();++i){
			tmpTempl[i].x -= offsetX;
			tmpTempl[i].y -= offsetY;
		}
		if(!thresholded.empty()){
			Helpers::plotTemplate2(thresholded,cv::Point2f(0,0),\
				cv::Scalar(0,255,0),tmpTempl);
			cv::imshow("part",thresholded);
			cv::waitKey(0);
		}
	}
}
//==============================================================================
/** Computes the motion vector for the current image given the tracks so far.
 */
float peopleDetector::motionVector(const cv::Point2f &head,\
const cv::Point2f &center,bool &moved){
	cv::Point2f prev = center;
	float angle      = 0;
	moved            = false;
	if(this->tracks.size()>1){
		for(unsigned i=0;i<this->tracks.size();++i){
			if(this->tracks[i].positions[this->tracks[i].positions.size()-1].x ==\
			center.x && this->tracks[i].positions[this->tracks[i].positions.size()-1].y\
			== center.y){
				if(this->tracks[i].positions.size()>1){
					moved  = true;
					prev.x = this->tracks[i].positions[this->tracks[i].positions.size()-2].x;
					prev.y = this->tracks[i].positions[this->tracks[i].positions.size()-2].y;
				}
				break;
			}
		}

		if(this->plot){
			cv::Mat tmp(cvCloneImage(this->borderedIpl));
			cv::line(tmp,prev,center,cv::Scalar(50,100,255),1,8,0);
			cv::imshow("tracks",tmp);
		}
	}
	if(moved){
		angle = std::atan2(center.y-prev.y,center.x-prev.x);
	}

	// FIX ANGLE WRT CAMERA
	angle = this->fixAngle(head,center,angle);
	std::cout<<"Motion angle>>> "<<(angle*180/M_PI)<<std::endl;
	return angle;
}
//==============================================================================
/** Starts running something (either the tracker or just mimics it).
 */
void peopleDetector::start(bool readFromFolder,bool useGT){
	this->useGroundTruth = useGT;
	// ((useGT & !GABOR & !EDGES) | EXTRACT) & ANNOS
	if(!this->targetAnno.empty() && (this->onlyExtract || this->useGroundTruth)){
		// READ THE FRAMES ONE BY ONE
		if(!this->producer){
			this->initProducer(readFromFolder);
		}
		IplImage *img     = this->producer->getFrame();
		Helpers::width    = img->width;
		Helpers::height   = img->height;
		Helpers::depth    = img->depth;
		Helpers::channels = img->nChannels;
		Helpers::halfresX = Helpers::width/2;
		Helpers::halfresY = Helpers::height/2;
		this->producer->backward(1);
		this->current     = new Image_t();
		unsigned index    = 0;
		this->borderedIpl = cvCreateImage(cvSize(img->width+this->border,\
			img->height+this->border),img->depth,img->nChannels);
		while((this->current->img = this->producer->getFrame())){
			this->current->sourceName = this->producer->getSource(-1);
			this->current->index      = index;
			std::cout<<index<<") Image... "<<this->current->sourceName<<std::endl;

			//1) EXTRACT FEATURES OD TRAINING/TESTING DATA
			if(this->onlyExtract){
				cvCopyMakeBorder(this->current->img,this->borderedIpl,cv::Point\
					(this->border/2,this->border/2),IPL_BORDER_REPLICATE,cvScalarAll(0));
				cv::Mat borderedMat(this->borderedIpl);
				this->extractor->extractFeatures(borderedMat,this->current->sourceName);
				borderedMat.release();
			}else{
				this->extractDataRow(NULL);
			}
			++index;
		}
		//5) RELEASE THE IMAGE AND THE BORDERED IMAGE
		if(this->borderedIpl){
			cvReleaseImage(&this->borderedIpl);
			this->borderedIpl = NULL;
		}
		if(this->current){
			cvReleaseImage(&this->current->img);
			this->current->img = NULL;
			delete this->current;
			this->current = NULL;
		}
	}else{
		// FOR THE EDGES AND GABOR I NEED A BACKGROUND MODEL EVEN IF I ONLY EXTRACT
		this->useGroundTruth = false;
		this->run(readFromFolder);
	}
}
//==============================================================================
/** Reads the locations at which there are people in the current frame (for the
 * case in which we do not want to use the tracker or build a bgModel).
 */
void peopleDetector::readLocations(){
	// FIND	THE INDEX FOR THE CURRENT IMAGE
	if(this->targetAnno.empty()){
		std::cerr<<"Annotations were not loaded."<<std::endl;
		exit(1);
	}
	std::deque<annotationsHandle::FULL_ANNOTATIONS>::iterator index = \
		std::find_if (this->targetAnno.begin(),this->targetAnno.end(),\
		compareImg(this->current->sourceName));

	if(index == this->targetAnno.end()){
		std::cerr<<"The image: "<<this->current->sourceName<<\
			" was not annotated"<<std::endl;
		exit(1);
	}
	// TRANSFORM THE LOCATION INTO UNSIGNED AND THEN PUSH IT IN THE VECTOR
	for(std::size_t l=0;l<(*index).annos.size();++l){
		// IF THE BORDERS OF THE TEMPLATE ARE OUTSIDE THE IMAGE THEN IGNORE IT
		std::vector<cv::Point2f> templ;
		if(!Helpers::genTemplate2((*index).annos[l].location,Helpers::persHeight,\
		Helpers::camHeight,templ)){
			continue;
		}
		// point.x + width*point.y
		cv::Point2f feet = (*index).annos[l].location;
		cv::Point2f head = cv::Point2f((templ[12].x+templ[14].x)/2,\
			(templ[12].y+templ[14].y)/2);

		// STORE THE LABELS FOR ALL THE LOCATIONS
		cv::Mat tmp = cv::Mat::zeros(cv::Size(4,1),CV_32FC1);
		// READ THE TARGET ANGLE FOR LONGITUDINAL ANGLE
		float angle = static_cast<float>((*index).annos[l].\
						poses[annotationsHandle::LONGITUDE]);
		angle = angle*M_PI/180.0;
		angle = this->fixAngle(head,feet,angle);
		tmp.at<float>(0,0) = std::sin(angle);
		tmp.at<float>(0,1) = std::cos(angle);

		// READ THE TARGET ANGLE FOR LATITUDINAL ANGLE
		angle = static_cast<float>((*index).annos[l].\
					poses[annotationsHandle::LATITUDE]);
		angle = angle*M_PI/180.0;
		tmp.at<float>(0,2) = std::sin(angle);
		tmp.at<float>(0,3) = std::cos(angle);

		// STORE THE LABELS IN THE TARGETS ON THE RIGHT POSITION
		peopleDetector::CLASSES groupNo = this->findImageClass(feet,head);
		peopleDetector::Existing tmpExisting(feet,static_cast<unsigned>(groupNo));
		this->existing.push_back(tmpExisting);
		if(this->targets[groupNo].empty()){
			tmp.copyTo(this->targets[groupNo]);
		}else{
			this->targets[groupNo].push_back(tmp);
		}
		tmp.release();
	}

	// SEE THE ANNOTATED LOCATIONS IN THE IMAGE
	if(this->plot){
		for(std::size_t i=0;i<this->existing.size();++i){
			std::vector<cv::Point2f> templ;
			Helpers::plotTemplate2(this->current->img,this->existing[i].location,\
				Helpers::persHeight,Helpers::camHeight,cv::Scalar(255,0,0),templ);
		}
		cvShowImage("AnnotatedLocations",this->current->img);
		cvWaitKey(0);
	}
}
//==============================================================================
/** Return rotation angle given the head and feet position.
 */
float peopleDetector::rotationAngle(const cv::Point2f &headLocation,\
const cv::Point2f &feetLocation){
	float rotAngle = -std::atan2((headLocation.y-feetLocation.y),\
						(headLocation.x-feetLocation.x));
	rotAngle  = M_PI/2.0-rotAngle;
	rotAngle *= (180.0/M_PI);
	return rotAngle;
}
//==============================================================================
/** Find the class in which we can store the current image (the data is split in
 * 3 classes depending on the position of the person wrt camera).
 */
peopleDetector::CLASSES peopleDetector::findImageClass(const cv::Point2f &feet,\
const cv::Point2f &head){
	if(this->classesRange.empty()){
		// GET THE CAMERA POSITION IN THE IMAGE PLANE
		cv::Point2f cam = (*Helpers::proj)(cv::Point3f(Helpers::camPosX,\
			Helpers::camPosY,0));
		float ratio,step;
		step  = (Helpers::maxPersHeightPixels-0.001)/3.0;
		float previous = 0.0;
		for(float in=step;in<=Helpers::maxPersHeightPixels;in+=step){
			this->classesRange.push_back(cv::Point2f(previous,in));
			std::cout<<"previousRange:"<<previous<<" currentRange"<<in<<std::endl;
			previous = in;
		}
		sleep(6);
	}

	float distance = Helpers::dist(feet,head);
	if(distance>this->classesRange[peopleDetector::CLOSE].x &&\
	distance<=this->classesRange[peopleDetector::CLOSE].y){
		return peopleDetector::CLOSE;
	}else if(distance>this->classesRange[peopleDetector::MEDIUM].x &&\
	distance<=this->classesRange[peopleDetector::MEDIUM].y){
		return peopleDetector::MEDIUM;
	}else if(distance>this->classesRange[peopleDetector::FAR].x){
		return peopleDetector::FAR;
	}
}
//==============================================================================
/** Get distance wrt the camera in the image.
 */
float peopleDetector::distanceWRTcamera(const cv::Point2f &feet){
	cv::Point2f cam = (*Helpers::proj)(cv::Point3f(Helpers::camPosX,\
		Helpers::camPosY,0.0));
	return Helpers::dist(cam,feet);
}
//==============================================================================
/*
int main(int argc,char **argv){
	peopleDetector feature(argc,argv,true,false,-1);
	feature.init(std::string(argv[1])+"annotated_train",\
		std::string(argv[1])+"annotated_train.txt",featureExtractor::EDGES,true);
	feature.start(true,false);
}
*/
