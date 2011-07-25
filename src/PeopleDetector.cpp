/* PeopleDetector.cpp
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
#include "PeopleDetector.h"
//==============================================================================
/** Define a post-fix increment operator for the enum \c POSE.
 */
void operator++(PeopleDetector::CLASSES &refClass){
	refClass = PeopleDetector::CLASSES(refClass+1);
}
//======================================================================
PeopleDetector::PeopleDetector(int argc,char** argv,bool extract,\
bool buildBg,int colorSp,FeatureExtractor::FEATUREPART part,bool flip):Tracker\
(argc,argv,126,buildBg,true,300){
	if(argc == 3){
		std::string dataPath  = std::string(argv[1]);
		std::string imgString = std::string(argv[2]);
		if(dataPath[dataPath.size()-1]!='/'){
			dataPath += "/";
		}
		this->isTest_         = false;
		this->flip_           = flip;
		this->plot_           = false;
		this->print_          = true;
		this->useGroundTruth_ = true;
		this->colorspaceCode_ = colorSp;
		this->featurePart_    = part;
		this->tracking_ 	  = 0;
		this->onlyExtract_    = extract;
		this->datasetPath_    = dataPath;
		this->imageString_    = imgString;
		this->extractor_      = std::tr1::shared_ptr<FeatureExtractor>\
			(new FeatureExtractor());
		this->initInvColoprSp();
	}else{
		std::cerr<<"Wrong number of arguments: command datasetPath/"<<\
			" imageString"<<std::endl;
		exit(1);
	}

	// INITIALIZE THE DATA MATRIX AND TARGETS MATRIX
	for(unsigned i=0;i<3;++i){
		this->data_.push_back(cv::Mat());
		this->targets_.push_back(cv::Mat());
		this->dataMotionVectors_.push_back(std::deque<float>());
	}
}
//==============================================================================
PeopleDetector::~PeopleDetector(){
	if(this->extractor_){
		this->extractor_.reset();
	}
	if(this->producer_){
		this->producer_.reset();
	}
	if(this->borderedIpl_){
		this->borderedIpl_.reset();
	}
	for(std::size_t i=0;i<this->data_.size();++i){
		if(!this->data_[i].empty()){
			this->data_[i].release();
		}
		if(!this->targets_[i].empty()){
			this->targets_[i].release();
		}
	}
	if(!this->targetAnno_.empty()){
		this->targetAnno_.clear();
	}
	if(!this->entireNext_.empty()){
		this->entireNext_.release();
	}
	this->templates_.clear();
	this->dataMotionVectors_.clear();
	this->existing_.clear();
	this->classesRange_.clear();
}
//==============================================================================
/** Initialize the inverse value of the color space used in feature extraction.
 */
void PeopleDetector::initInvColoprSp(){
	switch(this->colorspaceCode_){
		case(CV_BGR2XYZ):
			this->invColorspaceCode_ = CV_XYZ2BGR;
			break;
		case(CV_BGR2YCrCb):
			this->invColorspaceCode_ = CV_YCrCb2BGR;
			break;
		case(CV_BGR2HSV):
			this->invColorspaceCode_ = CV_HSV2BGR;
			break;
		case(CV_BGR2HLS):
			this->invColorspaceCode_ = CV_HLS2BGR;
			break;
		case(CV_BGR2Lab):
			this->invColorspaceCode_ = CV_Lab2BGR;
			break;
		case(CV_BGR2Luv):
			this->invColorspaceCode_ = CV_Luv2BGR;
			break;
	}
}
//==============================================================================
/** Checks the image name (used to find the corresponding labels for each image).
 */
struct compareImg{
	public:
		std::string imgName_;
		compareImg(std::string image){
			std::deque<std::string> parts = Helpers::splitLine\
				(const_cast<char*>(image.c_str()),'/');
			this->imgName_ = parts[parts.size()-1];
		}
		virtual ~compareImg(){};
		bool operator()(AnnotationsHandle::FULL_ANNOTATIONS anno)const{
			return (anno.imgFile_ == this->imgName_);
		}
		compareImg(const compareImg &comp){
			this->imgName_ = comp.imgName_;
		}
		compareImg& operator=(const compareImg &comp){
			if(this == &comp) return *this;
			this->imgName_ = comp.imgName_;
			return *this;
		}
};
//==============================================================================
/** Initializes the parameters of the tracker.
 */
void PeopleDetector::init(const std::string &dataFolder,\
const std::string &theAnnotationsFile,\
const std::deque<FeatureExtractor::FEATURE> &feat,bool test,\
bool readFromFolder){
	this->isTest_ = test;
	this->dataMotionVectors_.clear();
	this->existing_.clear();
	this->classesRange_.clear();
	this->initProducer(readFromFolder,const_cast<char*>(dataFolder.c_str()));
	if(this->borderedIpl_){
		this->borderedIpl_.reset();
	}
	if(!this->entireNext_.empty()){
		this->entireNext_.release();
	}
	for(std::size_t i=0;i<this->data_.size();++i){
		if(!this->targets_[i].empty()){
			this->targets_[i].release();
		}
		if(!this->data_[i].empty()){
			this->data_[i].release();
		}
	}
	if(!this->targetAnno_.empty()){
		this->targetAnno_.clear();
	}
	// LOAD THE DESIRED ANNOTATIONS FROM THE FILE
	if(!theAnnotationsFile.empty() && this->targetAnno_.empty()){
		AnnotationsHandle::loadAnnotations(const_cast<char*>\
			(theAnnotationsFile.c_str()),this->targetAnno_);
	}
	if(this->onlyExtract_){
		assert(!FeatureExtractor::isFeatureIn(feat,FeatureExtractor::HOG) &&\
			!FeatureExtractor::isFeatureIn(feat,FeatureExtractor::TEMPL_MATCHES) &&\
			!FeatureExtractor::isFeatureIn(feat,FeatureExtractor::RAW_PIXELS) &&\
			!FeatureExtractor::isFeatureIn(feat,FeatureExtractor::SKIN_BINS));
	}
	this->extractor_->init(feat,this->datasetPath_+"features/",this->colorspaceCode_,\
		this->invColorspaceCode_,this->featurePart_);
	if(FeatureExtractor::isFeatureIn(feat,FeatureExtractor::SIFT_DICT)){
		this->tracking_ = false;
	}
	this->templates_.clear();

	for(unsigned i=0;i<3;++i){
		this->data_.push_back(cv::Mat());
		this->targets_.push_back(cv::Mat());
		this->dataMotionVectors_.push_back(std::deque<float>());
	}
}
//==============================================================================
/** Get template extremities (if needed,considering some borders --
 * relative to the ROI).
 */
void PeopleDetector::templateExtremes(const std::vector<cv::Point2f> &templ,\
std::deque<float> &extremes,int minX,int minY){
	if(extremes.empty()){
		extremes = std::deque<float>(4,0.0f);
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
float PeopleDetector::getDistToTemplate(const int pixelX,const int pixelY,\
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
void PeopleDetector::templateWindow(const cv::Size &imgSize,int &minX,int &maxX,\
int &minY,int &maxY,const FeatureExtractor::templ &aTempl){
	// TRY TO ADD BORDERS TO MAKE IT 100
/*
	minX = aTempl.extremes_[0];
	maxX = aTempl.extremes_[1];
	minY = aTempl.extremes_[2];
	maxY = aTempl.extremes_[3];
	int diffX = (this->border_ - (maxX-minX))/2;
	int diffY = (this->border_ - (maxY-minY))/2;

	// WILL ALWAYS FIT BECAUSE WE ADDED THE BORDER
	minY = std::max(minY-diffY,0);
	maxY = std::min(maxY+diffY,imgSize.height);
	minX = std::max(minX-diffX,0);
	maxX = std::min(maxX+diffX,imgSize.width);

	// MAKE SUTE WE ARE NOT MISSING A PIXEL OR SO
	if(maxX-minX!=this->border_){
		int diffX2 = this->border_-(maxX-minX);
		if(minX>diffX2){
			minX -= diffX2;
		}else if(maxX<imgSize.width-diffX2){
			maxX += diffX2;
		}
	}
	if(maxY-minY!=this->border_){
		int diffY2 = this->border_-(maxY-minY);
		if(minY>diffY2){
			minY -= diffY2;
		}else if(maxY<imgSize.height-diffY2){
			maxY += diffY2;
		}
	}
*/
	minX = aTempl.center_.x - this->border_/2;
	maxX = aTempl.center_.x + this->border_/2;
	minY = aTempl.center_.y - this->border_/2;
	maxY = aTempl.center_.y + this->border_/2;
}
//==============================================================================
/** Assigns pixels to templates based on proximity.
 */
void PeopleDetector::pixels2Templates(int maxX,int minX,int maxY,int minY,\
int k,const cv::Mat &thresh,float tmplHeight,cv::Mat &colorRoi){
	// LOOP OVER THE AREA OF OUR TEMPLATE AND THERESHOLD ONLY THOSE PIXELS
	float tmpSize = std::max(this->templates_[k].extremes_[1]-\
		this->templates_[k].extremes_[0],this->templates_[k].extremes_[3]-\
		this->templates_[k].extremes_[2])/8;

	int x1Border = std::max(minX,static_cast<int>(this->templates_[k].extremes_[0]-\
		tmpSize));
	int x2Border = std::min(maxX,static_cast<int>(this->templates_[k].extremes_[1]+\
		tmpSize));
	int y1Border = std::max(minY,static_cast<int>(this->templates_[k].extremes_[2]-\
		tmpSize));
	int y2Border = std::min(maxY,static_cast<int>(this->templates_[k].extremes_[3]+\
		tmpSize));

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
	if(this->templates_.size()>1){
		for(int x=x1Border-minX;x<x2Border-minX;++x){
			for(int y=y1Border-minY;y<y2Border-minY;++y){
				if(static_cast<int>(threshPart.at<uchar>(y,x))==0){continue;}
				if(!FeatureExtractor::isInTemplate((x+minX),(y+minY),\
				this->templates_[k].points_)){
					cv::Point2f mid((this->templates_[k].center_.x+\
						this->templates_[k].head_.x)/2.0,(this->templates_[k].center_.y+\
						this->templates_[k].head_.y)/2.0);
					float minDist = Helpers::dist(cv::Point2f(x+minX,y+minY),mid);
					int label     = k;
					for(int l=0;l<this->templates_.size();++l){
						if(l==k){continue;}
						float tempDist = Helpers::dist(this->templates_[k].center_,\
							this->templates_[l].center_);
						if(tempDist >= tmplHeight){continue;}

						// IF IT IS IN ANOTHER TEMPLATE THEN IGNORE THE PIXEL
						if(FeatureExtractor::isInTemplate((x+minX),(y+minY),\
						this->templates_[l].points_)){
							minDist = 0;label = l;
							break;

						// ELSE COMPUTE THE DISTANCE FROM THE PIXEL TO THE TEMPLATE
						}else{
							cv::Point2f middle((this->templates_[l].center_.x+\
								this->templates_[l].head_.x)/2.0,\
								(this->templates_[l].center_.y+\
								this->templates_[l].head_.y)/2.0);
							float ptDist = Helpers::dist(cv::Point2f(x+minX,y+minY),\
								middle);
							if(minDist>ptDist){
								minDist = ptDist;label = l;
							}
						}
					}

					// IF THE PIXEL HAS A DIFFERENT LABEL THEN THE CURR TEMPL
					if(label != k || minDist >= tmplHeight/1.5){
						colorRoi.at<cv::Vec3b>(y,x) = cv::Vec3b(127,127,127);
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
void PeopleDetector::add2Templates(){
	this->templates_.clear();
	for(unsigned i=0;i<this->existing_.size();++i){
		FeatureExtractor::templ aTempl(this->existing_[i].location_);
		Helpers::genTemplate2(aTempl.center_,Helpers::persHeight(),Helpers::camHeight(),\
			aTempl.points_,this->border_/2,this->border_/2);
		aTempl.center_ = cv::Point2f((aTempl.points_[0].x+aTempl.points_[2].x)/2,\
			(aTempl.points_[0].y+aTempl.points_[2].y)/2);
		aTempl.head_ = cv::Point2f((aTempl.points_[12].x+aTempl.points_[14].x)/2,\
			(aTempl.points_[12].y+aTempl.points_[14].y)/2);
		this->templateExtremes(aTempl.points_,aTempl.extremes_);
		this->templates_.push_back(aTempl);
		aTempl.extremes_.clear();
		aTempl.points_.clear();
	}
}
//==============================================================================
/** Get the foreground pixels corresponding to each person.
 */
void PeopleDetector::allForegroundPixels(std::deque<FeatureExtractor::people>\
&allPeople,const IplImage *bg,float threshold){
	// INITIALIZING STUFF
	cv::Mat thsh,thrsh;
	if(bg){
		thsh  = cv::Mat(bg);
		cv::cvtColor(thsh,thsh,TO_IMG_FMT);
		cv::cvtColor(thsh,thrsh,CV_BGR2GRAY);
		cv::threshold(thrsh,thrsh,threshold,255,cv::THRESH_BINARY);
	}
	cv::Mat foregr(this->borderedIpl_.get());
	// FOR EACH EXISTING TEMPLATE LOOK ON AN AREA OF 100 PIXELS AROUND IT
	for(unsigned k=0;k<this->existing_.size();++k){
		allPeople[k].absoluteLoc_ = this->existing_[k].location_;
		float tmplHeight = Helpers::dist(this->templates_[k].head_,this->templates_[k].center_);
		float tmplArea   = tmplHeight*Helpers::dist(this->templates_[k].points_[0],\
			this->templates_[k].points_[1]);

		// GET THE 100X100 WINDOW ON THE TEMPLATE
		int minY=foregr.rows,maxY=0,minX=foregr.cols,maxX=0;
		this->templateWindow(cv::Size(foregr.cols,foregr.rows),minX,maxX,\
			minY,maxY,this->templates_[k]);
		int awidth  = maxX-minX;
		int aheight = maxY-minY;
		cv::Mat colorRoi = cv::Mat(foregr.clone(),cv::Rect(cv::Point2f(minX,minY),\
			cv::Size(awidth,aheight)));

		//IF THERE IS NO BACKGROUND THEN JUST COPY THE ROI
		if(!bg){
			colorRoi.copyTo(allPeople[k].pixels_);
		}else{
			// FOR MULTIPLE DISCONNECTED BLOBS KEEP THE CLOSEST TO CENTER
			this->pixels2Templates(maxX,minX,maxY,minY,k,thrsh,tmplHeight,colorRoi);
			if(this->colorspaceCode_!=-1){
				cv::cvtColor(colorRoi,colorRoi,this->invColorspaceCode_);
			}
			cv::Mat thrshRoi;
			cv::cvtColor(colorRoi,thrshRoi,CV_BGR2GRAY);
			cv::Point2f templateMid = cv::Point2f((this->templates_[k].head_.x+\
				this->templates_[k].center_.x)/2 - minX,(this->templates_[k].head_.y+\
				this->templates_[k].center_.y)/2 - minY);
			this->keepLargestBlob(thrshRoi,templateMid,tmplArea);

			allPeople[k].pixels_ = cv::Mat::zeros(colorRoi.size(),colorRoi.type());
			allPeople[k].pixels_ = cv::Scalar(127,127,127);
			colorRoi.copyTo(allPeople[k].pixels_,thrshRoi);
			thrshRoi.copyTo(allPeople[k].thresh_);
			thrshRoi.release();
		}
		// SAVE IT IN THE STRUCTURE OF FOREGOUND IMAGES
		allPeople[k].relativeLoc_ = cv::Point2f(this->existing_[k].location_.x-(minX),\
			this->existing_[k].location_.y-(minY));
		allPeople[k].borders_.assign(4,0);
		allPeople[k].borders_[0] = minX;
		allPeople[k].borders_[1] = maxX;
		allPeople[k].borders_[2] = minY;
		allPeople[k].borders_[3] = maxY;
		if(this->plot_){
			cv::imshow("people",allPeople[k].pixels_);
			if(!allPeople[k].thresh_.empty()){
				cv::imshow("threshold",allPeople[k].thresh_);
			}
			cv::waitKey(5);
		}
		colorRoi.release();
	}
	thrsh.release();
}
//==============================================================================
/** Keeps only the largest blob from the thresholded image.
 */
void PeopleDetector::keepLargestBlob(cv::Mat &thresh,const cv::Point2f &center,\
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
		float ptDist = Helpers::dist(center,cv::Point2f((maxX+minX)/2,(maxY+minY)/2));
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
void PeopleDetector::fixLocationsTracksBorderes(const std::deque<unsigned> &exi,\
bool flip){
	//1) FIRST GET THE CORRECT LOCATIONS ON THE ORIGINAL IMAGE
	if(this->useGroundTruth_){
		this->readLocations(flip);
	}else{
		if(this->targetAnno_.empty()){
			std::cerr<<"Annotations not loaded! "<<std::endl;
			std::abort();
		}
		this->fixLabels(exi,flip);
	}

	//2) THEN CREATE THE BORDERED IMAGE
	cvCopyMakeBorder(this->current_->img_.get(),this->borderedIpl_.get(),cv::Point\
		(this->border_/2,this->border_/2),IPL_BORDER_REPLICATE,cvScalarAll(0));

	//3) CREATE TEMPLATES
	this->add2Templates();

	//4) FIX THE LOCATIONS TO CORRESPOND TO THE BORDERED IMAGE:
	for(std::size_t i=0;i<this->existing_.size();++i){
		this->existing_[i].location_.x += this->border_/2;
		this->existing_[i].location_.y += this->border_/2;
	}

	//5) UPDATE THE TRACKS (ON THE MODIFIED DETECTED LOCATIONS)
	this->updateTracks(this->current_->index_,exi,this->current_->img_->width);

	if(this->plot_){
		IplImage *tmpSrc;
		tmpSrc = cvCloneImage(this->current_->img_.get());
		for(unsigned i=0;i<this->tracks_.size();++i){
			if(this->tracks_[i].imgID_.size() > MIN_TRACKLEN){
				if(this->current_->index_-this->tracks_[i].imgID_.back()<2){
					this->plotTrack(tmpSrc,this->tracks_[i],i,(unsigned)1);
				}
			}
		}
		cvShowImage("tracks",tmpSrc);
		cvWaitKey(5);
		cvReleaseImage(&tmpSrc);
	}
}
//==============================================================================
/** Creates on data row in the final data matrix by getting the feature
 * descriptors.
 */
void PeopleDetector::extractDataRow(const IplImage *oldBg,bool flip,\
const std::deque<unsigned> &exi,float threshVal){
	this->fixLocationsTracksBorderes(exi,flip);

	// ADD A BORDER AROUND THE BACKGROUND TO HAVE THE TEMPLATES CENTERED
	IplImage *bg = NULL;
	if(oldBg){
		bg = cvCreateImage(cvSize(oldBg->width+this->border_,oldBg->height+\
			this->border_),oldBg->depth,oldBg->nChannels);
		cvCopyMakeBorder(oldBg,bg,cv::Point(this->border_/2,this->border_/2),\
			IPL_BORDER_REPLICATE,cvScalarAll(0));
	}

	// PROCESS THE REST OF THE IMAGES
	cv::Mat image(this->borderedIpl_.get());
	if(this->colorspaceCode_!=-1){
		cv::cvtColor(image,image,this->colorspaceCode_);
	}
	if(this->tracking_ && this->current_->index_+1<this->producer_->filelist().size()){
		this->entireNext_ = cv::imread(this->producer_->filelist()\
							[this->current_->index_+1].c_str());
		if(this->colorspaceCode_!=-1){
			cv::cvtColor(this->entireNext_,this->entireNext_,this->colorspaceCode_);
		}
	}

	// REDUCE THE IMAGE TO ONLY THE INTERESTING AREA
	std::deque<FeatureExtractor::people> allPeople(this->existing_.size(),\
		FeatureExtractor::people());
	this->allForegroundPixels(allPeople,bg,threshVal);

	// GET ONLY THE IMAGE NAME OUT THE CURRENT IMAGE'S NAME
	unsigned pos1       = (this->current_->sourceName_).find_last_of("/\\");
	std::string imgName = (this->current_->sourceName_).substr(pos1+1);
	unsigned pos2       = imgName.find_last_of(".");
	imgName             = imgName.substr(0,pos2);

	// FOR EACH LOCATION IN THE IMAGE EXTRACT FEATURES,FILTER THEM AND RESHAPE
	std::deque<std::string> names;
	names.push_back("CLOSE");names.push_back("MEDIUM");	names.push_back("FAR");

	for(std::size_t i=0;i<this->existing_.size();++i){
		// DEFINE THE IMAGE ROI OF THE SAME SIZE AS THE FOREGROUND
		cv::Rect roi(allPeople[i].borders_[0],allPeople[i].borders_[2],\
			allPeople[i].borders_[1]-allPeople[i].borders_[0],\
			allPeople[i].borders_[3]-allPeople[i].borders_[2]);
		// COMPUTE THE ROTATION ANGLE ONCE AND FOR ALL
		float rotAngle = this->rotationAngle(this->templates_[i].head_,\
						this->templates_[i].center_);
		// ROTATE THE FOREGROUND PIXELS,THREHSOLD AND THE TEMPLATE
		cv::Point2f rotBorders,absRotCenter;
		std::vector<cv::Point2f> keys;
		if(!allPeople[i].thresh_.empty()){
			this->extractor_->rotate2Zero(rotAngle,FeatureExtractor::MATRIX,roi,\
				absRotCenter,rotBorders,keys,allPeople[i].thresh_);
		}
		this->extractor_->rotate2Zero(rotAngle,FeatureExtractor::MATRIX,roi,\
			absRotCenter,rotBorders,keys,allPeople[i].pixels_);
		cv::Mat aDummy;
		absRotCenter = cv::Point2f(absRotCenter.x+allPeople[i].borders_[0],\
			absRotCenter.y+allPeople[i].borders_[2]);
		this->extractor_->rotate2Zero(rotAngle,FeatureExtractor::TEMPLATE,roi,\
			absRotCenter,rotBorders,this->templates_[i].points_,aDummy);

		// RESET THE POSITION OF THE HEAD IN THE ROATED TEMPLATE
		this->templates_[i].center_ = cv::Point2f((this->templates_[i].points_[0].x+\
			this->templates_[i].points_[2].x)/2,(this->templates_[i].points_[0].y+\
			this->templates_[i].points_[2].y)/2);
		this->templates_[i].head_ = cv::Point2f((this->templates_[i].points_[12].x+\
			this->templates_[i].points_[14].x)/2,(this->templates_[i].points_[12].y+\
			this->templates_[i].points_[14].y)/2);
		this->templateExtremes(this->templates_[i].points_,this->templates_[i].extremes_);

		// IF THE PART TO BE CONSIDERED IS ONLY FEET OR ONLY HEAD
		if(this->featurePart_==FeatureExtractor::HEAD){
			this->extractHeadArea(i,allPeople[i]);
		}else{
			this->templatePart(i,allPeople[i]);
		}

		// ANF FINALLY EXTRACT THE DATA ROW
		this->extractor_->setImageClass(this->existing_[i].groupNo_);
		cv::Mat dataRow = this->extractor_->getDataRow(this->borderedIpl_->height,\
			this->templates_[i],roi,allPeople[i],imgName,absRotCenter,rotBorders,\
			rotAngle,flip,keys);
		dataRow.at<float>(0,dataRow.cols-2) = this->distanceWRTcamera\
			(this->templates_[i].center_);
		// CHECK THE DIRECTION OF THE MOVEMENT
		bool moved;
		float direction = this->motionVector(this->templates_[i].head_,\
			this->templates_[i].center_,flip,moved);
		if(!moved){direction = -1.0;}
		this->dataMotionVectors_[this->existing_[i].groupNo_].push_back(direction);

		if(this->tracking_ && !this->entireNext_.empty()){
			cv::Mat nextImg(this->entireNext_,roi);
			this->extractor_->rotate2Zero(rotAngle,FeatureExtractor::MATRIX,roi,\
				absRotCenter,rotBorders,keys,nextImg);
			dataRow.at<float>(0,dataRow.cols-1) = \
				this->opticalFlow(allPeople[i].pixels_,nextImg,keys,\
				this->templates_[i].head_,this->templates_[i].center_,false,flip);
			nextImg.release();
		}
		dataRow.convertTo(dataRow,CV_32FC1);

		// STORE THE EXTRACTED ROW INTO THE DATA MATRIX
		if(this->data_[this->existing_[i].groupNo_].empty()){
			dataRow.copyTo(this->data_[this->existing_[i].groupNo_]);
		}else{
			this->data_[this->existing_[i].groupNo_].push_back(dataRow);
		}

		if(this->isTest_){
			{
				boost::mutex::scoped_lock lock(PeopleDetector::dataMutex_);
				std::cout<<"Produce another test row..."<<std::endl;
				while(this->dataInfo_.size()>1){sleep(.1);}
				PeopleDetector::DataRow aRow(this->existing_[i].location_,\
					this->existing_[i].groupNo_,this->current_->sourceName_,\
					dataRow,this->targets_[this->existing_[i].groupNo_].row(i));
				this->dataInfo_.push_back(aRow);
			}
		}
		dataRow.release();

		// OUTPUT THE LABEL FOR THE CURRENT IMAGE
		float label = std::atan2(this->targets_[this->existing_[i].groupNo_].\
			at<float>(this->data_[this->existing_[i].groupNo_].rows-1,0),\
			this->targets_[this->existing_[i].groupNo_].\
			at<float>(this->data_[this->existing_[i].groupNo_].rows-1,1));
		Auxiliary::angle0to360(label);
		std::cout<<names[this->existing_[i].groupNo_]<<":"<<\
			this->templates_[i].center_<<">>>"<<"LONGITUDE>>>"<<(label*180.0/M_PI)<<\
			std::endl<<"-----------------------------------------------"<<std::endl;
	}
	if(bg){
		cvReleaseImage(&bg);
	}
}
//==============================================================================
/** Draws the target orientation and the predicted orientation on the image.
 */
void PeopleDetector::drawPredictions(const cv::Point2f &pred,\
std::tr1::shared_ptr<PeopleDetector::DataRow> dataRow){
	cv::Mat load = cv::imread(dataRow->imgName_);
	float pAngle = std::atan2(pred.y,pred.x);
	float tAngle = std::atan2(dataRow->testTarg_.at<float>(0,0),\
		dataRow->testTarg_.at<float>(0,1));
	std::vector<cv::Point2f> templ;
	dataRow->location_.x -= this->border_/2.0;
	dataRow->location_.y -= this->border_/2.0;
	Helpers::plotTemplate2(load,dataRow->location_,cv::Scalar(255,255,255),templ);
	cv::Point2f headLocation = cv::Point2f((templ[12].x+templ[14].x)/2,\
		(templ[12].y+templ[14].y)/2);
	pAngle = this->unfixAngle(headLocation,dataRow->location_,pAngle);
	tAngle = this->unfixAngle(headLocation,dataRow->location_,tAngle);

	cv::Point2f center((headLocation.x + dataRow->location_.x)/2.0,\
		(headLocation.y + dataRow->location_.y)/2.0);
	unsigned predA = static_cast<unsigned>((pAngle*180.0)/M_PI);
	unsigned targA = static_cast<unsigned>((tAngle*180.0)/M_PI);
	cv::Mat tmp = AnnotationsHandle::drawOrientation(center,predA,\
		load,cv::Scalar(0,255,0));
	cv::Mat fin = AnnotationsHandle::drawOrientation(center,targA,\
		tmp,cv::Scalar(0,0,255));
	cv::imshow("orientation_image",fin);
	cv::waitKey(10);
	load.release();
	fin.release();
	tmp.release();
}
//==============================================================================
/** Returns the last element in the data vector.
 */
std::tr1::shared_ptr<PeopleDetector::DataRow> PeopleDetector::popDataRow(){
	if(!this->dataInfo_.empty()){
		PeopleDetector::DataRow aRow = this->dataInfo_.back();
		std::tr1::shared_ptr<PeopleDetector::DataRow> ptRow = std::tr1::shared_ptr\
			<PeopleDetector::DataRow>(new PeopleDetector::DataRow(aRow));
		this->dataInfo_.pop_back();
		return ptRow;
	}
	return std::tr1::shared_ptr<PeopleDetector::DataRow>(static_cast\
		<PeopleDetector::DataRow*>(NULL));
}
//==============================================================================
/** Compute the dominant direction of the SIFT or SURF features.
 */
float PeopleDetector::opticalFlow(cv::Mat &currentImg,cv::Mat &nextImg,\
const std::vector<cv::Point2f> &keyPts,const cv::Point2f &head,\
const cv::Point2f &center,bool maxOrAvg,bool flip){
	// GET THE OPTICAL FLOW MATRIX FROM THE FEATURES
	float direction = -1;
	cv::Mat currGray,nextGray;
	if(this->colorspaceCode_!=-1){
		cv::cvtColor(currentImg,currentImg,this->invColorspaceCode_);
		cv::cvtColor(nextImg,nextImg,this->invColorspaceCode_);
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
	float flowX = 0.0f,flowY = 0.0f,resFlowX = 0.0f,resFlowY = 0.0f,magni = 0.0f;
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
		if(this->plot_){
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

	if(this->plot_){
		cv::line(currentImg,cv::Point2f(flowX,flowY),cv::Point2f(resFlowX,resFlowY),\
			cv::Scalar(255,0,0),1,8,0);
		cv::circle(currentImg,cv::Point2f(resFlowX,resFlowY),2,cv::Scalar(0,200,0),\
			1,8,0);
		cv::imshow("optical",currentImg);
		cv::waitKey(5);
	}

	// DIRECTION OF THE AVERAGE FLOW
	direction = std::atan2(resFlowY - flowY,resFlowX - flowX);
	std::cout<<"Flow direction: "<<direction*180/M_PI<<std::endl;
	currGray.release();
	nextGray.release();
	direction = this->fixAngle(head,center,direction,flip);
	return direction;
}
//==============================================================================
/** Checks to see if an annotation can be assigned to a detection.
 */
bool PeopleDetector::canBeAssigned(unsigned l,std::deque<float> &minDistances,\
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
float PeopleDetector::fixAngle(const cv::Point2f &headLocation,\
const cv::Point2f &feetLocation,float angle,bool flip){
//	return angle;
	float camAngle = std::atan2(headLocation.y-feetLocation.y,\
		headLocation.x-feetLocation.x);
	camAngle      += M_PI/2.0;
	float newAngle = angle+camAngle;
	Auxiliary::angle0to360(newAngle);

	float finalAngle = newAngle;
	if(flip){
		bool quadrant13 = ((finalAngle>=M_PI && finalAngle<M_PI*3.0/2.0) ||\
			(finalAngle>=0 && finalAngle<M_PI/2.0));
		int div90   = static_cast<int>(newAngle/(M_PI/2.0));
		float mod90 = newAngle-div90*(M_PI/2.0);
		float gamma = (quadrant13)?mod90:(M_PI/2.0-mod90);
		finalAngle  = (quadrant13)?(finalAngle+(M_PI-2.0*gamma)):\
			(finalAngle-(M_PI-2.0*gamma));
		Auxiliary::angle0to360(finalAngle);
	}
	std::cout<<"Angle: "<<(newAngle*180/M_PI)<<std::endl;
	return finalAngle;
}
//==============================================================================
/** Un-does the rotation with respect to the camera.
 */
float PeopleDetector::unfixAngle(const cv::Point2f &headLocation,\
const cv::Point2f &feetLocation,float angle){
	float camAngle = std::atan2(headLocation.y-feetLocation.y,\
		headLocation.x-feetLocation.x);
	camAngle      += M_PI/2.0;
	float newAngle = angle-camAngle;
	Auxiliary::angle0to360(newAngle);
	std::cout<<"Angle: "<<(newAngle*180/M_PI)<<std::endl;
	return newAngle;
}
//==============================================================================
/** For each row added in the data matrix (each person detected for which we
 * have extracted some features) find the corresponding label.
 */
void PeopleDetector::fixLabels(const std::deque<unsigned> &exi,bool flip){
	this->existing_.clear();
	// FIND	THE INDEX FOR THE CURRENT IMAGE
	std::deque<AnnotationsHandle::FULL_ANNOTATIONS>::iterator index = \
		std::find_if (this->targetAnno_.begin(),this->targetAnno_.end(),\
		compareImg(this->current_->sourceName_));
	if(index == this->targetAnno_.end()){
		std::cerr<<"The image: "<<this->current_->sourceName_<<\
			" was not annotated"<<std::endl;
		exit(1);
	}

	//GET THE VALID DETECTED LOCATIONS
	std::vector<cv::Point2f> points;
	for(std::size_t i=0;i<exi.size();++i){
		cv::Point2f center = this->cvPoint(exi[i],this->current_->img_->width);
		points.push_back(center);
	}

	// LOOP OVER ALL ANNOTATIONS FOR THE CURRENT IMAGE AND FIND THE CLOSEST ONES
	std::deque<int> assignments((*index).annos_.size(),-1);
	std::deque<float> minDistances((*index).annos_.size(),(float)INFINITY);

	std::vector<cv::Point2f> allTrueHeads((*index).annos_.size(),cv::Point2f());
	std::deque<bool> allTrueInside((*index).annos_.size(),true);
	std::deque<float> allTrueWidth((*index).annos_.size(),0.0f);
	unsigned canAssign = 1;
	while(canAssign){
		canAssign = 0;
		for(std::size_t l=0;l<(*index).annos_.size();++l){
			// IF WE ARE LOOPING THE FIRST TIME THEN STORE THE TEMPLATE POINTS
			if(!allTrueWidth[l]){
				std::vector<cv::Point2f> aTempl;
				allTrueInside[l] = Helpers::genTemplate2((*index).annos_[l].location_,\
					Helpers::persHeight(),Helpers::camHeight(),aTempl);
				allTrueHeads[l] = cv::Point2f((aTempl[12].x+aTempl[14].x)/2,\
					(aTempl[12].y+aTempl[14].y)/2);
				allTrueWidth[l] = dist(aTempl[0],aTempl[1]);
			}

			// EACH ANNOTATION NEEDS TO BE ASSIGNED TO THE CLOSEST DETECTION
			float distance     = (float)INFINITY;
			unsigned annoIndex = -1;
			for(std::size_t k=0;k<points.size();++k){
				float dstnc = dist((*index).annos_[l].location_,points[k]);
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
			cv::Point2f feet = (*index).annos_[position].location_;
			cv::Point2f head = allTrueHeads[position];

			// SAVE THE TARGET LABEL
			cv::Mat tmp = cv::Mat::zeros(1,4,CV_32FC1);
			// READ THE TARGET ANGLE FOR LONGITUDINAL ANGLE
			float angle = static_cast<float>((*index).annos_[position].\
							poses_[AnnotationsHandle::LONGITUDE]);
			angle = angle*M_PI/180.0;
			angle = this->fixAngle(head,feet,angle,flip);
			tmp.at<float>(0,0) = std::sin(angle);
			tmp.at<float>(0,1) = std::cos(angle);

			// READ THE TARGET ANGLE FOR LATITUDINAL ANGLE
			angle = static_cast<float>((*index).annos_[position].\
					poses_[AnnotationsHandle::LATITUDE]);
			angle = angle*M_PI/180.0;
			tmp.at<float>(0,2) = std::sin(angle);
			tmp.at<float>(0,3) = std::cos(angle);
			PeopleDetector::CLASSES groupNo=this->findImageClass(feet,head);
			PeopleDetector::Existing tmpExisting(feet,static_cast<unsigned>(groupNo));
			this->existing_.push_back(tmpExisting);
			if(this->targets_[groupNo].empty()){
				tmp.copyTo(this->targets_[groupNo]);
			}else{
				this->targets_[groupNo].push_back(tmp);
			}
			tmp.release();
		}
	}
}
//==============================================================================
/** Overwrites the \c doFindPeople function from the \c Tracker class to make it
 * work with the feature extraction.
 */
bool PeopleDetector::doFindPerson(unsigned imgNum,IplImage *src,\
const vnl_vector<float> &imgVec,vnl_vector<float> &bgVec,\
const float logBGProb,const vnl_vector<float> &logSumPixelBGProb){
	std::cout<<this->current_->index_<<") Image... "<<this->current_->sourceName_<<std::endl;

	//1) START THE TIMER & INITIALIZE THE PROBABLITIES,THE VECTOR OF POSITIONS
	stic();	std::deque<float> marginal;
	float lNone = logBGProb + this->logNumPrior_[0],lSum = -INFINITY;
	std::deque<unsigned> exi;
	std::deque< vnl_vector<float> > logPosProb;
	std::deque<Helpers::scanline_t> mask;
	marginal.push_back(lNone);
	logPosProb.push_back(vnl_vector<float>());

	//2) SCAN FOR POSSIBLE LOCATION OF PEOPLE GIVEN THE EXISTING ONES
	this->scanRest(exi,mask,Helpers::scanres(),logSumPixelBGProb,logPosProb,\
		marginal,lNone);

	//3) UPDATE THE PROBABILITIES
	for(unsigned i=0;i!=marginal.size();++i){
		lSum = log_sum_exp(lSum,marginal[i]);
	}

	//4) UPDATE THE MOST LIKELY NUMBER OF PEOPLE AND THE MARGINAL PROBABILITY
	unsigned mlnp = 0;// most likely number of people
	float mlprob  = -INFINITY;
	for(unsigned i=0;i!=marginal.size();++i) {
		if (marginal[i] > mlprob) {
			mlnp   = i;
			mlprob = marginal[i];
		}
	}

	//5) DILATE A BIT THE BACKGROUND SO THE BACKGROUND NOISE GOES NICELY
	IplImage *bg = Helpers::vec2img((imgVec-bgVec).apply(fabs));
	cvSmooth(bg,bg,CV_GAUSSIAN,31,31);
	for(unsigned l=0;l<10;++l){
		cvDilate(bg,bg,NULL,3);
		cvErode(bg,bg,NULL,3);
	}

	//7) SHOW THE FOREGROUND POSSIBLE LOCATIONS AND PLOT THE TEMPLATES
	cerr<<"no. of detected people: "<<exi.size()<<endl;
	if(this->plot_){
		IplImage *tmpBg,*tmpSrc;
		tmpBg  = cvCloneImage(bg);
		tmpSrc = cvCloneImage(src);
		// PLOT HULL
		this->plotHull(tmpSrc,this->priorHull_);

		// PLOT DETECTIONS
		for(unsigned i=0;i!=exi.size();++i){
			cv::Point2f pt = this->cvPoint(exi[i],this->current_->img_->width);
			std::vector<cv::Point2f> points;
			Helpers::plotTemplate2(tmpSrc,pt,Helpers::persHeight(),Helpers::camHeight(),\
				CV_RGB(255,255,255),points);
			//Helpers::plotScanLines(tmpBg,mask,CV_RGB(0,255,0),0.3);
		}
		cvShowImage("bg",tmpBg);
		cvShowImage("image",tmpSrc);
		cvWaitKey(5);
		cvReleaseImage(&tmpBg);
		cvReleaseImage(&tmpSrc);
	}

	//10) EXTRACT FEATURES FOR THE CURRENTLY DETECTED LOCATIONS
	cout<<"Number of templates: "<<exi.size()<<endl;
	this->extractDataRow(bg,false,exi);
	if(this->flip_){this->extractDataRow(bg,true,exi);}
	cout<<"Number of templates afterwards: "<<this->existing_.size()<<endl;

	//11) WHILE NOT q WAS PRESSED PROCESS THE NEXT IMAGES
	cvReleaseImage(&bg);
	return this->imageProcessingMenu();
}
//==============================================================================
/** Simple "menu" for skipping to the next image or quitting the processing.
 */
bool PeopleDetector::imageProcessingMenu(){
	if(this->plot_){
		cout<<" To quite press 'q'.\n To pause press 'p'.\n To skip 10 "<<\
			"images press 's'.\n To skip 100 images press 'f'.\n To go back 10 "<<\
			"images  press 'b'.\n To go back 100 images press 'r'.\n";
		int k = (char)cvWaitKey(this->waitTime_);
		cout<<"Press 'n' to go to the next frame"<<endl;
		while((char)k != 'n' && (char)k != 'q'){
			switch(k){
				case 'p':
					this->waitTime_ = 20-this->waitTime_;
					break;
				case 's':
					this->producer_->forward(10);
					break;
				case 'f':
					this->producer_->forward(100);
					break;
				case 'b':
					this->producer_->backward(10);
					break;
				case 'r':
					this->producer_->backward(100);
					break;
				default:
					break;
			}
			k = (char)cvWaitKey(this->waitTime_);
		}
		if(k == 'q'){return false;}
	}
	return true;
}
//==============================================================================
/** If only a part needs to be used to extract the features then the threshold
 * and the template need to be changed.
 */
void PeopleDetector::templatePart(int k,FeatureExtractor::people &person){
	// CHANGE THRESHOLD
	int minX,maxX,minY,maxY;
	if(!person.thresh_.empty()){
		this->extractor_->getThresholdBorderes(minX,maxX,minY,maxY,person.thresh_);
		for(unsigned i=0;i<this->templates_[k].points_.size();++i){
			if(this->templates_[k].points_[i].x-person.borders_[0]>maxX){
				this->templates_[k].points_[i].x = maxX+person.borders_[0];
			}
			if(this->templates_[k].points_[i].x-person.borders_[0]<minX){
				this->templates_[k].points_[i].x = minX+person.borders_[0];
			}
			if(this->templates_[k].points_[i].y-person.borders_[2]>maxY){
				this->templates_[k].points_[i].y = maxY+person.borders_[2];
			}
			if(this->templates_[k].points_[i].y-person.borders_[2]<minY){
				this->templates_[k].points_[i].y = minY+person.borders_[2];
			}
		}
	}else{
		minX = this->templates_[k].extremes_[0]-person.borders_[0];
		maxX = this->templates_[k].extremes_[1]-person.borders_[0];
		minY = this->templates_[k].extremes_[2]-person.borders_[2];
		maxY = this->templates_[k].extremes_[3]-person.borders_[2];
	}
	if(this->featurePart_ == FeatureExtractor::WHOLE ||\
	this->existing_[k].groupNo_==PeopleDetector::CLOSE){return;}

	// DEFINE THE PARTS (TOP & BOTTOM) AND CHANGE THE THRESHOLDED IMAGE
	float percent   = 0.50;
	float middleTop = (minY+maxY)*percent;
	float middleBot = (minY+maxY)*percent;
	if(!person.thresh_.empty()){
		for(int y=0;y<person.thresh_.rows;++y){
			// ERASE THE WHOLE ROW IF IT LOWER OR HIGHER THAN ALLOWED
			if(y>middleTop && this->featurePart_==FeatureExtractor::TOP){
				for(int x=0;x<person.thresh_.cols;++x){
					person.thresh_.at<uchar>(y,x) = static_cast<uchar>(0);
				}
			}else if(y<middleBot && this->featurePart_==FeatureExtractor::BOTTOM){
				for(int x=0;x<person.thresh_.cols;++x){
					person.thresh_.at<uchar>(y,x) = static_cast<uchar>(0);
				}
			}
		}
	}

	// CHANGE TEMPLATE TO CONTAIN ONLY THE WANTED PART
	for(unsigned i=0;i<this->templates_[k].points_.size();++i){
		if(this->featurePart_ == FeatureExtractor::TOP &&\
		(this->templates_[k].points_[i].y-person.borders_[2])>middleTop){
			this->templates_[k].points_[i].y = middleTop+person.borders_[2];
		}else if(this->featurePart_==FeatureExtractor::BOTTOM &&\
		(this->templates_[k].points_[i].y-person.borders_[2])<middleBot){
			this->templates_[k].points_[i].y = middleBot+person.borders_[2];
		}
	}
	this->templateExtremes(this->templates_[k].points_,this->templates_[k].extremes_);

	if(this->plot_){
		std::vector<cv::Point2f> tmpTempl = this->templates_[k].points_;
		for(unsigned i=0;i<tmpTempl.size();++i){
			tmpTempl[i].x -= person.borders_[0];
			tmpTempl[i].y -= person.borders_[2];
		}
		if(!person.thresh_.empty()){
			Helpers::plotTemplate2(person.thresh_,cv::Point2f(0,0),\
				cv::Scalar(0,255,0),tmpTempl);
			cv::imshow("part",person.thresh_);
			cv::waitKey(5);
		}
	}
}
//==============================================================================
/** Computes the motion vector for the current image given the tracks so far.
 */
float PeopleDetector::motionVector(const cv::Point2f &head,\
const cv::Point2f &center,bool flip,bool &moved){
	cv::Point2f prev = center;
	float angle      = 0;
	moved            = false;
	if(this->tracks_.size()>1){
		for(unsigned i=0;i<this->tracks_.size();++i){
			if(this->tracks_[i].positions_[this->tracks_[i].positions_.size()-1].x ==\
			center.x && this->tracks_[i].positions_[this->tracks_[i].positions_.size()-1].y\
			== center.y){
				if(this->tracks_[i].positions_.size()>1){
					moved  = true;
					prev.x = this->tracks_[i].positions_[this->tracks_[i].positions_.size()-2].x;
					prev.y = this->tracks_[i].positions_[this->tracks_[i].positions_.size()-2].y;
				}
				break;
			}
		}

		if(this->plot_){
			cv::Mat tmp(cvCloneImage(this->borderedIpl_.get()));
			cv::line(tmp,prev,center,cv::Scalar(50,100,255),1,8,0);
			cv::imshow("tracks",tmp);
		}
	}
	if(moved){
		angle = std::atan2(center.y-prev.y,center.x-prev.x);
	}

	// FIX ANGLE WRT CAMERA
	angle = this->fixAngle(head,center,angle,flip);
	std::cout<<"Motion angle>>> "<<(angle*180/M_PI)<<std::endl;
	return angle;
}
//==============================================================================
/** Starts running something (either the tracker or just mimics it).
 */
void PeopleDetector::start(bool readFromFolder,bool useGT){
	this->useGroundTruth_ = useGT;
	// ((useGT & !GABOR & !EDGES) | EXTRACT) & ANNOS
	if(!this->targetAnno_.empty() && (this->onlyExtract_ || this->useGroundTruth_)){
		// READ THE FRAMES ONE BY ONE
		if(!this->producer_){
			this->initProducer(readFromFolder);
		}
		IplImage *img = this->producer_->getFrame();
		Helpers::setWidth(img->width);
		Helpers::setHeight(img->height);
		Helpers::setDepth(img->depth);
		Helpers::setChannels(img->nChannels);
		Helpers::setHalfresX(Helpers::width()/2);
		Helpers::setHalfresY(Helpers::height()/2);
		this->producer_->backward(1);
		this->current_ = std::tr1::shared_ptr<Image_t>(new Image_t());
		unsigned index = 0;
		this->borderedIpl_ = std::tr1::shared_ptr<IplImage>\
			(cvCreateImage(cvSize(img->width+this->border_,img->height+this->border_),\
			img->depth,img->nChannels),Helpers::releaseImage);
		cvReleaseImage(&img);
		while(this->producer_->canProduce()){
			this->current_->img_ = std::tr1::shared_ptr<IplImage>\
				(this->producer_->getFrame(),Helpers::releaseImage);
			this->current_->sourceName_ = this->producer_->getSource(-1);
			this->current_->index_      = index;
			std::cout<<index<<") Image... "<<this->current_->sourceName_<<std::endl;

			//1) EXTRACT FEATURES OD TRAINING/TESTING DATA
			if(this->onlyExtract_){
				cvCopyMakeBorder(this->current_->img_.get(),this->borderedIpl_.get(),\
					cv::Point(this->border_/2,this->border_/2),IPL_BORDER_REPLICATE,\
					cvScalarAll(0));
				cv::Mat borderedMat(this->borderedIpl_.get());
				this->extractor_->extractFeatures(borderedMat,this->current_->sourceName_);
				borderedMat.release();
			}else{
				this->extractDataRow(NULL,false);
				if(this->flip_){this->extractDataRow(NULL,true);}
			}
			++index;
		}
	}else{
		// FOR THE EDGES AND GABOR I NEED A BACKGROUND MODEL EVEN IF I ONLY EXTRACT
		this->useGroundTruth_ = false;
		this->run(readFromFolder);
	}
	PeopleDetector::dataIsProduced_ = false;
}
//==============================================================================
/** Reads the locations at which there are people in the current frame (for the
 * case in which we do not want to use the tracker or build a bgModel).
 */
void PeopleDetector::readLocations(bool flip){
	this->existing_.clear();
	// FIND	THE INDEX FOR THE CURRENT IMAGE
	if(this->targetAnno_.empty()){
		std::cerr<<"Annotations were not loaded."<<std::endl;
		exit(1);
	}
	std::deque<AnnotationsHandle::FULL_ANNOTATIONS>::iterator index = \
		std::find_if(this->targetAnno_.begin(),this->targetAnno_.end(),\
		compareImg(this->current_->sourceName_));

	if(index == this->targetAnno_.end()){
		std::cerr<<"The image: "<<this->current_->sourceName_<<\
			" was not annotated"<<std::endl;
		exit(1);
	}
	// TRANSFORM THE LOCATION INTO UNSIGNED AND THEN PUSH IT IN THE VECTOR
	for(std::size_t l=0;l<(*index).annos_.size();++l){
		// IF THE BORDERS OF THE TEMPLATE ARE OUTSIDE THE IMAGE THEN IGNORE IT
		std::vector<cv::Point2f> templ;
		if(!Helpers::genTemplate2((*index).annos_[l].location_,Helpers::persHeight(),\
		Helpers::camHeight(),templ)){
			continue;
		}
		// point.x + width*point.y
		cv::Point2f feet = (*index).annos_[l].location_;
		cv::Point2f head = cv::Point2f((templ[12].x+templ[14].x)/2,\
			(templ[12].y+templ[14].y)/2);

		// STORE THE LABELS FOR ALL THE LOCATIONS
		cv::Mat tmp = cv::Mat::zeros(cv::Size(4,1),CV_32FC1);
		// READ THE TARGET ANGLE FOR LONGITUDINAL ANGLE
		float angle = static_cast<float>((*index).annos_[l].\
						poses_[AnnotationsHandle::LONGITUDE]);
		angle = angle*M_PI/180.0;
		angle = this->fixAngle(head,feet,angle,flip);
		tmp.at<float>(0,0) = std::sin(angle);
		tmp.at<float>(0,1) = std::cos(angle);

		// READ THE TARGET ANGLE FOR LATITUDINAL ANGLE
		angle = static_cast<float>((*index).annos_[l].\
					poses_[AnnotationsHandle::LATITUDE]);
		angle = angle*M_PI/180.0;
		tmp.at<float>(0,2) = std::sin(angle);
		tmp.at<float>(0,3) = std::cos(angle);

		// STORE THE LABELS IN THE TARGETS ON THE RIGHT POSITION
		PeopleDetector::CLASSES groupNo = this->findImageClass(feet,head);
		PeopleDetector::Existing tmpExisting(feet,static_cast<unsigned>(groupNo));
		this->existing_.push_back(tmpExisting);
		if(this->targets_[groupNo].empty()){
			tmp.copyTo(this->targets_[groupNo]);
		}else{
			this->targets_[groupNo].push_back(tmp);
		}
		tmp.release();
	}

	// SEE THE ANNOTATED LOCATIONS IN THE IMAGE
  	if(this->plot_){
		for(std::size_t i=0;i<this->existing_.size();++i){
			std::vector<cv::Point2f> templ;
			Helpers::plotTemplate2(this->current_->img_.get(),this->existing_[i].location_,\
				Helpers::persHeight(),Helpers::camHeight(),cv::Scalar(255,0,0),templ);
		}
		cvShowImage("AnnotatedLocations",this->current_->img_.get());
		cvWaitKey(5);
	}
}
//==============================================================================
/** Return rotation angle given the head and feet position.
 */
float PeopleDetector::rotationAngle(const cv::Point2f &headLocation,\
const cv::Point2f &feetLocation){
//	return 0;
	float rotAngle = std::atan2(headLocation.y-feetLocation.y,\
		headLocation.x-feetLocation.x);
	rotAngle += M_PI/2.0;
	rotAngle *= 180.0/M_PI;
	return rotAngle;
}
//==============================================================================
/** Find the class in which we can store the current image (the data is split in
 * 3 classes depending on the position of the person wrt camera).
 */
PeopleDetector::CLASSES PeopleDetector::findImageClass(const cv::Point2f &feet,\
const cv::Point2f &head,bool oneClass){
	if(oneClass){return PeopleDetector::FAR;}
	if(this->classesRange_.empty()){
		// GET THE CAMERA POSITION IN THE IMAGE PLANE
		cv::Point2f cam = (*Helpers::proj())(cv::Point3f(Helpers::camPosX(),\
			Helpers::camPosY(),0));
		float ratio,step;
		step  = (Helpers::maxPersHeightPixels()-0.001)/3.0;
		float previous = 0.0f;
		for(float in=step;in<=Helpers::maxPersHeightPixels();in+=step){
			this->classesRange_.push_back(cv::Point2f(previous,in));
			std::cout<<"previousRange:"<<previous<<" currentRange"<<in<<std::endl;
			previous = in;
		}
		sleep(6);
	}

	float distance = Helpers::dist(feet,head);
	if(distance>this->classesRange_[PeopleDetector::CLOSE].x &&\
	distance<=this->classesRange_[PeopleDetector::CLOSE].y){
		return PeopleDetector::CLOSE;
	}else if(distance>this->classesRange_[PeopleDetector::MEDIUM].x &&\
	distance<=this->classesRange_[PeopleDetector::MEDIUM].y){
		return PeopleDetector::MEDIUM;
	}else if(distance>this->classesRange_[PeopleDetector::FAR].x){
		return PeopleDetector::FAR;
	}
}
//==============================================================================
/** Get distance wrt the camera in the image.
 */
float PeopleDetector::distanceWRTcamera(const cv::Point2f &feet){
	cv::Point2f cam = (*Helpers::proj())(cv::Point3f(Helpers::camPosX(),\
		Helpers::camPosY(),0.0f));
	return Helpers::dist(cam,feet);
}
//==============================================================================
/** Extracts a circle around the predicted/annotated head positon.
 */
void PeopleDetector::extractHeadArea(int i,FeatureExtractor::people &person){
	cv::Mat headMask = cv::Mat::zeros(person.pixels_.size(),CV_8UC1);
	//IF THE THRESHOLD MATRIX IS THERE, FIX THE HEAD AND TEMPLATE AROUND IT
	cv::Point2f headPosition;
	float headSize1 = Helpers::dist(this->templates_[i].points_[12],\
		this->templates_[i].points_[13]);
	float headSize2 = Helpers::dist(this->templates_[i].points_[13],\
		this->templates_[i].points_[14]);
	int radius = static_cast<int>(std::max(headSize1,headSize2))*0.75;
	if(!person.thresh_.empty()){
		int minX,maxX,minY,maxY;
		this->extractor_->getThresholdBorderes(minX,maxX,minY,maxY,person.thresh_);
		minY += static_cast<int>(headSize1/3.0);
		headPosition = cv::Point2f((minX+maxX)/2,std::max(headSize1,headSize2)/2+minY);
		cv::circle(headMask,headPosition,radius,cv::Scalar(255,255,255),-1);
		cv::Mat tmpThresh;
		person.thresh_.copyTo(tmpThresh,headMask);
		person.thresh_.release();
		tmpThresh.copyTo(person.thresh_);
		tmpThresh.release();
	}else{
	//ELSE FIX ONLY THE TEMPLATE
		headPosition = cv::Point2f(this->templates_[i].head_.x-person.borders_[0],\
			this->templates_[i].head_.y-person.borders_[2]+\
			static_cast<int>(headSize1/3.0));
		cv::circle(headMask,headPosition,radius,cv::Scalar(255,255,255),-1);
	}
	//UPDATE THE TEMPLATE POINTS
	this->templates_[i].points_.clear();
	cv::Point2f top_left(headPosition.x+person.borders_[0]-radius,\
		headPosition.y+person.borders_[2]-radius);
	this->templates_[i].points_.push_back(top_left);
	this->templates_[i].points_.push_back(cv::Point2f(top_left.x+2*radius,top_left.y));
	this->templates_[i].points_.push_back(cv::Point2f(top_left.x,top_left.y+2*radius));
	this->templates_[i].points_.push_back(cv::Point2f(top_left.x+2*radius,\
		top_left.y+2*radius));
	this->templateExtremes(this->templates_[i].points_,this->templates_[i].extremes_);

	//UPDATE PERSON PIXELS, BORDERES AND THE REST
	cv::Mat tmpPerson;
	person.pixels_.copyTo(tmpPerson);
	person.pixels_.release();
	person.pixels_ = cv::Mat::zeros(tmpPerson.size(),tmpPerson.type());
	person.pixels_ = cv::Scalar(127,127,127);
	tmpPerson.copyTo(person.pixels_,headMask);
	if(this->plot_){
		cv::imshow("head_area",person.pixels_);
		cv::imshow("head_mask",headMask);
		if(!person.thresh_.empty()){
			cv::imshow("head_thresh",person.thresh_);
		}
		cv::waitKey(5);
	}
	tmpPerson.release();
	headMask.release();
}
//==============================================================================
std::vector<cv::Mat> PeopleDetector::data(){return this->data_;}
//==============================================================================
std::vector<cv::Mat> PeopleDetector::targets(){return this->targets_;}
//==============================================================================
std::deque<std::deque<float> > PeopleDetector::dataMotionVectors(){
	return this->dataMotionVectors_;
};
//==============================================================================
std::tr1::shared_ptr<FeatureExtractor> PeopleDetector::extractor(){
	return this->extractor_;
}
//==============================================================================
void PeopleDetector::setFlip(bool flip){
	this->flip_ = flip;
}
//==============================================================================
boost::mutex PeopleDetector::dataMutex_;
bool PeopleDetector::dataIsProduced_;
//==============================================================================
//==============================================================================
/*
int main(int argc,char **argv){
	PeopleDetector feature(argc,argv,true,false,-1);

	std::deque<FeatureExtractor::FEATURE> fe;
//	fe.push_back(FeatureExtractor::EDGES);
//	fe.push_back(FeatureExtractor::GABOR);
//	fe.push_back(FeatureExtractor::SURF);
//	fe.push_back(FeatureExtractor::IPOINTS);
	fe.push_back(FeatureExtractor::EDGES);

	for(std::size_t u=0;u<fe.size();++u){
		std::deque<FeatureExtractor::FEATURE> feat(1,fe[u]);
		feature.init(std::string(argv[1])+"annotated_train",\
			std::string(argv[1])+"annotated_train.txt",feat,true);
		feature.start(true,true);
	}
}
*/
