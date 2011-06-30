/* FeatureExtractor.cpp
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#include "Auxiliary.h"
#include "FeatureExtractor.h"
//==============================================================================
/** Define a post-fix increment operator for the enum \c FEATURE.
 */
void operator++(FeatureExtractor::FEATURE &feature){
	feature = FeatureExtractor::FEATURE(feature+1);
}
//==============================================================================
/** Checks to see if a pixel's x coordinate is on a scanline.
 */
struct onScanline{
	public:
		unsigned pixelY_;
		onScanline(const unsigned pixelY){this->pixelY_=pixelY;}
		virtual ~onScanline(){};
		bool operator()(const Helpers::scanline_t line)const{
			return (line.line_ == this->pixelY_);
		}
		onScanline(const onScanline &on){
			this->pixelY_ = on.pixelY_;
		}
		onScanline& operator=(const onScanline &on){
			if(this == &on) return *this;
			this->pixelY_ = on.pixelY_;
			return *this;
		}
};
//==============================================================================
FeatureExtractor::FeatureExtractor(){
	this->isInit_       = false;
	this->featureType_  = std::deque<FeatureExtractor::FEATURE>\
		(1,FeatureExtractor::EDGES);
	this->dictFilename_ = "none";
	this->noMeans_      = 500;
	this->meanSize_     = 128;
	this->featureFile_  = "none";
	this->print_        = false;
	this->plot_         = true;
}
//==============================================================================
FeatureExtractor::~FeatureExtractor(){
	if(!this->data_.empty()){
		this->data_.release();
	}
	if(!this->dictionarySIFT_.empty()){
		this->dictionarySIFT_.release();
	}
}
//==============================================================================
/** Initializes the class elements.
 */
void FeatureExtractor::init(const std::deque<FeatureExtractor::FEATURE> &fType,\
const std::string &featFile,int colorSp,int invColorSp,\
FeatureExtractor::FEATUREPART part){
	if(this->isInit_){this->reset();}
	this->featureType_       = fType;
	this->featureFile_       = featFile;
	this->isInit_            = true;
	this->colorspaceCode_    = colorSp;
	this->invColorspaceCode_ = invColorSp;
	this->bodyPart_          = part;
}
//==============================================================================
/** Find if a feature type is in the vector of features.
 */
bool FeatureExtractor::isFeatureIn(std::deque<FeatureExtractor::FEATURE> feats,\
FeatureExtractor::FEATURE feat){
	std::deque<FeatureExtractor::FEATURE>::iterator it = find\
		(feats.begin(),feats.end(),feat);
	return !(it == feats.end());
}
//==============================================================================
/** Initializes the settings for the SIFT dictionary.
 */
void FeatureExtractor::initSIFT(const std::string &dictName,unsigned means,\
unsigned size){
	if(!this->dictionarySIFT_.empty()){
		this->dictionarySIFT_.release();
	}
	this->dictFilename_ = "data/"+dictName;
	this->noMeans_      = means;
	this->meanSize_     = size;

	// ASSUME THAT THERE IS ALREADY A SIFT DICTIONARY AVAILABLE
	if(this->dictionarySIFT_.empty() && FeatureExtractor::isFeatureIn\
	(this->featureType_,FeatureExtractor::SIFT)){
		Auxiliary::binFile2mat(this->dictionarySIFT_,const_cast<char*>\
			(this->dictFilename_.c_str()));
	}
}
//==============================================================================
/** Resets the variables to the default values.
 */
void FeatureExtractor::reset(){
	if(!this->data_.empty()){
		this->data_.release();
	}
	if(!this->dictionarySIFT_.empty()){
		this->dictionarySIFT_.release();
	}
	this->isInit_ = false;
}
//==============================================================================
/** Compares SURF 2 descriptors and returns the boolean value of their comparison.
 */
bool FeatureExtractor::compareDescriptors(const FeatureExtractor::keyDescr &k1,\
const FeatureExtractor::keyDescr &k2){
	return (k1.keys_.response>k2.keys_.response);
}
//==============================================================================
/** Checks to see if a given pixel is inside a template.
 */
bool FeatureExtractor::isInTemplate(unsigned pixelX,unsigned pixelY,\
const std::vector<cv::Point2f> &templ){
	std::vector<cv::Point2f> hull;
	Helpers::convexHull(templ,hull);
	std::deque<Helpers::scanline_t> lines;
	Helpers::getScanLines(hull,lines);

	std::deque<Helpers::scanline_t>::iterator iter = std::find_if(lines.begin(),\
		lines.end(),onScanline(pixelY));
	if(iter == lines.end()){
		return false;
	}

	unsigned var = 5;
	if(std::abs(static_cast<int>(iter->line_)-static_cast<int>(pixelY))<=var &&\
	static_cast<int>(iter->start_)-var <= static_cast<int>(pixelX) &&\
	static_cast<int>(iter->end_)+var >= static_cast<int>(pixelX)){
		return true;
	}else{
		return false;
	}
}
//==============================================================================
/** Rotate a matrix/a template/keypoints wrt to the camera location.
 */
void FeatureExtractor::rotate2Zero(float rotAngle,FeatureExtractor::ROTATE what,\
const cv::Rect roi,cv::Point2f &rotCenter,cv::Point2f &rotBorders,\
std::vector<cv::Point2f> &pts,cv::Mat &toRotate){
	float diag;
	cv::Mat srcRotate,rotated,rotationMat,result;
	cv::Scalar threshColor(127,127,127);
	switch(what){
		case(FeatureExtractor::MATRIX):
			if(toRotate.channels()>1){threshColor = cv::Scalar(127,127,127);
			}else{threshColor = cv::Scalar(0,0,0);}

			diag = std::sqrt(toRotate.cols*toRotate.cols+toRotate.rows*\
				toRotate.rows);
			rotBorders.x = std::ceil((diag-toRotate.cols)/2.0);
			rotBorders.y = std::ceil((diag-toRotate.rows)/2.0);
			srcRotate = cv::Mat::zeros(cv::Size(toRotate.cols+2*rotBorders.x,\
				toRotate.rows+2*rotBorders.y),toRotate.type());
			cv::copyMakeBorder(toRotate,srcRotate,rotBorders.y,rotBorders.y,\
				rotBorders.x,rotBorders.x,cv::BORDER_CONSTANT,threshColor);
			rotCenter = cv::Point2f(srcRotate.cols/2.0,srcRotate.rows/2.0);
			rotationMat = cv::getRotationMatrix2D(rotCenter,rotAngle,1.0);
			rotationMat.convertTo(rotationMat,CV_32FC1);
			rotated     = cv::Mat::zeros(srcRotate.size(),toRotate.type());
			cv::warpAffine(srcRotate,rotated,rotationMat,srcRotate.size(),\
				cv::INTER_LINEAR,cv::BORDER_CONSTANT,threshColor);
			rotated.copyTo(result);
			break;
		case(FeatureExtractor::TEMPLATE):
			rotationMat = cv::getRotationMatrix2D(rotCenter,rotAngle,1.0);
			rotationMat.convertTo(rotationMat,CV_32FC1);
			toRotate = cv::Mat::ones(cv::Size(3,pts.size()),CV_32FC1);
			for(std::size_t i=0;i<pts.size();++i){
				toRotate.at<float>(i,0) = pts[i].x + rotBorders.x;
				toRotate.at<float>(i,1) = pts[i].y + rotBorders.y;
			}
			toRotate.convertTo(toRotate,CV_32FC1);
			rotated = toRotate*rotationMat.t();
			rotated.convertTo(rotated,CV_32FC1);
			pts.clear();
			pts = std::vector<cv::Point2f>(rotated.rows);
			for(int y=0;y<rotated.rows;++y){
				pts[y].x = rotated.at<float>(y,0);
				pts[y].y = rotated.at<float>(y,1);
			}
			break;
		case(FeatureExtractor::KEYS):
			rotationMat = cv::getRotationMatrix2D(rotCenter,rotAngle,1.0);
			rotationMat.convertTo(rotationMat,CV_32FC1);
			for(int y=0;y<toRotate.rows;++y){
				float ptX = toRotate.at<float>(y,toRotate.cols-2)+\
					rotBorders.x;
				float ptY = toRotate.at<float>(y,toRotate.cols-1)+\
					rotBorders.y;
				cv::Mat tmp = cv::Mat::zeros(cv::Size(3,1),CV_32FC1);
				tmp.at<float>(0,0) = ptX;
				tmp.at<float>(0,1) = ptY;
				tmp.at<float>(0,2) = 0.0f;
				if(srcRotate.empty()){
					tmp.copyTo(srcRotate);
				}else{
					srcRotate.push_back(tmp);
				}
				cv::Mat dummy = toRotate.row(y);
				if(result.empty()){
					dummy.copyTo(result);
				}else{
					result.push_back(dummy);
				}
				tmp.release();
				dummy.release();
			}
			srcRotate.convertTo(srcRotate,CV_32FC1);
			rotated = srcRotate*rotationMat.t();
			rotated.convertTo(rotated,CV_32FC1);
			for(int y=0;y<rotated.rows;++y){
				float ptX = rotated.at<float>(y,0);
				float ptY = rotated.at<float>(y,1);
				if(ptX<roi.x || ptY<roi.y || ptX>(roi.x+roi.width) || \
				ptY>(roi.y+roi.height)){
					continue;
				}
				result.at<float>(y,toRotate.cols-2);
				result.at<float>(y,toRotate.cols-1);
			}
			break;
	}
	rotationMat.release();
	srcRotate.release();
	rotated.release();
	toRotate.release();
	result.copyTo(toRotate);
	result.release();
}
//==============================================================================
/** Gets the plain pixels corresponding to the upper part of the body.
 */
cv::Mat FeatureExtractor::getTemplMatches(bool flip,\
const FeatureExtractor::people &person,const FeatureExtractor::templ &aTempl,\
const cv::Rect &roi){
	cv::Rect cutROI;
	if(!person.thresh_.empty()){
		this->getThresholdBorderes(cutROI.x,cutROI.width,cutROI.y,cutROI.height,\
			person.thresh_);
		cutROI.width  -= cutROI.x;
		cutROI.height -= cutROI.y;
	}else{
		cutROI.x      = aTempl.extremes_[0]-roi.x;
		cutROI.y      = aTempl.extremes_[2]-roi.y;
		cutROI.width  = aTempl.extremes_[1]-aTempl.extremes_[0];
		cutROI.height = aTempl.extremes_[3]-aTempl.extremes_[2];
	}
	cv::Mat large = this->cutAndResizeImage(cutROI,person.pixels_);
	if(flip){cv::flip(large,large,1);}
	if(this->colorspaceCode_!=-1){
		cv::cvtColor(large,large,this->invColorspaceCode_);
	}
	cv::Mat gray;
	cv::cvtColor(large,gray,CV_BGR2GRAY);
	large.release();
	if(this->plot_){
		cv::imshow("part",gray);
		cv::waitKey(5);
	}

	// MATCH SOME HEADS ON TOP AND GET THE RESULTS
	int radius     = Helpers::dist(aTempl.points_[12],aTempl.points_[13]);
	cv::Mat result = cv::Mat::zeros(cv::Size(12*20*30+2,1),CV_32FC1);
	for(int i=0;i<4;++i){
		std::string imgName = "templates/templ"+Auxiliary::int2string(i)+".jpg";
		cv::Mat tmple       = cv::imread(imgName.c_str(),0);
		if(tmple.empty()){
			std::cerr<<"In template matching FILE NOT FOUND: "<<imgName<<std::endl;
			exit(1);
		}
		cv::Mat small;
		cv::blur(tmple,tmple,cv::Size(3,3));
		cv::resize(tmple,small,cv::Size(radius,radius),0,0,cv::INTER_CUBIC);
		small.convertTo(small,CV_8UC1);
		cv::Mat tmp;
		cv::matchTemplate(gray,small,tmp,CV_TM_CCOEFF_NORMED);
		cv::Mat resized;
		cv::blur(tmp,tmp,cv::Size(3,3));
		cv::resize(tmp,resized,cv::Size(20,30),0,0,cv::INTER_CUBIC);
		if(this->plot_){
			cv::imshow("result"+Auxiliary::int2string(i),resized);
			cv::waitKey(5);
		}

		// RESHAPE THE RESULT AND CONVERT IT TO float
		resized = resized.reshape(0,1);
		resized.convertTo(resized,CV_32FC1);
		cv::Mat dummy = result.colRange(i*resized.cols*resized.rows,(i+1)*\
			resized.cols*resized.rows);
		resized.copyTo(dummy);
		tmp.release();
		resized.release();
		small.release();
		tmple.release();
		dummy.release();
	}
	result.convertTo(result,CV_32FC1);

	if(this->print_){
		std::cout<<"Size(PIXELS): ("<<result.size()<<std::endl;
		for(int i=0;i<std::min(10,result.cols);++i){
			std::cout<<result.at<float>(0,i)<<" ";
		}
		std::cout<<"..."<<std::endl;
	}
	gray.release();
	return result;
}
//==============================================================================
/** Get skin/non-skin ratio of the foreground area.
 */
cv::Mat FeatureExtractor::getSkinBins(bool flip,\
const FeatureExtractor::people &person,const FeatureExtractor::templ &aTempl,\
const cv::Rect &roi){
	cv::Rect cutROI;
	if(!person.thresh_.empty()){
		this->getThresholdBorderes(cutROI.x,cutROI.width,cutROI.y,cutROI.height,\
			person.thresh_);
		cutROI.width  -= cutROI.x;
		cutROI.height -= cutROI.y;
	}else{
		cutROI.x      = aTempl.extremes_[0]-roi.x;
		cutROI.y      = aTempl.extremes_[2]-roi.y;
		cutROI.width  = aTempl.extremes_[1]-aTempl.extremes_[0];
		cutROI.height = aTempl.extremes_[3]-aTempl.extremes_[2];
	}

	// GET THE SKIN PIXELS OUT OF THE IMAGE
	cv::Mat img;
	person.pixels_.copyTo(img);
	if(this->colorspaceCode_!=-1){
		cv::cvtColor(img,img,this->invColorspaceCode_);
	}
	cv::cvtColor(img,img,CV_BGR2HSV);
	std::vector<cv::Mat> threeChannels;
	cv::split(img,threeChannels);
	cv::Mat skin = threeChannels[1]/threeChannels[2];
	Auxiliary::normalizeMat(skin);
	cv::medianBlur(skin,skin,7);
	cv::erode(skin,skin,cv::Mat(),cv::Point(-1,-1),3);
	cv::dilate(skin,skin,cv::Mat(),cv::Point(-1,-1),3);

	// INVERT THE SKIN TO HAVE THE FACE REGION WHITE NOT BLACK
	skin = cv::Scalar(255,255,255) - skin;
	cv::Mat mask,thresh;
	person.thresh_.copyTo(thresh);
	cv::erode(thresh,thresh,cv::Mat(),cv::Point(-1,-1),5);
	skin.copyTo(mask,thresh);
	cv::Mat gray;
	// THRESHOLD THE SKIN IMAGE
	cv::threshold(mask,gray,250,255,cv::THRESH_BINARY);
	cv::Mat final = this->cutAndResizeImage(cutROI,gray);
	img.release();
	skin.release();
	mask.release();
	thresh.release();
	threeChannels.clear();

	if(flip){cv::flip(final,final,1);}
	if(this->plot_){
		cv::imshow("gray",person.pixels_);
		cv::imshow("final",final);
		cv::waitKey(5);
	}
	cv::Mat result = final.reshape(0,1);
/*
	cv::Mat result = cv::Mat::zeros(cv::Size(3,1),CV_32FC1);
	for(int c=0;c<final.cols;++c){
		for(int r=0;r<final.rows;++r){
			if(static_cast<int>(final.at<uchar>(r,c))!=0){
				++result.at<float>(0,0);
			}
		}
	}
	result /= static_cast<float>(final.cols*final.rows);
	final.release();
*/
	return result;
}
//==============================================================================
/** Gets the raw pixels corresponding to body of the person +/- background pixels.
 */
cv::Mat FeatureExtractor::getRawPixels(bool flip,\
const FeatureExtractor::people &person,const FeatureExtractor::templ &aTempl,\
const cv::Rect &roi,bool color){
	cv::Rect cutROI;
	if(!person.thresh_.empty()){
		this->getThresholdBorderes(cutROI.x,cutROI.width,cutROI.y,cutROI.height,\
			person.thresh_);
		cutROI.width  -= cutROI.x;
		cutROI.height -= cutROI.y;
	}else{
		cutROI.x      = aTempl.extremes_[0]-roi.x;
		cutROI.y      = aTempl.extremes_[2]-roi.y;
		cutROI.width  = aTempl.extremes_[1]-aTempl.extremes_[0];
		cutROI.height = aTempl.extremes_[3]-aTempl.extremes_[2];
	}
	cv::Mat large = this->cutAndResizeImage(cutROI,person.pixels_);
	if(flip){cv::flip(large,large,1);}
	if(this->colorspaceCode_!=-1){
		cv::cvtColor(large,large,this->invColorspaceCode_);
	}

	// IF THE IMAGE SHOULD BE GRAY THAN USE THE VALUE CHANNEL
	cv::Mat gray;
	std::vector<cv::Mat> threeChannels;
	if(!color){
		cv::cvtColor(large,large,CV_BGR2HSV);
		cv::split(large,threeChannels);
		threeChannels[2].copyTo(gray);
		Auxiliary::normalizeMat(gray);
	}else{
		gray = large.reshape(1,0);
	}
	if(this->plot_){
		cv::imshow("gray",gray);
		cv::waitKey(5);
	}
	cv::Mat result = cv::Mat::zeros(cv::Size(gray.cols*gray.rows+2,1),\
		CV_32FC1);
	result = gray.reshape(0,1);
	result.convertTo(result,CV_32FC1);

	if(this->print_){
		std::cout<<"Size(RAW_PIXELS): ("<<result.size()<<std::endl;
		for(int i=0;i<std::min(10,result.cols);++i){
			std::cout<<result.at<float>(0,i)<<" ";
		}
		std::cout<<"..."<<std::endl;
	}
	large.release();
	gray.release();
	return result;
}
//==============================================================================
/** Find the extremities of the thresholded image.
 */
void FeatureExtractor::getThresholdBorderes(int &minX,int &maxX,int &minY,\
int &maxY,const cv::Mat &thresh){
	minY = thresh.rows; maxY = 0;
	minX = thresh.cols; maxX = 0;
	for(int x=0;x<thresh.cols;++x){
		for(int y=0;y<thresh.rows;++y){
			if(static_cast<int>(thresh.at<uchar>(y,x))>0){
				if(y<=minY){minY = y;}
				if(y>=maxY){maxY = y;}
				if(x<=minX){minX = x;}
				if(x>=maxX){maxX = x;}
			}
		}
	}
}
//==============================================================================
/** Gets the edges in an image.
 */
cv::Mat FeatureExtractor::getEdges(bool flip,cv::Mat &feature,\
const cv::Mat &thresholded,const cv::Rect &roi,\
const FeatureExtractor::templ &aTempl,float rotAngle,bool contours){
	// EXTRACT THE EDGES AND ROTATE THE EDGES TO THE RIGHT POSSITION
	cv::Point2f rotBorders;
	feature.convertTo(feature,CV_8UC1);
	cv::Mat tmpFeat(feature.clone(),roi);
	std::vector<cv::Point2f> dummy;
	cv::Point2f rotCenter(tmpFeat.cols/2+roi.x,tmpFeat.rows/2+roi.y);
	this->rotate2Zero(rotAngle,FeatureExtractor::MATRIX,roi,rotCenter,rotBorders,\
		dummy,tmpFeat);

	// PICK OUT ONLY THE THRESHOLDED ARES RESHAPE IT AND RETURN IT
	cv::Rect cutROI;
	cv::Mat toResize;
	if(!thresholded.empty()){
		tmpFeat.copyTo(toResize,thresholded);
		this->getThresholdBorderes(cutROI.x,cutROI.width,cutROI.y,cutROI.height,\
			thresholded);
		cutROI.width   -= cutROI.x;
		cutROI.height  -= cutROI.y;
	}else{
		tmpFeat.copyTo(toResize);
		cutROI.x      = aTempl.extremes_[0]-roi.x;
		cutROI.y      = aTempl.extremes_[2]-roi.y;
		cutROI.width  = aTempl.extremes_[1]-aTempl.extremes_[0];
		cutROI.height = aTempl.extremes_[3]-aTempl.extremes_[2];
	}
	cv::Mat tmpEdge = this->cutAndResizeImage(cutROI,toResize);
	tmpFeat.release();
	toResize.release();
	if(flip){cv::flip(tmpEdge,tmpEdge,1);}

 	cv::Mat result;
	if(contours){
		cv::dilate(tmpEdge,tmpEdge,cv::Mat(),cv::Point(-1,-1),3);
		std::vector<std::vector<cv::Point> > contours;
		cv::findContours(tmpEdge,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
		std::vector<std::vector<cv::Point> >::iterator iter = std::max_element\
			(contours.begin(),contours.end(),Auxiliary::isLongerContours);
		cv::Mat contourMat = cv::Mat(contours[iter-contours.begin()]);
		contourMat.convertTo(contourMat,CV_32FC1);
		contourMat = contourMat.reshape(1,1);
		cv::Mat preResult;
		cv::blur(contourMat,contourMat,cv::Size(5,5));
		cv::resize(contourMat,preResult,cv::Size(100,1),0,0,cv::INTER_CUBIC);

		// WRITE IT ON ONE ROW
		result = cv::Mat::zeros(cv::Size(preResult.rows*preResult.cols+2,1),CV_32FC1);
		cv::Mat dumm = result.colRange(0,(preResult.cols*preResult.rows));
		preResult.copyTo(dumm);
		dumm.release();
		preResult.release();
	}else{
		cv::dilate(tmpEdge,tmpEdge,cv::Mat(),cv::Point(-1,-1),1);
		cv::medianBlur(tmpEdge,tmpEdge,3);
	 	if(this->plot_){
			cv::imshow("Edges",tmpEdge);
			cv::waitKey(5);
		}
		result = cv::Mat::zeros(cv::Size(tmpEdge.rows*tmpEdge.cols+2,1),CV_32FC1);
		cv::Mat dumm = result.colRange(0,(tmpEdge.cols*tmpEdge.rows));
		tmpEdge = tmpEdge.reshape(0,1);
		tmpEdge.copyTo(dumm);
		dumm.release();
	}
	result.convertTo(result,CV_32FC1);
	tmpEdge.release();
	if(this->print_){
		std::cout<<"Size(EDGES): "<<result.size()<<std::endl;
		for(int i=0;i<std::min(10,result.cols);++i){
			std::cout<<result.at<float>(0,i)<<" ";
		}
		std::cout<<"..."<<std::endl;
	}
	return result;
}
//==============================================================================
/** SURF descriptors (Speeded Up Robust Features).
 */
cv::Mat FeatureExtractor::getSURF(bool flip,cv::Mat &feature,\
const std::vector<cv::Point2f> &templ,const cv::Rect &roi,const cv::Mat &test,\
std::vector<cv::Point2f> &indices){
	// KEEP THE TOP 10 DESCRIPTORS WITHIN THE BORDERS OF THE TEMPLATE
	unsigned number  = 30;
	cv::Mat tmp      = cv::Mat::zeros(cv::Size(feature.cols,number),CV_32FC1);
	unsigned counter = 0;
	for(int y=0;y<feature.rows;++y){
		if(counter == number){
			break;
		}
		float ptX = feature.at<float>(y,feature.cols-2);
		float ptY = feature.at<float>(y,feature.cols-1);
		if(flip){ptX = (roi.width-ptX);}
		if(FeatureExtractor::isInTemplate(ptX,ptY,templ)){
			cv::Mat dummy1 = tmp.row(counter);
			cv::Mat dummy2 = feature.row(y);
			dummy2.copyTo(dummy1);
			dummy1.release();
			dummy2.release();
			indices.push_back(cv::Point2f(ptX-roi.x,ptY-roi.y));
			++counter;
		}
	}

	if(this->plot_ && !test.empty()){
		cv::Mat copyTest(test);
		for(std::size_t l=0;l<indices.size();++l){
			cv::circle(copyTest,indices[l],3,cv::Scalar(0,0,255));
		}
		cv::imshow("SURF",copyTest);
		cv::waitKey(5);
		copyTest.release();
	}

	// COPY THE DESCRIPTORS IN THE FINAL MATRIX
	cv::Mat result = cv::Mat::zeros(cv::Size(tmp.rows*tmp.cols+2,1),CV_32FC1);
	tmp            = tmp.reshape(0,1);
	tmp.convertTo(tmp,CV_32FC1);
	cv::Mat dummy = result.colRange(0,tmp.rows*tmp.cols);
	tmp.copyTo(dummy);
	tmp.release();
	dummy.release();
	result.convertTo(result,CV_32FC1);

	// IF WE WANT TO SEE SOME VALUES/IMAGES
	if(this->print_){
		std::cout<<"Size(SURF): "<<result.size()<<std::endl;
		for(int i=0;i<std::min(10,result.cols);++i){
			std::cout<<result.at<float>(0,i)<<" ";
		}
		std::cout<<"..."<<std::endl;
	}
	result.convertTo(result,CV_32FC1);
	return result;
}
//==============================================================================
/** Creates a "histogram" of interest points + number of blobs.
 */
cv::Mat FeatureExtractor::getPointsGrid(bool flip,const cv::Mat &feature,\
const cv::Rect &roi,const FeatureExtractor::templ &aTempl,const cv::Mat &test){
	// GET THE GRID SIZE FROM THE TEMPLATE SIZE
	unsigned no       = 30;
	cv::Mat preResult = cv::Mat::zeros(cv::Size(no*no,3),CV_32FC1);
	float rateX = (aTempl.extremes_[1]-aTempl.extremes_[0])/static_cast<float>(no);
	float rateY = (aTempl.extremes_[3]-aTempl.extremes_[2])/static_cast<float>(no);

	// KEEP ONLY THE KEYPOINTS THAT ARE IN THE TEMPLATE
	std::vector<cv::Point2f> indices;
	for(int y=0;y<feature.rows;++y){
		float ptX = feature.at<float>(y,0);
		float ptY = feature.at<float>(y,1);
		if(flip){ptX = (roi.width-ptX);}
		if(FeatureExtractor::isInTemplate(ptX,ptY,aTempl.points_)){
			cv::Point2f pt(ptX,ptY);
			indices.push_back(pt);

			// CHECK IN WHAT CELL OF THE GRID THE CURRENT POINTS FALL
			unsigned counter = 0;
			for(float ix=aTempl.extremes_[0];ix<aTempl.extremes_[1]-0.01;ix+=rateX){
				for(float iy=aTempl.extremes_[2];iy<aTempl.extremes_[3]-0.01;iy+=rateY){
					if(ix<=pt.x && pt.x<(ix+rateX) && iy<=pt.y && pt.y<(iy+rateY)){
						preResult.at<float>(0,counter) += 1.0;
						preResult.at<float>(1,counter) += pt.x;
						preResult.at<float>(2,counter) += pt.y;
					}
					counter +=1;
				}
			}
		}
	}
	if(this->plot_ && !test.empty()){
		cv::Mat copyTest(test);
		for(std::size_t l=0;l<indices.size();++l){
			cv::circle(copyTest,cv::Point2f(indices[l].x-roi.x,indices[l].y-roi.y),\
				3,cv::Scalar(0,0,255));
		}
		cv::imshow("IPOINTS",copyTest);
		cv::waitKey(5);
		copyTest.release();
	}
	preResult      = preResult.reshape(0,1);
	cv::Mat result = cv::Mat::zeros(cv::Size(preResult.cols+2,1),CV_32FC1);
	cv::Mat range  = result.colRange(0,preResult.cols);
	preResult.copyTo(range);
	preResult.release();
	range.release();

	// IF WE WANT TO SEE THE VALUES THAT WERE STORED
	if(this->print_){
		std::cout<<"Size(IPOINTS): ("<<result.size()<<std::endl;
		unsigned counter = 0;
		for(int i=0;i<result.cols,counter<10;++i){
			if(result.at<float>(0,i)!=0){
				std::cout<<result.at<float>(0,i)<<" ";
				++counter;
			}
		}
		std::cout<<std::endl;
	}
	result.convertTo(result,CV_32FC1);
	return result;
}
//==============================================================================
/** Creates a gabor with the parameters given by the parameter vector.
 */
void FeatureExtractor::createGabor(cv::Mat &gabor, float *params){
	// params[0] -- sigma: (3,68) // the actual size
	// params[1] -- gamma: (0.2,1) // how round the filter is
	// params[2] -- dimension: (1,10) // size
	// params[3] -- theta: (0,180) or (-90,90) // angle
	// params[4] -- lambda: (2,255) // thickness
	// params[5] -- psi: (0,180) // number of lines

	// SET THE PARAMTETERS OF THE GABOR FILTER
	if(params == NULL){
		params    = new float[6];
		params[0] = 10.0;params[1] = 0.9;params[2] = 2.0;
		params[3] = M_PI/4.0;params[4] = 50.0;params[5] = 12.0;
	}

	// CREATE THE GABOR FILTER OR WAVELET
	float sigmaX = params[0];
	float sigmaY = params[0]/params[1];
	float xMax   = std::max(std::abs(params[2]*sigmaX*std::cos(params[3])),\
							std::abs(params[2]*sigmaY*std::sin(params[3])));
	xMax         = std::ceil(std::max((float)1.0,xMax));
	float yMax  = std::max(std::abs(params[2]*sigmaX*std::cos(params[3])),\
							std::abs(params[2]*sigmaY*std::sin(params[3])));
	yMax         = std::ceil(std::max((float)1.0,yMax));
	float xMin  = -xMax;
	float yMin  = -yMax;
	gabor        = cv::Mat::zeros(cv::Size((int)(xMax-xMin),(int)(yMax-yMin)),\
					CV_32FC1);
	for(int x=(int)xMin;x<xMax;++x){
		for(int y=(int)yMin;y<yMax;++y){
			float xPrime = x*std::cos(params[3])+y*std::sin(params[3]);
			float yPrime = -x*std::sin(params[3])+y*std::cos(params[3]);
			gabor.at<float>((int)(y+yMax),(int)(x+xMax)) = \
				std::exp(-0.5*((xPrime*xPrime)/(sigmaX*sigmaX)+\
				(yPrime*yPrime)/(sigmaY*sigmaY)))*\
				std::cos(2.0 * M_PI/params[4]*xPrime*params[5]);
		}
	}
	gabor.convertTo(gabor,CV_32FC1);
}
//==============================================================================
/** Convolves an image with a Gabor filter with the given parameters and
 * returns the response image.
 */
cv::Mat FeatureExtractor::getGabor(bool flip,cv::Mat &feature,\
const cv::Mat &thresholded,const cv::Rect &roi,const cv::Size &foregrSize,\
const FeatureExtractor::templ &aTempl,const float rotAngle,int aheight){
	unsigned gaborNo    = feature.rows/aheight;
	int gaborRows       = feature.rows/gaborNo;
	cv::Mat result;
	cv::Point2f rotCenter(foregrSize.width/2+roi.x,foregrSize.height/2+roi.y);
	std::vector<cv::Point2f> dummy;

	// READ THE BORDERS OF THE THRESHOLDED IMAGE
	cv::Rect cutROI;
	if(!thresholded.empty()){
		this->getThresholdBorderes(cutROI.x,cutROI.width,cutROI.y,cutROI.height,\
			thresholded);
		cutROI.width  -= cutROI.x;
		cutROI.height -= cutROI.y;
	}else{
		cutROI.x      = aTempl.extremes_[0]-roi.x;
		cutROI.y      = aTempl.extremes_[2]-roi.y;
		cutROI.width  = aTempl.extremes_[1]-aTempl.extremes_[0];
		cutROI.height = aTempl.extremes_[3]-aTempl.extremes_[2];
	}
	for(unsigned i=0;i<gaborNo;++i){
		// GET THE ROI OUT OF THE iTH GABOR
		cv::Mat tmp = feature.rowRange(i*gaborRows,(i+1)*gaborRows);
		cv::Mat tmp1;
		tmp.copyTo(tmp1);
		tmp1.convertTo(tmp1,CV_8UC1);
		cv::Mat tmp2(tmp1.clone(),roi);

		// ROTATE EACH GABOR TO THE RIGHT POSITION
		cv::Point2f rotBorders;
		this->rotate2Zero(rotAngle,FeatureExtractor::MATRIX,roi,rotCenter,\
			rotBorders,dummy,tmp2);

		// KEEP ONLY THE THRESHOLDED VALUES
		cv::Mat toResize;
		if(!thresholded.empty()){
			tmp2.copyTo(toResize,thresholded);
		}else{
			tmp2.copyTo(toResize);
		}

		// READ THE ROI TO BE USED TO GET THE AREA AROUND THE IMAGE
		cv::Mat tmp3;
		tmp3 = this->cutAndResizeImage(cutROI,toResize);
		toResize.release();
		if(flip){cv::flip(tmp3,tmp3,1);}
		if(this->plot_){
			cv::imshow("WholeGaborResponse",tmp1);
			cv::imshow("GaborResponse",tmp3);
			cv::waitKey(5);
		}

		// RESHAPE AND STORE IN THE RIGHT PLACE
		tmp3 = tmp3.reshape(0,1);
		tmp3.convertTo(tmp3,CV_32FC1);
		if(result.empty()){
			result = cv::Mat::zeros(cv::Size(gaborNo*tmp3.cols+2,1),CV_32FC1);
		}
		cv::Mat dummy  = result.colRange(i*tmp3.cols,(i+1)*tmp3.cols);
		tmp3.copyTo(dummy);

		// RELEASE ALL THE TEMPS AND DUMMIES
		tmp.release();
		tmp1.release();
		tmp2.release();
		tmp3.release();
		dummy.release();
	}

	if(this->print_){
		std::cout<<"Size(GABOR): ("<<result.cols<<","<<result.rows<<")"<<std::endl;
		unsigned counter = 0;
		for(int i=0;i<result.cols,counter<10;++i){
			if(result.at<float>(0,i)!=0){
				++counter;
				std::cout<<result.at<float>(0,i)<<" ";
			}
		}
		std::cout<<"..."<<std::endl;
	}
	result.convertTo(result,CV_32FC1);
	return result;
}
//==============================================================================
/** Computes the distance from the first matrix to the second and the position on
 * which the minimum is found and the value of the minimum for each row.
 */
cv::Mat FeatureExtractor::dist2(const cv::Mat &mat1,const cv::Mat &mat2,\
cv::Mat &minDists,cv::Mat &minLabs){
	cv::Mat distances = cv::Mat::zeros(cv::Size(mat1.rows,mat2.rows),CV_32FC1);
	minDists          = cv::Mat::zeros(cv::Size(mat1.rows,1),CV_32FC1);
	minDists         -= 1;
	minLabs           = cv::Mat::zeros(cv::Size(mat1.rows,1),CV_32FC1);
	for(int j=0;j<mat1.rows;++j){
		for(int i=0;i<mat2.rows;++i){
			cv::Mat diff;
			cv::absdiff(mat2.row(i),mat1.row(j),diff);
			distances.at<float>(i,j) = std::sqrt(diff.dot(diff));
			diff.release();
			if(minDists.at<float>(0,j)==-1 || minDists.at<float>(0,j)>\
			distances.at<float>(i,j)){
				minDists.at<float>(0,j) = distances.at<float>(i,j);
				minLabs.at<float>(0,j)  = i;
			}
		}
	}
	return distances;
}
//==============================================================================
/** Compute the features from the SIFT descriptors by doing vector quantization.
 */
cv::Mat FeatureExtractor::getSIFT(bool flip,const cv::Mat &feature,\
const std::vector<cv::Point2f> &templ,const cv::Rect &roi,const cv::Mat &test,\
std::vector<cv::Point2f> &indices){
	// KEEP ONLY THE SIFT FEATURES THAT ARE WITHIN THE TEMPLATE
	cv::Mat tmp      = cv::Mat::zeros(cv::Size(feature.cols-2,feature.rows),CV_32FC1);
	unsigned counter = 0;
	for(int y=0;y<feature.rows;++y){
		float ptX = feature.at<float>(y,feature.cols-2);
		float ptY = feature.at<float>(y,feature.cols-1);
		if(flip){ptX = (roi.width-ptX);}
		if(FeatureExtractor::isInTemplate(ptX,ptY,templ)){
			cv::Mat dummy1 = tmp.row(counter);
			cv::Mat dummy2 = feature.row(y);
			cv::Mat dummy3 = dummy2.colRange(0,dummy2.cols-2);
			dummy3.copyTo(dummy1);
			dummy1.release();
			dummy2.release();
			dummy3.release();
			indices.push_back(cv::Point2f(ptX-roi.x,ptY-roi.y));
			++counter;
		}
	}

	// KEEP ONLY THE NON-ZEROS ROWS OUT OF tmp
	cv::Mat preFeature;
	cv::Mat dum = tmp.rowRange(0,counter);
	dum.copyTo(preFeature);
	dum.release();
	tmp.release();
	preFeature.convertTo(preFeature,CV_32FC1);

	// ASSUME THAT THERE IS ALREADY A SIFT DICTIONARY AVAILABLE
	if(this->dictionarySIFT_.empty()){
		Auxiliary::binFile2mat(this->dictionarySIFT_,const_cast<char*>\
			(this->dictFilename_.c_str()));
	}

	// COMPUTE THE DISTANCES FROM EACH NEW FEATURE TO THE DICTIONARY ONES
	cv::Mat minDists,minLabel;
	cv::Mat distances = FeatureExtractor::dist2(preFeature,this->dictionarySIFT_,\
		minDists,minLabel);

	// CREATE A HISTOGRAM(COUNT TO WHICH DICT FEATURE WAS ASSIGNED EACH NEW ONE)
	cv::Mat result = cv::Mat::zeros(cv::Size(this->dictionarySIFT_.rows+2,1),CV_32FC1);
	for(int i=0;i<minLabel.cols;++i){
		int which = minLabel.at<float>(0,i);
		result.at<float>(0,which) += 1.0;
	}

	// NORMALIZE THE HOSTOGRAM
	cv::Scalar scalar = cv::sum(result);
	result /= static_cast<float>(scalar[0]);

	if(this->plot_ && !test.empty()){
		cv::Mat copyTest(test);
		for(std::size_t l=0;l<indices.size();++l){
			cv::circle(copyTest,indices[l],3,cv::Scalar(0,0,255));
		}
		cv::imshow("SIFT",copyTest);
		cv::waitKey(5);
		copyTest.release();
	}
	preFeature.release();
	distances.release();
	minDists.release();
	minLabel.release();

	// IF WE WANT TO SEE THE VALUES THAT WERE STORED
	if(this->print_){
		std::cout<<"Size(SIFT): "<<result.size()<<std::endl;
		unsigned counter = 0;
		for(int i=0;i<result.cols,counter<10;++i){
			if(result.at<float>(0,i)!=0){
				std::cout<<result.at<float>(0,i)<<" ";
				++counter;
			}
		}
		std::cout<<std::endl;
	}
	result.convertTo(result,CV_32FC1);
	return result;
}
//==============================================================================
/** Creates a data matrix for each image and stores it locally.
 */
void FeatureExtractor::extractFeatures(cv::Mat &image,const std::string &sourceName){
	if(this->featureType_.size()>1){
		std::cerr<<"There can not be more than 1 features extracted in the same time"<<\
			std::endl;
		std::abort();
	}
	if(this->colorspaceCode_ != -1){
		cv::cvtColor(image,image,this->colorspaceCode_);
	}
	if(this->featureFile_[this->featureFile_.size()-1]!='/'){
		this->featureFile_ = this->featureFile_ + '/';
	}
	Helpers::file_exists(this->featureFile_.c_str(),true);
	std::cout<<"In extract features"<<std::endl;

	// FOR EACH LOCATION IN THE IMAGE EXTRACT FEATURES AND STORE
	cv::Mat feature;
	std::string toWrite = this->featureFile_;
	std::vector<cv::Point2f> dummyT;
	cv::Rect dummyR;
	for(FeatureExtractor::FEATURE f=FeatureExtractor::EDGES;\
	f<=FeatureExtractor::SKIN_BINS;++f){
		if(!FeatureExtractor::isFeatureIn(this->featureType_,f)){continue;}
		cv::Mat dummy;
		switch(f){
			case (FeatureExtractor::IPOINTS):
				toWrite += "IPOINTS/";
				Helpers::file_exists(toWrite.c_str(),true);
				dummy = this->extractPointsGrid(image);
				break;
			case FeatureExtractor::EDGES:
				toWrite += "EDGES/";
				Helpers::file_exists(toWrite.c_str(),true);
				dummy = this->extractEdges(image);
				break;
			case FeatureExtractor::SURF:
				toWrite += "SURF/";
				Helpers::file_exists(toWrite.c_str(),true);
				dummy = this->extractSURF(image);
				break;
			case FeatureExtractor::GABOR:
				toWrite += "GABOR/";
				Helpers::file_exists(toWrite.c_str(),true);
				dummy = this->extractGabor(image);
				break;
			case FeatureExtractor::SIFT:
				toWrite += "SIFT/";
				Helpers::file_exists(toWrite.c_str(),true);
				dummy = this->extractSIFT(image,dummyT,dummyR);
				break;
		}
		dummy.copyTo(feature);
		dummy.release();
		break;
	}
	feature.convertTo(feature,CV_32FC1);

	//WRITE THE FEATURE TO A BINARY FILE
	unsigned pos1       = sourceName.find_last_of("/\\");
	std::string imgName = sourceName.substr(pos1+1);
	unsigned pos2       = imgName.find_last_of(".");
	imgName             = imgName.substr(0,pos2);
	toWrite            += (imgName + ".bin");

	std::cout<<"Feature written to: "<<toWrite<<std::endl;
	Auxiliary::mat2BinFile(feature,const_cast<char*>(toWrite.c_str()));
	feature.release();
}
//==============================================================================
/** Extract the interest points in a gird and returns them.
 */
cv::Mat FeatureExtractor::extractPointsGrid(cv::Mat &image){
	// EXTRACT THE NORMALIZED GREY IMAGE
	cv::Mat gray;
	if(this->colorspaceCode_!=-1){
		cv::cvtColor(image,image,this->invColorspaceCode_);
	}
	cv::cvtColor(image,gray,CV_BGR2GRAY);

	// EXTRACT MAXIMALLY STABLE BLOBS
	std::vector<std::vector<cv::Point> > msers;
	cv::MSER aMSER;
	aMSER(gray,msers,cv::Mat());
	unsigned msersNo = 0;
	for(std::size_t x=0;x<msers.size();++x){
		msersNo += msers[x].size();
	}

	// NICE FEATURE POINTS
	std::vector<cv::Point2f> corners;
	cv::Ptr<cv::FeatureDetector> detector = new cv::GoodFeaturesToTrackDetector\
		(5000,0.00001,1.0,3.0);
	cv::GridAdaptedFeatureDetector gafd(detector,5000,10,10);
	std::vector<cv::KeyPoint> keys;
	std::deque<unsigned> indices;
	gafd.detect(gray,keys);

	// WRITE THE MSERS LOCATIONS IN THE MATRIX
	cv::Mat result = cv::Mat::zeros(cv::Size(2,msersNo+keys.size()),CV_32FC1);
	unsigned counts = 0;
	for(std::size_t x=0;x<msers.size();++x){
		for(std::size_t y=0;y<msers[x].size();++y){
			result.at<float>(counts,0) = msers[x][y].x;
			result.at<float>(counts,1) = msers[x][y].y;
			++counts;
		}
	}

	// WRITE THE KEYPOINTS IN THE MATRIX
	for(std::size_t i=0;i<keys.size();++i){
		result.at<float>(counts+i,0) = keys[i].pt.x;
		result.at<float>(counts+i,1) = keys[i].pt.y;
	}

	if(this->plot_){
		for(int y=0;y<result.rows;++y){
			cv::Scalar color;
			if(y<msersNo){
				color = cv::Scalar(0,0,255);
			}else{
				color = cv::Scalar(255,0,0);
			}
			float ptX = result.at<float>(y,0);
			float ptY = result.at<float>(y,1);
			cv::circle(image,cv::Point2f(ptX,ptY),3,color);
		}
		cv::imshow("IPOINTS",image);
		cv::waitKey(5);
	}
	result.convertTo(result,CV_32FC1);
	gray.release();
	return result;
}
//==============================================================================
/** Extract edges from the whole image.
 */
cv::Mat FeatureExtractor::extractEdges(cv::Mat &image){
	cv::Mat gray,result;
	if(this->colorspaceCode_!=-1){
		cv::cvtColor(image,image,this->invColorspaceCode_);
	}
	IplImage *im,*res;
	im = Auxiliary::mat2ipl(image);
	Auxiliary::myCvtColor(im,res,-1,true,false);
	image = Auxiliary::ipl2mat(res);
	cv::cvtColor(image,gray,CV_BGR2GRAY);
	Auxiliary::mean0Variance1(gray);
	gray *= 255;
	gray.convertTo(gray,CV_8UC1);
	cv::blur(gray,gray,cv::Size(3,3));
	cv::Canny(gray,result,250,150,3,true);
	if(this->plot_){
		cv::imshow("gray",gray);
		cv::imshow("edges",result);
		cv::waitKey(5);
	}
	gray.release();
	result.convertTo(result,CV_32FC1);
	return result;
}
//==============================================================================
/** Extracts all the surf descriptors from the whole image and writes them in a
 * matrix.
 */
cv::Mat FeatureExtractor::extractSURF(cv::Mat &image){
	std::vector<float> descriptors;
	std::vector<cv::KeyPoint> keypoints;
	cv::SURF aSURF = cv::SURF(0.01,4,2,false);

	// EXTRACT INTEREST POINTS FROM THE IMAGE
	cv::Mat gray,result;
	if(this->colorspaceCode_!=-1){
		cv::cvtColor(image,image,this->invColorspaceCode_);
	}
	cv::cvtColor(image,gray,CV_BGR2GRAY);
	cv::blur(gray,gray,cv::Size(3,3));
	aSURF(gray,cv::Mat(),keypoints,descriptors,false);
	gray.release();

	// WRITE ALL THE DESCRIPTORS IN THE STRUCTURE OF KEY-DESCRIPTORS
	std::deque<FeatureExtractor::keyDescr> kD;
	for(std::size_t i=0;i<keypoints.size();++i){
		FeatureExtractor::keyDescr tmp;
		for(int j=0;j<aSURF.descriptorSize();++j){
			tmp.descr_.push_back(descriptors[i*aSURF.descriptorSize()+j]);
		}
		tmp.keys_ = keypoints[i];
		kD.push_back(tmp);
		tmp.descr_.clear();
	}

	// SORT THE REMAINING DESCRIPTORS SO WE DON'T NEED TO DO THAT LATER
	std::sort(kD.begin(),kD.end(),(&FeatureExtractor::compareDescriptors));

	// WRITE THEM IN THE MATRIX AS FOLLOWS (ADD x,y COORD ON THE LAST 2 COLS):
	result = cv::Mat::zeros(cv::Size(aSURF.descriptorSize()+2,kD.size()),CV_32FC1);
	for(std::size_t i=0;i<kD.size();++i){
		if(i<10){
			std::cout<<kD[i].keys_.response<<std::endl;
		}
		result.at<float>(i,aSURF.descriptorSize()) = kD[i].keys_.pt.x;
		result.at<float>(i,aSURF.descriptorSize()+1) = kD[i].keys_.pt.y;
		for(int j=0;j<aSURF.descriptorSize();++j){
			result.at<float>(i,j) = kD[i].descr_[j];
		}
	}

	// PLOT THE KEYPOINTS TO SEE THEM
	if(this->plot_){
		for(std::size_t i=0;i<kD.size();++i){
			cv::circle(image,cv::Point2f(kD[i].keys_.pt.x,kD[i].keys_.pt.y),\
				3,cv::Scalar(0,0,255));
		}
		cv::imshow("SURFS",image);
		cv::waitKey(5);
	}
	result.convertTo(result,CV_32FC1);
	return result;
}
//==============================================================================
/** Convolves the whole image with some Gabors wavelets and then stores the
 * results.
 */
cv::Mat FeatureExtractor::extractGabor(cv::Mat &image){
	// DEFINE THE PARAMETERS FOR A FEW GABORS
	// params[0] -- sigma: (3,68) // the actual size
	// params[1] -- gamma: (0.2,1) // how round the filter is
	// params[2] -- dimension: (1,10) // size
	// params[3] -- theta: (0,180) or (-90,90) // angle
	// params[4] -- lambda: (2,255) // thickness
	// params[5] -- psi: (0,180) // number of lines
	std::deque<float*> allParams;
	float *params1 = new float[6];
	params1[0] = 1.0;params1[1] = 0.9;params1[2] = 2.0;
	params1[3] = M_PI/6.0;params1[4] = 200.0;params1[5] = 50.0;
	allParams.push_back(params1);
	float *params2 = new float[6];
	params2[0] = 1.0;params2[1] = 0.9;params2[2] = 2.0;
	params2[3] = M_PI/3.0;params2[4] = 200.0;params2[5] = 50.0;
	allParams.push_back(params2);
	float *params3 = new float[6];
	params3[0] = 1.0;params3[1] = 0.9;params3[2] = 2.0;
	params3[3] = 2.0*M_PI/3.0;params3[4] = 200.0;params3[5] = 50.0;
	allParams.push_back(params3);
	float *params4 = new float[6];
	params4[0] = 1.0;params4[1] = 0.9;params4[2] = 2.0;
	params4[3] = 5.0*M_PI/6.0;params4[4] = 200.0;params4[5] = 50.0;
	allParams.push_back(params4);

	// AND SOME BIGGER ONES
	float *params5 = new float[6];
	params5[0] = 3.0;params5[1] = 0.9;params5[2] = 2.0;
	params5[3] = M_PI/6.0;params5[4] = 200.0;params5[5] = 50.0;
	allParams.push_back(params5);
	float *params6 = new float[6];
	params6[0] = 3.0;params6[1] = 0.9;params6[2] = 2.0;
	params6[3] = M_PI/3.0;params6[4] = 200.0;params6[5] = 50.0;
	allParams.push_back(params6);
	float *params7 = new float[6];
	params7[0] = 3.0;params7[1] = 0.9;params7[2] = 2.0;
	params7[3] = 2.0*M_PI/3.0;params7[4] = 200.0;params7[5] = 50.0;
	allParams.push_back(params7);
	float *params8 = new float[6];
	params8[0] = 3.0;params8[1] = 0.9;params8[2] = 2.0;
	params8[3] = 5.0*M_PI/6.0;params8[4] = 200.0;params8[5] = 50.0;
	allParams.push_back(params8);

	// CONVERT THE IMAGE TO GRAYSCALE TO APPLY THE FILTER
	cv::Mat gray,result;
	if(this->colorspaceCode_!=-1){
		cv::cvtColor(image,image,this->invColorspaceCode_);
	}
	cv::cvtColor(image,gray,CV_BGR2GRAY);
	cv::blur(gray,gray,cv::Size(3,3));

	// CREATE EACH GABOR AND CONVOLVE THE IMAGE WITH IT
	result = cv::Mat::zeros(cv::Size(image.cols,image.rows*allParams.size()),\
		CV_32FC1);
	for(unsigned i=0;i<allParams.size();++i){
		cv::Mat response,agabor;

		this->createGabor(agabor,allParams[i]);
		cv::filter2D(gray,response,-1,agabor,cv::Point2f(-1,-1),0,\
			cv::BORDER_REPLICATE);
		// IF WE WANT TO SEE THE GABOR AND THE RESPONSE
		if(this->plot_){
			cv::imshow("Gray",gray);
			cv::imshow("GaborFilter",agabor);
			cv::imshow("GaborResponse",response);
			cv::waitKey(5);
		}
		cv::Mat temp = result.rowRange(i*response.rows,(i+1)*response.rows);
		response.convertTo(response,CV_32FC1);
		response.copyTo(temp);
		response.release();
		agabor.release();
		temp.release();
	}
	for(unsigned i=0;i<allParams.size();++i){
		delete [] allParams[i];
	}
	allParams.clear();
	gray.release();
	result.convertTo(result,CV_32FC1);
	return result;
}
//==============================================================================
/** Extracts SIFT features from the image and stores them in a matrix.
 */
cv::Mat FeatureExtractor::extractSIFT(cv::Mat &image,\
const std::vector<cv::Point2f> &templ,const cv::Rect &roi){
	// DEFINE THE SURF KEYPOINTS AND THE DESCRIPTORS
	std::vector<cv::KeyPoint> keypoints;
	cv::SIFT::DetectorParams detectP  = cv::SIFT::DetectorParams(0.0001,10.0);
	cv::SIFT::DescriptorParams descrP = cv::SIFT::DescriptorParams();
	cv::SIFT::CommonParams commonP    = cv::SIFT::CommonParams();
	cv::SIFT aSIFT(commonP,detectP,descrP);

	// EXTRACT SIFT FEATURES IN THE IMAGE
	cv::Mat gray, result;
	if(this->colorspaceCode_!=-1){
		cv::cvtColor(image,image,this->invColorspaceCode_);
	}
	cv::cvtColor(image,gray,CV_BGR2GRAY);
	cv::blur(gray,gray,cv::Size(3,3));
	aSIFT(gray,cv::Mat(),keypoints);

	// WE USE THE SAME FUNCTION TO BUILD THE DICTIONARY ALSO
	if(FeatureExtractor::isFeatureIn(this->featureType_,FeatureExtractor::SIFT_DICT)){
		cv::Mat preResult = cv::Mat::zeros(keypoints.size(),aSIFT.descriptorSize(),CV_32FC1);
		std::vector<cv::KeyPoint> goodKP;
		result = cv::Mat::zeros(cv::Size(preResult.cols+2,result.rows),CV_32FC1);
		for(std::size_t i=0;i<keypoints.size();++i){
			if(FeatureExtractor::isInTemplate(keypoints[i].pt.x+roi.x,\
			keypoints[i].pt.y+roi.y,templ)){
				goodKP.push_back(keypoints[i]);
				result.at<float>(goodKP.size()-1,result.cols-2) = keypoints[i].pt.x;
				result.at<float>(goodKP.size()-1,result.cols-1) = keypoints[i].pt.y;
			}
		}
		aSIFT(gray,cv::Mat(),goodKP,preResult,true);
		cv::Mat dummy =	result.colRange(0,preResult.cols);
		preResult.copyTo(dummy);
		preResult.release();
		if(this->plot_){
			for(std::size_t i=0;i<goodKP.size();++i){
				cv::circle(image,goodKP[i].pt,3,cv::Scalar(0,0,255));
			}
			cv::imshow("SIFT_DICT",image);
			cv::waitKey(5);
		}
	// IF WE ONLY WANT TO STORE THE SIFT FEATURE WE NEED TO ADD THE x-S AND y-S
	}else if(FeatureExtractor::isFeatureIn(this->featureType_,FeatureExtractor::SIFT)){
		result = cv::Mat::zeros(keypoints.size(),aSIFT.descriptorSize()+4,CV_32FC1);
		cv::Mat dummy1 = result.colRange(0,aSIFT.descriptorSize());
		cv::Mat dummy2;
		aSIFT(gray,cv::Mat(),keypoints,dummy2,true);
		dummy2.convertTo(dummy2,CV_32FC1);
		dummy2.copyTo(dummy1);
		dummy1.release();
		dummy2.release();
		for(std::size_t i=0;i<keypoints.size();++i){
			result.at<float>(i,result.cols-2) = keypoints[i].pt.x;
			result.at<float>(i,result.cols-1) = keypoints[i].pt.y;
		}
		// PLOT THE KEYPOINTS TO SEE THEM
		if(this->plot_){
			for(std::size_t i=0;i<keypoints.size();++i){
				cv::circle(image,keypoints[i].pt,3,cv::Scalar(0,0,255));
			}
			cv::imshow("SIFT",image);
			cv::waitKey(5);
		}
	}

	// NORMALIZE THE FEATURE
	result.convertTo(result,CV_32FC1);
	for(int i=0;i<result.rows;++i){
		cv::Mat rowsI = result.row(i);
		rowsI         = rowsI/cv::norm(rowsI);
		rowsI.release();

		// IF WE WANT TO STORE THE SIFT FEATURES THEN, WE NEED TO STORE x AND y
		if(!FeatureExtractor::isFeatureIn(this->featureType_,FeatureExtractor::SIFT)){
			result.at<float>(i,aSIFT.descriptorSize())   = keypoints[i].pt.x;
			result.at<float>(i,aSIFT.descriptorSize()+1) = keypoints[i].pt.y;
		}
	}
	gray.release();
	return result;
}
//==============================================================================
/** Returns the row corresponding to the indicated feature type.
 */
cv::Mat FeatureExtractor::getDataRow(int imageRows,\
const FeatureExtractor::templ &aTempl,const cv::Rect &roi,\
const FeatureExtractor::people &person,const std::string &imgName,\
cv::Point2f &absRotCenter,cv::Point2f &rotBorders,float rotAngle,bool flip,\
std::vector<cv::Point2f> &keys){
	cv::Mat feature,result,dictImg;
	std::string toRead;
	std::cout<<"Image class (CLOSE/MEDIUM/FAR): "<<this->imageClass_<<std::endl;
	for(FeatureExtractor::FEATURE f=FeatureExtractor::EDGES;\
	f<=FeatureExtractor::SKIN_BINS;++f){
		if(!FeatureExtractor::isFeatureIn(this->featureType_,f)){continue;}
		cv::Mat dummy;
		switch(f){
			case (FeatureExtractor::IPOINTS):
				toRead = (this->featureFile_+"IPOINTS/"+imgName+".bin");
				Auxiliary::binFile2mat(feature,const_cast<char*>(toRead.c_str()));
				this->rotate2Zero(rotAngle,FeatureExtractor::KEYS,roi,absRotCenter,\
					rotBorders,keys,feature);
				dummy = this->getPointsGrid(flip,feature,roi,aTempl,person.pixels_);
				break;
			case FeatureExtractor::EDGES:
				toRead = (this->featureFile_+"EDGES/"+imgName+".bin");
				Auxiliary::binFile2mat(feature,const_cast<char*>(toRead.c_str()));
				dummy = this->getEdges(flip,feature,person.thresh_,roi,aTempl,rotAngle);
				break;
			case FeatureExtractor::SURF:
				toRead = (this->featureFile_+"SURF/"+imgName+".bin");
				Auxiliary::binFile2mat(feature,const_cast<char*>(toRead.c_str()));
				this->rotate2Zero(rotAngle,FeatureExtractor::KEYS,roi,absRotCenter,\
					rotBorders,keys,feature);
				dummy = this->getSURF(flip,feature,aTempl.points_,roi,person.pixels_,keys);
				break;
			case FeatureExtractor::GABOR:
				toRead = (this->featureFile_+"GABOR/"+imgName+".bin");
				Auxiliary::binFile2mat(feature,const_cast<char*>(toRead.c_str()));
				dummy = this->getGabor(flip,feature,person.thresh_,roi,\
					person.pixels_.size(),aTempl,rotAngle,imageRows);
				break;
			case FeatureExtractor::SIFT_DICT:
				person.pixels_.copyTo(dictImg);
				dummy = this->extractSIFT(dictImg,aTempl.points_,roi);
				break;
			case FeatureExtractor::SIFT:
				toRead = (this->featureFile_+"SIFT/"+imgName+".bin");
				Auxiliary::binFile2mat(feature,const_cast<char*>(toRead.c_str()));
				this->rotate2Zero(rotAngle,FeatureExtractor::KEYS,roi,absRotCenter,\
					rotBorders,keys,feature);
				dummy = this->getSIFT(flip,feature,aTempl.points_,roi,person.pixels_,keys);
				break;
			case FeatureExtractor::TEMPL_MATCHES:
				// NO NEED TO STORE ANY FEATURE,ONLY THE PIXEL VALUES ARE NEEDED
				dummy = this->getTemplMatches(flip,person,aTempl,roi);
				break;
			case FeatureExtractor::HOG:
				// CAN ONLY EXTRACT THEM OVER AN IMAGE SO NO FEATURES CAN BE STORED
				dummy = this->getHOG(flip,person,aTempl,roi);
				break;
			case FeatureExtractor::RAW_PIXELS:
				// NO NEED TO STORE ANY FEATURE,ONLY THE PIXEL VALUES ARE NEEDED
				dummy = this->getRawPixels(flip,person,aTempl,roi);
				break;
			case FeatureExtractor::SKIN_BINS:
				// NO NEED TO STORE ANY FEATURE,ONLY THE PIXEL VALUES ARE NEEDED
				dummy = this->getSkinBins(flip,person,aTempl,roi);
				break;
			default:
				std::cerr<<"Feature type "<<f<<" unknown"<<std::endl;
				std::abort();
				break;
		}
		cv::Mat tmp = cv::Mat::zeros(cv::Size(result.cols+dummy.cols,1),CV_32FC1);
		cv::Mat dum = tmp.colRange(result.cols,result.cols+dummy.cols);
		dummy.copyTo(dum);
		result.release();
		tmp.copyTo(result);
		tmp.release();
		dum.release();
		dummy.release();
	}
	dictImg.release();
	if(!feature.empty()){
		feature.release();
	}
	result.convertTo(result,CV_32FC1);
	return result;
}
//==============================================================================
/** Gets the HOG descriptors over an image.
 */
cv::Mat FeatureExtractor::getHOG(bool flip,\
const FeatureExtractor::people &person,const FeatureExtractor::templ &aTempl,\
const cv::Rect &roi){
	// JUST COPY THE PIXELS THAT ARE LARGER THAN 0 INTO
	cv::Rect cutROI;
	if(!person.thresh_.empty()){
		this->getThresholdBorderes(cutROI.x,cutROI.width,cutROI.y,cutROI.height,\
			person.thresh_);
		cutROI.width   -= cutROI.x;
		cutROI.height  -= cutROI.y;
	}else{
		cutROI.x      = aTempl.extremes_[0]-roi.x;
		cutROI.y      = aTempl.extremes_[2]-roi.y;
		cutROI.width  = aTempl.extremes_[1]-aTempl.extremes_[0];
		cutROI.height = aTempl.extremes_[3]-aTempl.extremes_[2];
	}
	cv::Mat large = this->cutAndResizeImage(cutROI,person.pixels_);
	if(flip){cv::flip(large,large,1);}
	if(this->colorspaceCode_!=-1){
		cv::cvtColor(large,large,this->invColorspaceCode_);
	}
	cv::Mat gray;
	cv::cvtColor(large,gray,CV_BGR2GRAY);
	//cv::equalizeHist(gray,gray);
	large.release();
	cv::blur(gray,gray,cv::Size(3,3));
	unsigned stepX = (gray.cols%10==0?10:(10+gray.cols%10));
	unsigned stepY = (gray.rows%10==0?10:(10+gray.rows%10));
	cv::HOGDescriptor hogD(gray.size(),cv::Size(stepX,stepY),cv::Size(10,10),\
		cv::Size(stepX,stepY),15,1,-1,cv::HOGDescriptor::L2Hys,0.2,true);
	std::vector<float> descriptors;
	if(this->plot_){
		cv::imshow("image4HOG",gray);
		cv::waitKey(5);
	}
	hogD.compute(gray,descriptors,cv::Size(stepX,stepY),cv::Size(0,0),\
		std::vector<cv::Point>());
	cv::Mat dummy1(descriptors);
	cv::Mat result = cv::Mat::zeros(cv::Size(dummy1.cols,dummy1.rows+2),\
		CV_32FC1);
	cv::Mat dummy2 = result.colRange(0,dummy1.cols);
	dummy1.copyTo(dummy2);
	dummy1.release();
	dummy2.release();
	result.convertTo(result,CV_32FC1);

	std::cout<<"In HOG: descriptorSize = "<<descriptors.size()<<std::endl;
	if(this->print_){
		unsigned counts = 0;
		std::cout<<"Size(HOG): "<<result.size()<<std::endl;
		for(int i=0;i<result.rows,counts<10;++i){
			if(result.at<float>(i,0)!=0){
				std::cout<<result.at<float>(i,0)<<" ";
				++counts;
			}
		}
		std::cout<<"..."<<std::endl;
	}
	gray.release();
	return result.t();
}
//==============================================================================
/**Return number of means.
 */
unsigned FeatureExtractor::readNoMeans(){
	return this->noMeans_;
}
//==============================================================================
/**Return name of the SIFT dictionary.
 */
std::string FeatureExtractor::readDictName(){
	return this->dictFilename_;
}
//==============================================================================
/** Sets the image class and resets the dictionary name.
 */
unsigned FeatureExtractor::setImageClass(unsigned aClass){
	std::deque<std::string> names;
	names.push_back("CLOSE");names.push_back("MEDIUM");	names.push_back("FAR");
	this->imageClass_ = names[aClass];

	// WE NEED A SIFT DICTIONARY FOR EACH CLASS
	if(FeatureExtractor::isFeatureIn(this->featureType_,FeatureExtractor::SIFT) || \
	FeatureExtractor::isFeatureIn(this->featureType_,FeatureExtractor::SIFT_DICT)){
		std::string dictName = this->imageClass_+"_SIFT"+".bin";
		this->initSIFT(dictName);
	}
}
//==============================================================================
/** Cut the image around the template or bg bordered depending on which is used
 * and resize to a common size.
 */
cv::Mat FeatureExtractor::cutAndResizeImage(const cv::Rect &roiCut,\
const cv::Mat &img){
	cv::Mat tmp(img.clone(),roiCut),large;
	cv::Size aSize;
	if(this->bodyPart_ == FeatureExtractor::HEAD){
		aSize = cv::Size(10,10);
		cv::blur(tmp,tmp,cv::Size(3,3));
		cv::resize(tmp,large,aSize,0,0,cv::INTER_NEAREST);
		tmp.release();
		return large;
	}else{
		unsigned sHeight = 10;
		if(this->imageClass_=="CLOSE"){
			aSize.height = sHeight;
			aSize.width  = aSize.height*6/5;
		}else if(this->imageClass_=="FAR"){
			if(this->bodyPart_ == FeatureExtractor::WHOLE){
				aSize.height = 3*sHeight;
			}else{
				aSize.height = 3/2*sHeight;
			}
			aSize.width  = 3/2*sHeight;
		}else if(this->imageClass_=="MEDIUM"){
			if(this->bodyPart_ == FeatureExtractor::WHOLE){
				aSize.height = 2*sHeight;
			}else{
				aSize.height = sHeight;
			}
			aSize.width  = 3/2*sHeight;
		}
		cv::blur(tmp,tmp,cv::Size(3,3));
		cv::resize(tmp,large,aSize,0,0,cv::INTER_NEAREST);
		tmp.release();
		return large;
	}
}
//==============================================================================






