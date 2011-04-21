/* peopleDetector.cpp
 * Author: Silvia-Laura Pintea
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
#include <exception>
#include <opencv2/opencv.hpp>
#include "eigenbackground/src/Tracker.hh"
#include "eigenbackground/src/Helpers.hh"
#include "eigenbackground/src/defines.hh"
#include "featureExtractor.h"
#include "annotationsHandle.h"
#include "peopleDetector.h"
//======================================================================
peopleDetector::peopleDetector(int argc,char** argv, bool extract=false,\
bool buildBg=false):Tracker(argc, argv, 10, buildBg, true){
	if(argc == 3){
		std::string dataPath  = std::string(argv[1]);
		std::string imgString = std::string(argv[2]);
		if(datasetPath[dataPath.size()-1]!='/'){
			dataPath += '/';
		}
		this->plot           = true;
		this->print          = true;
		this->useGroundTruth = true;
		this->lastIndex      = 0;
		this->producer       = NULL;
		this->colorspaceCode = CV_BGR2Lab;
		this->featurePart    = peopleDetector::WHOLE;
		this->tracking 	     = 0;
		this->onlyExtract    = extract;
		this->datasetPath    = dataPath;
		this->imageString    = imgString;
		this->extractor      = new featureExtractor();
	}else{
		std::cerr<"Wrong number of arguments: command datasetPath/"<<\
			" imageString"<<std::endl;
		exit(1);
	}
}
//==============================================================================
virtual peopleDetector::~peopleDetector(){
	delete this->extractor;
	if(this->producer){
		delete this->producer;
		this->producer = NULL;
	}
	if(!this->targets.empty()){
		this->targets.release();
	}
	if(!this->data.empty()){
		this->data.release();
	}
	if(!this->targetAnno.empty()){
		this->targetAnno.clear();
	}
	if(!this->entireNext.empty()){
		this->entireNext.release();
	}
	this->templates.clear();
}
//==============================================================================
/** Checks to see if a pixel's x coordinate is on a scanline.
 */
struct onScanline{
	public:
		unsigned pixelY;
		onScanline(const unsigned pixelY){this->pixelY=pixelY;}
		bool operator()(const scanline_t line)const{
			return (line.line == this->pixelY);
		}
};
//==============================================================================
/** Checks the image name (used to find the corresponding labels for each image).
 */
struct compareImg{
	public:
		std::string imgName;
		compareImg(std::string image){
			std::deque<std::string> parts = splitLine(\
											const_cast<char*>(image.c_str()),'/');
			this->imgName = parts[parts.size()-1];
		}
		bool operator()(annotationsHandle::FULL_ANNOTATIONS anno)const{
			return (anno.imgFile == this->imgName);
		}
};
//==============================================================================
/** Initializes the parameters of the tracker.
 */
void peopleDetector::init(std::string dataFolder, std::string theAnnotationsFile,\
featureExtractor::FEATURE feat, bool readFromFolder){
	this->initProducer(readFromFolder, const_cast<char*>(dataFolder.c_str()));
	if(!this->entireNext.empty()){
		this->entireNext.release();
	}
	if(!this->targets.empty()){
		this->targets.release();
	}
	if(!this->data.empty()){
		this->data.release();
	}
	if(!this->targetAnno.empty()){
		this->targetAnno.clear();
	}
	// LOAD THE DESIRED ANNOTATIONS FROM THE FILE
	if(!theAnnotationsFile.empty() && this->targetAnno.empty()){
		annotationsHandle::loadAnnotations(const_cast<char*>\
			(theAnnotationsFile.c_str()), this->targetAnno);
	}
	this->lastIndex = 0;
	if(feat == featureExtractor::SIFT_DICT){
		this->tracking = false;
		this->extractor->initSIFT(this->datasetPath+"SIFT_"+this->imageString+".bin");
	}
	this->extractor->init(feat,this->datasetPath+"features/");
	this->templates.clear();
}
//==============================================================================
/** Compares SURF 2 descriptors and returns the boolean value of their comparison.
 */
bool peopleDetector::compareDescriptors(const peopleDetector::keyDescr k1,\
const peopleDetector::keyDescr k2){
	return (k1.keys.response>k2.keys.response);
}
//==============================================================================
/** Get template extremities (if needed, considering some borders --
 * relative to the ROI).
 */
std::deque<double> peopleDetector::templateExtremes(\
std::vector<cv::Point2f> templ, int minX, int minY){
	std::deque<double> extremes(4,0.0);
	extremes[0] = std::max(0.0f, templ[0].x-minX);
	extremes[1] = std::max(0.0f, templ[0].x-minX);
	extremes[2] = std::max(0.0f, templ[0].y-minY);
	extremes[3] = std::max(0.0f, templ[0].y-minY);
	for(std::size_t i=0; i<templ.size();i++){
		if(extremes[0]>=templ[i].x-minX) extremes[0] = templ[i].x - minX;
		if(extremes[1]<=templ[i].x-minX) extremes[1] = templ[i].x - minX;
		if(extremes[2]>=templ[i].y-minY) extremes[2] = templ[i].y - minY;
		if(extremes[3]<=templ[i].y-minY) extremes[3] = templ[i].y - minY;
	}
	return extremes;
}
//==============================================================================
/** Gets the distance to the given template from a given pixel location.
 */
double peopleDetector::getDistToTemplate(int pixelX, int pixelY, \
std::vector<cv::Point2f> templ){
	std::vector<cv::Point2f> hull;
	convexHull(templ, hull);
	double minDist=-1;
	unsigned i=0, j=1;
	while(i<hull.size() && j<hull.size()-1){
		double midX  = (hull[i].x + hull[j].x)/2;
		double midY  = (hull[i].y + hull[j].y)/2;
		double aDist = std::sqrt((midX - pixelX)*(midX - pixelX) + \
						(midY - pixelY)*(midY - pixelY));
		if(minDist == -1 || minDist>aDist){
			minDist = aDist;
		}
		i++; j++;
	}
	return minDist;
}
//==============================================================================
/** Checks to see if a given pixel is inside a template.
 */
bool peopleDetector::isInTemplate(unsigned pixelX, unsigned pixelY,\
std::vector<cv::Point2f> templ){
	std::vector<cv::Point2f> hull;
	convexHull(templ, hull);
	std::deque<scanline_t> lines;
	getScanLines(hull, lines);

	std::deque<scanline_t>::iterator iter = std::find_if(lines.begin(),\
		lines.end(), onScanline(pixelY));
	if(iter == lines.end()){
		return false;
	}

	if(std::abs(static_cast<int>(iter->line)-static_cast<int>(pixelY))<5 &&\
	static_cast<int>(iter->start) <= static_cast<int>(pixelX) &&\
	static_cast<int>(iter->end) >= static_cast<int>(pixelX)){
		return true;
	}else{
		return false;
	}
}
//==============================================================================
/** Returns the size of a window around a template centered in a given point.
 */
void peopleDetector::templateWindow(cv::Size imgSize, int &minX, int &maxX,\
int &minY, int &maxY, peopleDetector::templ aTempl, int tplBorder){
	// TRY TO ADD BORDERS TO MAKE IT 100
	minX = aTempl.extremes[0];
	maxX = aTempl.extremes[1];
	minY = aTempl.extremes[2];
	maxY = aTempl.extremes[3];
	int diffX = (tplBorder - (maxX-minX))/2;
	int diffY = (tplBorder - (maxY-minY))/2;
	minY = std::max(minY-diffY,0);
	maxY = std::min(maxY+diffY,imgSize.height);
	minX = std::max(minX-diffX,0);
	maxX = std::min(maxX+diffX,imgSize.width);

	if(maxX-minX!=tplBorder){
		int diffX2 = tplBorder-(maxX-minX);
		if(minX>diffX2){
			minX -= diffX2;
		}else if(maxX<imgSize.width-diffX2){
			maxX += diffX2;
		}
	}
	if(maxY-minY!=tplBorder){
		int diffY2 = tplBorder-(maxY-minY);
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
int k, cv::Mat thresh, cv::Mat &colorRoi, double tmplHeight){
	// LOOP OVER THE AREA OF OUR TEMPLATE AND THERESHOLD ONLY THOSE PIXELS
	for(unsigned x=0; x<maxX-minX; x++){
		for(unsigned y=0; y<maxY-minY; y++){
			if((int)(thresh.at<uchar>((int)(y+minY),(int)(x+minX)))>0){

				// IF THE PIXEL IS NOT INSIDE OF THE TEMPLATE
				if(!this->isInTemplate((x+minX),(y+minY),this->templates[k].points)\
				&& this->templates.size()>1){
					double minDist = thrsh.rows*thrsh.cols;
					unsigned label = -1;
					for(unsigned l=0; l<this->templates.size(); l++){
						if(k==l){continue;}

						// IF IT IS IN ANOTHER TEMPLATE THEN IGNORE THE PIXEL
						if(this->isInTemplate((x+minX),(y+minY),this->templates[l].points)){
							minDist = 0;label = l;
							break;

						// ELSE COMPUTE THE DISTANCE FROM THE PIXEL TO THE TEMPLATE
						}else{
							double ptDist = dist(cv::Point2f(x+minX,y+minY),\
								this->templates[l].center);
							if(minDist>ptDist){
								minDist = ptDist;label = l;
							}
						}
					}

					// IF THE PIXEL HAS A DIFFERENT LABEL THEN THE CURR TEMPL
					if(label != k || minDist>=tmplHeight){
						colorRoi.at<cv::Vec3b>((int)y,(int)x) = cv::Vec3b(0,0,0);
					}
				}
			}else{
				colorRoi.at<cv::Vec3b>((int)y,(int)x) = cv::Vec3b(0,0,0);
			}
		}
	}
}
//==============================================================================
/** Adds a templates to the vector of templates at detected positions.
 */
void peopleDetector::add2Templates(std::deque<unsigned> existing){
	this->templates.clear();
	for(unsigned i=0; i<existing.size(); i++){
		peopleDetector::templ aTempl;
		aTempl.center = this->cvPoint(existing[i]);
		genTemplate2(aTempl.center, persHeight, camHeight, aTempl.points);
		aTempl.head = cv::Point2f((aTempl.points[12].x+aTempl.points[14].x)/2,\
						(aTempl.points[12].y+aTempl.points[14].y)/2);
		aTempl.extremes = this->templateExtremes(aTempl.points);
		this->templates.push_back(aTempl);
	}
}
//==============================================================================
/** Get the foreground pixels corresponding to each person.
 */
void peopleDetector::allForegroundPixels(std::deque<peopleDetector::people>\
&allPeople, std::deque<unsigned> existing, IplImage *bg, double threshold){
	// INITIALIZING STUFF
	cv::Mat thsh, thrsh;
	if(bg){
		thsh  = cv::Mat(bg);
		thrsh = cv::Mat(thsh.rows, thsh.cols, CV_8UC1);
		cv::cvtColor(thsh, thrsh, CV_BGR2GRAY);
		cv::threshold(thrsh, thrsh, threshold, 255, cv::THRESH_BINARY);
	}
	cv::Mat foregr(this->current->img);

	// FOR EACH EXISTING TEMPLATE LOOK ON AN AREA OF 100 PIXELS AROUND IT
	for(unsigned k=0; k<existing.size();k++){
		cv::Point2f center       = this->cvPoint(existing[k]);
		allPeople[k].absoluteLoc = center;
		double tmplHeight = dist(this->templates[k].head,this->templates[k].head);
		double tmplArea   = tmplHeight*dist(this->templates[k].points[0],\
							this->templates[k].points[1]);

		// GET THE 100X100 WINDOW ON THE TEMPLATE
		int minY=foregr.rows, maxY=0, minX=foregr.cols, maxX=0;
		this->templateWindow(cv::Size(foregr.cols,foregr.rows),minX, maxX,\
				minY, maxY, this->templates[k]);
		int width  = maxX-minX;
		int height = maxY-minY;
		cv::Mat colorRoi = cv::Mat(foregr.clone(),cv::Rect(cv::Point2f(minX,minY),\
							cv::Size(width,height)));

		//IF THERE IS NO BACKGROUND THEN JUST COPY THE ROI
		cv::Mat thrshRoi;
		if(!bg){
			colorRoi.copyTo(allPeople[k].pixels);
		}else{
		// FOR MULTIPLE DISCONNECTED BLOBS KEEP THE CLOSEST TO CENTER
			this->pixels2Templates(maxX,minX,maxY,minY,k,thrsh,colorRoi,tmplHeight);
			cv::cvtColor(colorRoi, thrshRoi, CV_BGR2GRAY);
			this->keepLargestBlob(thrshRoi,cv::Point2f(center.x-minX,center.y-minY),
					tmplArea);
			colorRoi.copyTo(allPeople[k].pixels,thrshRoi);
		}
		// SAVE IT IN THE STRUCTURE OF FOREGOUND IMAGES
		allPeople[k].relativeLoc = cv::Point2f(center.x-(minX),center.y-(minY));
		allPeople[k].borders.assign(4,0);
		allPeople[k].borders[0] = minX;
		allPeople[k].borders[1] = maxX;
		allPeople[k].borders[2] = minY;
		allPeople[k].borders[3] = maxY;
		if(this->plot){
			cv::imshow("people", allPeople[k].pixels);
			cv::imshow("threshold", thrshRoi);
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
void peopleDetector::keepLargestBlob(cv::Mat &thresh,cv::Point2f center,\
double tmplArea){
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(thresh,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
	cv::drawContours(thresh,contours,-1,cv::Scalar(255,255,255),CV_FILLED);

	int contourIdx =-1;
	double minDist = thresh.cols*thresh.rows;
	for(size_t i=0; i<contours.size(); i++){
		unsigned minX=contours[i][0].x,maxX=contours[i][0].x,minY=contours[i][0].y,\
			maxY=contours[i][0].y;
		for(size_t j=1; j<contours[i].size(); j++){
			if(minX>=contours[i][j].x){minX = contours[i][j].x;}
			if(maxX<contours[i][j].x){maxX = contours[i][j].x;}
			if(minY>=contours[i][j].y){minY = contours[i][j].y;}
			if(maxY<contours[i][j].y){maxY = contours[i][j].y;}
		}
		double ptDist = dist(center,cv::Point2f((maxX+minX)/2,(maxY+minY)/2));
		double area   = (maxX-minX)*(maxY-minY);
		if(ptDist<minDist & area>=tmplArea){
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
/** Creates on data row in the final data matrix by getting the feature
 * descriptors.
 */
void peopleDetector::extractDataRow(std::deque<unsigned> &existing, IplImage *bg){
	// FIX THE LABELS TO CORRESPOND TO THE PEOPLE DETECTED IN THE IMAGE
	if(!this->targetAnno.empty() && !this->useGroundTruth){
		existing = this->fixLabels(existing);
	}

	// PROCESS THE REST OF THE IMAGES
	this->lastIndex = this->data.rows;
	cv::Mat image(this->current->img);
	cv::cvtColor(image, image, this->colorspaceCode);
	if(this->tracking && this->current->index+1<this->producer->filelist.size()){
		this->entireNext = cv::imread(this->producer->filelist\
							[this->current->index+1].c_str());
		cv::cvtColor(this->entireNext, this->entireNext, this->colorspaceCode);
	}

	// REDUCE THE IMAGE TO ONLY THE INTERESTING AREA
	this->add2Templates(existing);
	std::deque<peopleDetector::people> allPeople(existing.size(),\
		peopleDetector::people());
	this->allForegroundPixels(allPeople, existing, bg, 7.0);

	//GET ONLY THE IMAGE NAME OUT THE CURRENT IMAGE'S NAME
	unsigned pos1       = (this->current->sourceName).find_last_of("/\\");
	std::string imgName = (this->current->sourceName).substr(pos1+1);
	unsigned pos2       = imgName.find_last_of(".");
	imgName             = imgName.substr(0,pos2);

	// FOR EACH LOCATION IN THE IMAGE EXTRACT FEATURES, FILTER THEM AND RESHAPE
	for(std::size_t i=0; i<existing.size(); i++){
		// DEFINE THE IMAGE ROI OF THE SAME SIZE AS THE FOREGROUND
		cv::Rect roi(allPeople[i].borders[0],allPeople[i].borders[2],\
			allPeople[i].borders[1]-allPeople[i].borders[0],\
			allPeople[i].borders[3]-allPeople[i].borders[2]);

		// ROTATE THE FOREGROUND PIXELS, THREHSOLD AND THE TEMPLATE
		cv::Mat thresholded;
		cv::Point2f rotBorders;
		cv::Point2f absRotCenter(allPeople[i].pixels.cols/2.0+allPeople[i].borders[0],\
			allPeople[i].pixels.rows/2.0+allPeople[i].borders[2]);
		allPeople[i].pixels = peopleDetector::rotate2Zero(head,center,\
								allPeople[i].pixels,rotBorders);
		this->templates[i].points = peopleDetector::rotatePoints2Zero(head,center,\
									this->templates[i].points,rotBorders,absRotCenter);
		this->templates[i].extremes = this->templateExtremes(this->templates[i].points);

		// IF WE CAN THRESHOLD THE IMAGE USING THE BACKGROUND MODEL
		if(bg){
			cv::inRange(allPeople[i].pixels,cv::Scalar(1,1,1),cv::Scalar(255,225,225),\
					thresholded);
			cv::dilate(thresholded,thresholded,cv::Mat());
		}

		// IF THE PART TO BE CONSIDERED IS ONLY FEET OR ONLY HEAD
		if(this->featurePart != ' '){
			this->templatePart(thresholded,i,allPeople[i].borders[0],\
				allPeople[i].borders[2]);
		}

		// ANF FINALLY EXTRACT THE DATA ROW
		cv::vector<cv::Point2f> keys;
		cv::Mat dataRow = this->extractor->getDataRow(this->templates[i],roi,\
							allPeople[i],thresholded,keys,imgName,absRotCenter,\
							rotBorders);
		if(this->tracking && !this->entireNext.empty()){
			dataRow.at<double>(0,dataRow.cols-1) = this->motionVector(\
				this->templates[i].head,this->templates[i].center);
			cv::Mat nextImg(this->entireNext,roi);
			nextImg = peopleDetector::rotate2Zero(this->templates[i].head,\
						this->templates[i].center,nextImg.clone(),rotBorders);
			dataRow.at<double>(0,dataRow.cols-2) = \
				this->opticalFlow(feature,allPeople[i].pixels,nextImg,keys,\
				this->templates[i].head,this->templates[i].center,false);
			nextImg.release();
		}
		dataRow.convertTo(dataRow, cv::DataType<double>::type);
		normalizeMat(dataRow);
		if(this->data.empty()){
			dataRow.copyTo(this->data);
		}else{
			this->data.push_back(dataRow);
		}
		thresholded.release();
		dataRow.release();
	}
}
//==============================================================================
/** Compute the dominant direction of the SIFT or SURF features.
 */
double peopleDetector::opticalFlow(cv::Mat keys, cv::Mat currentImg,\
cv::Mat nextImg,std::vector<cv::Point2f> keyPts,cv::Point2f head,\
cv::Point2f center,bool maxOrAvg){
	// GET THE OPTICAL FLOW MATRIX FROM THE FEATURES
	double direction = -1;
	cv::Mat currGray, nextGray;
	cv::cvtColor(currentImg,currGray,CV_RGB2GRAY);
	cv::cvtColor(nextImg,nextGray,CV_RGB2GRAY);
	std::vector<cv::Point2f> flow;
	std::vector<uchar> status;
	std::vector<float> error;
	cv::calcOpticalFlowPyrLK(currGray,nextGray,keyPts,flow,status,error,\
		cv::Size(15,15),3,cv::TermCriteria(cv::TermCriteria::COUNT+\
		cv::TermCriteria::EPS,30,0.01),0.5,0);

	//GET THE DIRECTION OF THE MAXIMUM OPTICAL FLOW
	double flowX = 0.0, flowY = 0.0, resFlowX = 0.0, resFlowY = 0.0, magni = 0.0;
	for(std::size_t i=0;i<flow.size();i++){
		double ix,iy,fx,fy;
		ix = keyPts[i].x; iy = keyPts[i].y; fx = flow[i].x; fy = flow[i].y;

		// IF WE WANT THE MAXIMUM OPTICAL FLOW
		if(maxOrAvg){
			double newMagni = std::sqrt((fx-ix)*(fx-ix) + (fy-iy)*(fy-iy));
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
		flowX /= keyPts.size(); flowY /= keyPts.size();
		resFlowX /= keyPts.size(); resFlowY /= keyPts.size();
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
bool peopleDetector::canBeAssigned(unsigned l,std::deque<double> &minDistances,\
unsigned k,double distance, std::deque<int> &assignment){
	unsigned isThere = 1;
	while(isThere){
		isThere = 0;
		for(std::size_t i=0; i<assignment.size(); i++){
			// IF THERE IS ANOTHER ASSIGNMENT FOR K WITH A LARGER DIST
			if(assignment[i] == k && i!=l && minDistances[i]>distance){
				assignment[i]   = -1;
				minDistances[i] = (double)INFINITY;
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
/** Rotate the points wrt to the camera location.
 */
std::vector<cv::Point2f> peopleDetector::rotatePoints2Zero(cv::Point2f \
headLocation, cv::Point2f feetLocation, std::vector<cv::Point2f> pts,\
cv::Point2f rotBorders, cv::Point2f rotCenter){
	// GET THE ANGLE WITH WHICH WE NEED TO ROTATE
	double rotAngle = std::atan2((headLocation.y-feetLocation.y),\
						(headLocation.x-feetLocation.x));
	rotAngle -= M_PI;
	angle0to360(rotAngle);

	// THE ROTATION ANGLE NEEDS TO BE IN DEGREES
	rotAngle *= (180.0/M_PI);

	// GET THE ROTATION MATRIX WITH RESPECT TO THE GIVEN CENTER
	cv::Mat rotationMat = cv::getRotationMatrix2D(rotCenter, rotAngle, 1.0);

	// BUILD A MATRIX OUT OF THE TEMPLATE POINTS
	cv::Mat toRotate = cv::Mat::ones(cv::Size(3, pts.size()),\
						cv::DataType<double>::type);
	for(std::size_t i=0; i<pts.size(); i++){
		toRotate.at<double>(i,0) = pts[i].x + rotBorders.x;
		toRotate.at<double>(i,1) = pts[i].y + rotBorders.y;
	}

	// MULTIPLY THE TEMPLATE MATRIX WITH THE ROTATION MATRIX
	toRotate.convertTo(toRotate, cv::DataType<double>::type);
	cv::Mat rotated = toRotate*rotationMat.t();
	rotated.convertTo(rotated, cv::DataType<double>::type);

	// COPY THE RESULT BACK INTO A TEMPLATE SHAPE
	std::vector<cv::Point2f> newPts(rotated.rows);
	for(int y=0; y<rotated.rows; y++){
		newPts[y].x = rotated.at<double>(y,0);
		newPts[y].y = rotated.at<double>(y,1);
	}
	rotationMat.release();
	toRotate.release();
	rotated.release();
	return newPts;
}
//==============================================================================
/** Rotate the keypoints wrt to the camera location.
 */
void peopleDetector::rotateKeypts2Zero(cv::Point2f headLocation, cv::Point2f \
feetLocation, cv::Mat &keys, cv::Point2f rotBorders, cv::Point2f rotCenter){
	// GET THE ANGLE WITH WHICH WE NEED TO ROTATE
	double rotAngle = std::atan2((headLocation.y-feetLocation.y),\
						(headLocation.x-feetLocation.x));
	rotAngle -= M_PI;
	angle0to360(rotAngle);

	// THE ROTATION ANGLE NEEDS TO BE IN DEGREES
	rotAngle *= (180.0/M_PI);

	// GET THE ROTATION MATRIX WITH RESPECT TO THE GIVEN CENTER
	cv::Mat rotationMat = cv::getRotationMatrix2D(rotCenter, rotAngle, 1.0);

	// BUILD A MATRIX OUT OF THE TEMPLATE POINTS
	cv::Mat toRotate = cv::Mat::ones(cv::Size(3,keys.rows),\
						cv::DataType<double>::type);
	for(int y=0; y<keys.rows; y++){
		toRotate.at<double>(y,0) = keys.at<double>(y,keys.cols-2)+rotBorders.x;
		toRotate.at<double>(y,1) = keys.at<double>(y,keys.cols-1)+rotBorders.y;
	}

	// MULTIPLY THE TEMPLATE MATRIX WITH THE ROTATION MATRIX
	toRotate.convertTo(toRotate, cv::DataType<double>::type);
	cv::Mat rotated = toRotate*rotationMat.t();
	rotated.convertTo(rotated, cv::DataType<double>::type);

	// COPY THE RESULT BACK INTO A TEMPLATE SHAPE
	for(int y=0; y<keys.rows; y++){
		keys.at<double>(y,keys.cols-2) = rotated.at<double>(y,0);
		keys.at<double>(y,keys.cols-1) = rotated.at<double>(y,1);
	}
	rotationMat.release();
	toRotate.release();
	rotated.release();
}
//==============================================================================
/** Rotate matrix wrt to the camera location.
 */
cv::Mat peopleDetector::rotate2Zero(cv::Point2f headLocation,\
cv::Point2f feetLocation, cv::Mat toRotate, cv::Point2f &borders){
	// GET THE ANGLE TO ROTATE WITH
	double rotAngle = std::atan2((headLocation.y-feetLocation.y),\
						(headLocation.x-feetLocation.x));
	rotAngle -= M_PI;
	angle0to360(rotAngle);

	// THE ANGLE NEEDS TO BE IN DEGREES TO ROTATE WITH IT
	rotAngle *= (180.0/M_PI);

	// ADD A BLACK BORDER TO THE ORIGINAL IMAGE
	double diag       = std::sqrt(toRotate.cols*toRotate.cols+toRotate.rows*\
						toRotate.rows);
	borders.x         = std::ceil((diag-toRotate.cols)/2.0);
	borders.y         = std::ceil((diag-toRotate.rows)/2.0);
	cv::Mat srcRotate = cv::Mat::zeros(cv::Size(toRotate.cols+2*borders.x,\
						toRotate.rows+2*borders.y),toRotate.type());
	cv::copyMakeBorder(toRotate,srcRotate,borders.y,borders.y,borders.x,\
		borders.x,cv::BORDER_CONSTANT);

	// GET THE ROTATION MATRIX
	cv::Mat rotationMat = cv::getRotationMatrix2D(cv::Point2f(\
		srcRotate.cols/2.0,srcRotate.rows/2.0),rotAngle, 1.0);

	// ROTATE THE IMAGE WITH THE ROTATION MATRIX
	cv::Mat rotated = cv::Mat::zeros(srcRotate.size(),toRotate.type());
	cv::warpAffine(srcRotate, rotated, rotationMat, srcRotate.size());
	rotationMat.release();
	srcRotate.release();
	return rotated;
}
//==============================================================================
/** Fixes the angle to be relative to the camera position with respect to the
 * detected position.
 */
double peopleDetector::fixAngle(cv::Point2f headLocation, cv::Point2f feetLocation,\
double angle){
	// GET THE CAMERA ANGLE IN RADIANDS IN [-pi,pi)
	double cameraAngle = std::atan2((headLocation.y - feetLocation.y),\
						(headLocation.x - feetLocation.x));
	cameraAngle -= M_PI;

	double newAngle;
	newAngle = angle + cameraAngle;
	angle0to360(newAngle);
	return newAngle;
}
//==============================================================================
/** For each row added in the data matrix (each person detected for which we
 * have extracted some features) find the corresponding label.
 */
std::deque<unsigned> peopleDetector::fixLabels(std::deque<unsigned> existing){
	// FIND	THE INDEX FOR THE CURRENT IMAGE
	std::deque<annotationsHandle::FULL_ANNOTATIONS>::iterator index = \
		std::find_if (this->targetAnno.begin(), this->targetAnno.end(),\
		compareImg(this->current->sourceName));
	if(index == this->targetAnno.end()){
		std::cerr<<"The image: "<<this->current->sourceName<<\
			" was not annotated"<<std::endl;
		exit(1);
	}

	//GET THE EXTREMES OF THE DETECTED LOCATIONS
	std::vector<cv::Point2f> points;
	for(std::size_t i=0; i<existing.size(); i++){
		cv::Point2f center = this->cvPoint(existing[i]);
		points.push_back(center);
	}

	// LOOP OVER ALL ANNOTATIONS FOR THE CURRENT IMAGE AND FIND THE CLOSEST ONES
	std::deque<int> assignments((*index).annos.size(),-1);
	std::deque<double> minDistances((*index).annos.size(),(double)INFINITY);
	unsigned canAssign = 1;
	while(canAssign){
		canAssign = 0;
		for(std::size_t l=0; l<(*index).annos.size(); l++){
			// EACH ANNOTATION NEEDS TO BE ASSIGNED TO THE CLOSEST DETECTION
			double distance    = (double)INFINITY;
			unsigned annoIndex = -1;
			for(std::size_t k=0; k<points.size(); k++){
				double dstnc = dist((*index).annos[l].location,points[k]);
				if(distance>dstnc && this->canBeAssigned(l,minDistances,k,dstnc,\
				assignments)){
					distance  = dstnc;
					annoIndex = k;
				}
			}
			assignments[l]  = annoIndex;
			minDistances[l] = distance;
		}
	}

	// DELETE DETECTED LOCATIONS THAT ARE NOT LABELLED ARE IGNORED
	std::deque<unsigned> finExisting;
	for(std::size_t k=0; k<points.size(); k++){
		// SEARCH FOR A DETECTED POSITION IN ASSIGNMENTS
		std::deque<int>::iterator targetPos = \
			std::find(assignments.begin(),assignments.end(),k);

		// IF THE POSITION IS FOUND READ THE LABEL AND SAVE IT
		if(targetPos != assignments.end()){
			finExisting.push_back(existing[k]);
			int position = targetPos-assignments.begin();
			cv::Point2f feet = (*index).annos[position].location;
			cv::Point2f head = this->headLocation(feet);

			// SAVE THE TARGET LABEL
			cv::Mat tmp = cv::Mat::zeros(1,4,cv::DataType<double>::type);
			// READ THE TARGET ANGLE FOR LONGITUDINAL ANGLE
			double angle = static_cast<double>((*index).annos[position].\
							poses[annotationsHandle::LONGITUDE]);
			angle = angle*M_PI/180.0;
			angle = this->fixAngle(head,feet,angle);
			std::cout<<"Longitude: "<<angle*(180/M_PI)<<std::endl;
			tmp.at<double>(0,0) = std::sin(angle);
			tmp.at<double>(0,1) = std::cos(angle);

			// READ THE TARGET ANGLE FOR LATITUDINAL ANGLE
			angle = static_cast<double>((*index).annos[position].\
					poses[annotationsHandle::LATITUDE]);
			std::cout<<"Latitude: "<<angle<<std::endl;
			angle = angle*M_PI/180.0;
			tmp.at<double>(0,2) = std::sin(angle);
			tmp.at<double>(0,3) = std::cos(angle);

			// STORE THE LABELS IN THE TARGETS ON THE RIGHT POSITION
			if(this->targets.empty()){
				tmp.copyTo(this->targets);
			}else{
				this->targets.push_back(tmp);
			}
			tmp.release();
		}
	}
	return finExisting;
}
//==============================================================================
/** Overwrites the \c doFindPeople function from the \c Tracker class to make it
 * work with the feature extraction.
 */
bool peopleDetector::doFindPerson(unsigned imgNum, IplImage *src,\
const vnl_vector<FLOAT> &imgVec, vnl_vector<FLOAT> &bgVec,\
const FLOAT logBGProb,const vnl_vector<FLOAT> &logSumPixelBGProb){
	std::cout<<this->current->index<<") Image... "<<this->current->sourceName<<std::endl;

	//1) START THE TIMER & INITIALIZE THE PROBABLITIES, THE VECTOR OF POSITIONS
	stic();	std::deque<FLOAT> marginal;
	FLOAT lNone = logBGProb + this->logNumPrior[0], lSum = -INFINITY;
	std::deque<unsigned> existing;
	std::deque< vnl_vector<FLOAT> > logPosProb;
	std::deque<scanline_t> mask;
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
	if(this->plot){
		for(unsigned i=0; i<tracks.size(); ++i){
			if(this->tracks[i].imgID.size() > MIN_TRACKLEN){
				if(imgNum-this->tracks[i].imgID.back() < 2){
					this->plotTrack(src,tracks[i],i,(unsigned)1);
				}
			}
		}
	}
	IplImage *bg = vec2img((imgVec-bgVec).apply(fabs));

	//7') EXTRACT FEATURES FOR THE CURRENTLY DETECTED LOCATIONS
	cout<<"Number of templates: "<<existing.size()<<endl;
	if(this->onlyExtract){
		this->extractor->extractFeatures();
	}else{
		this->extractDataRow(existing, bg);
	}

	//7) SHOW THE FOREGROUND POSSIBLE LOCATIONS AND PLOT THE TEMPLATES
	cerr<<"no. of detected people: "<<existing.size()<<endl;
	if(this->plot){
		this->plotHull(bg, this->priorHull);
		for(unsigned i=0; i!=existing.size(); ++i){
			cv::Point2f pt = this->cvPoint(existing[i]);
			plotTemplate2(bg, pt, persHeight, camHeight, CV_RGB(255,255,255));
			plotScanLines(src, mask, CV_RGB(0,255,0), 0.3);
		}
		cvShowImage("bg", bg);
		cvShowImage("image",src);
		cv::waitKey(0);
	}

	//8) WHILE NOT q WAS PRESSED PROCESS THE NEXT IMAGES
	cvReleaseImage(&bg);
	return this->imageProcessingMenu();
}
//==============================================================================
/** Simple "menu" for skipping to the next image or quitting the processing.
 */
bool peopleDetector::imageProcessingMenu(){
	if(this->print){
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
void peopleDetector::templatePart(cv::Mat &thresholded, int k, double offsetX,\
double offsetY){
	unsigned minsize = 50;
	// CHANGE THRESHOLD
	double minX, maxX, minY, maxY;
	if(!thresholded.empty()){
		minX = thresholded.cols, maxX = 0, minY = thresholded.rows, maxY = 0;
		for(int x=0; x<thresholded.cols; x++){
			for(int y=0; y<thresholded.rows; y++){
				if(thresholded.at<uchar>(y,x)>0){
					if(y<=minY){minY = y;}
					if(y>=maxY){maxY = y;}
					if(x<=minX){minX = x;}
					if(x>=maxX){maxX = x;}
				}
			}
		}
	}else{
		minX = this->templates[k].extremes[0];
		maxX = this->templates[k].extremes[1];
		minY = this->templates[k].extremes[2];
		maxY = this->templates[k].extremes[3];
	}

	unsigned middleTop = (minX+maxX)/2;
	unsigned middleBot = (minX+maxX)/2;
	if((middleTop-minX)/2.0>minsize){
		middleTop = (minX + middleTop)/2.0;
	}
	if((maxX-middleBot)/2.0>minsize){
		middleBot = (maxX + middleBot)/2.0;
	}

	if(!thresholded.empty()){
		for(int x=0; x<thresholded.cols; x++){
			for(int y=0; y<thresholded.rows; y++){
				if(x>middleTop && this->featurePart=='t'){
					thresholded.at<uchar>(y,x) = 0;
				}else if(x<middleBot && this->featurePart=='b'){
					thresholded.at<uchar>(y,x) = 0;
				}
			}
		}
	}

	// CHANGE TEMPLATE
	for(unsigned i=0; i<this->templates[k].points.size(); i++){
		if(this->featurePart=='t' && (this->templates[k].points[i].x-offsetX)\
		>=middleTop){
			this->templates[k].points[i].x = middleTop+offsetX;
		}else if(this->featurePart=='b' && (this->templates[k].points[i].x-offsetX)\
		<=middleBot){
			this->templates[k].points[i].x = middleBot+offsetX;
		}
	}

	if(this->plot){
		std::vector<cv::Point2f> tmpTempl = this->templates[k].points;
		for(unsigned i=0; i<tmpTempl.size(); i++){
			tmpTempl[i].x -= offsetX;
			tmpTempl[i].y -= offsetY;
		}
		if(!thresholded.empty()){
			IplImage *toSee = new IplImage(thresholded);
			plotTemplate2(toSee, cv::Point2f(0,0), persHeight,\
				camHeight, cvScalar(150,0,0),tmpTempl);
			cvShowImage("part", toSee);
			cvWaitKey(0);
		}
	}
}
//==============================================================================
/** Computes the motion vector for the current image given the tracks so far.
 */
double peopleDetector::motionVector(cv::Point2f head, cv::Point2f center){
	cv::Point2f prev = center;
	double angle = 0;
	if(this->tracks.size()>1){
		for(unsigned i=0; i<this->tracks.size();i++){
			if(this->tracks[i].positions[this->tracks[i].positions.size()-1].x ==\
			center.x && this->tracks[i].positions[this->tracks[i].positions.size()-1].y\
			== center.y){
				if(this->tracks[i].positions.size()>1){
					prev.x = this->tracks[i].positions[this->tracks[i].positions.size()-2].x;
					prev.y = this->tracks[i].positions[this->tracks[i].positions.size()-2].y;
				}
				break;
			}
		}

		if(this->plot){
			cv::Mat tmp(this->current->img);
			cv::line(tmp,prev,center,cv::Scalar(50,100,255),1,8,0);
			cv::imshow("tracks",tmp);
		}
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
void peopleDetector::start(bool readFromFolder, bool useGT){
	this->useGroundTruth = useGT;
	// ((useGT & !GABOR & !EDGES) | EXTRACT) & ANNOS
	if(!this->targetAnno.empty() && (this->onlyExtract || this->useGroundTruth)){
		// READ THE FRAMES ONE BY ONE
		if(!this->producer){
			this->initProducer(readFromFolder);
		}
		IplImage *img = this->producer->getFrame();
		width         = img->width;
		height        = img->height;
		depth         = img->depth;
		channels      = img->nChannels;
		halfresX      = width/2;
		halfresY      = height/2;
		this->producer->backward(1);
		this->current = new Image_t();

		unsigned index = 0;
		while((this->current->img = this->producer->getFrame())){
			this->current->sourceName = this->producer->getSource(-1);
			this->current->index      = index;
			std::cout<<index<<") Image... "<<this->current->sourceName<<std::endl;
			if(this->onlyExtract){
				this->extractor->extractFeatures();
			}else{
				// READ THE LOCATION AT WICH THERE ARE PEOPLE
				std::deque<unsigned> existing = this->readLocations();
				cout<<"Number of templates: "<<existing.size()<<endl;
				this->extractDataRow(existing,NULL);
			}
			index++;
		}
	}else{
		// FOR THE EDGES AND GABOR I NEED A BACKGROUND MODEL EVEN IF I ONLY EXTRACT
		this->useGroundTruth = false;
		this->run(readFromFolder);
	}
}
//==============================================================================
/** Gets the location of the head given the feet location.
 */
cv::Point2f peopleDetector::headLocation(cv::Point2f center){
	std::vector<cv::Point2f> templ;
	genTemplate2(center, persHeight, camHeight, templ);
	return cv::Point2f((templ[12].x+templ[14].x)/2,(templ[12].y+templ[14].y)/2);
}
//==============================================================================
/** Reads the locations at which there are people in the current frame (for the
 * case in which we do not want to use the tracker or build a bgModel).
 */
std::deque<unsigned> peopleDetector::readLocations(){
	// FIND	THE INDEX FOR THE CURRENT IMAGE
	if(this->targetAnno.empty()){
		std::cerr<<"Annotations were not loaded."<<std::endl;
		exit(1);
	}
	std::deque<annotationsHandle::FULL_ANNOTATIONS>::iterator index = \
		std::find_if (this->targetAnno.begin(), this->targetAnno.end(),\
		compareImg(this->current->sourceName));

	if(index == this->targetAnno.end()){
		std::cerr<<"The image: "<<this->current->sourceName<<\
			" was not annotated"<<std::endl;
		exit(1);
	}
	// TRANSFORM THE LOCATION INTO UNSIGNED AND THEN PUSH IT IN THE VECTOR
	std::deque<unsigned> locations;
	for(std::size_t l=0; l<(*index).annos.size(); l++){
		// point.x + width*point.y
		cv::Point2f feet = (*index).annos[l].location;
		cv::Point2f head = this->headLocation(feet);
		unsigned location = feet.x + width*feet.y;
		locations.push_back(location);

		// STORE THE LABELS FOR ALL THE LOCATIONS
		cv::Mat tmp = cv::Mat::zeros(cv::Size(4,1),cv::DataType<double>::type);
		// READ THE TARGET ANGLE FOR LONGITUDINAL ANGLE
		double angle = static_cast<double>((*index).annos[l].\
						poses[annotationsHandle::LONGITUDE]);
		angle = angle*M_PI/180.0;
		angle = this->fixAngle(head,feet,angle);
		std::cout<<"Longitude: "<<(angle*180/M_PI)<<std::endl;
		tmp.at<double>(0,0) = std::sin(angle);
		tmp.at<double>(0,1) = std::cos(angle);

		// READ THE TARGET ANGLE FOR LATITUDINAL ANGLE
		angle = static_cast<double>((*index).annos[l].\
					poses[annotationsHandle::LATITUDE]);
		std::cout<<"Latitude: "<<angle<<std::endl;
		angle = angle*M_PI/180.0;
		tmp.at<double>(0,2) = std::sin(angle);
		tmp.at<double>(0,3) = std::cos(angle);

		// STORE THE LABELS IN THE TARGETS ON THE RIGHT POSITION
		if(this->targets.empty()){
			tmp.copyTo(this->targets);
		}else{
			this->targets.push_back(tmp);
		}
		tmp.release();
	}

	// SEE THE ANNOTATED LOCATIONS IN THE IMAGE
	if(this->plot){
		for(std::size_t i=0; i<locations.size(); i++){
			cv::Point2f center = this->cvPoint(locations[i]);
			std::vector<cv::Point2f> templ;
			plotTemplate2(this->current->img, center, persHeight,\
				camHeight, cv::Scalar(255,0,0),templ);
		}
		cvShowImage("AnnotatedLocations", this->current->img);
		cvWaitKey(0);
	}

	// UPDATE THE TRACKS IN THE CASE WE WANT TO USE TRACKING
	this->updateTracks(this->current->index, locations);
	return locations;
}
//==============================================================================
int main(int argc, char **argv){
	peopleDetector feature(argc,argv,true,false);
	feature.init(std::string(argv[1])+"annotated_train",\
		std::string(argv[1])+"annotated_train.txt",featureExtractor::HOG,true);
	feature.start(true, true);
}

