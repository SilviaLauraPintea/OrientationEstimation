/* featureExtractor.cpp
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#include "Auxiliary.h"
#include "featureExtractor.h"
//==============================================================================
/** Checks to see if a pixel's x coordinate is on a scanline.
 */
struct onScanline{
	public:
		unsigned pixelY;
		onScanline(const unsigned pixelY){this->pixelY=pixelY;}
		virtual ~onScanline(){};
		bool operator()(const Helpers::scanline_t line)const{
			return (line.line == this->pixelY);
		}
		onScanline(const onScanline &on){
			this->pixelY = on.pixelY;
		}
		onScanline& operator=(const onScanline &on){
			if(this == &on) return *this;
			this->pixelY = on.pixelY;
			return *this;
		}
};
//==============================================================================
featureExtractor::featureExtractor(){
	this->isInit       = false;
	this->featureType  = featureExtractor::EDGES;
	this->dictFilename = "none";
	this->noMeans      = 500;
	this->meanSize     = 128;
	this->featureFile  = "none";
	this->print        = false;
	this->plot         = true;
}
//==============================================================================
featureExtractor::~featureExtractor(){
	if(!this->data.empty()){
		this->data.release();
	}
	if(!this->dictionarySIFT.empty()){
		this->dictionarySIFT.release();
	}
}
//==============================================================================
/** Initializes the class elements.
 */
void featureExtractor::init(featureExtractor::FEATURE fType,\
const std::string &featFile,int colorSp,int invColorSp,bool part){
	if(this->isInit){this->reset();}
	this->featureType       = fType;
	this->featureFile       = featFile;
	this->isInit            = true;
	this->colorspaceCode    = colorSp;
	this->invColorspaceCode = invColorSp;
	this->bodyPart          = part;
}
//==============================================================================
/** Initializes the settings for the SIFT dictionary.
 */
void featureExtractor::initSIFT(const std::string &dictName,unsigned means,\
unsigned size){
	if(!this->dictionarySIFT.empty()){
		this->dictionarySIFT.release();
	}
	this->dictFilename = "data/"+dictName;
	this->noMeans      = means;
	this->meanSize     = size;

	// ASSUME THAT THERE IS ALREADY A SIFT DICTIONARY AVAILABLE
	if(this->featureType == featureExtractor::SIFT_DICT){
		std::cout<<"Build dictionary .."<<std::endl;
	}
	if(this->dictionarySIFT.empty() && this->featureType!=featureExtractor::SIFT_DICT){
		Auxiliary::binFile2mat(this->dictionarySIFT,const_cast<char*>\
			(this->dictFilename.c_str()));
	}
}
//==============================================================================
/** Resets the variables to the default values.
 */
void featureExtractor::reset(){
	if(!this->data.empty()){
		this->data.release();
	}
	if(!this->dictionarySIFT.empty()){
		this->dictionarySIFT.release();
	}
	this->isInit = false;
}
//==============================================================================
/** Compares SURF 2 descriptors and returns the boolean value of their comparison.
 */
bool featureExtractor::compareDescriptors(const featureExtractor::keyDescr &k1,\
const featureExtractor::keyDescr &k2){
	return (k1.keys.response>k2.keys.response);
}
//==============================================================================
/** Checks to see if a given pixel is inside a template.
 */
bool featureExtractor::isInTemplate(unsigned pixelX,unsigned pixelY,\
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
	if(std::abs(static_cast<int>(iter->line)-static_cast<int>(pixelY))<=var &&\
	static_cast<int>(iter->start)-var <= static_cast<int>(pixelX) &&\
	static_cast<int>(iter->end)+var >= static_cast<int>(pixelX)){
		return true;
	}else{
		return false;
	}
}
//==============================================================================
/** Rotate a matrix/a template/keypoints wrt to the camera location.
 */
void featureExtractor::rotate2Zero(float rotAngle,featureExtractor::ROTATE what,\
const cv::Rect roi,cv::Point2f &rotCenter,cv::Point2f &rotBorders,\
std::vector<cv::Point2f> &pts,cv::Mat &toRotate){
	float diag;
	cv::Mat srcRotate,rotated,rotationMat,result;
	switch(what){
		case(featureExtractor::MATRIX):
			diag = std::sqrt(toRotate.cols*toRotate.cols+toRotate.rows*\
				toRotate.rows);
			rotBorders.x = std::ceil((diag-toRotate.cols)/2.0);
			rotBorders.y = std::ceil((diag-toRotate.rows)/2.0);
			srcRotate = cv::Mat::zeros(cv::Size(toRotate.cols+2*rotBorders.x,\
				toRotate.rows+2*rotBorders.y),toRotate.type());
			cv::copyMakeBorder(toRotate,srcRotate,rotBorders.y,rotBorders.y,\
				rotBorders.x,rotBorders.x,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
			rotCenter = cv::Point2f(srcRotate.cols/2.0,srcRotate.rows/2.0);
			rotationMat = cv::getRotationMatrix2D(rotCenter,rotAngle,1.0);
			rotationMat.convertTo(rotationMat,CV_32FC1);
			rotated     = cv::Mat::zeros(srcRotate.size(),toRotate.type());
			cv::warpAffine(srcRotate,rotated,rotationMat,srcRotate.size(),\
				cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
			rotated.copyTo(result);
			break;
		case(featureExtractor::TEMPLATE):
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
		case(featureExtractor::KEYS):
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
cv::Mat featureExtractor::getPixels(const cv::Mat &feature,const featureExtractor::templ \
&aTempl,const cv::Rect &roi,const cv::Mat &thresh){
	cv::Rect cutROI;
	if(!thresh.empty()){
		this->getThresholdBorderes(cutROI.x,cutROI.width,cutROI.y,cutROI.height,\
			thresh);
		cutROI.width  -= cutROI.x;
		cutROI.height -= cutROI.y;
	}else{
		cutROI.x      = aTempl.extremes[0]-roi.x;
		cutROI.y      = aTempl.extremes[2]-roi.y;
		cutROI.width  = aTempl.extremes[1]-aTempl.extremes[0];
		cutROI.height = aTempl.extremes[3]-aTempl.extremes[2];
	}
	cv::Mat large = this->cutAndResizeImage(cutROI,feature);
	if(this->colorspaceCode!=-1){
		cv::cvtColor(large,large,this->invColorspaceCode);
	}
	cv::Mat gray;
	cv::cvtColor(large,gray,CV_BGR2GRAY);
	large.release();
	if(this->plot){
		cv::imshow("part",large);
		cv::waitKey(0);
	}

	// MATCH SOME HEADS ON TOP AND GET THE RESULTS
	int radius     = Helpers::dist(aTempl.points[12],aTempl.points[13]);
	cv::Mat result = cv::Mat::zeros(cv::Size(12*20*30+3,1),CV_32FC1);
	cv::medianBlur(gray,gray,3);
	for(int i=0;i<12;++i){
		std::string imgName = "templates/templ"+Auxiliary::int2string(i)+".jpg";
		cv::Mat tmple       = cv::imread(imgName.c_str(),0);
		if(tmple.empty()){
			std::cerr<<"In template matching FILE NOT FOUND: "<<imgName<<std::endl;
			exit(1);
		}
		cv::Mat small;
		cv::resize(tmple,small,cv::Size(radius,radius),0,0,cv::INTER_CUBIC);
		small.convertTo(small,CV_8UC1);
		cv::Mat tmp;
		cv::matchTemplate(gray,small,tmp,CV_TM_CCOEFF_NORMED);
		cv::Mat resized;
		cv::resize(tmp,resized,cv::Size(20,30),0,0,cv::INTER_CUBIC);
		if(this->plot){
			cv::imshow("result"+Auxiliary::int2string(i),resized);
			cv::waitKey(0);
		}

		// RESHAPE THE RESULT AND CONVERT IT TO FLOAT
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

	if(this->print){
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
/** Find the extremities of the thresholded image.
 */
void featureExtractor::getThresholdBorderes(int &minX,int &maxX,int &minY,\
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
cv::Mat featureExtractor::getEdges(cv::Mat &feature,const cv::Mat &thresholded,\
const cv::Rect &roi,const featureExtractor::templ &aTempl,float rotAngle){
	// EXTRACT THE EDGES AND ROTATE THE EDGES TO THE RIGHT POSSITION
	cv::Point2f rotBorders;
	feature.convertTo(feature,CV_8UC1);
	cv::Mat tmpFeat(feature.clone(),roi);
	std::vector<cv::Point2f> dummy;
	cv::Point2f rotCenter(tmpFeat.cols/2+roi.x,tmpFeat.rows/2+roi.y);
	this->rotate2Zero(rotAngle,featureExtractor::MATRIX,roi,rotCenter,rotBorders,\
		dummy,tmpFeat);

	// PICK OUT ONLY THE THRESHOLDED ARES RESHAPE IT AND RETURN IT
	cv::Mat toResize;
	cv::Rect cutROI;
	if(!thresholded.empty()){
		tmpFeat.copyTo(toResize,thresholded);
		this->getThresholdBorderes(cutROI.x,cutROI.width,cutROI.y,cutROI.height,\
			thresholded);
		cutROI.width   -= cutROI.x;
		cutROI.height  -= cutROI.y;
	}else{
		tmpFeat.copyTo(toResize);
		cutROI.x      = aTempl.extremes[0]-roi.x;
		cutROI.y      = aTempl.extremes[2]-roi.y;
		cutROI.width  = aTempl.extremes[1]-aTempl.extremes[0];
		cutROI.height = aTempl.extremes[3]-aTempl.extremes[2];
	}
	tmpFeat.release();
	cv::Mat tmpEdge = this->cutAndResizeImage(cutROI,toResize);
	toResize.release();

	// IF WE WANT TO SEE HOW THE EXTRACTED EDGES LOOK LIKE
	if(this->plot){
		cv::imshow("Edges",tmpEdge);
		cv::waitKey(0);
	}

	// WRITE IT ON ONE ROW
	cv::Mat result = cv::Mat::zeros(cv::Size(tmpEdge.cols*tmpEdge.rows+3,1),CV_32FC1);
	cv::Mat dumm   = result.colRange(0,tmpEdge.cols*tmpEdge.rows);
	tmpEdge        = tmpEdge.reshape(0,1);
	tmpEdge.convertTo(tmpEdge,CV_32FC1);
	tmpEdge.copyTo(dumm);
	dumm.release();
	tmpEdge.release();
	if(this->print){
		std::cout<<"Size(EDGES): "<<result.size()<<std::endl;
		for(int i=0;i<std::min(10,result.cols);++i){
			std::cout<<result.at<float>(0,i)<<" ";
		}
		std::cout<<"..."<<std::endl;
	}
	result.convertTo(result,CV_32FC1);
	return result;
}
//==============================================================================
/** SURF descriptors (Speeded Up Robust Features).
 */
cv::Mat featureExtractor::getSURF(cv::Mat &feature,const std::vector<cv::Point2f>\
&templ,const cv::Rect &roi,const cv::Mat &test,std::vector<cv::Point2f> &indices){
	// KEEP THE TOP 10 DESCRIPTORS WITHIN THE BORDERS OF THE TEMPLATE
	unsigned number  = 30;
	cv::Mat tmp      = cv::Mat::zeros(cv::Size(feature.cols-2,number),CV_32FC1);
	unsigned counter = 0;
	for(int y=0;y<feature.rows;++y){
		if(counter == number){
			break;
		}
		float ptX = feature.at<float>(y,feature.cols-2);
		float ptY = feature.at<float>(y,feature.cols-1);
		if(featureExtractor::isInTemplate(ptX,ptY,templ)){
			cv::Mat dummy1 = tmp.row(counter);
			cv::Mat dummy2 = feature.row(y);
			cv::Mat dummy3 = dummy2.colRange(0,feature.cols-2);
			dummy3.copyTo(dummy1);
			dummy1.release();
			dummy2.release();
			dummy3.release();
			indices.push_back(cv::Point2f(ptX-roi.x,ptY-roi.y));
			++counter;
		}
	}

	if(this->plot && !test.empty()){
		cv::Mat copyTest(test);
		for(std::size_t l=0;l<indices.size();++l){
			cv::circle(copyTest,indices[l],3,cv::Scalar(0,0,255));
		}
		cv::imshow("SURF",copyTest);
		cv::waitKey(0);
		copyTest.release();
	}

	// COPY THE DESCRIPTORS IN THE FINAL MATRIX
	cv::Mat result = cv::Mat::zeros(cv::Size(tmp.rows*tmp.cols+3,1),CV_32FC1);
	tmp            = tmp.reshape(0,1);
	tmp.convertTo(tmp,CV_32FC1);
	cv::Mat dummy = result.colRange(0,tmp.rows*tmp.cols);
	tmp.copyTo(dummy);
	tmp.release();
	dummy.release();
	result.convertTo(result,CV_32FC1);

	// IF WE WANT TO SEE SOME VALUES/IMAGES
	if(this->print){
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
cv::Mat featureExtractor::getPointsGrid(const cv::Mat &feature,const cv::Rect &roi,\
const featureExtractor::templ &aTempl,const cv::Mat &test){
	// GET THE GRID SIZE FROM THE TEMPLATE SIZE
	unsigned no    = 30;
	cv::Mat result = cv::Mat::zeros(cv::Size(no*no+3,1),CV_32FC1);
	float rateX    = (aTempl.extremes[1]-aTempl.extremes[0])/static_cast<float>(no);
	float rateY    = (aTempl.extremes[3]-aTempl.extremes[2])/static_cast<float>(no);

	// KEEP ONLY THE KEYPOINTS THAT ARE IN THE TEMPLATE
	std::vector<cv::Point2f> indices;
	for(int y=0;y<feature.rows;++y){
		float ptX = feature.at<float>(y,0);
		float ptY = feature.at<float>(y,1);
		if(featureExtractor::isInTemplate(ptX,ptY,aTempl.points)){
			cv::Point2f pt(ptX,ptY);
			indices.push_back(pt);

			// CHECK IN WHAT CELL OF THE GRID THE CURRENT POINTS FALL
			unsigned counter = 0;
			for(float ix=aTempl.extremes[0];ix<aTempl.extremes[1]-0.01;ix+=rateX){
				for(float iy=aTempl.extremes[2];iy<aTempl.extremes[3]-0.01;iy+=rateY){
					if(ix<=pt.x && pt.x<(ix+rateX) && iy<=pt.y && pt.y<(iy+rateY)){
						result.at<float>(0,counter) += 1.0;
					}
					counter +=1;
				}
			}
		}
	}
	if(this->plot && !test.empty()){
		cv::Mat copyTest(test);
		for(std::size_t l=0;l<indices.size();++l){
			cv::circle(copyTest,cv::Point2f(indices[l].x-roi.x,indices[l].y-roi.y),\
				3,cv::Scalar(0,0,255));
		}
		cv::imshow("IPOINTS",copyTest);
		cv::waitKey(0);
		copyTest.release();
	}

	// IF WE WANT TO SEE THE VALUES THAT WERE STORED
	if(this->print){
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
void featureExtractor::createGabor(cv::Mat &gabor, float *params){
	// params[0] -- sigma: (3,68) // the actual size
	// params[1] -- gamma: (0.2,1) // how round the filter is
	// params[2] -- dimension: (1,10) // size
	// params[3] -- theta: (0,180) or (-90,90) // angle
	// params[4] -- lambda: (2,256) // thickness
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
cv::Mat featureExtractor::getGabor(cv::Mat &feature,const cv::Mat &thresholded,\
const cv::Rect &roi,const cv::Size &foregrSize,\
const featureExtractor::templ &aTempl,const float rotAngle,int aheight){
	unsigned gaborNo    = std::ceil(feature.rows/aheight);
	int gaborRows       = std::ceil(feature.rows/gaborNo);
	unsigned resultCols = foregrSize.width*foregrSize.height;
	cv::Mat result      = cv::Mat::zeros(cv::Size(gaborNo*resultCols+3,1),CV_32FC1);
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
		cutROI.x      = aTempl.extremes[0]-roi.x;
		cutROI.y      = aTempl.extremes[2]-roi.y;
		cutROI.width  = aTempl.extremes[1]-aTempl.extremes[0];
		cutROI.height = aTempl.extremes[3]-aTempl.extremes[2];
	}
	for(unsigned i=0;i<gaborNo;++i){
		// GET THE ROI OUT OF THE iTH GABOR
		cv::Mat tmp1 = feature.rowRange(i*gaborRows,(i+1)*gaborRows);
		tmp1.convertTo(tmp1,CV_8UC1);
		cv::Mat tmp2(tmp1.clone(),roi);

		// ROTATE EACH GABOR TO THE RIGHT POSITION
		cv::Point2f rotBorders;
		this->rotate2Zero(rotAngle,featureExtractor::MATRIX,roi,rotCenter,\
			rotBorders,dummy,tmp2);

		// KEEP ONLY THE THRESHOLDED VALUES
		cv::Mat toResize;
		if(!thresholded.empty()){
			tmp2.copyTo(toResize,thresholded);
		}else{
			tmp2.copyTo(toResize);
		}

		// READ THE ROI TO BE USED TO GET THE AREA AROUND THE IMAGE
		cv::Mat tmp3 = this->cutAndResizeImage(cutROI,toResize);
		toResize.release();
		if(this->plot){
			cv::imshow("GaborResponse",tmp3);
			cv::waitKey(0);
		}

		// RESHAPE AND STORE IN THE RIGHT PLACE
		tmp3 = tmp3.reshape(0,1);
		tmp3.convertTo(tmp3,CV_32FC1);
		cv::Mat dummy = result.colRange(i*resultCols,(i+1)*resultCols);
		tmp3.copyTo(dummy);

		// RELEASE ALL THE TEMPS AND DUMMIES
		tmp1.release();
		tmp2.release();
		tmp3.release();
		dummy.release();
	}

	if(this->print){
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
/** Compute the features from the SIFT descriptors by doing vector quantization.
 */
cv::Mat featureExtractor::getSIFT(const cv::Mat &feature,\
const std::vector<cv::Point2f> &templ,const cv::Rect &roi,const cv::Mat &test,\
std::vector<cv::Point2f> &indices){
	// KEEP ONLY THE SIFT FEATURES THAT ARE WITHIN THE TEMPLATE
	cv::Mat tmp      = cv::Mat::zeros(cv::Size(feature.cols-2,feature.rows),CV_32FC1);
	unsigned counter = 0;
	for(int y=0;y<feature.rows;++y){
		float ptX = feature.at<float>(y,feature.cols-2);
		float ptY = feature.at<float>(y,feature.cols-1);
		if(featureExtractor::isInTemplate(ptX,ptY,templ)){
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
	if(this->dictionarySIFT.empty()){
		Auxiliary::binFile2mat(this->dictionarySIFT,const_cast<char*>\
			(this->dictFilename.c_str()));
	}

	// COMPUTE THE DISTANCES FROM EACH NEW FEATURE TO THE DICTIONARY ONES
	cv::Mat distances = cv::Mat::zeros(cv::Size(preFeature.rows,\
		this->dictionarySIFT.rows),CV_32FC1);
	cv::Mat minDists  = cv::Mat::zeros(cv::Size(preFeature.rows,1),CV_32FC1);
	cv::Mat minLabel  = cv::Mat::zeros(cv::Size(preFeature.rows,1),CV_32FC1);
	minDists -= 1;
	for(int j=0;j<preFeature.rows;++j){
		for(int i=0;i<this->dictionarySIFT.rows;++i){
			cv::Mat diff;
			cv::absdiff(this->dictionarySIFT.row(i),preFeature.row(j),diff);
			distances.at<float>(i,j) = diff.dot(diff);
			diff.release();
			if(minDists.at<float>(0,j)==-1 ||\
			minDists.at<float>(0,j)>distances.at<float>(i,j)){
				minDists.at<float>(0,j) = distances.at<float>(i,j);
				minLabel.at<float>(0,j) = i;
			}
		}
	}

	// CREATE A HISTOGRAM(COUNT TO WHICH DICT FEATURE WAS ASSIGNED EACH NEW ONE)
	cv::Mat result = cv::Mat::zeros(cv::Size(this->dictionarySIFT.rows+3,1),CV_32FC1);
	for(int i=0;i<minLabel.cols;++i){
		int which = minLabel.at<float>(0,i);
		result.at<float>(0,which) += 1.0;
	}

	// NORMALIZE THE HOSTOGRAM
	cv::Scalar scalar = cv::sum(result);
	result /= static_cast<float>(scalar[0]);

	if(this->plot && !test.empty()){
		cv::Mat copyTest(test);
		for(std::size_t l=0;l<indices.size();++l){
			cv::circle(copyTest,indices[l],3,cv::Scalar(0,0,255));
		}
		cv::imshow("SIFT",copyTest);
		cv::waitKey(0);
		copyTest.release();
	}
	preFeature.release();
	distances.release();
	minDists.release();
	minLabel.release();

	// IF WE WANT TO SEE THE VALUES THAT WERE STORED
	if(this->print){
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
void featureExtractor::extractFeatures(cv::Mat &image,const std::string &sourceName){
	if(this->colorspaceCode != -1){
		cv::cvtColor(image,image,this->colorspaceCode);
	}
	if(this->featureFile[this->featureFile.size()-1]!='/'){
		this->featureFile = this->featureFile + '/';
	}
	Helpers::file_exists(this->featureFile.c_str(),true);
	std::cout<<"In extract features"<<std::endl;

	// FOR EACH LOCATION IN THE IMAGE EXTRACT FEATURES AND STORE
	cv::Mat feature;
	std::string toWrite = this->featureFile;
	std::vector<cv::Point2f> dummyT;
	cv::Rect dummyR;
	switch(this->featureType){
		case (featureExtractor::IPOINTS):
			toWrite += "IPOINTS/";
			Helpers::file_exists(toWrite.c_str(),true);
			feature = this->extractPointsGrid(image);
			break;
		case featureExtractor::EDGES:
			toWrite += "EDGES/";
			Helpers::file_exists(toWrite.c_str(),true);
			feature = this->extractEdges(image);
			break;
		case featureExtractor::SURF:
			toWrite += "SURF/";
			Helpers::file_exists(toWrite.c_str(),true);
			feature = this->extractSURF(image);
			break;
		case featureExtractor::GABOR:
			toWrite += "GABOR/";
			Helpers::file_exists(toWrite.c_str(),true);
			feature = this->extractGabor(image);
			break;
		case featureExtractor::SIFT:
			toWrite += "SIFT/";
			Helpers::file_exists(toWrite.c_str(),true);
			feature = this->extractSIFT(image,dummyT,dummyR);
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
cv::Mat featureExtractor::extractPointsGrid(cv::Mat &image){
	// EXTRACT THE NORMALIZED GREY IMAGE
	cv::Mat gray;
	if(this->colorspaceCode!=-1){
		cv::cvtColor(image,image,this->invColorspaceCode);
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

	if(this->plot){
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
		cv::waitKey(0);
	}
	result.convertTo(result,CV_32FC1);
	gray.release();
	return result;
}
//==============================================================================
/** Extract edges from the whole image.
 */
cv::Mat featureExtractor::extractEdges(cv::Mat &image){
	cv::Mat gray,result;
	if(this->colorspaceCode!=-1){
		cv::cvtColor(image,image,this->invColorspaceCode);
	}
	cv::cvtColor(image,gray,CV_BGR2GRAY);
	Auxiliary::mean0Variance1(gray);
	gray *= 255;
	gray.convertTo(gray,CV_8UC1);
	cv::medianBlur(gray,gray,3);
	cv::Canny(gray,result,250,150,3,true);
	if(this->plot){
		cv::imshow("gray",gray);
		cv::imshow("edges",result);
		cv::waitKey(0);
	}
	gray.release();
	result.convertTo(result,CV_32FC1);
	return result;
}
//==============================================================================
/** Extracts all the surf descriptors from the whole image and writes them in a
 * matrix.
 */
cv::Mat featureExtractor::extractSURF(cv::Mat &image){
	std::vector<float> descriptors;
	std::vector<cv::KeyPoint> keypoints;
	cv::SURF aSURF = cv::SURF(0.01,4,2,false);

	// EXTRACT INTEREST POINTS FROM THE IMAGE
	cv::Mat gray,result;
	if(this->colorspaceCode!=-1){
		cv::cvtColor(image,image,this->invColorspaceCode);
	}
	cv::cvtColor(image,gray,CV_BGR2GRAY);
	cv::medianBlur(gray,gray,3);
	aSURF(gray,cv::Mat(),keypoints,descriptors,false);
	gray.release();

	// WRITE ALL THE DESCRIPTORS IN THE STRUCTURE OF KEY-DESCRIPTORS
	std::deque<featureExtractor::keyDescr> kD;
	for(std::size_t i=0;i<keypoints.size();++i){
		featureExtractor::keyDescr tmp;
		for(int j=0;j<aSURF.descriptorSize();++j){
			tmp.descr.push_back(descriptors[i*aSURF.descriptorSize()+j]);
		}
		tmp.keys = keypoints[i];
		kD.push_back(tmp);
		tmp.descr.clear();
	}

	// SORT THE REMAINING DESCRIPTORS SO WE DON'T NEED TO DO THAT LATER
	std::sort(kD.begin(),kD.end(),(&featureExtractor::compareDescriptors));

	// WRITE THEM IN THE MATRIX AS FOLLOWS (ADD x,y COORD ON THE LAST 2 COLS):
	result = cv::Mat::zeros(cv::Size(aSURF.descriptorSize()+2,kD.size()),CV_32FC1);
	for(std::size_t i=0;i<kD.size();++i){
		if(i<10){
			std::cout<<kD[i].keys.response<<std::endl;
		}
		result.at<float>(i,aSURF.descriptorSize()) = kD[i].keys.pt.x;
		result.at<float>(i,aSURF.descriptorSize()+1) = kD[i].keys.pt.y;
		for(int j=0;j<aSURF.descriptorSize();++j){
			result.at<float>(i,j) = kD[i].descr[j];
		}
	}

	// PLOT THE KEYPOINTS TO SEE THEM
	if(this->plot){
		for(std::size_t i=0;i<kD.size();++i){
			cv::circle(image,cv::Point2f(kD[i].keys.pt.x,kD[i].keys.pt.y),\
				3,cv::Scalar(0,0,255));
		}
		cv::imshow("SURFS",image);
		cv::waitKey(0);
	}
	result.convertTo(result,CV_32FC1);
	return result;
}
//==============================================================================
/** Convolves the whole image with some Gabors wavelets and then stores the
 * results.
 */
cv::Mat featureExtractor::extractGabor(cv::Mat &image){
	// DEFINE THE PARAMETERS FOR A FEW GABORS
	// params[0] -- sigma: (3,68) // the actual size
	// params[1] -- gamma: (0.2,1) // how round the filter is
	// params[2] -- dimension: (1,10) // size
	// params[3] -- theta: (0,180) or (-90,90) // angle
	// params[4] -- lambda: (2,256) // thickness
	// params[5] -- psi: (0,180) // number of lines
	std::deque<float*> allParams;
	float *params1 = new float[6];
	params1[0] = 10.0;params1[1] = 0.9;params1[2] = 2.0;
	params1[3] = M_PI/4.0;params1[4] = 50.0;params1[5] = 15.0;
	allParams.push_back(params1);

	// THE PARAMETERS FOR THE SECOND GABOR
	float *params2 = new float[6];
	params2[0] = 10.0;params2[1] = 0.9;params2[2] = 2.0;
	params2[3] = 3.0*M_PI/4.0;params2[4] = 50.0;params2[5] = 15.0;
	allParams.push_back(params2);

	// CONVERT THE IMAGE TO GRAYSCALE TO APPLY THE FILTER
	cv::Mat gray,result;
	if(this->colorspaceCode!=-1){
		cv::cvtColor(image,image,this->invColorspaceCode);
	}
	cv::cvtColor(image,gray,CV_BGR2GRAY);
	cv::medianBlur(gray,gray,3);

	// CREATE EACH GABOR AND CONVOLVE THE IMAGE WITH IT
	result = cv::Mat::zeros(cv::Size(image.cols,image.rows*allParams.size()),\
		CV_32FC1);
	for(unsigned i=0;i<allParams.size();++i){
		cv::Mat response,agabor;
		this->createGabor(agabor,allParams[i]);
		cv::filter2D(gray,response,-1,agabor,cv::Point2f(-1,-1),0,\
			cv::BORDER_REPLICATE);
		cv::Mat temp = result.rowRange(i*response.rows,(i+1)*response.rows);
		response.convertTo(response,CV_32FC1);
		response.copyTo(temp);

		// IF WE WANT TO SEE THE GABOR AND THE RESPONSE
		if(this->plot){
			cv::imshow("GaborFilter",agabor);
			cv::imshow("GaborResponse",temp);
			cv::waitKey(0);
		}
		response.release();
		agabor.release();
		temp.release();
	}
	delete [] allParams[0];
	delete [] allParams[1];
	allParams.clear();
	gray.release();
	result.convertTo(result,CV_32FC1);
	return result;
}
//==============================================================================
/** Extracts SIFT features from the image and stores them in a matrix.
 */
cv::Mat featureExtractor::extractSIFT(cv::Mat &image,\
const std::vector<cv::Point2f> &templ,const cv::Rect &roi){
	// DEFINE THE SURF KEYPOINTS AND THE DESCRIPTORS
	std::vector<cv::KeyPoint> keypoints;
	cv::SIFT::DetectorParams detectP  = cv::SIFT::DetectorParams(0.0001,10.0);
	cv::SIFT::DescriptorParams descrP = cv::SIFT::DescriptorParams();
	cv::SIFT::CommonParams commonP    = cv::SIFT::CommonParams();
	cv::SIFT aSIFT(commonP,detectP,descrP);

	// EXTRACT SIFT FEATURES IN THE IMAGE
	cv::Mat gray, result;
	if(this->colorspaceCode!=-1){
		cv::cvtColor(image,image,this->invColorspaceCode);
	}
	cv::cvtColor(image,gray,CV_BGR2GRAY);
	cv::medianBlur(gray,gray,3);
	aSIFT(gray,cv::Mat(),keypoints);

	// WE USE THE SAME FUNCTION TO BUILD THE DICTIONARY ALSO
	if(this->featureType==featureExtractor::SIFT_DICT){
		result = cv::Mat::zeros(keypoints.size(),aSIFT.descriptorSize(),CV_32FC1);
		std::vector<cv::KeyPoint> goodKP;
		for(std::size_t i=0;i<keypoints.size();++i){
			if(featureExtractor::isInTemplate(keypoints[i].pt.x+roi.x,\
			keypoints[i].pt.y+roi.y,templ)){
				goodKP.push_back(keypoints[i]);
			}
		}
		aSIFT(gray,cv::Mat(),goodKP,result,true);
		if(this->plot){
			for(std::size_t i=0;i<goodKP.size();++i){
				cv::circle(image,goodKP[i].pt,3,cv::Scalar(0,0,255));
			}
			cv::imshow("SIFT_DICT",image);
			cv::waitKey(0);
		}
	// IF WE ONLY WANT TO STORE THE SIFT FEATURE WE NEED TO ADD THE x-S AND y-S
	}else if(this->featureType == featureExtractor::SIFT){
		result = cv::Mat::zeros(keypoints.size(),aSIFT.descriptorSize()+2,CV_32FC1);
		cv::Mat dummy1 = result.colRange(0,aSIFT.descriptorSize());
		cv::Mat dummy2;
		aSIFT(gray,cv::Mat(),keypoints,dummy2,true);
		dummy2.convertTo(dummy2,CV_32FC1);
		dummy2.copyTo(dummy1);
		dummy1.release();
		dummy2.release();
		// PLOT THE KEYPOINTS TO SEE THEM
		if(this->plot){
			for(std::size_t i=0;i<keypoints.size();++i){
				cv::circle(image,keypoints[i].pt,3,cv::Scalar(0,0,255));
			}
			cv::imshow("SIFT",image);
			cv::waitKey(0);
		}
	}

	// NORMALIZE THE FEATURE
	result.convertTo(result,CV_32FC1);
	for(int i=0;i<result.rows;++i){
		cv::Mat rowsI = result.row(i);
		rowsI         = rowsI/cv::norm(rowsI);
		rowsI.release();

		// IF WE WANT TO STORE THE SIFT FEATURES THEN, WE NEED TO STORE x AND y
		if(this->featureType==featureExtractor::SIFT){
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
cv::Mat featureExtractor::getDataRow(int imageRows,\
const featureExtractor::templ &aTempl,const cv::Rect &roi,\
const featureExtractor::people &person,const cv::Mat &thresholded,\
const std::string &imgName,cv::Point2f &absRotCenter,cv::Point2f &rotBorders,\
float rotAngle,std::vector<cv::Point2f> &keys){
	cv::Mat feature,result,dictImg;
	std::string toRead;
	std::cout<<"Image class (CLOSE/MEDIUM/FAR): "<<this->imageClass<<std::endl;
	switch(this->featureType){
		case (featureExtractor::IPOINTS):
			toRead = (this->featureFile+"IPOINTS/"+imgName+".bin");
			Auxiliary::binFile2mat(feature,const_cast<char*>(toRead.c_str()));
			this->rotate2Zero(rotAngle,featureExtractor::KEYS,roi,absRotCenter,\
				rotBorders,keys,feature);
			result = this->getPointsGrid(feature,roi,aTempl,person.pixels);
			break;
		case featureExtractor::EDGES:
			toRead = (this->featureFile+"EDGES/"+imgName+".bin");
			Auxiliary::binFile2mat(feature,const_cast<char*>(toRead.c_str()));
			result = this->getEdges(feature,thresholded,roi,aTempl,rotAngle);
			break;
		case featureExtractor::SURF:
			toRead = (this->featureFile+"SURF/"+imgName+".bin");
			Auxiliary::binFile2mat(feature,const_cast<char*>(toRead.c_str()));
			this->rotate2Zero(rotAngle,featureExtractor::KEYS,roi,absRotCenter,\
				rotBorders,keys,feature);
			result = this->getSURF(feature,aTempl.points,roi,person.pixels,keys);
			break;
		case featureExtractor::GABOR:
			toRead = (this->featureFile+"GABOR/"+imgName+".bin");
			Auxiliary::binFile2mat(feature,const_cast<char*>(toRead.c_str()));
			result = this->getGabor(feature,thresholded,roi,person.pixels.size(),\
				aTempl,rotAngle,imageRows);
			break;
		case featureExtractor::SIFT_DICT:
			person.pixels.copyTo(dictImg);
			result = this->extractSIFT(dictImg,aTempl.points,roi);
			break;
		case featureExtractor::SIFT:
			toRead = (this->featureFile+"SIFT/"+imgName+".bin");
			Auxiliary::binFile2mat(feature,const_cast<char*>(toRead.c_str()));
			this->rotate2Zero(rotAngle,featureExtractor::KEYS,roi,absRotCenter,\
				rotBorders,keys,feature);
			result = this->getSIFT(feature,aTempl.points,roi,person.pixels,keys);
			break;
		case featureExtractor::PIXELS:
			// NO NEED TO STORE ANY FEATURE,ONLY THE PIXEL VALUES ARE NEEDED
			result = this->getPixels(person.pixels,aTempl,roi,thresholded);
			break;
		case featureExtractor::HOG:
			// CAN ONLY EXTRACT THEM OVER AN IMAGE SO NO FEATURES CAN BE STORED
			result = this->getHOG(person.pixels,aTempl,roi,thresholded);
			break;
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
cv::Mat featureExtractor::getHOG(const cv::Mat &feature,\
const featureExtractor::templ &aTempl,const cv::Rect &roi,const cv::Mat &thresh){
	// JUST COPY THE PIXELS THAT ARE LARGER THAN 0 INTO
	cv::Rect cutROI;
	if(!thresh.empty()){
		this->getThresholdBorderes(cutROI.x,cutROI.width,cutROI.y,cutROI.height,\
			thresh);
		cutROI.width   -= cutROI.x;
		cutROI.height  -= cutROI.y;
	}else{
		cutROI.x      = aTempl.extremes[0]-roi.x;
		cutROI.y      = aTempl.extremes[2]-roi.y;
		cutROI.width  = aTempl.extremes[1]-aTempl.extremes[0];
		cutROI.height = aTempl.extremes[3]-aTempl.extremes[2];
	}

std::cout<<"image_size="<<feature.size()<<std::endl;
std::cout<<"roi="<<roi<<std::endl;
std::cout<<"cut_roi="<<cutROI<<std::endl;


	cv::Mat large = this->cutAndResizeImage(cutROI,feature);



	if(this->colorspaceCode!=-1){
		cv::cvtColor(large,large,this->invColorspaceCode);
	}
	cv::Mat gray;
	cv::cvtColor(large,gray,CV_BGR2GRAY);
	large.release();
	cv::medianBlur(gray,gray,3);
	unsigned stepX = (gray.cols%10==0?10:(10+gray.cols%10));
	unsigned stepY = (gray.rows%10==0?10:(10+gray.rows%10));
	cv::HOGDescriptor hogD(gray.size(),cv::Size(stepX,stepY),cv::Size(10,10),\
		cv::Size(stepX,stepY),15,1,-1,cv::HOGDescriptor::L2Hys,0.2,true);
	std::vector<float> descriptors;

	if(this->plot){
		cv::imshow("image4HOG",gray);
		cv::waitKey(0);
	}
	hogD.compute(gray,descriptors,cv::Size(stepX,stepY),cv::Size(0,0),\
		std::vector<cv::Point>());
	cv::Mat dummy1(descriptors);
	cv::Mat result = cv::Mat::zeros(cv::Size(dummy1.cols+3,dummy1.rows),CV_32FC1);
	cv::Mat dummy2 = result.colRange(0,dummy1.cols);
	dummy1.copyTo(dummy2);
	dummy1.release();
	dummy2.release();
	result.convertTo(result,CV_32FC1);

	std::cout<<"In HOG: descriptorSize = "<<descriptors.size()<<std::endl;
	if(this->print){
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
unsigned featureExtractor::readNoMeans(){
	return this->noMeans;
}
//==============================================================================
/**Return name of the SIFT dictionary.
 */
std::string featureExtractor::readDictName(){
	return this->dictFilename;
}
//==============================================================================
/** Sets the image class and resets the dictionary name.
 */
unsigned featureExtractor::setImageClass(unsigned aClass){
	std::deque<std::string> names;
	names.push_back("CLOSE");names.push_back("MEDIUM");	names.push_back("FAR");
	this->imageClass = names[aClass];

	// WE NEED A SIFT DICTIONARY FOR EACH CLASS
	std::string dictName = this->imageClass+"_SIFT"+".bin";
	this->initSIFT(dictName);
}
//==============================================================================
/** Cut the image around the template or bg bordered depending on which is used
 * and resize to a common size.
 */
cv::Mat featureExtractor::cutAndResizeImage(const cv::Rect &roiCut,\
const cv::Mat &img){
	cv::Mat tmp(img.clone(),roiCut),large;
	cv::Size aSize;
	if(this->imageClass=="CLOSE"){
		aSize.height = 50;
		aSize.width  = aSize.height*6/5;
	}else if(this->imageClass=="FAR"){
		int pHeight = 150;
		if(this->bodyPart){
			aSize.height = pHeight/2;
		}else{
			aSize.height = pHeight;
		}
		aSize.width  = pHeight*1/2;
	}else if(this->imageClass=="MEDIUM"){
		int pHeight = 100;
		if(this->bodyPart){
			aSize.height = pHeight/2;
		}else{
			aSize.height = pHeight;
		}
		aSize.width  = pHeight*3/4;
	}
	cv::medianBlur(tmp,tmp,3);
	cv::resize(tmp,large,aSize,0,0,cv::INTER_CUBIC);
	return large;
}
//==============================================================================






