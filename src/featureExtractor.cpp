/* featureExtractor.cpp
 * Author: Silvia-Laura Pintea
 */
#include "eigenbackground/src/Helpers.hh"
#include "featureExtractor.h"
#include "Auxiliary.h"
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
featureExtractor::featureExtractor(){
	this->isInit      = false;
	this->featureType = featureExtractor::EDGES;

	this->dictFilename = "none";
	this->noMeans      = 500;
	this->meanSize     = 128;
	this->featureFile  = "none";
	this->print        = true;
	this->plot         = false;
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
void featureExtractor::init(featureExtractor::FEATURE fType, std::string featFile){
	if(this->isInit){this->reset();}
	this->featureType = fType;
	this->featureFile = featFile;
	this->isInit      = true;
}
//==============================================================================
/** Initializes the settings for the SIFT dictionary.
 */
void featureExtractor::initSIFT(std::string dictName, unsigned means, unsigned size){
	if(!this->dictionarySIFT.empty()){
		this->dictionarySIFT.release();
	}
	this->dictFilename = dictName;
	this->noMeans      = means;
	this->meanSize     = size;

	// ASSUME THAT THERE IS ALREADY A SIFT DICTIONARY AVAILABLE
	if(this->dictionarySIFT.empty()){
		binFile2mat(this->dictionarySIFT, const_cast<char*>(this->dictFilename.c_str()));
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
bool featureExtractor::compareDescriptors(const featureExtractor::keyDescr k1,\
const featureExtractor::keyDescr k2){
	return (k1.keys.response>k2.keys.response);
}
//==============================================================================
/** Checks to see if a given pixel is inside a template.
 */
bool featureExtractor::isInTemplate(unsigned pixelX, unsigned pixelY,\
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

	if(std::abs(static_cast<int>(iter->line)==static_cast<int>(pixelY)) &&\
	static_cast<int>(iter->start) <= static_cast<int>(pixelX) &&\
	static_cast<int>(iter->end) >= static_cast<int>(pixelX)){
		return true;
	}else{
		return false;
	}
}
//==============================================================================
/** Rotate a matrix/a template/keypoints wrt to the camera location.
 */
cv::Mat featureExtractor::rotate2Zero(cv::Point2f headLocation, cv::Point2f \
feetLocation, cv::Mat toRotate, cv::Point2f &rotBorders, cv::Point2f rotCenter,\
featureExtractor::ROTATE what, std::vector<cv::Point2f> &pts){
	// GET THE ANGLE WITH WHICH WE NEED TO ROTATE
	double rotAngle = std::atan2((headLocation.y-feetLocation.y),\
						(headLocation.x-feetLocation.x));
	rotAngle -= M_PI;
	angle0to360(rotAngle);

	// THE ROTATION ANGLE NEEDS TO BE IN DEGREES
	rotAngle *= (180.0/M_PI);

	double diag;
	cv::Mat srcRotate, rotated, rotationMat, result;
	switch(what){
		case(featureExtractor::MATRIX):
			diag      = std::sqrt(toRotate.cols*toRotate.cols+toRotate.rows*\
						toRotate.rows);
			rotBorders.x = std::ceil((diag-toRotate.cols)/2.0);
			rotBorders.y = std::ceil((diag-toRotate.rows)/2.0);
			srcRotate = cv::Mat::zeros(cv::Size(toRotate.cols+2*rotBorders.x,\
						toRotate.rows+2*rotBorders.y),toRotate.type());
			cv::copyMakeBorder(toRotate,srcRotate,rotBorders.y,rotBorders.y,\
				rotBorders.x,rotBorders.x,cv::BORDER_CONSTANT);
			rotCenter = cv::Point2f(srcRotate.cols/2.0,srcRotate.rows/2.0);
			rotationMat = cv::getRotationMatrix2D(rotCenter, rotAngle, 1.0);
			rotated     = cv::Mat::zeros(srcRotate.size(),toRotate.type());
			cv::warpAffine(srcRotate, rotated, rotationMat, srcRotate.size());
			rotated.copyTo(result);
			break;
		case(featureExtractor::TEMPLATE):
			rotationMat = cv::getRotationMatrix2D(rotCenter, rotAngle, 1.0);
			toRotate    = cv::Mat::ones(cv::Size(3, pts.size()),\
							cv::DataType<double>::type);
			for(std::size_t i=0; i<pts.size(); i++){
				toRotate.at<double>(i,0) = pts[i].x + rotBorders.x;
				toRotate.at<double>(i,1) = pts[i].y + rotBorders.y;
			}
			toRotate.convertTo(toRotate, cv::DataType<double>::type);
			rotated = toRotate*rotationMat.t();
			rotated.convertTo(rotated, cv::DataType<double>::type);
			pts.clear();
			pts = std::vector<cv::Point2f>(rotated.rows);
			for(int y=0; y<rotated.rows; y++){
				pts[y].x = rotated.at<double>(y,0);
				pts[y].y = rotated.at<double>(y,1);
			}
			break;
		case(featureExtractor::KEYS):
			rotationMat = cv::getRotationMatrix2D(rotCenter, rotAngle, 1.0);
			srcRotate = cv::Mat::ones(cv::Size(3,toRotate.rows),\
						cv::DataType<double>::type);
			for(int y=0; y<toRotate.rows; y++){
				srcRotate.at<double>(y,0) = toRotate.at<double>(y,toRotate.cols-2)+\
											rotBorders.x;
				srcRotate.at<double>(y,1) = toRotate.at<double>(y,toRotate.cols-1)+\
											rotBorders.y;
			}
			srcRotate.convertTo(srcRotate, cv::DataType<double>::type);
			rotated = srcRotate*rotationMat.t();
			rotated.convertTo(rotated, cv::DataType<double>::type);
			toRotate.copyTo(result);
			for(int y=0; y<toRotate.rows; y++){
				result.at<double>(y,toRotate.cols-2) = rotated.at<double>(y,0);
				result.at<double>(y,toRotate.cols-1) = rotated.at<double>(y,1);
			}
			break;
	}
	toRotate.release();
	rotationMat.release();
	srcRotate.release();
	rotated.release();
	return result;
}
//==============================================================================
/** Gets the plain pixels corresponding to the upper part of the body.
 */
cv::Mat featureExtractor::getPixels(cv::Mat thresholded,featureExtractor::templ \
aTempl){
	// JUST COPY THE PIXELS THAT ARE LARGER THAN 0 INTO
	cv::Mat tmp(thresholded, cv::Rect(aTempl.extremes[0],aTempl.extremes[2],
		aTempl.extremes[1]-aTempl.extremes[0],aTempl.extremes[3]-\
		aTempl.extremes[2]));

cv::imshow("tmp",tmp);
cv::waitKey(0);

	// RESIZE TO A COMMON SIZE SUCH THAT ALL THE ROWS HAVE THE SAME LENGTH

	// RESHAPE AND RETURN
}
//==============================================================================
/** Gets the edges in an image.
 */
cv::Mat featureExtractor::getEdges(cv::Mat feature, cv::Mat thresholded,\
cv::Rect roi, cv::Point2f head, cv::Point2f center){
	if(thresholded.empty()){
		std::cerr<<"Edge-feature needs a background model"<<std::endl;
		exit(1);
	}

	// EXTRACT THE EDGES AND ROTATE THE EDGES TO THE RIGHT POSSITION
	cv::Point2f rotBorders;
	feature.convertTo(feature,CV_8UC1);
	cv::Mat tmpFeat(feature.clone(),roi);
	std::vector<cv::Point2f> dummy;
	cv::Point2f rotCenter(tmpFeat.cols/2+roi.x,tmpFeat.rows/2+roi.y);
	tmpFeat = this->rotate2Zero(head,center,tmpFeat,rotBorders,rotCenter,\
				featureExtractor::MATRIX,dummy);

	// PICK OUT ONLY THE THRESHOLDED ARES RESHAPE IT AND RETURN IT
	cv::Mat tmpEdge;
	tmpFeat.copyTo(tmpEdge,thresholded);
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(thresholded,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);

	// IF WE WANT TO SEE HOW THE EXTRACTED EDGES LOOK LIKE
	if(this->plot){
		cv::drawContours(tmpEdge,contours,-1,cv::Scalar(255,255,255));
		cv::imshow("Edges", tmpEdge);
		cv::waitKey(0);
	}

	// WRITE IT ON ONE ROW
	cv::Mat edge = cv::Mat::zeros(cv::Size(2*(tmpEdge.cols*tmpEdge.rows)+1,1),\
					cv::DataType<double>::type);

	// ADD THE CONTOURS ALSO
	cv::drawContours(thresholded,contours,-1,cv::Scalar(255,255,255));
	thresholded = (thresholded).reshape(0,1);
	thresholded.convertTo(thresholded,cv::DataType<double>::type);
	cv::Mat dummy1 = edge.colRange(0, tmpEdge.cols*tmpEdge.rows);
	thresholded.copyTo(dummy1);
	dummy1.release();

	// ADD THE EDGES
	cv::Mat dummy2 = edge.colRange(tmpEdge.cols*tmpEdge.rows, 2*\
						(tmpEdge.cols*tmpEdge.rows));
	tmpEdge = (tmpEdge).reshape(0,1);
	tmpEdge.convertTo(tmpEdge,cv::DataType<double>::type);
	tmpEdge.copyTo(dummy2);
	dummy2.release();

	tmpFeat.release();
	tmpEdge.release();
	if(this->print){
		std::cout<<"Size(EDGES): ("<<edge.cols<<","<<edge.rows<<")"<<std::endl;
		unsigned counter = 0;
		for(int i=0; i<edge.cols,counter<10;i++){
			if(edge.at<double>(0,i)!=0){
				counter++;
				std::cout<<edge.at<double>(0,i)<<" ";
			}
		}
		std::cout<<"..."<<std::endl;
	}
	return edge;
}
//==============================================================================
/** SURF descriptors (Speeded Up Robust Features).
 */
cv::Mat featureExtractor::getSURF(cv::Mat feature, std::vector<cv::Point2f> templ,\
std::vector<cv::Point2f> &indices, cv::Rect roi,cv::Mat test){
	// KEEP THE TOP 10 DESCRIPTORS WITHIN THE BORDERS OF THE TEMPLATE
	unsigned number = 30;
	cv::Mat tmp     = cv::Mat::zeros(cv::Size(feature.cols-2,number),\
						cv::DataType<double>::type);
	unsigned counter = 0;
	for(int y=0; y<feature.rows; y++){
		if(counter == number){
			break;
		}
		double ptX = feature.at<double>(y,feature.cols-2);
		double ptY = feature.at<double>(y,feature.cols-1);
		if(featureExtractor::isInTemplate(ptX, ptY, templ)){
			cv::Mat dummy1 = tmp.row(counter);
			cv::Mat dummy2 = feature.row(y);
			cv::Mat dummy3 = dummy2.colRange(0,feature.cols-2);
			dummy3.copyTo(dummy1);
			dummy1.release();
			dummy2.release();
			dummy3.release();
			indices.push_back(cv::Point2f(ptX-roi.x,ptY-roi.y));
			counter++;
		}
	}

	if(this->plot && !test.empty()){
		for(std::size_t l=0; l<indices.size(); l++){
			cv::circle(test,indices[l],3,cv::Scalar(0,0,255));
		}
		cv::imshow("SURF",test);
		cv::waitKey(0);
	}

	// COPY THE DESCRIPTORS IN THE FINAL MATRIX
	cv::Mat surf = cv::Mat::zeros(cv::Size(tmp.rows*tmp.cols+2,1),\
					cv::DataType<double>::type);
	tmp = tmp.reshape(0,1);
	tmp.convertTo(tmp,cv::DataType<double>::type);
	cv::Mat dummy = surf.colRange(0,tmp.rows*tmp.cols);
	tmp.copyTo(dummy);
	tmp.release();
	dummy.release();
	surf.convertTo(surf,cv::DataType<double>::type);

	// IF WE WANT TO SEE SOME VALUES/IMAGES
	if(this->print){
		std::cout<<"Size(SURF): ("<<surf.rows<<","<<surf.cols<<")"<<std::endl;
		for(int i=0; i<std::min(10,surf.cols);i++){
			std::cout<<surf.at<double>(0,i)<<" ";
		}
		std::cout<<"..."<<std::endl;
	}
	return surf;
}
//==============================================================================
/** Creates a "histogram" of interest points + number of blobs.
 */
cv::Mat featureExtractor::getPointsGrid(cv::Mat feature, cv::Rect roi,\
featureExtractor::templ aTempl, cv::Mat test){
	// GET THE GRID SIZE FROM THE TEMPLATE SIZE
	unsigned no     = 10;
	cv::Mat rowData = cv::Mat::zeros(cv::Size(no*no+1,1),cv::DataType<double>::type);
	double rateX    = (aTempl.extremes[1]-aTempl.extremes[0])/static_cast<double>(no);
	double rateY    = (aTempl.extremes[3]-aTempl.extremes[2])/static_cast<double>(no);

	// KEEP ONLY THE KEYPOINTS THAT ARE IN THE TEMPLATE
	std::vector<cv::Point2f> indices;
	for(int y=0; y<feature.rows; y++){
		double ptX = feature.at<double>(y,0);
		double ptY = feature.at<double>(y,1);
		if(featureExtractor::isInTemplate(ptX, ptY, aTempl.points)){
			cv::Point2f pt(ptX, ptY);
			indices.push_back(pt);

			// CHECK IN WHAT CELL OF THE GRID THE CURRENT POINTS FALL
			unsigned counter = 0;
			for(double ix=aTempl.extremes[0]; ix<aTempl.extremes[1]-0.01; ix+=rateX){
				for(double iy=aTempl.extremes[2]; iy<aTempl.extremes[3]-0.01; iy+=rateY){
					if(ix<=pt.x && pt.x<(ix+rateX) && iy<=pt.y && pt.y<(iy+rateY)){
						rowData.at<double>(0,counter) += 1.0;
					}
					counter +=1;
				}
			}
		}
	}
	if(this->plot && !test.empty()){
		for(std::size_t l=0; l<indices.size(); l++){
			cv::circle(test,cv::Point2f(indices[l].x-roi.x,indices[l].y-roi.y),\
				3,cv::Scalar(0,0,255));
		}
		cv::imshow("IPOINTS",test);
		cv::waitKey(0);
	}

	// IF WE WANT TO SEE THE VALUES THAT WERE STORED
	if(this->print){
		std::cout<<"Size(IPOINTS): ("<<rowData.cols<<","<<rowData.rows<<")"<<std::endl;
		unsigned counter = 0;
		for(int i=0; i<rowData.cols,counter<10; i++){
			if(rowData.at<double>(0,i)!=0){
				std::cout<<rowData.at<double>(0,i)<<" ";
				counter++;
			}
		}
		std::cout<<std::endl;
	}
	return rowData;
}
//==============================================================================
/** Creates a gabor with the parameters given by the parameter vector.
 */
cv::Mat featureExtractor::createGabor(double *params){
	// params[0] -- sigma: (3, 68) // the actual size
	// params[1] -- gamma: (0.2, 1) // how round the filter is
	// params[2] -- dimension: (1, 10) // size
	// params[3] -- theta: (0, 180) or (-90, 90) // angle
	// params[4] -- lambda: (2, 256) // thickness
	// params[5] -- psi: (0, 180) // number of lines

	// SET THE PARAMTETERS OF THE GABOR FILTER
	if(params == NULL){
		params    = new double[6];
		params[0] = 10.0; params[1] = 0.9; params[2] = 2.0;
		params[3] = M_PI/4.0; params[4] = 50.0; params[5] = 12.0;
	}

	// CREATE THE GABOR FILTER OR WAVELET
	cv::Mat gabor;
	float sigmaX = params[0];
	float sigmaY = params[0]/params[1];
	float xMax   = std::max(std::abs(params[2]*sigmaX*std::cos(params[3])), \
							std::abs(params[2]*sigmaY*std::sin(params[3])));
	xMax         = std::ceil(std::max((float)1.0, xMax));
	float yMax  = std::max(std::abs(params[2]*sigmaX*std::cos(params[3])), \
							std::abs(params[2]*sigmaY*std::sin(params[3])));
	yMax         = std::ceil(std::max((float)1.0, yMax));
	float xMin  = -xMax;
	float yMin  = -yMax;
	gabor        = cv::Mat::zeros(cv::Size((int)(xMax-xMin),(int)(yMax-yMin)),\
					cv::DataType<float>::type);
	for(int x=(int)xMin; x<xMax; x++){
		for(int y=(int)yMin; y<yMax; y++){
			float xPrime = x*std::cos(params[3])+y*std::sin(params[3]);
			float yPrime = -x*std::sin(params[3])+y*std::cos(params[3]);
			gabor.at<float>((int)(y+yMax),(int)(x+xMax)) = \
				std::exp(-0.5*((xPrime*xPrime)/(sigmaX*sigmaX)+\
				(yPrime*yPrime)/(sigmaY*sigmaY)))*\
				std::cos(2.0 * M_PI/params[4]*xPrime*params[5]);
		}
	}
	return gabor;
}
//==============================================================================
/** Convolves an image with a Gabor filter with the given parameters and
 * returns the response image.
 */
cv::Mat featureExtractor::getGabor(cv::Mat feature, cv::Mat thresholded,\
cv::Rect roi, cv::Point2f center, cv::Point2f head, cv::Size foregrSize){
	unsigned gaborNo    = std::ceil(feature.rows/height);
	int gaborRows       = std::ceil(feature.rows/gaborNo);
	unsigned resultCols = foregrSize.width*foregrSize.height;
	cv::Mat result      = cv::Mat::zeros(cv::Size(2*resultCols+1,1),\
								cv::DataType<double>::type);
	cv::Point2f rotCenter(foregrSize.width/2+roi.x,foregrSize.height/2+roi.y);
	std::vector<cv::Point2f> dummy;
	for(unsigned i=0; i<gaborNo; i++){
		// GET THE ROI OUT OF THE iTH GABOR
		cv::Mat tmp1 = feature.rowRange(i*gaborRows,(i+1)*gaborRows);
		tmp1.convertTo(tmp1,CV_8UC1);
		cv::Mat tmp2(tmp1, roi);

		// ROTATE EACH GABOR TO THE RIGHT POSITION
		cv::Point2f rotBorders;
		tmp2 = this->rotate2Zero(head,center,tmp2.clone(),rotBorders,rotCenter,\
				featureExtractor::MATRIX,dummy);
		// KEEP ONLY THE THRESHOLDED VALUES
		cv::Mat tmp3;
		tmp2.copyTo(tmp3,thresholded);
		if(this->plot){
			cv::imshow("GaborResponse",tmp3);
			cv::waitKey(0);
		}

		// RESHAPE AND STORE IN THE RIGHT PLACE
		tmp3 = tmp3.reshape(0,1);
		tmp3.convertTo(tmp3,cv::DataType<double>::type);
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
		for(int i=0; i<result.cols,counter<10;i++){
			if(result.at<double>(0,i)!=0){
				counter++;
				std::cout<<result.at<double>(0,i)<<" ";
			}
		}
		std::cout<<"..."<<std::endl;
	}
	return result;
}
//==============================================================================
/** Compute the features from the SIFT descriptors by doing vector quantization.
 */
cv::Mat featureExtractor::getSIFT(cv::Mat feature,std::vector<cv::Point2f> templ,
std::vector<cv::Point2f> &indices, cv::Rect roi, cv::Mat test){
	// KEEP ONLY THE SIFT FEATURES THAT ARE WITHIN THE TEMPLATE
	cv::Mat tmp = cv::Mat::zeros(cv::Size(feature.cols-2,feature.rows),\
					cv::DataType<double>::type);
	unsigned counter = 0;
	for(int y=0; y<feature.rows; y++){
		double ptX = feature.at<double>(y,feature.cols-2);
		double ptY = feature.at<double>(y,feature.cols-1);
		if(featureExtractor::isInTemplate(ptX, ptY, templ)){
			cv::Mat dummy1 = tmp.row(counter);
			cv::Mat dummy2 = feature.row(y);
			cv::Mat dummy3 = dummy2.colRange(0,dummy2.cols-2);
			dummy3.copyTo(dummy1);
			dummy1.release();
			dummy2.release();
			dummy3.release();
			indices.push_back(cv::Point2f(ptX-roi.x,ptY-roi.y));
			counter++;
		}
	}

	// KEEP ONLY THE NON-ZEROS ROWS OUT OF tmp
	cv::Mat preFeature;
	cv::Mat dum = tmp.rowRange(0,counter);
	dum.copyTo(preFeature);
	dum.release();
	tmp.release();
	preFeature.convertTo(preFeature,cv::DataType<double>::type);

	// ASSUME THAT THERE IS ALREADY A SIFT DICTIONARY AVAILABLE
	if(this->dictionarySIFT.empty()){
		binFile2mat(this->dictionarySIFT, const_cast<char*>(this->dictFilename.c_str()));
	}

	// COMPUTE THE DISTANCES FROM EACH NEW FEATURE TO THE DICTIONARY ONES
	cv::Mat distances = cv::Mat::zeros(cv::Size(preFeature.rows,\
						this->dictionarySIFT.rows),cv::DataType<double>::type);
	cv::Mat minDists  = cv::Mat::zeros(cv::Size(preFeature.rows, 1),\
						cv::DataType<double>::type);
	cv::Mat minLabel  = cv::Mat::zeros(cv::Size(preFeature.rows, 1),\
						cv::DataType<double>::type);
	minDists -= 1;
	for(int j=0; j<preFeature.rows; j++){
		for(int i=0; i<this->dictionarySIFT.rows; i++){
			cv::Mat diff;
			cv::absdiff(this->dictionarySIFT.row(i),preFeature.row(j),diff);
			distances.at<double>(i,j) = diff.dot(diff);
			diff.release();
			if(minDists.at<double>(0,j)==-1 ||\
			minDists.at<double>(0,j)>distances.at<double>(i,j)){
				minDists.at<double>(0,j) = distances.at<double>(i,j);
				minLabel.at<double>(0,j) = i;
			}
		}
	}

	// CREATE A HISTOGRAM(COUNT TO WHICH DICT FEATURE WAS ASSIGNED EACH NEW ONE)
	cv::Mat sift = cv::Mat::zeros(cv::Size(this->dictionarySIFT.rows+2, 1),\
					cv::DataType<double>::type);
	for(int i=0; i<minLabel.cols; i++){
		int which = minLabel.at<double>(0,i);
		sift.at<double>(0,which) += 1.0;
	}

	// NORMALIZE THE HOSTOGRAM
	cv::Scalar scalar = cv::sum(sift);
	sift /= static_cast<double>(scalar[0]);

	if(this->plot && !test.empty()){
		for(std::size_t l=0; l<indices.size(); l++){
			cv::circle(test,indices[l],3,cv::Scalar(0,0,255));
		}
		cv::imshow("SIFT",test);
		cv::waitKey(0);
	}
	preFeature.release();
	distances.release();
	minDists.release();
	minLabel.release();

	// IF WE WANT TO SEE THE VALUES THAT WERE STORED
	if(this->print){
		std::cout<<"Size(SIFT): ("<<sift.cols<<","<<sift.rows<<")"<<std::endl;
		unsigned counter = 0;
		for(int i=0; i<sift.cols,counter<10; i++){
			if(sift.at<double>(0,i)!=0){
				std::cout<<sift.at<double>(0,i)<<" ";
				counter++;
			}
		}
		std::cout<<std::endl;
	}
	return sift;
}
//==============================================================================
/** Creates a data matrix for each image and stores it locally.
 */
void featureExtractor::extractFeatures(cv::Mat image, std::string sourceName,\
int colorspaceCode){
	cv::cvtColor(image, image, colorspaceCode);

	if(this->featureFile[this->featureFile.size()-1]!='/'){
		this->featureFile = this->featureFile + '/';
	}
	std::cout<<"In extract features"<<std::endl;

	// FOR EACH LOCATION IN THE IMAGE EXTRACT FEATURES AND STORE
	cv::Mat feature;
	std::string toWrite = this->featureFile;
	switch(this->featureType){
		case (featureExtractor::IPOINTS):
			toWrite += "IPOINTS/";
			file_exists(toWrite.c_str(), true);
			feature = this->extractPointsGrid(image);
			break;
		case featureExtractor::EDGES:
			toWrite += "EDGES/";
			file_exists(toWrite.c_str(), true);
			feature = this->extractEdges(image);
			break;
		case featureExtractor::SURF:
			toWrite += "SURF/";
			file_exists(toWrite.c_str(), true);
			feature = this->extractSURF(image);
			break;
		case featureExtractor::GABOR:
			toWrite += "GABOR/";
			file_exists(toWrite.c_str(), true);
			feature = this->extractGabor(image);
			break;
		case featureExtractor::SIFT:
			toWrite += "SIFT/";
			file_exists(toWrite.c_str(), true);
			feature = this->extractSIFT(image);
			break;
	}
	feature.convertTo(feature, cv::DataType<double>::type);

	//WRITE THE FEATURE TO A BINARY FILE
	unsigned pos1       = sourceName.find_last_of("/\\");
	std::string imgName = sourceName.substr(pos1+1);
	unsigned pos2       = imgName.find_last_of(".");
	imgName             = imgName.substr(0,pos2);
	toWrite += (imgName + ".bin");

	std::cout<<"Feature written to: "<<toWrite<<std::endl;
	mat2BinFile(feature, const_cast<char*>(toWrite.c_str()));
	feature.release();
}
//==============================================================================
/** Extract the interest points in a gird and returns them.
 */
cv::Mat featureExtractor::extractPointsGrid(cv::Mat image){
	// EXTRACT MAXIMALLY STABLE BLOBS
	std::vector<std::vector<cv::Point> > msers;
	cv::MSER aMSER;
	aMSER(image, msers, cv::Mat());
	unsigned msersNo = 0;
	for(std::size_t x=0; x<msers.size(); x++){
		msersNo += msers[x].size();
	}

	// NICE FEATURE POINTS
	std::vector<cv::Point2f> corners;
	cv::Ptr<cv::FeatureDetector> detector = new cv::GoodFeaturesToTrackDetector(\
											5000, 0.00001, 1.0, 3.0);
	cv::GridAdaptedFeatureDetector gafd(detector, 5000, 10, 10);
	std::vector<cv::KeyPoint> keys;
	std::deque<unsigned> indices;
	gafd.detect(image, keys);

	cv::Mat interestPoints = cv::Mat::zeros(cv::Size(2,msersNo+keys.size()),\
								cv::DataType<double>::type);
	// WRITE THE MSERS LOCATIONS IN THE MATRIX
	unsigned counts = 0;
	for(std::size_t x=0; x<msers.size(); x++){
		for(std::size_t y=0; y<msers[x].size(); y++){
			interestPoints.at<double>(counts,0) = msers[x][y].x;
			interestPoints.at<double>(counts,1) = msers[x][y].y;
			counts++;
		}
	}

	// WRITE THE KEYPOINTS IN THE MATRIX
	for(std::size_t i=0; i<keys.size(); i++){
		interestPoints.at<double>(counts+i,0) = keys[i].pt.x;
		interestPoints.at<double>(counts+i,1) = keys[i].pt.y;
	}

	if(this->plot){
		for(int y=0; y<interestPoints.rows; y++){
			cv::Scalar color;
			if(y<msersNo){
				color = cv::Scalar(0,0,255);
			}else{
				color = cv::Scalar(255,0,0);
			}
			double ptX = interestPoints.at<double>(y,0);
			double ptY = interestPoints.at<double>(y,1);
			cv::circle(image, cv::Point2f(ptX,ptY),3,color);
		}
		cv::imshow("IPOINTS", image);
		cv::waitKey(0);
	}
	return interestPoints;
}
//==============================================================================
/** Extract edges from the whole image.
 */
cv::Mat featureExtractor::extractEdges(cv::Mat image){
	cv::Mat gray, edges;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	normalizeMat(gray);
	gray *= 255.0;
	cv::equalizeHist(gray,gray);
	cv::medianBlur(gray, gray, 3);
	cv::Canny(gray, edges, 100, 0, 3, true);
	edges.convertTo(edges,cv::DataType<double>::type);

	if(this->plot){
		cv::imshow("edges",edges);
		cv::waitKey(0);
	}
	gray.release();
	return edges;
}
//==============================================================================
/** Extracts all the surf descriptors from the whole image and writes them in a
 * matrix.
 */
cv::Mat featureExtractor::extractSURF(cv::Mat image){
	std::vector<float> descriptors;
	std::vector<cv::KeyPoint> keypoints;
	cv::SURF aSURF = cv::SURF(0.01,4,2,false);

	// EXTRACT INTEREST POINTS FROM THE IMAGE
	cv::Mat gray;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	normalizeMat(gray);
	gray *= 255.0;
	gray.convertTo(gray,CV_8UC1);
	cv::equalizeHist(gray, gray);
	cv::medianBlur(gray, gray, 3);
	aSURF(gray, cv::Mat(), keypoints, descriptors, false);
	gray.release();

	// WRITE ALL THE DESCRIPTORS IN THE STRUCTURE OF KEY-DESCRIPTORS
	std::deque<featureExtractor::keyDescr> kD;
	for(std::size_t i=0; i<keypoints.size();i++){
		featureExtractor::keyDescr tmp;
		for(int j=0; j<aSURF.descriptorSize();j++){
			tmp.descr.push_back(descriptors[i*aSURF.descriptorSize()+j]);
		}
		tmp.keys = keypoints[i];
		kD.push_back(tmp);
		tmp.descr.clear();
	}

	// SORT THE REMAINING DESCRIPTORS SO WE DON'T NEED TO DO THAT LATER
	std::sort(kD.begin(),kD.end(),(&featureExtractor::compareDescriptors));

	// WRITE THEM IN THE MATRIX AS FOLLOWS:
	cv::Mat surfs = cv::Mat::zeros(cv::Size(aSURF.descriptorSize()+2,kD.size()),\
					cv::DataType<double>::type);
	for(std::size_t i=0; i<kD.size();i++){
		if(i<10){
			std::cout<<kD[i].keys.response<<std::endl;
		}
		surfs.at<double>(i,aSURF.descriptorSize()) = kD[i].keys.pt.x;
		surfs.at<double>(i,aSURF.descriptorSize()+1) = kD[i].keys.pt.y;
		for(int j=0; j<aSURF.descriptorSize();j++){
			surfs.at<double>(i,j) = kD[i].descr[j];
		}
	}

	// PLOT THE KEYPOINTS TO SEE THEM
	if(this->plot){
		for(std::size_t i=0; i<kD.size();i++){
			cv::circle(image, cv::Point2f(kD[i].keys.pt.x,kD[i].keys.pt.y),\
				3,cv::Scalar(0,0,255));
		}
		cv::imshow("SURFS", image);
		cv::waitKey(0);
	}
	return surfs;
}
//==============================================================================
/** Convolves the whole image with some Gabors wavelets and then stores the
 * results.
 */
cv::Mat featureExtractor::extractGabor(cv::Mat image){
	// DEFINE THE PARAMETERS FOR A FEW GABORS
	// params[0] -- sigma: (3, 68) // the actual size
	// params[1] -- gamma: (0.2, 1) // how round the filter is
	// params[2] -- dimension: (1, 10) // size
	// params[3] -- theta: (0, 180) or (-90, 90) // angle
	// params[4] -- lambda: (2, 256) // thickness
	// params[5] -- psi: (0, 180) // number of lines
	std::deque<double*> allParams;
	double *params1 = new double[6];
	params1[0] = 10.0; params1[1] = 0.9; params1[2] = 2.0;
	params1[3] = M_PI/4.0; params1[4] = 50.0; params1[5] = 15.0;
	allParams.push_back(params1);

	// THE PARAMETERS FOR THE SECOND GABOR
	double *params2 = new double[6];
	params2[0] = 10.0; params2[1] = 0.9; params2[2] = 2.0;
	params2[3] = 3.0*M_PI/4.0; params2[4] = 50.0; params2[5] = 15.0;
	allParams.push_back(params2);

	// CONVERT THE IMAGE TO GRAYSCALE TO APPLY THE FILTER
	cv::Mat gray;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	normalizeMat(gray);
	gray *= 255.0;
	cv::equalizeHist(gray, gray);
	cv::medianBlur(gray, gray, 3);

	// CREATE EACH GABOR AND CONVOLVE THE IMAGE WITH IT
	cv::Mat gabors = cv::Mat::zeros(cv::Size(image.cols,image.rows*allParams.size()),\
						cv::DataType<double>::type);
	for(unsigned i=0; i<allParams.size(); i++){
		cv::Mat response;
		cv::Mat agabor = this->createGabor(allParams[i]);
		cv::filter2D(gray, response, -1, agabor, cv::Point2f(-1,-1), 0,\
			cv::BORDER_REPLICATE);
		cv::Mat temp = gabors.rowRange(i*response.rows,(i+1)*response.rows);
		response.convertTo(response,cv::DataType<double>::type);
		response.copyTo(temp);

		// IF WE WANT TO SEE THE GABOR AND THE RESPONSE
		if(this->plot){
			cv::imshow("GaborFilter", agabor);
			cv::imshow("GaborResponse", temp);
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
	return gabors;
}
//==============================================================================
/** Extracts SIFT features from the image and stores them in a matrix.
 */
cv::Mat featureExtractor::extractSIFT(cv::Mat image, std::vector<cv::Point2f> templ,\
cv::Rect roi){
	// DEFINE THE SURF KEYPOINTS AND THE DESCRIPTORS
	std::vector<cv::KeyPoint> keypoints;
	cv::SIFT::DetectorParams detectP  = cv::SIFT::DetectorParams(0.0001,10.0);
	cv::SIFT::DescriptorParams descrP = cv::SIFT::DescriptorParams();
	cv::SIFT::CommonParams commonP    = cv::SIFT::CommonParams();
	cv::SIFT aSIFT(commonP, detectP, descrP);

	// EXTRACT SIFT FEATURES IN THE IMAGE
	cv::Mat gray, sift;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	normalizeMat(gray);
	gray *= 255.0;
	cv::equalizeHist(gray, gray);
	cv::medianBlur(gray, gray, 3);
	aSIFT(gray, cv::Mat(), keypoints);

	// WE USE THE SAME FUNCTION TO BUILD THE DICTIONARY ALSO
	if(templ.size()!=0){
		sift = cv::Mat::zeros(keypoints.size(),aSIFT.descriptorSize(),\
				cv::DataType<double>::type);
		std::vector<cv::KeyPoint> goodKP;
		for(std::size_t i=0; i<keypoints.size();i++){
			if(featureExtractor::isInTemplate(keypoints[i].pt.x+roi.x,\
			keypoints[i].pt.y+roi.y,templ)){
				goodKP.push_back(keypoints[i]);
			}
		}
		aSIFT(gray, cv::Mat(), goodKP, sift, true);
		if(this->plot){
			for(std::size_t i=0; i<goodKP.size(); i++){
				cv::circle(image,goodKP[i].pt,3,cv::Scalar(0,0,255));
			}
			cv::imshow("SIFT_DICT", image);
			cv::waitKey(0);
		}

	// IF WE ONLY WANT TO STORE THE SIFT FEATURE WE NEED TO ADD THE x-S AND y-S
	}else{
		sift = cv::Mat::zeros(keypoints.size(),aSIFT.descriptorSize()+2,\
				cv::DataType<double>::type);
		cv::Mat dummy1 = sift.colRange(0,aSIFT.descriptorSize());
		cv::Mat dummy2;
		aSIFT(gray, cv::Mat(), keypoints, dummy2, true);
		dummy2.convertTo(dummy2, cv::DataType<double>::type);
		dummy2.copyTo(dummy1);
		dummy1.release();
		dummy2.release();
		// PLOT THE KEYPOINTS TO SEE THEM
		if(this->plot){
			for(std::size_t i=0; i<keypoints.size();i++){
				cv::circle(image, keypoints[i].pt,3,cv::Scalar(0,0,255));
			}
			cv::imshow("SIFT", image);
			cv::waitKey(0);
		}
	}

	// NORMALIZE THE FEATURE
	sift.convertTo(sift, cv::DataType<double>::type);
	for(int i=0; i<sift.rows; i++){
		cv::Mat rowsI = sift.row(i);
		rowsI         = rowsI/cv::norm(rowsI);
		rowsI.release();
	}

	// IF WE WANT TO STORE THE SIFT FEATURES THEN WE NEED TO STORE x AND y
	if(templ.size()==0){
		for(std::size_t i=0; i<keypoints.size();i++){
			sift.at<double>(i,aSIFT.descriptorSize())   = keypoints[i].pt.x;
			sift.at<double>(i,aSIFT.descriptorSize()+1) = keypoints[i].pt.y;
		}
	}
	gray.release();
	return sift;
}
//==============================================================================
/** Returns the row corresponding to the indicated feature type.
 */
cv::Mat featureExtractor::getDataRow(cv::Mat image, featureExtractor::templ aTempl,\
cv::Rect roi,featureExtractor::people person, cv::Mat thresholded,\
cv::vector<cv::Point2f> &keys, std::string imgName, cv::Point2f absRotCenter,\
cv::Point2f rotBorders){
	cv::Mat dataRow, feature;
	std::string toRead;
	cv::Mat dictImage;
	switch(this->featureType){
		case (featureExtractor::IPOINTS):
			toRead = (this->featureFile+"IPOINTS/"+imgName+".bin");
			binFile2mat(feature, const_cast<char*>(toRead.c_str()));
			feature.convertTo(feature,cv::DataType<double>::type);
			feature = this->rotate2Zero(aTempl.head,aTempl.center,feature,rotBorders,\
						absRotCenter,featureExtractor::KEYS,keys);
			dataRow = this->getPointsGrid(feature,roi,aTempl,person.pixels);
			break;
		case featureExtractor::EDGES:
			toRead = (this->featureFile+"EDGES/"+imgName+".bin");
			binFile2mat(feature, const_cast<char*>(toRead.c_str()));
			dataRow = this->getEdges(feature,thresholded,roi,aTempl.head,\
						aTempl.center);
			break;
		case featureExtractor::PIXELS:
			dataRow = this->getPixels(thresholded,aTempl);
			break;
		case featureExtractor::SURF:
			toRead = (this->featureFile+"SURF/"+imgName+".bin");
			binFile2mat(feature, const_cast<char*>(toRead.c_str()));
			feature.convertTo(feature,cv::DataType<double>::type);
			feature = this->rotate2Zero(aTempl.head,aTempl.center,feature,rotBorders,\
						absRotCenter,featureExtractor::KEYS,keys);
			dataRow = this->getSURF(feature,aTempl.points,keys,roi,person.pixels);
			break;
		case featureExtractor::GABOR:
			toRead = (this->featureFile+"GABOR/"+imgName+".bin");
			binFile2mat(feature, const_cast<char*>(toRead.c_str()));
			dataRow = this->getGabor(feature,thresholded,roi,aTempl.center,\
					aTempl.head,person.pixels.size());
			break;
		case featureExtractor::SIFT_DICT:
			dictImage = cv::Mat(image, roi);
			dictImage = this->rotate2Zero(aTempl.head,aTempl.center,dictImage,\
						rotBorders,absRotCenter,featureExtractor::MATRIX,keys);
			dataRow   = this->extractSIFT(dictImage, aTempl.points, roi);
			break;
		case featureExtractor::SIFT:
			toRead = (this->featureFile+"SIFT/"+imgName+".bin");
			binFile2mat(feature, const_cast<char*>(toRead.c_str()));
			feature.convertTo(feature,cv::DataType<double>::type);
			feature = this->rotate2Zero(aTempl.head,aTempl.center,feature,rotBorders,\
						absRotCenter,featureExtractor::KEYS,keys);
			dataRow = this->getSIFT(feature,aTempl.points,keys,roi,person.pixels);
			break;
	}
	dictImage.release();
	feature.release();
	return dataRow;
}
//==============================================================================
