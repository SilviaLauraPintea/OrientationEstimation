/* featureDetector.cpp
 * Author: Silvia-Laura Pintea
 */
#include "featureDetector.h"
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
void featureDetector::init(std::string dataFolder, std::string theAnnotationsFile,\
bool readFromFolder){
	this->initProducer(readFromFolder, const_cast<char*>(dataFolder.c_str()));
	if(!this->entireNext.empty()){
		this->entireNext.release();
	}

	// CLEAR DATA AND TARGETS
	if(!this->targets.empty()){
		this->targets.release();
	}
	if(!this->data.empty()){
		this->data.release();
	}

	// CLEAR THE ANNOTATIONS
	for(std::size_t i=0; i<this->targetAnno.size(); i++){
		for(std::size_t j=0; j<this->targetAnno[i].annos.size(); j++){
			this->targetAnno[i].annos[j].poses.clear();
		}
		this->targetAnno[i].annos.clear();
	}
	this->targetAnno.clear();
	// LOAD THE DESIRED ANNOTATIONS FROM THE FILE
	if(!theAnnotationsFile.empty() && this->targetAnno.empty()){
		annotationsHandle::loadAnnotations(const_cast<char*>\
			(theAnnotationsFile.c_str()), this->targetAnno);
	}
	this->lastIndex = 0;
}
//==============================================================================
/** Get perpendicular to a line given by 2 points A, B in point C.
 */
void featureDetector::getLinePerpendicular(cv::Point2f A, cv::Point2f B,
cv::Point2f C, double &m, double &b){
	double slope = (double)(B.y - A.y)/(double)(B.x - A.x);
	m            = -1.0/slope;
	b            = C.y - m * C.x;
}
//==============================================================================
/** Checks to see if a point is on the same side of a line like another given point.
 */
bool featureDetector::sameSubplane(cv::Point2f test, cv::Point2f point,\
double m, double b){
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
void featureDetector::gaussianKernel(cv::Mat &gauss, cv::Size size,\
double sigma, cv::Point2f offset){
	int xmin = -1 * size.width/2, xmax = size.width/2;
	int ymin = -1 * size.height/2, ymax = size.height/2;

	gauss = cv::Mat::zeros(size.height,size.width,cv::DataType<float>::type);
	for(int x=xmin; x<xmax; x++){
		for(int y=ymin; y<ymax; y++){
			gauss.at<float>(y+ymax,x+xmax) = \
				std::exp(-0.5*((x+offset.x)*(x+offset.x)+(y+offset.y)*\
						(y+offset.y))/(sigma*sigma));
		}
	}
}
//==============================================================================
/** Gets the edges in an image.
 */
cv::Mat featureDetector::getEdges(cv::Mat feature, cv::Mat thresholded,\
cv::Rect roi, cv::Point2f head, cv::Point2f center){
	// EXTRACT THE EDGES AND ROTATE THE EDGES TO THE RIGHT POSSITION
	cv::Point2f rotBorders;
	feature.convertTo(feature,CV_8UC1);
	cv::Mat tmpFeat(feature.clone(),roi);
	tmpFeat = this->rotate2Zero(head,center,tmpFeat.clone(),rotBorders);

	// PICK OUT ONLY THE THRESHOLDED ARES RESHAPE IT AND RETURN IT
	cv::Mat tmpEdge;
	tmpFeat.copyTo(tmpEdge,thresholded);

	// IF WE WANT TO SEE HOW THE EXTRACTED EDGES LOOK LIKE
	if(this->plotTracks){
		cv::imshow("Edges", tmpEdge);
		cv::waitKey(0);
	}

	// WRITE IT ON ONE ROW
	cv::Mat edge = cv::Mat::zeros(cv::Size(tmpEdge.cols*tmpEdge.rows+1,1),\
					cv::DataType<double>::type);
	cv::Mat dummy = edge.colRange(0,tmpEdge.cols*tmpEdge.rows);
	tmpEdge = (tmpEdge).reshape(0,1);
	tmpEdge.convertTo(tmpEdge,cv::DataType<double>::type);
	tmpEdge.copyTo(dummy);
	dummy.release();
	tmpFeat.release();
	tmpEdge.release();

	if(this->printValues){
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
/** Compares SURF 2 descriptors and returns the boolean value of their comparison.
 */
bool featureDetector::compareDescriptors(const featureDetector::keyDescr k1,\
const featureDetector::keyDescr k2){
	return (k1.keys.response>k2.keys.response);
}
//==============================================================================
/** SURF descriptors (Speeded Up Robust Features).
 */
cv::Mat featureDetector::getSURF(cv::Mat feature, std::vector<cv::Point2f> templ,\
std::vector<cv::Point2f> &indices, cv::Rect roi,cv::Mat test){
	// KEEP THE TOP 10 DESCRIPTORS WITHIN THE BORDERS OF THE TEMPLATE
	cv::Mat tmp = cv::Mat::zeros(cv::Size(feature.cols-2,10),\
					cv::DataType<double>::type);
	unsigned counter = 0;
	for(int y=0; y<feature.rows; y++){
		if(counter == 10){
			break;
		}
		double ptX = feature.at<double>(y,feature.cols-2);
		double ptY = feature.at<double>(y,feature.cols-1);
		if(this->isInTemplate(ptX, ptY, templ)){
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

	if(this->plotTracks && !test.empty()){
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
	std::cout<<"Size(SURF): ("<<surf.rows<<","<<surf.cols<<")"<<std::endl;
	if(this->printValues){
		for(int i=0; i<std::min(10,surf.cols);i++){
			std::cout<<surf.at<double>(0,i)<<" ";
		}
		std::cout<<"..."<<std::endl;
	}
	return surf;
}
//==============================================================================
/** Get template extremities (if needed, considering some borders --
 * relative to the ROI).
 */
std::deque<double> featureDetector::templateExtremes(\
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
/** Creates a "histogram" of interest points + number of blobs.
 */
cv::Mat featureDetector::getPointsGrid(cv::Mat feature,cv::Rect roi,\
std::vector<cv::Point2f> templ, std::deque<double> templExtremes,cv::Mat test){
	// GET THE GRID SIZE FROM THE TEMPLATE SIZE
	unsigned no     = 10;
	cv::Mat rowData = cv::Mat::zeros(cv::Size(no*no+1,1),cv::DataType<double>::type);
	double rateX    = (templExtremes[1]-templExtremes[0])/static_cast<double>(no);
	double rateY    = (templExtremes[3]-templExtremes[2])/static_cast<double>(no);

	// KEEP ONLY THE KEYPOINTS THAT ARE IN THE TEMPLATE
	std::vector<cv::Point2f> indices;
	for(int y=0; y<feature.rows; y++){
		double ptX = feature.at<double>(y,0);
		double ptY = feature.at<double>(y,1);
		if(this->isInTemplate(ptX, ptY, templ)){
			indices.push_back(cv::Point(ptX, ptY));
		}
	}

	if(this->plotTracks && !test.empty()){
		for(std::size_t l=0; l<indices.size(); l++){
			cv::circle(test,cv::Point2f(indices[l].x-roi.x,indices[l].y-roi.y),\
				3,cv::Scalar(0,0,255));
		}
		cv::imshow("IPOINTS",test);
		cv::waitKey(0);
	}

	// FOR EACH GRID SLICE COUNT HOW MANY POINTS ARE IN IT
	unsigned counter = 0;
	for(double x=templExtremes[0]; x<templExtremes[1]-0.01; x+=rateX){
		for(double y=templExtremes[2]; y<templExtremes[3]-0.01; y+=rateY){
			for(std::size_t j=0; j<indices.size(); j++){
				if(x<=indices[j].x && indices[j].x<(x+rateX) &&\
				y<=indices[j].y && indices[j].y<(y+rateY)){
					rowData.at<double>(0,counter) += 1.0;
				}
			}
			counter +=1;
		}
	}

	// IF WE WANT TO SEE THE VALUES THAT WERE STORED
	if(this->printValues){
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
/** Just displaying an image a bit larger to visualize it better.
 */
void featureDetector::showZoomedImage(cv::Mat image, const std::string title){
	cv::Mat large;
	cv::resize(image, large, cv::Size(0,0), 5, 5, cv::INTER_CUBIC);
	cv::imshow(title, large);
	cv::waitKey(0);
	cvDestroyWindow(title.c_str());
	large.release();
}
//==============================================================================
/** Shows a ROI in a given image.
 */
void featureDetector::showROI(cv::Mat image, cv::Point2f top_left, cv::Size ROI_size){
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
std::vector<cv::Point2f> templ){
	std::vector<cv::Point2f> hull;
	convexHull(templ, hull);
	double minDist = -1;
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
bool featureDetector::isInTemplate(unsigned pixelX, unsigned pixelY,\
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
	static_cast<int>(iter->start)-5 <= static_cast<int>(pixelX) &&\
	static_cast<int>(iter->end)+5 >= static_cast<int>(pixelX)){
		return true;
	}else{
		return false;
	}
}
//==============================================================================
/** Returns the size of a window around a template centered in a given point.
 */
void featureDetector::templateWindow(cv::Size imgSize, int &minX, int &maxX,\
int &minY, int &maxY, std::vector<cv::Point2f> &templ, int tplBorder){
	// GET THE MIN/MAX SIZE OF THE TEMPLATE
	for(unsigned i=0; i<templ.size(); i++){
		if(minX>=templ[i].x){minX = templ[i].x;}
		if(maxX<=templ[i].x){maxX = templ[i].x;}
		if(minY>=templ[i].y){minY = templ[i].y;}
		if(maxY<=templ[i].y){maxY = templ[i].y;}
	}

	// TRY TO ADD BORDERS TO MAKE IT 100
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
/** Get the foreground pixels corresponding to each person.
 */
void featureDetector::allForegroundPixels(std::deque<featureDetector::people>\
&allPeople, std::deque<unsigned> existing, IplImage *bg, double threshold){
	// INITIALIZING STUFF
	cv::Mat thsh(bg);
	cv::Mat thrsh(thsh.rows, thsh.cols, CV_8UC1);
	cv::cvtColor(thsh, thrsh, CV_BGR2GRAY);
	cv::threshold(thrsh, thrsh, threshold, 255, cv::THRESH_BINARY);
	cv::Mat foregr(this->current->img);

	// FOR EACH EXISTING TEMPLATE LOOK ON AN AREA OF 100 PIXELS AROUND IT
	cout<<"number of templates: "<<existing.size()<<endl;
	for(unsigned k=0; k<existing.size();k++){
		cv::Point2f center       = this->cvPoint(existing[k]);
		allPeople[k].absoluteLoc = center;
		std::vector<cv::Point2f> templ;
		genTemplate2(center, persHeight, camHeight, templ);
		cv::Point2f head((templ[12].x+templ[14].x)/2,(templ[12].y+templ[14].y)/2);
		double tmplHeight = dist(head,center);
		double tmplArea   = tmplHeight*dist(templ[0],templ[1]);

		// GET THE 100X100 WINDOW ON THE TEMPLATE
		int minY=thrsh.rows, maxY=0, minX=thrsh.cols, maxX=0;
		this->templateWindow(cv::Size(foregr.cols,foregr.rows),minX, maxX,\
			minY, maxY, templ);
		int width  = maxX-minX;
		int height = maxY-minY;
		cv::Mat colorRoi = cv::Mat(foregr.clone(),cv::Rect(cv::Point2f(minX,minY),\
							cv::Size(width,height)));

		// LOOP OVER THE AREA OF OUR TEMPLATE AND THERESHOLD ONLY THOSE PIXELS
		for(unsigned x=0; x<maxX-minX; x++){
			for(unsigned y=0; y<maxY-minY; y++){
				if((int)(thrsh.at<uchar>((int)(y+minY),(int)(x+minX)))>0){

					// IF THE PIXEL IS NOT INSIDE OF THE TEMPLATE
					if(!this->isInTemplate((x+minX),(y+minY),templ) &&\
					existing.size()>1){
						double minDist = thrsh.rows*thrsh.cols;
						unsigned label = -1;
						for(unsigned l=0; l<existing.size(); l++){
							if(k!=l){
								cv::Point2f aCenter = this->cvPoint(existing[l]);
								std::vector<cv::Point2f> aTempl;
								genTemplate2(aCenter,persHeight,camHeight,aTempl);

								// IF IT IS IN ANOTHER TEMPLATE THEN IGNORE THE PIXEL
								if(this->isInTemplate((x+minX),(y+minY),aTempl)){
									minDist = 0;
									label   = l;
									break;
								// ELSE COMPUTE THE DISTANCE FROM THE PIXEL TO THE TEMPLATE
								}else{
									double ptDist = dist(cv::Point2f(x+minX,\
													y+minY),aCenter);
									if(minDist>ptDist){
										minDist = ptDist;
										label   = l;
									}
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
		// FOR MULTIPLE DISCONNECTED BLOBS KEEP THE CLOSEST TO CENTER
		cv::Mat thrshRoi;
		cv::cvtColor(colorRoi, thrshRoi, CV_BGR2GRAY);
		this->keepLargestBlob(thrshRoi,cv::Point2f(center.x-minX,center.y-minY),
				tmplArea);
		colorRoi.copyTo(allPeople[k].pixels,thrshRoi);

		// SAVE IT IN THE STRUCTURE OF FOREGOUND IMAGES
		allPeople[k].relativeLoc = cv::Point2f(center.x-(minX),center.y-(minY));
		allPeople[k].borders.assign(4,0);
		allPeople[k].borders[0] = minX;
		allPeople[k].borders[1] = maxX;
		allPeople[k].borders[2] = minY;
		allPeople[k].borders[3] = maxY;
		if(this->plotTracks){
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
/** Creates a gabor with the parameters given by the parameter vector.
 */
cv::Mat featureDetector::createGabor(double *params){
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
cv::Mat featureDetector::getGabor(cv::Mat feature, cv::Mat thresholded,\
cv::Rect roi, cv::Point2f center, cv::Point2f head, cv::Size foregrSize){
	unsigned gaborNo    = std::ceil(feature.rows/height);
	int gaborRows       = std::ceil(feature.rows/gaborNo);
	unsigned resultCols = foregrSize.width*foregrSize.height;
	cv::Mat result      = cv::Mat::zeros(cv::Size(2*resultCols+1,1),\
								cv::DataType<double>::type);
	for(unsigned i=0; i<gaborNo; i++){
		// GET THE ROI OUT OF THE iTH GABOR
		cv::Mat tmp1 = feature.rowRange(i*gaborRows,(i+1)*gaborRows);
		tmp1.convertTo(tmp1,CV_8UC1);
		cv::Mat tmp2(tmp1, roi);

		// ROTATE EACH GABOR TO THE RIGHT POSITION
		cv::Point2f rotBorders;
		tmp2 = this->rotate2Zero(head,center,tmp2.clone(),rotBorders);

		// KEEP ONLY THE THRESHOLDED VALUES
		cv::Mat tmp3;
		tmp2.copyTo(tmp3,thresholded);
		if(this->plotTracks){
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

	if(this->printValues){
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
/** Set what kind of features to extract.
 */
void featureDetector::setFeatureType(featureDetector::FEATURE type){
	this->featureType = type;
}
//==============================================================================
/** Set the name of the file where the SIFT dictionary is stored.
 */
void featureDetector::setSIFTDictionary(std::string fileSIFT){
	this->dictFileName = fileSIFT;
}
//==============================================================================
/** Keeps only the largest blob from the thresholded image.
 */
void featureDetector::keepLargestBlob(cv::Mat &thresh,cv::Point2f center,\
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
void featureDetector::extractDataRow(std::deque<unsigned> existing, IplImage *bg){
	this->lastIndex = this->data.rows;
	cv::Mat image(this->current->img);
	cv::cvtColor(image, image, this->colorspaceCode);
	if(this->tracking && (this->featureType==featureDetector::SIFT ||\
	this->featureType==featureDetector::SURF)){
		std::cout<<this->current->index+1<<" "<<std::endl;
		if(this->current->index+1<this->producer->filelist.size()){
			this->entireNext = cv::imread(this->producer->filelist\
								[this->current->index+1].c_str());
			cv::cvtColor(this->entireNext, this->entireNext, this->colorspaceCode);
		}
	}

	// REDUCE THE IMAGE TO ONLY THE INTERESTING AREA
	std::deque<featureDetector::people> allPeople(existing.size(),\
		featureDetector::people());
	this->allForegroundPixels(allPeople, existing, bg, 7.0);

	//GET ONLY THE IMAGE NAME OUT THE CURRENT IMAGE'S NAME
	unsigned pos1       = (this->current->sourceName).find_last_of("/\\");
	std::string imgName = (this->current->sourceName).substr(pos1+1);
	unsigned pos2       = imgName.find_last_of(".");
	imgName             = imgName.substr(0,pos2);
	// FOR EACH LOCATION IN THE IMAGE EXTRACT FEATURES, FILTER THEM AND RESHAPE

	std::vector< std::vector<cv::Point2f> > allLocations;
	for(std::size_t i=0; i<existing.size(); i++){
		cv::Point2f center = this->cvPoint(existing[i]);
		std::vector<cv::Point2f> templ;
		std::vector<cv::Point2f> templPoints;
		genTemplate2(center, persHeight, camHeight, templ);
		cv::Point2f head((templ[12].x+templ[14].x)/2,(templ[12].y+templ[14].y)/2);
		templPoints.push_back(center);
		templPoints.push_back(head);
		allLocations.push_back(templPoints);
		templPoints.clear();

		// DEFINE THE IMAGE ROI OF THE SAME SIZE AS THE FOREGROUND
		cv::Rect roi(allPeople[i].borders[0],allPeople[i].borders[2],\
			allPeople[i].borders[1]-allPeople[i].borders[0],\
			allPeople[i].borders[3]-allPeople[i].borders[2]);

		// ROTATE THE FOREGROUND PIXELS, THREHSOLD AND THE TEMPLATE
		cv::Mat thresholded, dummy;
		cv::Point2f rotBorders;
		allPeople[i].pixels = this->rotate2Zero(head,center,allPeople[i].pixels,rotBorders);
		cv::inRange(allPeople[i].pixels,cv::Scalar(1,1,1),cv::Scalar(255,225,225),\
			thresholded);
		cv::dilate(thresholded,thresholded,cv::Mat());
		cv::Point2f absRotCenter(allPeople[i].pixels.cols/2.0+allPeople[i].borders[0],\
			allPeople[i].pixels.rows/2.0+allPeople[i].borders[2]);
		templ = this->rotatePoints2Zero(head,center,templ,rotBorders,absRotCenter);

		// GET THE EXTREME POINTS OF THE TEMPLATE
		std::deque<double> extremes = this->templateExtremes(templ);

		// IF THE PART TO BE CONSIDERED IS ONLY FEET OR ONLY HEAD
		if(this->featurePart != ' '){
			this->onlyPart(thresholded,templ,allPeople[i].borders[0],\
				allPeople[i].borders[2]);
		}

		// ANF FINALLY EXTRACT THE DATA ROW
		cv::Mat dataRow, feature;
		std::string toRead;
		cv::vector<cv::Point2f> keys;
		switch(this->featureType){
			case (featureDetector::IPOINTS):
				toRead = (this->featureFile+"IPOINTS/"+imgName+".bin");
				binFile2mat(feature, const_cast<char*>(toRead.c_str()));
				feature.convertTo(feature,cv::DataType<double>::type);
				this->rotateKeypts2Zero(head,center,feature,absRotCenter,rotBorders);
				dataRow = this->getPointsGrid(feature,roi,templ,extremes,\
							allPeople[i].pixels);
				break;
			case featureDetector::EDGES:
				toRead = (this->featureFile+"EDGES/"+imgName+".bin");
				binFile2mat(feature, const_cast<char*>(toRead.c_str()));
				dataRow = this->getEdges(feature,thresholded,roi,head,center);
				break;
			case featureDetector::SURF:
				toRead = (this->featureFile+"SURF/"+imgName+".bin");
				binFile2mat(feature, const_cast<char*>(toRead.c_str()));
				feature.convertTo(feature,cv::DataType<double>::type);
				this->rotateKeypts2Zero(head,center,feature,absRotCenter,rotBorders);
				dataRow = this->getSURF(feature,templ,keys,roi,allPeople[i].pixels);
				if(this->tracking && !this->entireNext.empty()){
					double flow = this->opticalFlowFeature(feature,this->current->img,\
									this->entireNext,keys,roi,head,center,false);
					dataRow.at<double>(0,dataRow.cols-2) = \
						this->fixAngle(head,center,flow);
				}
				break;
			case featureDetector::GABOR:
				toRead = (this->featureFile+"GABOR/"+imgName+".bin");
				binFile2mat(feature, const_cast<char*>(toRead.c_str()));
				dataRow = this->getGabor(feature,thresholded,roi,center,head,\
					allPeople[i].pixels.size());
				break;
			case featureDetector::SIFT_DICT:
				dataRow = this->extractSIFT(image, templ);
				break;
			case featureDetector::SIFT:
				toRead = (this->featureFile+"SIFT/"+imgName+".bin");
				binFile2mat(feature, const_cast<char*>(toRead.c_str()));
				feature.convertTo(feature,cv::DataType<double>::type);
				this->rotateKeypts2Zero(head,center,feature,absRotCenter,rotBorders);
				dataRow = this->getSIFT(feature,templ,keys,roi,allPeople[i].pixels);
				if(this->tracking && !this->entireNext.empty()){
					double flow = this->opticalFlowFeature(feature,this->current->img,\
									this->entireNext,keys,roi,head,center,false);
					dataRow.at<double>(0,dataRow.cols-2) = \
						this->fixAngle(head,center,flow);
				}
				break;
		}
		dataRow.convertTo(dataRow, cv::DataType<double>::type);
		if(this->tracking && this->featureType!=featureDetector::SIFT_DICT){
			dataRow.at<double>(0,dataRow.cols-1) = this->motionVector(head,center);
		}
		if(this->data.empty()){
			dataRow.copyTo(this->data);
		}else{
			this->data.push_back(dataRow);
		}
		thresholded.release();
		feature.release();
		dataRow.release();
	}

	// FIX THE LABELS TO CORRESPOND TO THE PEOPLE DETECTED IN THE IMAGE
	if(!this->targetAnno.empty()){
		this->fixLabels(allLocations);
	}
}
//==============================================================================
/** Compute the dominant direction of the SIFT or SURF features.
 */
double featureDetector::opticalFlowFeature(cv::Mat keys, cv::Mat currentImg,\
cv::Mat nextImg,std::vector<cv::Point2f> keyPts,cv::Rect roi,cv::Point2f head,\
cv::Point2f center,bool maxOrAvg){
	// ROTATE THE IMAGES
	cv::Point2f rotBorders;
	cv::Mat tmp1(currentImg,roi), tmp2(nextImg,roi);
	tmp1 = this->rotate2Zero(head,center,tmp1.clone(),rotBorders);
	tmp2 = this->rotate2Zero(head,center,tmp2.clone(),rotBorders);

	// GET THE OPTICAL FLOW MATRIX FROM THE FEATURES
	double direction = -1;
	cv::Mat currGray, nextGray;
	cv::cvtColor(tmp1,currGray,CV_RGB2GRAY);
	cv::cvtColor(tmp2,nextGray,CV_RGB2GRAY);
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
		if(this->plotTracks){
			cv::line(tmp1,cv::Point2f(ix,iy),cv::Point2f(fx,fy),\
				cv::Scalar(0,0,255),1,8,0);
			cv::circle(tmp1,cv::Point2f(fx,fy),2,cv::Scalar(0,200,0),1,8,0);
		}
	}
	//	IF WE WANT THE AVERAGE OPTICAL FLOW
	if(!maxOrAvg){
		flowX /= keyPts.size(); flowY /= keyPts.size();
		resFlowX /= keyPts.size(); resFlowY /= keyPts.size();
	}

	if(this->plotTracks){
		cv::line(tmp1,cv::Point2f(flowX,flowY),cv::Point2f(resFlowX,resFlowY),\
			cv::Scalar(255,0,0),1,8,0);
		cv::circle(tmp1,cv::Point2f(resFlowX,resFlowY),2,cv::Scalar(0,200,0),\
			1,8,0);
		cv::imshow("optical",tmp1);
		cv::waitKey(0);
	}

	// DIRECTION OF THE AVERAGE FLOW
	direction = std::atan2(resFlowY - flowY,resFlowX - flowX);
	std::cout<<"flow direction: "<<direction*180/M_PI<<std::endl;
	currGray.release();
	nextGray.release();
	tmp1.release();
	tmp2.release();
	return direction;
}
//==============================================================================
/** Compute the features from the SIFT descriptors by doing vector quantization.
 */
cv::Mat featureDetector::getSIFT(cv::Mat feature,std::vector<cv::Point2f> templ,
std::vector<cv::Point2f> &indices, cv::Rect roi, cv::Mat test){
	// KEEP ONLY THE SIFT FEATURES THAT ARE WITHIN THE TEMPLATE
	cv::Mat tmp = cv::Mat::zeros(cv::Size(feature.cols-2,feature.rows),\
					cv::DataType<double>::type);
	unsigned counter = 0;
	for(int y=0; y<feature.rows; y++){
		double ptX = feature.at<double>(y,feature.cols-2);
		double ptY = feature.at<double>(y,feature.cols-1);
		if(this->isInTemplate(ptX, ptY, templ)){
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
		binFile2mat(this->dictionarySIFT, const_cast<char*>(this->dictFileName.c_str()));
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

	if(this->plotTracks && !test.empty()){
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
	if(this->printValues){
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
/** Checks to see if an annotation can be assigned to a detection.
 */
bool featureDetector::canBeAssigned(unsigned l,std::deque<double> &minDistances,\
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
std::vector<cv::Point2f> featureDetector::rotatePoints2Zero(cv::Point2f \
headLocation, cv::Point2f feetLocation, std::vector<cv::Point2f> pts,\
cv::Point2f rotBorders, cv::Point2f rotCenter){
	// GET THE ANGLE WITH WHICH WE NEED TO ROTATE
	double rotAngle = std::atan2((headLocation.y-feetLocation.y),\
						(headLocation.x-feetLocation.x));
	rotAngle = (rotAngle+M_PI/2.0);
	if(rotAngle>2.0*M_PI){
		rotAngle -= 2.0*M_PI;
	}
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
void featureDetector::rotateKeypts2Zero(cv::Point2f headLocation, cv::Point2f \
feetLocation, cv::Mat &keys, cv::Point2f rotCenter, cv::Point2f rotBorders){
	// GET THE ANGLE WITH WHICH WE NEED TO ROTATE
	double rotAngle = std::atan2((headLocation.y-feetLocation.y),\
						(headLocation.x-feetLocation.x));
	rotAngle = (rotAngle+M_PI/2.0);
	if(rotAngle>2.0*M_PI){
		rotAngle -= 2.0*M_PI;
	}
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
cv::Mat featureDetector::rotate2Zero(cv::Point2f headLocation,\
cv::Point2f feetLocation, cv::Mat toRotate, cv::Point2f &borders){
	// GET THE ANGLE TO ROTATE WITH
	double rotAngle = std::atan2((headLocation.y-feetLocation.y),\
						(headLocation.x-feetLocation.x));
	rotAngle = (rotAngle+M_PI/2.0);
	if(rotAngle>2.0*M_PI){
		rotAngle -= 2.0*M_PI;
	}
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
double featureDetector::fixAngle(cv::Point2f headLocation, cv::Point2f feetLocation,\
double angle){
	double cameraAngle = std::atan2((headLocation.y-feetLocation.y),\
						(headLocation.x-feetLocation.x));
	angle = angle-cameraAngle;
	if(angle >= 2.0*M_PI){
		angle -= 2.0*M_PI;
	}else if(angle < 0){
		angle += 2.0*M_PI;
	}
	return angle;
}
//==============================================================================
/** For each row added in the data matrix (each person detected for which we
 * have extracted some features) find the corresponding label.
 */
void featureDetector::fixLabels(std::vector< std::vector<cv::Point2f> > points){
	// FIND	THE INDEX FOR THE CURRENT IMAGE
	std::deque<annotationsHandle::FULL_ANNOTATIONS>::iterator index = \
		std::find_if (this->targetAnno.begin(), this->targetAnno.end(),\
		compareImg(this->current->sourceName));

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
				double dstnc = dist((*index).annos[l].location,points[k][0]);
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

	// DELETE DETECTED LOCATIONS THAT ARE NOT LABELLED
	unsigned extra = 0;
	std::deque<unsigned> unlabelled;
	for(std::size_t k=0; k<points.size(); k++){
		if(std::find(assignments.begin(),assignments.end(),k) == assignments.end()){
			unlabelled.push_back(k);
		}
	}
	// UNDETECTED LOCATIONS THAT ARE LABELLED
	for(std::size_t i=0; i<assignments.size(); i++){
		if(assignments[i] != -1){
			extra++;
		}
	}
	extra += unlabelled.size();


	// STORE THE LABELS FOR ALL THE DETECTED LOCATIONS (+UNLABELLED ONES)
	if(this->targets.empty()){
		this->targets = cv::Mat::zeros(1,4,cv::DataType<double>::type);
	}
	for(unsigned i=this->targets.rows; i<this->lastIndex+extra; i++){
		cv::Mat tmp = cv::Mat::zeros(1,4,cv::DataType<double>::type);
		this->targets.push_back(tmp);
		tmp.release();
	}
	for(std::size_t i=0; i<assignments.size(); i++){
		if(assignments[i] != -1){
			cv::Mat tmp = cv::Mat::zeros(1,4,cv::DataType<double>::type);
			// READ THE TARGET ANGLE FOR LONGITUDINAL ANGLE
			double angle = static_cast<double>\
				((*index).annos[i].poses[annotationsHandle::LONGITUDE]);
			std::cout<<"Longitude: "<<angle<<std::endl;
			angle = angle*M_PI/180.0;
			angle = this->fixAngle((*index).annos[i].location,\
					points[assignments[i]][1],angle);
			tmp.at<double>(0,0) = std::sin(angle);
			tmp.at<double>(0,1) = std::cos(angle);

			// READ THE TARGET ANGLE FOR LATITUDINAL ANGLE
			angle = static_cast<double>\
				((*index).annos[i].poses[annotationsHandle::LATITUDE]);
			std::cout<<"Latitude: "<<angle<<std::endl;
			angle = angle*M_PI/180.0;
			angle = this->fixAngle((*index).annos[i].location,\
					points[assignments[i]][1],angle);
			tmp.at<double>(0,2) = std::sin(angle);
			tmp.at<double>(0,3) = std::cos(angle);

			// STORE THE LABELS IN THE TARGETS ON THE RIGHT POSITION
			cv::Mat tmp2 = this->targets.row(this->lastIndex+assignments[i]);
			tmp.copyTo(tmp2);
			tmp.release();
			tmp2.release();
		}
	}

	// REMOVE THE DETECTED, UNLABELLED POSITIONS FROM TARGETS AND TRAIN DATA
	if(!unlabelled.empty()){
		double dataDiff    = this->data.rows-unlabelled.size();
		double targetsDiff = this->targets.rows-unlabelled.size();
		cv::Mat tmpData = cv::Mat::zeros(cv::Size(this->data.cols,\
							dataDiff),cv::DataType<double>::type);
		cv::Mat tmpTargets = cv::Mat::zeros(cv::Size(this->targets.cols,\
							targetsDiff),cv::DataType<double>::type);
		// COPY THE MATRIXES IN 2 TEMPORARY MATRIXES
		if(this->lastIndex>0){
			cv::Mat tmp1 = this->data.rowRange(0,this->lastIndex);
			cv::Mat tmp2 = this->targets.rowRange(0,this->lastIndex);
			tmp1.copyTo(tmpData); tmp1.release();
			tmp2.copyTo(tmpTargets); tmp2.release();
		}
		// ADD ONLY THE LABELLED ROWS OVER ON TOP
		for(std::size_t k=0; k<points.size(); k++){
			if(std::find(unlabelled.begin(),unlabelled.end(),k) == unlabelled.end()){
				cv::Mat dummy1 = this->data.row(this->lastIndex+k);
				tmpData.push_back(dummy1);
				dummy1.release();

				cv::Mat dummy2 = this->targets.row(this->lastIndex+k);
				tmpTargets.push_back(dummy2);
				dummy2.release();
			}
		}
		// COPY FROM THE TEMPORARY MATRIXES TO THE ACTUAL MATRIXES
		this->data.release();
		tmpData.copyTo(this->data);
		this->targets.release();
		tmpTargets.copyTo(this->targets);
		tmpData.release();
		tmpTargets.release();
	}
}
//==============================================================================
/** Overwrites the \c doFindPeople function from the \c Tracker class to make it
 * work with the feature extraction.
 */
bool featureDetector::doFindPerson(unsigned imgNum, IplImage *src,\
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
	if(this->plotTracks){
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
	if(this->onlyExtract){
		this->extractFeatures();
	}else{
		this->extractDataRow(existing, bg);
	}

	//7) SHOW THE FOREGROUND POSSIBLE LOCATIONS AND PLOT THE TEMPLATES
	cerr<<"no. of detected people: "<<existing.size()<<endl;
	if(this->plotTracks){
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
bool featureDetector::imageProcessingMenu(){
	if(this->plotTracks){
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
void featureDetector::onlyPart(cv::Mat &thresholded, std::vector<cv::Point2f> &templ,\
double offsetX, double offsetY){
	// CHANGE THRESHOLD
	unsigned minY = thresholded.rows, maxY = 0;
	for(int x=0; x<thresholded.cols; x++){
		for(int y=0; y<thresholded.rows; y++){
			if(thresholded.at<uchar>(y,x)>0){
				if(y<=minY){minY = y;}
				if(y>=maxY){maxY = y;}
			}
		}
	}
	unsigned middleTop = (minY+maxY)/2;
	unsigned middleBot = (minY+maxY)/2;
	if((middleTop-minY)/2.0>40){
		middleTop = (minY + middleTop)/2.0;
	}
	if((maxY-middleBot)/2.0>40){
		middleBot = (maxY + middleBot)/2.0;
	}

	for(int x=0; x<thresholded.cols; x++){
		for(int y=0; y<thresholded.rows; y++){
			if(y>middleTop && this->featurePart=='t'){
				thresholded.at<uchar>(y,x) = 0;
			}else if(y<middleBot && this->featurePart=='b'){
				thresholded.at<uchar>(y,x) = 0;
			}
		}
	}

	// CHANGE TEMPLATE
	for(unsigned i=0; i<templ.size(); i++){
		if(this->featurePart=='t' && (templ[i].y-offsetY)>=middleTop){
			templ[i].y = middleTop+offsetY;
		}else if(this->featurePart=='b' && (templ[i].y-offsetY)<=middleBot){
			templ[i].y = middleBot+offsetY;
		}
	}

	if(this->plotTracks){
		std::vector<cv::Point2f> tmpTempl = templ;
		for(unsigned i=0; i<tmpTempl.size(); i++){
			tmpTempl[i].x -= offsetX;
			tmpTempl[i].y -= offsetY;
		}
		IplImage *toSee = new IplImage(thresholded);
		plotTemplate2(toSee, cv::Point2f(0,0), persHeight,\
			camHeight, cvScalar(150,0,0),tmpTempl);
		cvShowImage("part", toSee);
		cvWaitKey(0);
	}
}
//==============================================================================
/** Computes the motion vector for the current image given the tracks so far.
 */
double featureDetector::motionVector(cv::Point2f head, cv::Point2f center){
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

		if(this->plotTracks){
			cv::Mat tmp(this->current->img);
			cv::line(tmp,prev,center,cv::Scalar(50,100,255),1,8,0);
			cv::imshow("tracks",tmp);
		}
		angle = std::atan2(center.y-prev.y,center.x-prev.x);
		std::cout<<"Motion angle>>> "<<(angle*180/M_PI)<<std::endl;
	}

	// FIX ANGLE WRT CAMERA
	angle = this->fixAngle(head,center,angle);
	return angle;
}
//==============================================================================
/** Creates a data matrix for each image and stores it locally.
 */
void featureDetector::extractFeatures(){
	cv::Mat image(this->current->img);
	cv::cvtColor(image, image, this->colorspaceCode);

	if(this->featureFile[this->featureFile.size()-1]!='/'){
		this->featureFile = this->featureFile + '/';
	}
	std::cout<<"In extract features"<<std::endl;

	// FOR EACH LOCATION IN THE IMAGE EXTRACT FEATURES AND STORE
	cv::Mat feature;
	std::string toWrite = this->featureFile;
	switch(this->featureType){
		case (featureDetector::IPOINTS):
			toWrite += "IPOINTS/";
			file_exists(toWrite.c_str(), true);
			feature = this->extractPointsGrid(image);
			break;
		case featureDetector::EDGES:
			toWrite += "EDGES/";
			file_exists(toWrite.c_str(), true);
			feature = this->extractEdges(image);
			break;
		case featureDetector::SURF:
			toWrite += "SURF/";
			file_exists(toWrite.c_str(), true);
			feature = this->extractSURF(image);
			break;
		case featureDetector::GABOR:
			toWrite += "GABOR/";
			file_exists(toWrite.c_str(), true);
			feature = this->extractGabor(image);
			break;
		case featureDetector::SIFT:
			toWrite += "SIFT/";
			file_exists(toWrite.c_str(), true);
			feature = this->extractSIFT(image);
			break;
	}
	feature.convertTo(feature, cv::DataType<double>::type);

	//WRITE THE FEATURE TO A BINARY FILE
	unsigned pos1      = (this->current->sourceName).find_last_of("/\\");
	std::string imgName = (this->current->sourceName).substr(pos1+1);
	unsigned pos2      = imgName.find_last_of(".");
	imgName             = imgName.substr(0,pos2);
	toWrite += (imgName + ".bin");

	std::cout<<"Feature written to: "<<toWrite<<std::endl;
	mat2BinFile(feature, const_cast<char*>(toWrite.c_str()));
	feature.release();
}
//==============================================================================
/** Extract the interest points in a gird and returns them.
 */
cv::Mat featureDetector::extractPointsGrid(cv::Mat image){
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

	if(this->plotTracks){
		cv::Mat toSee(this->current->img);
		for(int y=0; y<interestPoints.rows; y++){
			cv::Scalar color;
			if(y<msersNo){
				color = cv::Scalar(0,0,255);
			}else{
				color = cv::Scalar(255,0,0);
			}
			double ptX = interestPoints.at<double>(y,0);
			double ptY = interestPoints.at<double>(y,1);
			cv::circle(toSee, cv::Point2f(ptX,ptY),3,color);
		}
		cv::imshow("IPOINTS", toSee);
		cv::waitKey(0);
		toSee.release();
	}
	return interestPoints;
}
//==============================================================================
/** Extract edges from the whole image.
 */
cv::Mat featureDetector::extractEdges(cv::Mat image){
	cv::Mat gray, edges;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::equalizeHist(gray,gray);
	cv::medianBlur(gray, gray, 3);
	cv::Canny(gray, edges, 100, 0, 3, true);
	edges.convertTo(edges,cv::DataType<double>::type);

	if(this->plotTracks){
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
cv::Mat featureDetector::extractSURF(cv::Mat image){
	std::vector<float> descriptors;
	std::vector<cv::KeyPoint> keypoints;
	cv::SURF aSURF = cv::SURF(10,3,4,false);

	// EXTRACT INTEREST POINTS FROM THE IMAGE
	cv::Mat gray;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::equalizeHist(gray, gray);
	cv::medianBlur(gray, gray, 3);
	aSURF(gray, cv::Mat(), keypoints, descriptors, false);
	gray.release();

	// WRITE ALL THE DESCRIPTORS IN THE STRUCTURE OF KEY-DESCRIPTORS
	std::deque<featureDetector::keyDescr> kD;
	for(std::size_t i=0; i<keypoints.size();i++){
		featureDetector::keyDescr tmp;
		for(int j=0; j<aSURF.descriptorSize();j++){
			tmp.descr.push_back(descriptors[i*aSURF.descriptorSize()+j]);
		}
		tmp.keys = keypoints[i];
		kD.push_back(tmp);
		tmp.descr.clear();
	}

	// SORT THE REMAINING DESCRIPTORS SO WE DON'T NEED TO DO THAT LATER
	std::sort(kD.begin(),kD.end(),(&featureDetector::compareDescriptors));

	// WRITE THEM IN THE MATRIX AS FOLLOWS:
	cv::Mat surfs = cv::Mat::zeros(cv::Size(aSURF.descriptorSize()+2,kD.size()),\
					cv::DataType<double>::type);
	for(std::size_t i=0; i<kD.size();i++){
		surfs.at<double>(i,aSURF.descriptorSize()) = kD[i].keys.pt.x;
		surfs.at<double>(i,aSURF.descriptorSize()+1) = kD[i].keys.pt.y;
		for(int j=0; j<aSURF.descriptorSize();j++){
			surfs.at<double>(i,j) = kD[i].descr[j];
		}
	}

	// PLOT THE KEYPOINTS TO SEE THEM
	if(this->plotTracks){
		cv::Mat toSee(this->current->img);
		for(std::size_t i=0; i<kD.size();i++){
			cv::circle(toSee, cv::Point2f(kD[i].keys.pt.x,kD[i].keys.pt.y),\
				3,cv::Scalar(0,0,255));
		}
		cv::imshow("SURFS", toSee);
		cv::waitKey(0);
		toSee.release();
	}
	return surfs;
}
//==============================================================================
/** Convolves the whole image with some Gabors wavelets and then stores the
 * results.
 */
cv::Mat featureDetector::extractGabor(cv::Mat image){
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
		if(this->plotTracks){
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
cv::Mat featureDetector::extractSIFT(cv::Mat image, std::vector<cv::Point2f> templ){
	// DEFINE THE SURF KEYPOINTS AND THE DESCRIPTORS
	std::vector<cv::KeyPoint> keypoints;
	cv::SIFT::DetectorParams detectP  = cv::SIFT::DetectorParams(0.0001,10.0);
	cv::SIFT::DescriptorParams descrP = cv::SIFT::DescriptorParams();
	cv::SIFT::CommonParams commonP    = cv::SIFT::CommonParams();
	cv::SIFT aSIFT(commonP, detectP, descrP);

	// EXTRACT SIFT FEATURES IN THE IMAGE
	cv::Mat gray, sift;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::equalizeHist(gray, gray);
	cv::medianBlur(gray, gray, 3);
	aSIFT(gray, cv::Mat(), keypoints);

	// WE USE THE SAME FUNCTION TO BUILD THE DICTIONARY ALSO
	if(templ.size()!=0){
		std::cout<<"EXtracting SIFT features for... "<<this->current->sourceName<<std::endl;
		sift = cv::Mat::zeros(keypoints.size(),aSIFT.descriptorSize(),\
				cv::DataType<double>::type);
		std::vector<cv::KeyPoint> goodKP;
		for(std::size_t i=0; i<keypoints.size();i++){
			if(this->isInTemplate(keypoints[i].pt.x,keypoints[i].pt.y,templ)){
				goodKP.push_back(keypoints[i]);
			}
		}
		aSIFT(gray, cv::Mat(), goodKP, sift, true);
		if(this->plotTracks){
			cv::Mat toSee(this->current->img);
			for(std::size_t i=0; i<goodKP.size(); i++){
				cv::circle(toSee,goodKP[i].pt,3,cv::Scalar(0,0,255));
			}
			cv::imshow("SIFT_DICT", toSee);
			cv::waitKey(0);
			toSee.release();
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
		if(this->plotTracks){
			cv::Mat toSee(this->current->img);
			for(std::size_t i=0; i<keypoints.size();i++){
				cv::circle(toSee, keypoints[i].pt,3,cv::Scalar(0,0,255));
			}
			cv::imshow("SIFT", toSee);
			cv::waitKey(0);
			toSee.release();
		}
	}

	// NORMALIZE THE FEATURE
	sift.convertTo(sift, cv::DataType<double>::type);
	for(int i=0; i<sift.rows; i++){
		cv::Mat rowsI = sift.row(i);
		rowsI         = rowsI/cv::norm(rowsI);
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
/*
int main(int argc, char **argv){
	featureDetector feature(argc,argv,true,false);
	feature.setFeatureType(featureDetector::SIFT);
	feature.run();
}
*/
