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
			std::vector<std::string> parts = splitLine(\
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
	this->initProducer(readFromFolder, dataFolder.c_str());

	// CLEAR DATA AND TARGETS
	if(!this->targets.empty()){
		for(std::size_t i=0; i<this->targets.size(); i++){
			this->targets[i].release();
		}
		this->targets.clear();
	}
	if(!this->data.empty()){
		for(std::size_t i=0; i<this->data.size(); i++){
			this->data[i].release();
		}
		this->data.clear();
	}

	// CLEAR THE ANNOTATIONS
	for(std::size_t i=0; i<this->targetAnno.size(); i++){
		for(std::size_t j=0; j<this->targetAnno[i].annos.size(); j++){
			this->targetAnno[i].annos[j].poses.clear();
		}
		this->targetAnno[i].annos.clear();
	}
	this->targetAnno.clear();
	this->lastIndex = 0;

	// LOAD THE DESIRED ANNOTATIONS FROM THE FILE
	if(!theAnnotationsFile.empty() && this->targetAnno.empty()){
		annotationsHandle::loadAnnotations(const_cast<char*>\
			(theAnnotationsFile.c_str()), this->targetAnno);
	}
}
//==============================================================================
/** Get perpendicular to a line given by 2 points A, B in point C.
 */
void featureDetector::getLinePerpendicular(cv::Point A, cv::Point B, cv::Point C,\
double &m, double &b){
	double slope = (double)(B.y - A.y)/(double)(B.x - A.x);
	m            = -1.0/slope;
	b            = C.y - m * C.x;
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
void featureDetector::getEdges(cv::Point center, cv::Mat &feature, cv::Mat image,\
unsigned reshape, cv::Mat thresholded){
	//cv::Mat up, low;
	//this->upperLowerROI(image,up,low,0,0);

	// GET THE EDGES
	cv::Mat gray, edges;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::equalizeHist(gray,gray);
	cv::medianBlur(gray, gray, 3);
	cv::Canny(gray, edges, 100, 0, 3, true);
	edges.convertTo(edges,cv::DataType<double>::type);
	feature = cv::Mat::zeros(edges.size(),cv::DataType<double>::type);

	// CONSIDER ONLY THE EDGES WITHIN THE THRESHOLDED AREA
	if(!thresholded.empty()){
		edges.copyTo(feature,thresholded);
	}else{
		edges.copyTo(feature);
	}
	if(this->plotTracks){
		cv::imshow("Edges", feature);
		cv::waitKey(0);
		cvDestroyWindow("Edges");
	}

    // RESHAPE IF NEEDED
	if(reshape){
		feature = (feature.clone()).reshape(0,1);
	}
	gray.release();
	edges.release();
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
void featureDetector::getSURF(cv::Mat &feature, cv::Mat image, int minX,\
int minY, std::vector<CvPoint> templ, cv::Point center){
	// EXTRACT THE SURF KEYPOINTS AND THE DESCRIPTORS
	std::vector<float> descriptors;
	std::vector<cv::KeyPoint> keypoints;
	cv::SURF aSURF = cv::SURF(10,3,4,false);

	// EXTRACT INTEREST POINTS FROM THE IMAGE
	cv::Mat gray;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::equalizeHist(gray, gray);
	cv::medianBlur(gray, gray, 3);
	aSURF(gray, cv::Mat(), keypoints, descriptors, false);

	// KEEP THE DESCRIPTORS WITHIN THE BORDERS ONLY
	std::vector<featureDetector::keyDescr> kD;
	for(std::size_t i=0; i<keypoints.size();i++){
		if(this->isInTemplate(keypoints[i].pt.x+minX,keypoints[i].pt.y+minY,templ)){
			featureDetector::keyDescr tmp;
			for(int j=0; j<aSURF.descriptorSize();j++){
				tmp.descr.push_back(descriptors[i*aSURF.descriptorSize()+j]);
			}
			tmp.keys = keypoints[i];
			kD.push_back(tmp);
			tmp.descr.clear();
		}
	}

	// SORT THE REMAINING DESCRIPTORS AND KEEP ONLY THE FIRST ?
	std::sort(kD.begin(),kD.end(),(&featureDetector::compareDescriptors));

	// COPY TO FEATURE THE FIRST 10 DESCRIPTORS
	feature = cv::Mat::zeros(cv::Size(10*aSURF.descriptorSize(),1),\
				cv::DataType<double>::type);
	for(unsigned i=0; i<std::min(10,static_cast<int>(kD.size())); i++){
		for(std::size_t k=0; k<kD[i].descr.size(); k++){
			feature.at<double>(0,i*aSURF.descriptorSize()+k) = \
				static_cast<double>(kD[i].descr[k]);
		}
	}

	std::cout<<"key-size:"<<kD.size()<<" feat-size:"<<feature.cols<<std::endl;
	if(this->plotTracks){
		for(std::size_t i=0; i<kD.size(); i++){
			cv::circle(image, cv::Point(kD[i].keys.pt.x, kD[i].keys.pt.y),\
				kD[i].keys.size, cv::Scalar(0,0,255), 1, 8, 0);
		}
		cv::imshow("SURFS", image);
		cv::waitKey(0);
	}
	gray.release();
}

//==============================================================================
/** Get template extremities (if needed, considering some borders --
 * relative to the ROI).
 */
void featureDetector::templateExtremes(std::vector<CvPoint> templ, double\
&minTmplX, double &maxTmplX, double &minTmplY, double &maxTmplY, int minX,\
int minY){
	minTmplX = std::max(0,templ[0].x-minX),\
	maxTmplX = std::max(0,templ[0].x-minX),\
	minTmplY = std::max(0,templ[0].y-minY),\
	maxTmplY = std::max(0,templ[0].y-minY);
	for(std::size_t i=0; i<templ.size();i++){
		if(minTmplY>=templ[i].y-minY) minTmplY  = templ[i].y - minY;
		if(maxTmplY<=templ[i].y-minY) maxTmplY = templ[i].y - minY;
		if(minTmplX>=templ[i].x-minX) minTmplX  = templ[i].x - minX;
		if(maxTmplX<=templ[i].x-minX) maxTmplX = templ[i].x - minX;
	}
}
//==============================================================================
/** Creates a "histogram" of interest points + number of blobs.
 */
void featureDetector::interestPointsGrid(cv::Mat &feature, cv::Mat image,\
std::vector<CvPoint> templ, int minX, int minY, cv::Point center){
	double minTmplX,maxTmplX,minTmplY,maxTmplY;
	this->templateExtremes(templ,minTmplX,maxTmplX,minTmplY,maxTmplY,minX,minY);

	// EXTRACT MAXIMALLY STABLE BLOBS
	std::vector<std::vector<cv::Point> > msers;
	cv::MSER aMSER;
	aMSER(image, msers, cv::Mat());

	// COUNTE THE CONTOURS INSIDE THE TEMPLATE
	uchar msersTmpl = 0;
	for(std::size_t x=0; x<msers.size(); x++){
		for(std::size_t y=0; y<msers[x].size(); y++){
			if(!this->isInTemplate(msers[x][y].x+minX,msers[x][y].y+minY,templ)){
				msersTmpl++;
			}
		}
	}

	if(this->plotTracks){
		cv::drawContours(image, msers, -1, cv::Scalar(255,255,255), 1, 8);
		cv::imshow("Blobs",image);
		cv::waitKey(0);
	}

	unsigned no             = 10;
	feature                 = cv::Mat::zeros(1,no*no+1,cv::DataType<double>::type);
	feature.at<double>(0,0) = static_cast<double>(msersTmpl);

	// HISTOGRAM OF NICE FEATURES
	std::vector<cv::Point2f> corners;
	cv::Ptr<cv::FeatureDetector> detector = \
		new cv::GoodFeaturesToTrackDetector(5000, 0.00001, 1.0, 3.0);
	cv::GridAdaptedFeatureDetector gafd(detector, 5000, no, no);
	std::vector<cv::KeyPoint> keys;
	std::vector<unsigned> indices;
	gafd.detect(image, keys);

	//COUNT THE INTEREST POINTS IN EACH CELL
	unsigned contor  = 0;
	cv::Mat histoMat = cv::Mat::zeros(1,no*no,cv::DataType<double>::type);
	double rateX     = (maxTmplX-minTmplX)/static_cast<double>(no);
	double rateY     = (maxTmplY-minTmplY)/static_cast<double>(no);
	for(double x=minTmplX; x<maxTmplX-0.01; x+=rateX){
		for(double y=minTmplY; y<maxTmplY-0.01; y+=rateY){
			if(indices.empty()){
				for(std::size_t i=0; i<keys.size(); i++){
					if(this->isInTemplate(keys[i].pt.x+minX,keys[i].pt.y+minY,templ)){
						indices.push_back(i);
						if(x<=keys[i].pt.x && keys[i].pt.x<x+rateX &&\
						y<=keys[i].pt.y && keys[i].pt.y<y+rateY){
							histoMat.at<double>(0,contor) += 1.0;
						}
					}
				}
			}else{
				for(std::size_t j=0; j<indices.size(); j++){
					unsigned i = indices[j];
					if(x<=keys[i].pt.x && keys[i].pt.x<x+rateX &&\
					y<=keys[i].pt.y && keys[i].pt.y<y+rateY){
						histoMat.at<double>(0,contor) += 1.0;
					}
				}
			}
			contor +=1;
		}
	}
	cv::Mat stupid = feature.colRange(1,no*no+1);
	histoMat.copyTo(stupid);
	histoMat.release();
	//-----------------REMOVE--------------------------
	std::cout<<"IPOINTS-FEATURES: "<<std::endl;
	for(int i=0; i<feature.cols; i++){
		std::cout<<feature.at<double>(0,i)<<" ";
	}
	std::cout<<std::endl;
	//-------------------------------------------------
}
//==============================================================================
/** Just displaying an image a bit larger to visualize it better.
 */
void featureDetector::showZoomedImage(cv::Mat image, const std::string title){
	cv::Mat large;
	cv::resize(image, large, cv::Size(0,0), 5, 5, cv::INTER_CUBIC);
	cv::imshow(title, image);
	cv::waitKey(0);
	cvDestroyWindow(title.c_str());
	large.release();
}
//==============================================================================
/** Head detection by fitting ellipses (if \i templateCenter is relative to the
 * \i img the offset needs to be used).
 */
void featureDetector::skinEllipses(cv::RotatedRect &finalBox, cv::Mat img,\
cv::Point templateCenter,cv::Point offset,double minHeadSize,double maxHeadSize){
	// TRANSFORM FROM BGR TO HSV TO BE ABLE TO DETECT MORE SKIN-LIKE PIXELS
	cv::Mat hsv;
	cv::cvtColor(img, hsv, CV_BGR2HSV);

	// THRESHOLD THE HSV TO KEEP THE HIGH-GREEN PIXELS => brown+green+skin
	for(int x=0; x<hsv.cols; x++){
		for(int y=0; y<hsv.rows; y++){
			if(((hsv.at<cv::Vec3b>(y,x)[0]<255 && hsv.at<cv::Vec3b>(y,x)[0]>50) && \
				(hsv.at<cv::Vec3b>(y,x)[1]<200 && hsv.at<cv::Vec3b>(y,x)[1]>0) && \
				(hsv.at<cv::Vec3b>(y,x)[2]<200 && hsv.at<cv::Vec3b>(y,x)[2]>0))){
				img.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,0);
			}
		}
	}

	// GET EDGES OF THE IMAGE
	cv::Mat_<uchar> edges;
	this->getEdges(cv::Point(0,0), edges, img, 0);

	// GROUP THE EDGES INTO CONTOURS
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(edges, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	cv::drawContours(edges, contours, -1, cv::Scalar::all(255), 1, 8);
	cv::imshow("Contours", edges);
	cv::waitKey(0);
	cvDestroyWindow("Contours");

	// GET THE HEAD CENTER
	std::vector<CvPoint> templ;
	genTemplate2(templateCenter, persHeight, camHeight, templ);
	cv::Point headCenter((templ[12].x+templ[14].x)/2-offset.x, \
		(templ[12].y+templ[14].y)/2-offset.y);

	// FIND THE BEST ELLIPSE YOU CAN FIND
	double bestDistance = edges.cols*edges.rows;
	for(std::size_t i=0; i<contours.size(); i++){
		size_t count=contours[i].size();
		if(count<6) continue;
		cv::Mat pointsf;
		cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
		cv::RotatedRect box = fitEllipse(pointsf);
		box.angle           = -box.angle;
		if(std::max(box.size.width, box.size.height)>\
		std::min(box.size.width, box.size.height)*30) continue;

		// IF IT IS AN ACCEPTABLE BLOB
		double aDist = dist(box.center, headCenter);
		if(aDist<bestDistance && minHeadSize<box.size.width && \
		box.size.width<maxHeadSize && minHeadSize<box.size.height &&\
		box.size.height<maxHeadSize){
			bestDistance = aDist;
			finalBox     = box;
		}
		pointsf.release();
	}
	if(bestDistance != edges.cols*edges.rows){
		cv::ellipse(hsv, finalBox, cv::Scalar(0,0,255), 1, CV_AA);
		cv::imshow("HSV", hsv);
		cv::waitKey(0);
		cvDestroyWindow("HSV");
	}
	edges.release();
	hsv.release();
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
std::vector<CvPoint> templ){
	std::vector<CvPoint> hull;
	convexHull(templ, hull);
	std::vector<scanline_t> lines;
	getScanLines(hull, lines);

	std::vector<scanline_t>::iterator iter = std::find_if(lines.begin(),\
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
int &minY, int &maxY, std::vector<CvPoint> &templ, int tplBorder){
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

		std::cout<<"diff on X:"<<diffX2<<std::endl;

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
void featureDetector::allForegroundPixels(std::vector<featureDetector::people>\
&allPeople, std::vector<unsigned> existing, IplImage *bg, double threshold){
	// INITIALIZING STUFF
	cv::Mat thsh(bg);
	cv::Mat thrsh(thsh.rows, thsh.cols, CV_8UC1);
	cv::cvtColor(thsh, thrsh, CV_BGR2GRAY);
	cv::threshold(thrsh, thrsh, threshold, 255, cv::THRESH_BINARY);
	cv::Mat foregr(this->current->img);

	// FOR EACH EXISTING TEMPLATE LOOK ON AN AREA OF 100 PIXELS AROUND IT
	cout<<"number of templates: "<<existing.size()<<endl;
	for(unsigned k=0; k<existing.size();k++){
		cv::Point center         = this->cvPoint(existing[k]);
		allPeople[k].absoluteLoc = center;
		std::vector<CvPoint> templ;
		genTemplate2(center, persHeight, camHeight, templ);

		// GET THE 100X100 WINDOW ON THE TEMPLATE
		int minY=thrsh.rows, maxY=0, minX=thrsh.cols, maxX=0;
		this->templateWindow(cv::Size(foregr.cols,foregr.rows),minX, maxX,\
			minY, maxY, templ);
		int width        = maxX-minX;
		int height       = maxY-minY;
		cv::Mat colorRoi = cv::Mat(foregr.clone(),cv::Rect(cv::Point(minX,minY),\
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
						}
					}
				}else{
					colorRoi.at<cv::Vec3b>((int)y,(int)x) = cv::Vec3b(0,0,0);
				}
			}
		}
		allPeople[k].relativeLoc = cv::Point(center.x - (int)(minX),\
									center.y - (int)(minY));
		allPeople[k].borders.assign(4,0);
		allPeople[k].borders[0] = minX;
		allPeople[k].borders[1] = maxX;
		allPeople[k].borders[2] = minY;
		allPeople[k].borders[3] = maxY;
		cv::Point rotBorders;
		allPeople[k].pixels = this->rotateWrtCamera(center, cv::Point(camPosX,\
								camPosY),colorRoi.clone(),rotBorders);
		if(this->plotTracks){
			cv::imshow("people",allPeople[k].pixels);
			cv::waitKey(0);
		}
		colorRoi.release();
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
	float yMax   = std::max(std::abs(params[2]*sigmaX*std::cos(params[3])), \
							std::abs(params[2]*sigmaY*std::sin(params[3])));
	yMax         = std::ceil(std::max((float)1.0, yMax));
	float xMin   = -xMax;
	float yMin   = -yMax;
	gabor        = cv::Mat::zeros((int)(xMax-xMin),(int)(yMax-yMin),\
					cv::DataType<float>::type);
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
	return gabor;
}
//==============================================================================
/** Convolves an image with a Gabor filter with the given parameters and
 * returns the response image.
 */
void featureDetector::getGabor(cv::Mat &feature,cv::Mat image,cv::Mat thresholded,\
cv::Point center){
	// DEFINE THE PARAMETERS FOR A FEW GABORS
	// params[0] -- sigma: (3, 68) // the actual size
	// params[1] -- gamma: (0.2, 1) // how round the filter is
	// params[2] -- dimension: (1, 10) // size
	// params[3] -- theta: (0, 180) or (-90, 90) // angle
	// params[4] -- lambda: (2, 256) // thickness
	// params[5] -- psi: (0, 180) // number of lines
	std::vector<double*> allParams;
	double *params1 = new double[6];
	params1[0] = 10.0; params1[1] = 0.9; params1[2] = 2.0;
	params1[3] = M_PI/4.0; params1[4] = 50.0; params1[5] = 15.0;
	allParams.push_back(params1);

	double *params2 = new double[6];
	params2[0] = 10.0; params2[1] = 0.9; params2[2] = 2.0;
	params2[3] = 3.0*M_PI/4.0; params2[4] = 50.0; params2[5] = 15.0;
	allParams.push_back(params2);

	// CREATE EACH GABOR AND CONVOLVE THE IMAGE WITH IT
	feature = cv::Mat::zeros(1,(image.cols*image.rows*allParams.size()),\
				cv::DataType<double>::type);
	for(unsigned i=0; i<allParams.size(); i++){
		cv::Mat agabor = this->createGabor(allParams[i]);

		// CONVERT THE IMAGE TO GRAYSCALE TO APPLY THE FILTER
		cv::Mat gray;
		cv::cvtColor(image, gray, CV_BGR2GRAY);
		cv::equalizeHist(gray, gray);
		cv::medianBlur(gray, gray, 3);

		// FILTER THE IMAGE WITH THE GABOR FILTER
		cv::Mat response;
		cv::filter2D(gray, response, -1, agabor, cv::Point(-1,-1), 0,\
			cv::BORDER_REPLICATE);

		// CONSIDER ONLY THE RESPONSE WITHIN THE THRESHOLDED AREA
		response = response.reshape(0,1);
		cv::Mat temp = feature.colRange(i*response.cols, (i+1)*response.cols);
		if(!thresholded.empty()){
			response.copyTo(temp,thresholded);
		}else{
			response.copyTo(temp);
		}

		if(this->plotTracks){
			cv::imshow("GaborFilter", agabor);
			cv::imshow("GaborResponse", temp.reshape(1,142));
			cv::waitKey(0);
		}
		response.release();
		gray.release();
		agabor.release();
		temp.release();
	}
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
void featureDetector::setSIFTDictionary(char* fileSIFT){
	this->dictFileName = const_cast<char*>(fileSIFT);
}
//==============================================================================
/** Creates on data row in the final data matrix by getting the feature
 * descriptors.
 */
void featureDetector::extractDataRow(std::vector<unsigned> existing, IplImage *bg){
	cv::Mat image(this->current->img);
	cv::cvtColor(image, image, this->colorspaceCode);

	// REDUCE THE IMAGE TO ONLY THE INTERESTING AREA
	std::vector<featureDetector::people> allPeople(existing.size(),\
		featureDetector::people());
	this->allForegroundPixels(allPeople, existing, bg, 7.0);
	this->lastIndex = this->data.size();

	// FOR EACH LOCATION IN THE IMAGE EXTRACT FEATURES, FILTER THEM AND RESHAPE
	std::vector<cv::Point> allLocations;
	for(std::size_t i=0; i<existing.size(); i++){
		// READ THE TEMPLATE
		cv::Point center = this->cvPoint(existing[i]);
		allLocations.push_back(center);
		std::vector<CvPoint> templ;
		genTemplate2(center, persHeight, camHeight, templ);

		// GET THE AREA OF 100/100 AROUND THE TEAMPLATE
		std::vector<unsigned> borders = allPeople[i].borders;
		unsigned width  = allPeople[i].borders[1]-allPeople[i].borders[0];
		unsigned height = allPeople[i].borders[3]-allPeople[i].borders[2];
		cv::Mat imgRoi(image.clone(),cv::Rect(cv::Point(\
			allPeople[i].borders[0],allPeople[i].borders[2]),\
			cv::Size(width,height)));

		//	ROTATE ROI, ROTATE TEMPLATE & ROATE PERSON PIXELS
		cv::Point rotBorders;
		cv::Mat tmp = this->rotateWrtCamera(center,cv::Point(camPosX,camPosY),\
			imgRoi,rotBorders);
		tmp.copyTo(imgRoi);
		tmp.release();
		templ = this->rotateTemplWrtCamera(center,cv::Point(camPosX,camPosY),\
				templ, rotBorders, cv::Point2f(\
				imgRoi.cols/2.0+allPeople[i].borders[0],\
				imgRoi.rows/2.0+allPeople[i].borders[2]));
		cv::Mat thresholded;
		cv::inRange(allPeople[i].pixels,cv::Scalar(1,1,1),cv::Scalar(255,225,225),\
			thresholded);
		cv::dilate(thresholded,thresholded,cv::Mat());

		// IF THE PART TO BE CONSIDERED IS ONLY FEET OR ONLY HEAD
		if(this->featurePart != ' '){
			this->onlyPart(thresholded,templ,allPeople[i].borders[0],\
				allPeople[i].borders[2]);
		}

		// EXTRACT FEATURES
		cv::Mat feature;
		switch(this->featureType){
			case (featureDetector::IPOINTS):
				this->interestPointsGrid(feature, imgRoi, templ,\
					allPeople[i].borders[0],allPeople[i].borders[2],center);
				break;
			case featureDetector::EDGES:
				this->getEdges(center, feature, imgRoi, 1, thresholded);
				break;
			case featureDetector::SURF:
				this->getSURF(feature, imgRoi, allPeople[i].borders[0],\
					allPeople[i].borders[2], templ, center);
				break;
			case featureDetector::GABOR:
				this->getGabor(feature, imgRoi, thresholded, center);
				break;
			case featureDetector::SIFT_DICT:
				this->extractSIFT(feature, imgRoi, allPeople[i].borders[0],\
					allPeople[i].borders[2], templ, center);
				break;
			case featureDetector::SIFT:
				this->getSIFT(feature, imgRoi, allPeople[i].borders[0],\
					allPeople[i].borders[2], templ, center);
				break;
		}
		feature.convertTo(feature, cv::DataType<double>::type);
		cv::Mat cloneFeature = cv::Mat::zeros(feature.rows,feature.cols+1,\
								cv::DataType<double>::type);
		cloneFeature.at<double>(0,0) = this->motionVector(center);
		cv::Mat tmpClone = cloneFeature.colRange(1,cloneFeature.cols);
		feature.copyTo(tmpClone);
		cloneFeature.convertTo(feature, cv::DataType<double>::type);
		this->data.push_back(cloneFeature);

		thresholded.release();
		cloneFeature.release();
		tmpClone.release();
		feature.release();
		imgRoi.release();
	}
	// FIX THE LABELS TO CORRESPOND TO THE PEOPLE DETECTED IN THE IMAGE
	if(!this->targetAnno.empty()){
		this->fixLabels(allLocations);
	}
}
//==============================================================================
/** SIFT descriptors (Scale Invariant Feature Transform).
 */
void featureDetector::extractSIFT(cv::Mat &feature, cv::Mat image, int minX,\
int minY, std::vector<CvPoint> templ, cv::Point center){
	// EXTRACT THE SURF KEYPOINTS AND THE DESCRIPTORS
	std::vector<cv::KeyPoint> keypoints;
	cv::SIFT::DetectorParams detectP  = cv::SIFT::DetectorParams(0.0001,10.0);
	cv::SIFT::DescriptorParams descrP = cv::SIFT::DescriptorParams();
	cv::SIFT::CommonParams commonP    = cv::SIFT::CommonParams();
	cv::SIFT aSIFT(commonP, detectP, descrP);

	// EXTRACT SIFT FEATURES IN THE IMAGE
	cv::Mat gray;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::equalizeHist(gray, gray);
	cv::medianBlur(gray, gray, 3);
	aSIFT(gray, cv::Mat(), keypoints);

	// KEEP THE DESCRIPTORS WITHIN THE BORDERS ONLY
	cv::vector<cv::KeyPoint> goodKP;
	for(std::size_t i=0; i<keypoints.size();i++){
		if(this->isInTemplate(keypoints[i].pt.x+minX,keypoints[i].pt.y+minY,templ)){
			goodKP.push_back(keypoints[i]);
		}
	}
	aSIFT(gray, cv::Mat(), goodKP, feature, true);
	std::cout<<"SIFTS: "<<feature.cols<<" "<<feature.rows<<std::endl;

	if(this->plotTracks){
		for(std::size_t i=0; i<goodKP.size(); i++){
			cv::circle(image, cv::Point(goodKP[i].pt.x, goodKP[i].pt.y),\
				goodKP[i].response, cv::Scalar(0,0,255), 1, 8, 0);
		}
		cv::imshow("SIFT", image);
		cv::waitKey(0);
	}
	gray.release();
}
//==============================================================================
/** Compute the features from the SIFT descriptors by doing vector quantization.
 */
void featureDetector::getSIFT(cv::Mat &feature, cv::Mat image, int minX,\
int minY, std::vector<CvPoint> templ, cv::Point center){
	// EXTRACT SIFT FEATURES
	cv::Mat preFeature;
	this->extractSIFT(preFeature, image, minX, minY, templ, center);
	preFeature.convertTo(preFeature, cv::DataType<double>::type);

	// NORMALIZE THE FEATURES ALSO
	for(int i=0; i<preFeature.rows; i++){
		cv::Mat rowsI = preFeature.row(i);
		rowsI         = rowsI/cv::norm(rowsI);
	}

	// IF DICTIONARY EXISTS THEN LOAD IT, ELSE CREATE IT AND STORE IT.
	if(this->dictionarySIFT.empty()){
		binFile2mat(this->dictionarySIFT, this->dictFileName);
	}

	std::cout<<"SIFT size:"<<this->dictionarySIFT.cols<<" "<<\
		this->dictionarySIFT.rows<<std::endl;

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
	feature = cv::Mat::zeros(cv::Size(this->dictionarySIFT.rows, 1),\
				cv::DataType<double>::type);
	for(int i=0; i<minLabel.cols; i++){
		int which = minLabel.at<double>(0,i);
		feature.at<double>(0,which) += 1.0;
	}

	// NORMALIZE THE HOSTOGRAM
	cv::Scalar scalar = cv::sum(feature);
	feature /= static_cast<double>(scalar[0]);

	preFeature.release();
	distances.release();
	minDists.release();
	minLabel.release();
}
//==============================================================================
/** Checks to see if an annotation can be assigned to a detection.
 */
bool featureDetector::canBeAssigned(unsigned l,std::vector<double> &minDistances,\
unsigned k,double distance, std::vector<int> &assignment){
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
/** Rotate the points corresponding to the template wrt to the camera location.
 */
std::vector<CvPoint> featureDetector::rotateTemplWrtCamera(cv::Point feetLocation,\
cv::Point cameraLocation, std::vector<CvPoint> templ, cv::Point rotBorders,\
cv::Point2f rotCenter){
	// GET THE ANGLE WITH WHICH WE NEED TO ROTATE
	double cameraAngle = std::atan2((feetLocation.y-cameraLocation.y),\
						(feetLocation.x-cameraLocation.x));
	cameraAngle = (cameraAngle+M_PI/2.0);
	if(cameraAngle>2.0*M_PI){
		cameraAngle -= 2.0*M_PI;
	}
	cameraAngle *= (180.0/M_PI);

	// GET THE ROTATION MATRIX WITH RESPECT TO THE GIVEN CENTER

	cv::Mat rotationMat = cv::getRotationMatrix2D(rotCenter, cameraAngle, 1.0);

	// BUILD A MATRIX OUT OF THE TEMPLATE POINTS
	cv::Mat toRotate = cv::Mat::ones(cv::Size(3,templ.size()),\
						cv::DataType<double>::type);
	for(std::size_t i=0; i<templ.size(); i++){
		toRotate.at<double>(i,0) = templ[i].x + rotBorders.x;
		toRotate.at<double>(i,1) = templ[i].y + rotBorders.y;
	}

	// MULTIPLY THE TEMPLATE MATRIX WITH THE ROTATION MATRIX
	toRotate.convertTo(toRotate, cv::DataType<double>::type);
	cv::Mat rotated = toRotate*rotationMat.t();
	rotated.convertTo(rotated, cv::DataType<double>::type);

	// COPY THE RESULT BACK INTO A TEMPLATE SHAPE
	std::vector<CvPoint> newTempl(rotated.rows);
	for(int y=0; y<rotated.rows; y++){
		newTempl[y].x = rotated.at<double>(y,0);
		newTempl[y].y = rotated.at<double>(y,1);
	}
	rotationMat.release();
	toRotate.release();
	rotated.release();
	return newTempl;
}
//==============================================================================
//!!!!!!!!!!!!!!!!! NEEDS HELP
/*
cv::Mat featureDetector::vertProject(cv::Mat img){
	cv::Mat homogr(cv::Size(3,3),cv::DataType<double>::type);
	homogr.at<double>(0,0) = 431; homogr.at<double>(1,0) = 148;
	homogr.at<double>(0,1) = 431; homogr.at<double>(1,1) = 148;
	homogr.at<double>(0,2) = 320; homogr.at<double>(1,1) = 240;

	cv::Mat invHomo = homogr.inv();
	invHomo.convertTo(invHomo,cv::DataType<double>::type);
	cv::Mat pointu(cv::Size(1,3),cv::DataType<double>::type);
	for(int xi=0; xi<foregr.cols; xi++){
		for(int yi=0; yi<foregr.rows; yi++){
			pointu.at<double>(0,0) = double(xi);
			pointu.at<double>(1,0) = double(yi);
			pointu.at<double>(2,0) = double(1);
			cv::Mat result = invHomo * pointu;
			std::cout<<abs((int)result.at<double>(1,0))<<" "<<
					abs((int)result.at<double>(0,0))<<std::endl;

			bkprojected.at<cv::Vec3d>(abs((int)result.at<double>(1,0)),\
			abs((int)result.at<double>(0,0))) = foregr.at<cv::Vec3d>(xi,yi);
		}
	}
	cv::imshow("bkproj",bkprojected);
	cv::imshow("foregr",colorRoi);
	cv::waitKey(0);
}
*/
//==============================================================================
/** Rotate matrix wrt to the camera location.
 */
cv::Mat featureDetector::rotateWrtCamera(cv::Point feetLocation,\
cv::Point cameraLocation, cv::Mat toRotate, cv::Point &borders){
	// GET THE ANGLE TO ROTATE WITH
	double cameraAngle = std::atan2((feetLocation.y-cameraLocation.y),\
						(feetLocation.x-cameraLocation.x));
	cameraAngle = (cameraAngle+M_PI/2.0);
	if(cameraAngle>2.0*M_PI){
		cameraAngle -= 2.0*M_PI;
	}
	cameraAngle *= (180.0/M_PI);


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
		srcRotate.cols/2.0,srcRotate.rows/2.0),cameraAngle, 1.0);

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
double featureDetector::fixAngle(cv::Point feetLocation, cv::Point cameraLocation,\
double angle){
	double cameraAngle = std::atan2((feetLocation.y-cameraLocation.y),\
						(feetLocation.x-cameraLocation.x));
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
void featureDetector::fixLabels(std::vector<cv::Point> feetPos){
	// FIND	THE INDEX FOR THE CURRENT IMAGE
	std::vector<annotationsHandle::FULL_ANNOTATIONS>::iterator index = \
		std::find_if (this->targetAnno.begin(), this->targetAnno.end(),\
		compareImg(this->current->sourceName));

	// LOOP OVER ALL ANNOTATIONS FOR THE CURRENT IMAGE AND FIND THE CLOSEST ONES
	std::vector<int> assignments((*index).annos.size(),-1);
	std::vector<double> minDistances((*index).annos.size(),\
		(double)INFINITY);
	unsigned canAssign = 1;
	while(canAssign){
		canAssign = 0;
		for(std::size_t l=0; l<(*index).annos.size(); l++){
			// EACH ANNOTATION NEEDS TO BE ASSIGNED TO THE CLOSEST DETECTION
			double distance    = (double)INFINITY;
			unsigned annoIndex = -1;
			for(std::size_t k=0; k<feetPos.size(); k++){
				double dstnc = dist((*index).annos[l].location,\
								feetPos[k]);
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
	std::vector<unsigned> unlabelled;
	for(std::size_t k=0; k<feetPos.size(); k++){
		bool inThere = false;
		for(std::size_t i=0; i<assignments.size(); i++){
			if(assignments[i]==k){
				inThere = true;
				break;
			}
		}
		if(!inThere){
			this->data.erase(this->data.begin()+(this->lastIndex+k));
			unlabelled.push_back(k);
		}
	}

	unsigned extra = 0;
	for(std::size_t i=0; i<assignments.size(); i++){
		if(assignments[i] != -1){
			extra++;
		}
	}
	extra += unlabelled.size();

	// STORE THE CPRRESPONDING LABELS
	this->targets.resize(this->lastIndex+extra,\
		cv::Mat::zeros(1,4,cv::DataType<double>::type));
	for(std::size_t i=0; i<assignments.size(); i++){
		if(assignments[i] != -1){
			cv::Mat tmp = cv::Mat::zeros(1,4,cv::DataType<double>::type);
			// READ THE TARGET ANGLE FOR LONGITUDE
			double angle = static_cast<double>\
				((*index).annos[i].poses[annotationsHandle::LONGITUDE]);

			std::cout<<"Longitude: "<<angle<<std::endl;

			angle = angle*M_PI/180.0;
			angle = this->fixAngle((*index).annos[i].location,\
					cv::Point(camPosX,camPosY),angle);
			tmp.at<double>(0,0) = std::sin(angle)*std::sin(angle);
			tmp.at<double>(0,1) = std::cos(angle)*std::cos(angle);

			// READ THE TARGET ANGLE FOR LATITUDE
			angle = static_cast<double>\
				((*index).annos[i].poses[annotationsHandle::LATITUDE]);
			std::cout<<"Latitude: "<<angle<<std::endl;

			angle = angle*M_PI/180.0;
			angle = this->fixAngle((*index).annos[i].location,\
					cv::Point(camPosX,camPosY),angle);
			tmp.at<double>(0,2) = std::sin(angle)*std::sin(angle);
			tmp.at<double>(0,3) = std::cos(angle)*std::cos(angle);

			this->targets[this->lastIndex+assignments[i]] = tmp.clone();
			tmp.release();
		}
	}

	for(std::size_t i=0; i<unlabelled.size(); i++){
		this->targets.erase(this->targets.begin()+(this->lastIndex+unlabelled[i]));
	}
	//-------------------------------------------------
	/*
	std::cout<<"current image/index: "<<this->current->sourceName<<" "<<std::endl;
	std::cout<<"The current image name is: "<<(*index).imgFile<<\
		" =?= "<<this->current->sourceName<<" (corresponding?)"<<std::endl;
	std::cout<<"Annotations: "<<std::endl;
	for(std::size_t i=0; i<(*index).annos.size(); i++){
		std::cout<<i<<":("<<(*index).annos[i].location.x<<","<<\
			(*index).annos[i].location.y<<") ";
	}
	std::cout<<std::endl;
	std::cout<<"Detections: "<<std::endl;
	for(std::size_t i=0; i<feetPos.size(); i++){
		std::cout<<i<<":("<<feetPos[i].x<<","<<feetPos[i].y<<") ";
	}
	std::cout<<std::endl;
	std::cout<<"Assignments: "<<std::endl;
	for(std::size_t i=0; i<assignments.size(); i++){
		std::cout<<"annot["<<i<<"]=>"<<assignments[i]<<" ";
	}
	std::cout<<std::endl;
	*/
	//-------------------------------------------------
}
//==============================================================================
/** Overwrites the \c doFindPeople function from the \c Tracker class to make it
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
	this->extractDataRow(existing, bg);

	//7) SHOW THE FOREGROUND POSSIBLE LOCATIONS AND PLOT THE TEMPLATES
	cerr<<"no. of detected people: "<<existing.size()<<endl;
	if(this->plotTracks){
		this->plotHull(bg, this->priorHull);
		for(unsigned i=0; i!=existing.size(); ++i){
			cv::Point pt = this->cvPoint(existing[i]);
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
void featureDetector::onlyPart(cv::Mat &thresholded, std::vector<CvPoint> &templ,\
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
		std::vector<CvPoint> tmpTempl = templ;
		for(unsigned i=0; i<tmpTempl.size(); i++){
			tmpTempl[i].x -= offsetX;
			tmpTempl[i].y -= offsetY;
		}
		plotTemplate2(new IplImage(thresholded), cv::Point(0,0), persHeight,\
				camHeight, cvScalar(150,0,0),tmpTempl);
		cv::imshow("templ", thresholded);
		cv::imshow("Part", thresholded);
		cv::waitKey(0);
	}
}
//==============================================================================
/** Computes the motion vector for the current image given the tracks so far.
 */
double featureDetector::motionVector(cv::Point center){
	cv::Point prev = center;
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
	return angle;
}
//==============================================================================
/*
int main(int argc, char **argv){
	featureDetector feature(argc,argv,true);
	feature.run();
}
*/

