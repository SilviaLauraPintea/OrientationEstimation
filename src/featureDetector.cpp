/* featureDetector.cpp
 * Author: Silvia-Laura Pintea
 */
#include "featureDetector.h"

//==============================================================================
struct onScanline{
	public:
		unsigned pixelY;
		onScanline(const unsigned pixelY){this->pixelY=pixelY;}
		bool operator()(const scanline_t line)const{
			return (line.line == this->pixelY);
		}
};
//==============================================================================
/** Initializes the parameters of the tracker.
 */
void featureDetector::init(std::string dataFolder, std::string theAnnotationsFile){
	std::cout<<"I can;t delete the producer"<<std::endl;
	this->initProducer(dataFolder.c_str(), true);
	std::cout<<"I can;t delete the producer???"<<std::endl;

	this->targets.clear();
	this->data.clear();

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

	std::cout<<"after init"<<std::endl;
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
/** Gets strong corner points in an image.
 */
void featureDetector::getCornerPoints(cv::Mat &feature, cv::Mat image,\
std::vector<unsigned> borders, cv::Mat thresholded){
	std::vector<cv::Point2f> corners;
	cv::Mat_<uchar> gray;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::goodFeaturesToTrack(gray, corners, 100, 0.001, image.rows/4);

	for(std::size_t i = 0; i <corners.size(); i++){
		// IF THE CORNERS ARE OUTSIDE THE BORDERS OF FOREGROUND PATCH
		if(borders[0]>corners[i].x || borders[1]<corners[i].x ||\
		borders[2]>corners[i].y || borders[3]<corners[i].y){
			corners.erase(corners.begin()+i);
		}
		if(this->plotTracks){
			cv::circle(image, cv::Point(corners[i].x, corners[i].y), 1,\
				cv::Scalar(0,0,255), 1, 8, 0);
		}
	}

	// some feature extractor!!!

	if(this->plotTracks){
		cv::imshow("Corners", image);
		cv::waitKey(0);
		cvDestroyWindow("Corners");
	}

	gray.release();
}
//==============================================================================
/** Gets the edges in an image.
 */
void featureDetector::getEdges(cv::Mat &feature, cv::Mat image, unsigned reshape,\
cv::Mat thresholded){
	// GET THE EDGES
	cv::Mat_<double> gray, edges;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::medianBlur(gray, gray, 3);
	cv::Canny(gray, edges, 60, 30, 3, true);
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
int minY, std::vector<CvPoint> templ){
	// EXTRACT THE SURF KEYPOINTS AND THE DESCRIPTORS
	std::vector<float> descriptors;
	cv::Mat mask;
	std::vector<cv::KeyPoint> keypoints;
	cv::SURF aSURF = cv::SURF(10,3,4,false);

	// FINDS MIN AND MAX IN TEMPLATE
	double minTmplX = std::max(0,templ[0].x-minX),
		maxTmplX = std::max(0,templ[0].x-minX),\
		minTmplY = std::max(0,templ[0].y-minY),\
		maxTmplY = std::max(0,templ[0].y-minY);
	for(std::size_t i=0; i<templ.size();i++){
		templ[i].y = std::max(0, templ[i].y - minY);
		templ[i].x = std::max(0, templ[i].x - minX);
		if(minTmplY>templ[i].y) minTmplY  = templ[i].y;
		if(maxTmplY<=templ[i].y) maxTmplY = templ[i].y;
		if(minTmplX>templ[i].x) minTmplX  = templ[i].x;
		if(maxTmplX<=templ[i].x) maxTmplX = templ[i].x;
	}
	cv::Mat_<double> gray;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	aSURF(gray, mask, keypoints, descriptors, false);

	// KEEP THE DESCRIPTORS WITHIN THE BORDERS ONLY
	std::vector<featureDetector::keyDescr> kD;
	for(std::size_t i=0; i<descriptors.size();i++){
		if(this->isInTemplate(keypoints[i].pt.x,keypoints[i].pt.y,templ)){
			featureDetector::keyDescr tmp;
			for(int j=0; j<aSURF.descriptorSize();j++){
				tmp.descr.push_back(descriptors[i*aSURF.descriptorSize()+j]);
			}
			tmp.keys = keypoints[i];
			kD.push_back(tmp);
		}
	}

	// SORT THE REMAINING DESCRIPTORS AND KEEP ONLY THE FIRST ?
	std::sort(kD.begin(),kD.end(),(&featureDetector::compareDescriptors));
	std::vector<float> keptDescr;
	feature = cv::Mat::zeros(cv::Size(10*aSURF.descriptorSize(),1),\
				cv::DataType<double>::type);
	for(unsigned i=0; i<std::min(10,static_cast<int>(kD.size())); i++){
		cv::Mat stupid1 = cv::Mat(kD[i].descr).t();
		cv::Mat stupid2 = feature.colRange(i*aSURF.descriptorSize(),\
							i*aSURF.descriptorSize()+aSURF.descriptorSize());
		stupid1.convertTo(stupid2,cv::DataType<double>::type);
		stupid1.release();
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
	mask.release();
	gray.release();
}
//==============================================================================
/** Creates a "histogram" of interest points + number of blobs.
 */
void featureDetector::interestPointsGrid(cv::Mat &feature, cv::Mat image,\
std::vector<CvPoint> templ, int minX, int minY){
	// FINDS MIN AND MAX IN TEMPLATE
	double minTmplX = std::max(0,templ[0].x-minX),
		maxTmplX = std::max(0,templ[0].x-minX),\
		minTmplY = std::max(0,templ[0].y-minY),\
		maxTmplY = std::max(0,templ[0].y-minY);
	for(std::size_t i=0; i<templ.size();i++){
		if(minTmplY>templ[i].y-minY) minTmplY  = templ[i].y - minY;
		if(maxTmplY<=templ[i].y-minY) maxTmplY = templ[i].y - minY;
		if(minTmplX>templ[i].x-minX) minTmplX  = templ[i].x - minX;
		if(maxTmplX<=templ[i].x-minX) maxTmplX = templ[i].x - minX;
	}

	// EXTRACT MAXIMALLY STABLE BLOBS
	std::vector<std::vector<cv::Point> > msers;
	cv::MSER aMSER;
	cv::Mat mask;
	aMSER(image, msers, mask);
	mask.release();

	// COUNTE THE CONTOURS INSIDE THE TEMPLATE
	uchar msersTmpl = 0;
	for(std::size_t x=0; x<msers.size(); x++){
		for(std::size_t y=0; y<msers[x].size(); y++){
			if(this->isInTemplate(msers[x][y].x+minX,msers[x][y].y+minY,templ)){
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
	feature                 = cv::Mat(1,no*no+1,cv::DataType<double>::type);
	feature.at<double>(0,0) = static_cast<double>(msersTmpl);

	// HISTOGRAM OF NICE FEATURES
	std::vector<cv::Point2f> corners;
	cv::Ptr<cv::FeatureDetector> detector = \
		new cv::GoodFeaturesToTrackDetector(5000, 0.001, 0.1);
	cv::GridAdaptedFeatureDetector gafd(detector, 5000, no, no);
	std::vector<cv::KeyPoint> keys;
	std::vector<unsigned> indices;
	gafd.detect(image, keys);

	//COUNT THE INTEREST POINTS IN EACH CELL
	unsigned contor = 0;
	cv::Mat histo   = cv::Mat::zeros(1,no*no,cv::DataType<double>::type);
	for(double x=minTmplX; x<maxTmplX; x+=(maxTmplX-minTmplX)/no){
		for(double y=minTmplY; y<maxTmplY; y+=(maxTmplY-minTmplY)/no){
			if(indices.empty()){
				for(std::size_t i=0; i<keys.size(); i++){
					if(this->isInTemplate(keys[i].pt.x+minX,keys[i].pt.y+minY,templ)){
						indices.push_back(i);
						if(x<=keys[i].pt.x && keys[i].pt.x<x+(maxTmplX-minTmplX)/no\
						&& y<=keys[i].pt.y && keys[i].pt.y<y+(maxTmplY-minTmplY)/no){
							histo.at<double>(0,contor) += 1.0;
						}
					}
				}
			}else{
				for(std::size_t j=0; j<indices.size(); j++){
					unsigned i = indices[j];
					if(x<=keys[i].pt.x && keys[i].pt.x<x+(maxTmplX-minTmplX)/no\
					&& y<=keys[i].pt.y && keys[i].pt.y<y+(maxTmplY-minTmplY)/no){
						histo.at<double>(0,contor) += 1.0;
					}
				}
			}
			contor +=1;
		}
	}
	cv::Mat stupid = feature.colRange(1,no*no+1);
	histo.copyTo(stupid);
	histo.release();

	//-----------------REMOVE--------------------------
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
	this->getEdges(edges, img, 0);

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
	if(iter == lines.end()) return false;

/*
	cout << "iter->line=" << iter->line << endl;
	cout << "PixelX=" << pixelX << endl;
	cout << "PixelY=" << pixelY << endl;
	cout << "iter->start=" << iter->start << endl;
	cout << "iter->end=" << iter->end << endl;
*/

	if(iter->line==pixelY && iter->start<=pixelX && iter->end>=pixelX){
		return true;
	}else{
		return false;
	}
}
//==============================================================================
/** Returns the size of a window around a template centered in a given point.
 */
void featureDetector::templateWindow(cv::Size imgSize, int &minX, int &maxX,\
int &minY, int &maxY, std::vector<CvPoint> &templ, unsigned tplBorder){
	// GET THE MIN/MAX SIZE OF THE TEMPLATE
	for(unsigned i=0; i<templ.size(); i++){
		if(minX>templ[i].x){minX = templ[i].x;}
		if(maxX<templ[i].x){maxX = templ[i].x;}
		if(minY>templ[i].y){minY = templ[i].y;}
		if(maxY<templ[i].y){maxY = templ[i].y;}
	}

	// TRY TO ADD BORDERS TO MAKE IT 100
	int diffX = (tplBorder - (maxX-minX))/2;
	int diffY = (tplBorder - (maxY-minY))/2;
	minY = std::max(minY-diffY,0);
	maxY = std::min(maxY+diffY,imgSize.height);
	minX = std::max(minX-diffX,0);
	maxX = std::min(maxX+diffX,imgSize.width);

	if(minX-maxX!=tplBorder){
		int diffX2 = tplBorder-(maxX-minX);
		if(minX>diffX2){
			minX += diffX2;
		}else if(maxX<imgSize.width-diffX2){
			minX += diffX2;
		}
	}
	if(minY-maxY!=tplBorder){
		int diffY2 = tplBorder-(maxY-minY);
		if(minY>diffY2){
			minY += diffY2;
		}else if(maxY<imgSize.height-diffY2){
			minY += diffY2;
		}
	}
}
//==============================================================================
/** Get the foreground pixels corresponding to each person.
 */
void featureDetector::allForegroundPixels(std::vector<featureDetector::people>\
&allPeople, std::vector<unsigned> existing, IplImage *bg, double threshold){
	// INITIALIZING STUFF
	cv::Mat thsh(cvCloneImage(bg));
	cv::Mat thrsh(thsh.rows, thsh.cols, CV_8UC1);
	cv::cvtColor(thsh, thrsh, CV_BGR2GRAY);
	cv::threshold(thrsh, thrsh, threshold, 255, cv::THRESH_BINARY);
	cv::Mat foregr(cvCloneImage(this->current->img));

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
		allPeople[k].pixels     = cv::Mat(colorRoi.clone());
		if(this->plotTracks){
			cv::imshow("people",allPeople[k].pixels);
			cv::waitKey(0);
		}
		colorRoi.release();

		//-------------REMOVE--------------------
		/*
		plotTemplate2(this->current->img,center,persHeight,camHeight,\
				cv::Scalar(255,0,0));
		cv::imshow("template",this->current->img);
		cv::imshow("PPL",allPeople[k].pixels);
		cv::waitKey(0);
		*/
		//---------------------------------------------
	}
	thrsh.release();
	foregr.release();
}
//==============================================================================
/** Convolves an image with a Gabor filter with the given parameters and
 * returns the response image.
 */
void featureDetector::getGabor(cv::Mat &response, cv::Mat image, float *params){
	// params[0] -- sigma: (3, 68)
	// params[1] -- gamma: (0.2, 1)
	// params[2] -- dimension: (1, 10)
	// params[3] -- theta: (0, 180) or (-90, 90)
	// params[4] -- lambda: (2, 256)
	// params[5] -- psi: (0, 180)

	if(!params){
		params = new float[6];
		params[0] = 3.0; params[1] = 0.4;
		params[2] = 2.0; params[3] = M_PI/4.0;
		params[4] = 4.0; params[5] = 20;
	}

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

	cv::Mat gabor = cv::Mat::zeros((int)(xMax-xMin),(int)(yMax-yMin),\
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
	cv::imshow("Gabor",gabor);
	cv::waitKey(0);
	cvDestroyWindow("Gabor");

	cv::filter2D(image,response,-1,gabor,cv::Point(-1,-1),0,cv::BORDER_REPLICATE);
	cv::imshow("GaborResponse",response);
	cv::waitKey(0);
	cvDestroyWindow("GaborResponse");

	gabor.release();
}
//==============================================================================
/** Function that gets the ROI corresponding to a head/feet of a person in
 * an image.
 */
void featureDetector::upperLowerROI(featureDetector::people someone,
double variance, cv::Mat &upperRoi, cv::Mat &lowerRoi){
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
	upperRoi = someone.pixels.clone();
	lowerRoi = someone.pixels.clone();
	for(int x=0; x<someone.pixels.cols; x++){
		for(int y=0; y<someone.pixels.rows; y++){
			if(!this->sameSubplane(cv::Point(x,y),A,m,b)){
				upperRoi.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,0);
			}else{
				lowerRoi.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,0);
			}
		}
	}
	cv::imshow("UpperPart", upperRoi);
	cv::imshow("LowerPart", lowerRoi);
	cv::waitKey(0);
	cvDestroyWindow("LowerPart");
	cvDestroyWindow("UpperPart");
}
//==============================================================================
/** Set what kind of features to extract.
 */
void featureDetector::setFeatureType(featureDetector::FEATURE type){
	this->featureType = type;
}
//==============================================================================
/** Creates on data row in the final data matrix by getting the feature
 * descriptors.
 */
void featureDetector::extractDataRow(std::vector<unsigned> existing, IplImage *bg){
	cv::Mat image(cvCloneImage(this->current->img));
	cv::cvtColor(image, image, CV_BGR2Lab);

	// REDUCE THE IMAGE TO ONLY THE INTERESTING AREA
	std::vector<featureDetector::people> allPeople(existing.size());
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
		cv::Mat thresholded;
		cv::inRange(allPeople[i].pixels,cv::Scalar(1,1,1),cv::Scalar(255,225,225),\
			thresholded);

		//-----------REMOVE------------------------
		/*
		cv::imshow("imgRoi", imgRoi);
		cv::imshow("pixels", allPeople[i].pixels);
		cv::imshow("thresh", thresholded);
		cv::waitKey(0);
		*/
		//-----------------------------------------

		// EXTRACT FEATURES
		cv::Mat feature;
		switch(this->featureType){
			case (featureDetector::IPOINTS):
				this->interestPointsGrid(feature, imgRoi, templ,\
					allPeople[i].borders[0],allPeople[i].borders[2]);
				break;
			case featureDetector::EDGES:
				this->getEdges(feature, imgRoi, 1, thresholded);
				break;
			case featureDetector::SURF:
				this->getSURF(feature, imgRoi, allPeople[i].borders[0],\
					allPeople[i].borders[2], templ);
				break;
			case featureDetector::CORNER:
				this->getCornerPoints(feature, imgRoi, allPeople[i].borders, thresholded);
				break;

			/*
			case featureDetector::ELLIPSE:
				cv::Point2f boxCenter(0.f,0.f);
				cv::Size2f boxSize(0,0);
				cv::RotatedRect finalBox(boxCenter, boxSize, 0);
				this->skinEllipses(finalBox,imgRoi,center,cv::Point(minX,minY),\
				20,60);
				if(!(finalBox.center == boxCenter && finalBox.angle == 0 && \
				finalBox.size == boxSize)){
					break;
				}
			case featureDetector::GABOR:
				cv::Mat response;
				this->getGabor(response, imgRoi);
				break;
			 */
		}
		feature.convertTo(feature, cv::DataType<double>::type);
		this->data.push_back(feature.clone());

		thresholded.release();
		feature.release();
		imgRoi.release();
	}
	// FIX THE LABELS TO CORRESPOND TO THE PEOPLE DETECTED IN THE IMAGE
	this->fixLabels(allLocations, this->current->sourceName, this->current->index);
	image.release();
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
/** Fixes the angle to be relative to the camera position with respect to the
 * detected position.
 */
double featureDetector::fixAngle(cv::Point feetLocation, cv::Point cameraLocation,\
double angle){
	double cameraAngle = std::atan2((feetLocation.y-cameraLocation.y),\
						(feetLocation.x-cameraLocation.x));
	angle = angle-cameraAngle;
	if(angle>=2.0*M_PI){
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
void featureDetector::fixLabels(std::vector<cv::Point> feetPos, string imageName,\
unsigned index){
	// LOOP OVER ALL ANNOTATIONS FOR THE CURRENT IMAGE AND FIND THE CLOSEST ONES
	std::vector<int> assignments(this->targetAnno[index].annos.size(),-1);
	std::vector<double> minDistances(this->targetAnno[index].annos.size(),\
		(double)INFINITY);
	unsigned canAssign = 1;
	while(canAssign){
		canAssign = 0;
		for(std::size_t l=0; l<this->targetAnno[index].annos.size(); l++){
			// EACH ANNOTATION NEEDS TO BE ASSIGNED TO THE CLOSEST DETECTION
			double distance    = (double)INFINITY;
			unsigned annoIndex = -1;
			for(std::size_t k=0; k<feetPos.size(); k++){
				double dstnc = dist(this->targetAnno[index].annos[l].location,\
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
		cv::Mat::zeros(1,2,cv::DataType<double>::type));
	for(std::size_t i=0; i<assignments.size(); i++){
		if(assignments[i] != -1){
			cv::Mat tmp = cv::Mat::zeros(1,2,cv::DataType<double>::type);
			double angle = static_cast<double>\
				(targetAnno[index].annos[i].poses[annotationsHandle::ORIENTATION]);
			angle = angle*M_PI/180.0;
			angle = this->fixAngle(targetAnno[index].annos[i].location,\
					cv::Point(camPosX,camPosY),angle);
			tmp.at<double>(0,0) = std::sin(angle);
			tmp.at<double>(0,1) = std::cos(angle);
			this->targets[this->lastIndex+assignments[i]] = tmp.clone();
			tmp.release();
		}
	}

	for(std::size_t i=0; i<unlabelled.size(); i++){
		this->targets.erase(this->targets.begin()+(this->lastIndex+unlabelled[i]));
	}

	//-------------------------------------------------
	/*
	std::cout<<"current image/index: "<<imageName<<" "<<index<<" "<<std::endl;
	std::cout<<"The current image name is: "<<this->targetAnno[index].imgFile<<\
		" =?= "<<imageName<<" (corresponding?)"<<std::endl;
	std::cout<<"Annotations: "<<std::endl;
	for(std::size_t i=0; i<this->targetAnno[index].annos.size(); i++){
		std::cout<<i<<":("<<targetAnno[index].annos[i].location.x<<","<<\
			targetAnno[index].annos[i].location.y<<") ";
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
	for(std::size_t i=0; i<this->targets.size(); i++){
		std::cout<<"("<<this->targets[i].at<double>(0,0)<<","<<\
			this->targets[i].at<double>(0,1)<<")";
	}
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
			plotTemplate2(bg,pt,persHeight,camHeight,CV_RGB(255,255,255));
			plotScanLines(this->current->img,mask,CV_RGB(0,255,0),0.3);
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
/*int main(int argc, char **argv){
	featureDetector feature(argc,argv,true);
	feature.run();
}*/
