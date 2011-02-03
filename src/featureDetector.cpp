/* featureDetector.cpp
 * Author: Silvia-Laura Pintea
 */
#include "featureDetector.h"
//==============================================================================
/** Initializes the parameters of the tracker.
 */
void featureDetector::init(std::string dataFolder){
	this->producer = NULL;
	this->producer = new ImgProducer(dataFolder.c_str(), true);
	for(size_t i=0; i<this->data.size(); i++){
		this->data[i].release();
	}
	this->data.clear();
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
void featureDetector::getEdges(cv::Mat &feature, cv::Mat image, std::vector<unsigned> \
borders, unsigned reshape, cv::Mat thresholded){
	cv::Mat_<double> gray, edges;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::medianBlur(gray, gray, 3);

	cv::Canny(gray, edges, 60, 30, 3, true);
	edges = edges(cv::Range(borders[2],borders[3]),cv::Range(borders[0],borders[1]));

	if(reshape){
		cv::Mat continuousEdges = cv::Mat::zeros(cv::Size(edges.cols,edges.rows),\
									cv::DataType<double>::type);
		if(!thresholded.empty()){
			edges.copyTo(continuousEdges,thresholded);
			//-------------------------------------
			cv::imshow("edges",continuousEdges);
			cv::waitKey(0);
			//-------------------------------------
		}else{
			edges.copyTo(continuousEdges);
		}
		feature = (continuousEdges.clone()).reshape(0,1);
		continuousEdges.release();
	}else{
		if(!thresholded.empty()){
			edges.copyTo(feature,thresholded);
		}else{
			edges.copyTo(feature);
		}
	}
	if(this->plotTracks){
		cv::imshow("Edges", edges);
		cv::waitKey(0);
		cvDestroyWindow("Edges");
	}
	gray.release();
	edges.release();
}
//==============================================================================
/** SURF descriptors (Speeded Up Robust Features).
 */
void featureDetector::getSURF(std::vector<float>& descriptors, cv::Mat image){
	cv::Mat mask;
	std::vector<cv::KeyPoint> keypoints;
	cv::SURF aSURF = cv::SURF();
	aSURF(image, mask, keypoints, descriptors, false);

	/*
	Point2f pt;     ==> coordinates of the key-points
	float size;     ==> diameter of the meaning-full key-point neighborhood
	float angle;    ==> computed orientation of the key-point(-1 if not applicable)
	float response; ==> the response by which the most strong key-points
						have been selected. Can be used for the further sorting
						or sub-sampling
	int octave;     ==> octave (pyramid layer) from which the key-point has
						been extracted
	int class_id;   ==> object class (if the key-points need to be clustered by
						an object they belong to)
	*/

	for(std::size_t i = 0; i <keypoints.size(); i++){
		cv::circle(image, cv::Point(keypoints[i].pt.x, keypoints[i].pt.y),\
			keypoints[i].size, cv::Scalar(0,0,255), 1, 8, 0);
	}
	cv::imshow("SURFS", image);
	cv::waitKey(0);
	cvDestroyWindow("SURFS");

	mask.release();
}
//==============================================================================
/** Blob detector in RGB color space.
 */
void featureDetector::blobDetector(cv::Mat &feature, cv::Mat image,\
std::vector<unsigned> borders, cv::Mat thresholded, string featType){
	std::vector<std::vector<cv::Point> > msers;
	/*
	int _delta, _min_area, _max_area, _max_evolution, _edge_blur_size;
	float _max_variation, _min_diversity;
	double _area_threshold, _min_margin;
	cv::MSER aMSER = cv::MSER(_delta, _min_area, _max_area, _max_variation,\
		_min_diversity, _max_evolution, _area_threshold, _min_margin, _edge_blur_size );
	 */
	cv::MSER aMSER;

	// runs the extractor on the specified image; returns the MSERs,
	// each encoded as a contour the optional mask marks the area where MSERs
	// are searched for
	cv::Mat mask;
	aMSER(image, msers, mask);
	mask.release();

	// KEEP ONLY THE DESCRIPTORS WITHIN THE BORDERS
	uchar removed = 1;
	while(removed){
		removed = 0;
		for(std::size_t x=0; x<msers.size(); x++){
			for(std::size_t y=0; y<msers[x].size(); y++){
				if(borders[0]>msers[x][y].x || borders[1]<msers[x][y].x ||\
				borders[2]>msers[x][y].y || borders[3]<msers[x][y].y){
					msers[x].erase(msers[x].begin()+y);
					removed = 1;
					break;
				}
			}
			if(msers[x].empty()){ msers.erase(msers.begin()+x);
				removed = 1;
				break;
			}
		}
	}

	// SAVE IN A MATRIX
	if(this->plotTracks){
		cv::Mat tmp = cv::Mat::zeros(cv::Size(borders[3]-borders[2],\
						borders[1]-borders[0]), CV_8UC1);
		cv::drawContours(tmp, msers, -1, cv::Scalar(255,255,255), 1, 8);
		cv::imshow("feature",tmp);
		cv::waitKey(0);
		tmp.release();
	}

	if(featType == "2d"){
		feature = cv::Mat(msers).reshape(0,1);
		// THE VALUES IN THE FEATURES
		for(int x=0; x<feature.cols; x++){
			for(int y=0; y<feature.rows; y++){
				cout<<"feature values: "<<(int)(feature.at<cv::Vec2b>(y,x)[0])<<" "<<\
					(int)(feature.at<cv::Vec2b>(y,x)[1])<<endl;
			}
		}
	}else{
		//cv::Mat tmp = cv::Mat::zeros(cv::Size(borders[1]-borders[0],\
						borders[3]-borders[2]), CV_8UC1);

		cv::Mat tmp = cv::Mat::zeros(image.size(), CV_8UC1);
		cv::drawContours(tmp, msers, -1, cv::Scalar(255,255,255), 1, 8);

		cv::imshow("feature",tmp);
		cv::waitKey(0);


		feature = tmp.reshape(0,1);
		/*
		// THE VALUES IN THE FEATURES
		for(int x=0; x<feature.cols; x++){
			for(int y=0; y<feature.rows; y++){
				cout<<"feature vales: "<<(int)(feature.at<uchar>(y,x))<<endl;
			}
		}
		*/
	}
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
	this->getEdges(edges, img, std::vector<unsigned>(4,0), 0);

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
	for(std::vector<scanline_t>::const_iterator i=lines.begin();i!=lines.end();++i){
		if(i->line==pixelY && i->start<=pixelX && i->end>=pixelX){
			return true;
		};
	}
	return false;
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
	minY = std::max((int)(minY-tplBorder),0);
	maxY = std::min(imgSize.height,(int)(maxY+tplBorder));
	minX = std::max((int)(minX-tplBorder),0);
	maxX = std::min(imgSize.width,(int)(maxX+tplBorder));
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

	std::vector<int> tmpminX(existing.size(),thrsh.cols);
	std::vector<int> tmpmaxX(existing.size(),0);
	std::vector<int> tmpminY(existing.size(),thrsh.rows);
	std::vector<int> tmpmaxY(existing.size(),0);

	// FOR EACH EXISTING TEMPLATE LOOK ON AN AREA OF 100 PIXELS AROUND IT
	cout<<"number of templates: "<<existing.size()<<endl;
	for(unsigned k=0; k<existing.size();k++){
		cv::Point center         = this->cvPoint(existing[k]);
		allPeople[k].absoluteLoc = center;
		std::vector<CvPoint> templ;
		genTemplate2(center, persHeight, camHeight, templ);

		int minY=thrsh.rows, maxY=0, minX=thrsh.cols, maxX=0;
		this->templateWindow(cv::Size(foregr.cols,foregr.rows),minX, maxX,\
			minY, maxY, templ);

		int width   = maxX-minX;
		int height  = maxY-minY;
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

		width  = tmpmaxX[k]-tmpminX[k];
		height = tmpmaxY[k]-tmpminY[k];
		if(width!=100){
			tmpmaxX[k] += static_cast<int>((100-width)/2.0);
			tmpminX[k] -= static_cast<int>((100-width)/2.0);
		}
		if(height!=100){
			tmpmaxY[k] += static_cast<int>((100-height)/2.0);
			tmpminY[k] -= static_cast<int>((100-height)/2.0);
		}
		if(tmpmaxX[k]-tmpminX[k]!=100){tmpmaxX[k] += 100-(tmpmaxX[k]-tmpminX[k]);}
		if(tmpmaxY[k]-tmpminY[k]!=100){tmpmaxY[k] += 100-(tmpmaxY[k]-tmpminY[k]);}

		cout<<tmpmaxX[k]<<" "<<tmpminX[k]<<endl;
		cout<<tmpmaxY[k]<<" "<<tmpminY[k]<<endl;

		allPeople[k].relativeLoc = cv::Point(center.x - (int)(minX + tmpminX[k]),\
									center.y - (int)(minY + tmpminY[k]));
		allPeople[k].borders.assign(4,0);
		allPeople[k].borders[0] = tmpminX[k]+minX;
		allPeople[k].borders[1] = tmpmaxX[k]+minX;
		allPeople[k].borders[2] = tmpminY[k]+minY;
		allPeople[k].borders[3] = tmpmaxY[k]+minY;

		width  = tmpmaxX[k]-tmpminX[k];
		height = tmpmaxY[k]-tmpminY[k];
		allPeople[k].pixels = cv::Mat(colorRoi.clone(),\
			cv::Rect(cv::Point(tmpminX[k],tmpminY[k]),cv::Size(width,height)));
		if(this->plotTracks){
			cv::imshow("people",allPeople[k].pixels);
			cv::waitKey(0);
		}
		colorRoi.release();
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
	cv::Mat image(this->current->img);
	// REDUCE THE IMAGE TO ONLY THE INTERESTING AREA
	std::vector<featureDetector::people> allPeople(existing.size());
	this->allForegroundPixels(allPeople, existing, bg, 7.0);

	// FOR EACH LOCATION IN THE IMAGE EXTRACT FEATURES, FILTER THEM AND RESHAPE
	for(std::size_t i=0; i<existing.size(); i++){

		// CONSIDER ONLY A WINDOW OF 100X100 AROUND THE TEMPLATE
		cv::Point center = this->cvPoint(existing[i]);
		std::vector<CvPoint> templ;
		genTemplate2(center, persHeight, camHeight, templ);
		int minY=image.rows, maxY=0, minX=image.cols, maxX=0;
		this->templateWindow(cv::Size(image.cols,image.rows), minX, maxX,\
			minY, maxY, templ);
		unsigned width  = maxX-minX;
		unsigned height = maxY-minY;
		cv::Mat imgRoi  = cv::Mat(image.clone(),cv::Rect(cv::Point(minX,minY),\
							cv::Size(width,height)));

		// EXTRACT FEATURES
		cv::Mat feature;
		std::vector<unsigned> borders = allPeople[i].borders;
		borders[0] -= minX; borders[1] -= minX;
		borders[2] -= minY; borders[3] -= minY;

		cv::Mat thresholded;
		cv::inRange(allPeople[i].pixels,cv::Scalar(1,1,1),cv::Scalar(255,225,225),\
			thresholded);
		//------------------------------------------
		cv::imshow("mask",thresholded);
		cv::imshow("foreground",allPeople[i].pixels);
		cv::waitKey(0);
		//------------------------------------------

		switch(this->featureType){
			case (featureDetector::BLOB):
				this->blobDetector(feature, imgRoi, borders, thresholded);
				break;
			case featureDetector::CORNER:
				this->getCornerPoints(feature, imgRoi, borders, thresholded);
				break;
			case featureDetector::EDGES:
				this->getEdges(feature, imgRoi, borders, 1, thresholded);
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
			case featureDetector::SURF:
				std::vector<float> descriptors;
				this->getSURF(descriptors, imgRoi);
				break;
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
			cv::Mat tmp = cv::Mat(cvCloneImage(this->current->img));
			char buffer[50];
			sprintf(buffer,"%u",i);
			cv::putText(tmp,(string)buffer,pt,3,3.0,cv::Scalar(0,0,255),1,8,false);
			tmp.release();
		}
		cvShowImage("bg", bg);
		cvShowImage("image",src);
	}
	//8) WHILE NOT q WAS PRESSED PROCESS THE NEXT IMAGES
	return this->imageProcessingMenu();
	cvReleaseImage(&bg);
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
