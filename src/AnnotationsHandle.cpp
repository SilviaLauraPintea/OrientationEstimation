/* AnnotationsHandle.cpp
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#include <err.h>
#include <iostream>
#include <exception>
#include <cmath>
#include <fstream>
#include <string>
#include <stdio.h>
#include <boost/version.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#if BOOST_VERSION < 103500
	#include <boost/thread/detail/lock.hpp>
#endif
#include <boost/thread/xtime.hpp>
#include "eigenbackground/src/Annotate.hh"
#include "Auxiliary.h"
#include "AnnotationsHandle.h"
//==============================================================================
/** Initializes all the values of the class variables.
 */
void AnnotationsHandle::init(){
	if(image_){
		image_.reset();
	}
	choice_    = ' ';
	withPoses_ = false;
	poseSize_  = 5;
	poseNames_.push_back("SITTING");
	poseNames_.push_back("STANDING");
	poseNames_.push_back("BENDING");
	poseNames_.push_back("LONGITUDE");
	poseNames_.push_back("LATITUDE");
}
//==============================================================================
/** Define a post-fix increment operator for the enum \c POSE.
 */
void operator++(AnnotationsHandle::POSE &refPose){
	refPose = (AnnotationsHandle::POSE)(refPose+1);
}
//==============================================================================
/** Mouse handler for annotating people's positions and poses.
 */
void AnnotationsHandle::mouseHandlerAnn(int event,int x,int y,int flags,void *param){
	static bool down = false;
	switch (event){
		case CV_EVENT_LBUTTONDOWN:
			if(choice_ == 'c'){
				down = true;
				cout<<"Left button down at >>> ("<<x<<","<<y<<")"<<endl;
				Annotate::plotAreaTmp(image_.get(),(float)x,(float)y);
			}
			break;
		case CV_EVENT_MOUSEMOVE:
			if(down){
				Annotate::plotAreaTmp(image_.get(),(float)x,(float)y);
			}
			break;
		case CV_EVENT_LBUTTONUP:
			if(choice_ == 'c'){
				cout<<"Left button up at >>> ("<<x<<","<<y<<")"<<endl;
				choice_ = ' ';
				down = false;
				AnnotationsHandle::ANNOTATION temp;
				temp.location_ = cv::Point2f(x,y);
				std::cout<<"Saved location: "<<temp.location_<<std::endl;
				temp.id_       = annotations_.size();
				temp.poses_.assign(poseSize_,0);
				temp.poses_[(int)LATITUDE] = 90;
				annotations_.push_back(temp);
				cv::Point2f aCenter(x,y);
				AnnotationsHandle::showMenu(aCenter);
				for(unsigned i=0;i!=annotations_.size();++i){
					Annotate::plotArea(image_.get(),(float)annotations_[i].location_.x,\
						(float)annotations_[i].location_.y);
				}
			}
			break;
	}
}
//==============================================================================
/** Rotate matrix wrt to the camera location.
 */
cv::Mat AnnotationsHandle::rotateWrtCamera(const cv::Point2f &headLocation,\
const cv::Point2f &feetLocation,const cv::Mat &toRotate,cv::Point2f &borders){
	// GET THE ANGLE TO ROTATE WITH
	float cameraAngle = std::atan2((headLocation.y-feetLocation.y),\
						(headLocation.x-feetLocation.x));
	cameraAngle = (cameraAngle+M_PI/2.0);
	Auxiliary::angle0to360(cameraAngle);

	// THE ROTATION ANGLE NEEDS TO BE IN DEGREES
	cameraAngle *= (180.0/M_PI);

	// ADD A BLACK BORDER TO THE ORIGINAL IMAGE
	float diag       = std::sqrt(toRotate.cols*toRotate.cols+toRotate.rows*\
						toRotate.rows);
	borders.x         = std::ceil((diag-toRotate.cols)/2.0);
	borders.y         = std::ceil((diag-toRotate.rows)/2.0);
	cv::Mat srcRotate = cv::Mat::zeros(cv::Size(toRotate.cols+2*borders.x,\
						toRotate.rows+2*borders.y),toRotate.type());
	cv::copyMakeBorder(toRotate,srcRotate,borders.y,borders.y,borders.x,\
		borders.x,cv::BORDER_CONSTANT);

	// GET THE ROTATION MATRIX
	cv::Mat rotationMat = cv::getRotationMatrix2D(cv::Point2f(\
		srcRotate.cols/2.0,srcRotate.rows/2.0),cameraAngle,1.0);

	// ROTATE THE IMAGE WITH THE ROTATION MATRIX
	cv::Mat rotated = cv::Mat::zeros(srcRotate.size(),toRotate.type());
	cv::warpAffine(srcRotate,rotated,rotationMat,srcRotate.size());

	rotationMat.release();
	srcRotate.release();
	return rotated;
}
//==============================================================================
/** Shows how the selected orientation looks on the image.
 */
void AnnotationsHandle::drawLatitude(const cv::Point2f &head,const cv::Point2f &feet,\
unsigned int orient,AnnotationsHandle::POSE pose){
	unsigned int length = 80;
	float angle = (M_PI * orient)/180;

	// GET THE TEMPLATE AND DETERMINE ITS SIZE
	std::vector<cv::Point2f> points;
	Helpers::genTemplate2(feet,Helpers::persHeight(),Helpers::camHeight(),points);
	int maxX=0,maxY=0,minX=image_->width,minY=image_->height;
	for(unsigned i=0;i<points.size();++i){
		if(maxX<points[i].x){maxX = points[i].x;}
		if(maxY<points[i].y){maxY = points[i].y;}
		if(minX>points[i].x){minX = points[i].x;}
		if(minY>points[i].y){minY = points[i].y;}
	}

	minX = std::max(minX-10,0);
	minY = std::max(minY-10,0);
	maxX = std::min(maxX+10,image_->width);
	maxY = std::min(maxY+10,image_->height);
	// ROTATE THE TEMPLATE TO HORIZONTAL SO WE CAN SEE IT
	cv::Mat tmpImage((image_.get()),cv::Rect(cv::Point2f(minX,minY),\
		cv::Size(maxX-minX,maxY-minY)));
	cv::Point2f stupid;
	cv::Mat tmpImg = AnnotationsHandle::rotateWrtCamera(head,feet,tmpImage,stupid);
	cv::Mat large;
	cv::resize(tmpImg,large,cv::Size(0,0),1.5,1.5,cv::INTER_CUBIC);

	// DRAW THE LINE ON WHICH THE ARROW SITS
	cv::Point2f center(large.cols*1/4,large.rows/2);
	cv::Point point1,point2;
	point1.x = center.x - 0.3*length * cos(angle + M_PI/2.0);
	point1.y = center.y + 0.3*length * sin(angle + M_PI/2.0);
	point2.x = center.x - length * cos(angle + M_PI/2.0);
	point2.y = center.y + length * sin(angle + M_PI/2.0);
	cv::clipLine(large.size(),point1,point2);
	cv::line(large,point1,point2,cv::Scalar(100,50,255),2,8,0);

	// DRAW THE TOP OF THE ARROW
	cv::Point2f point3,point4,point5;
	point3.x = center.x - length * 4/5 * cos(angle + M_PI/2.0);
	point3.y = center.y + length * 4/5 * sin(angle + M_PI/2.0);
	point4.x = point3.x - 7 * cos(M_PI + angle);
	point4.y = point3.y + 7 * sin(M_PI + angle);
	point5.x = point3.x - 7 * cos(M_PI + angle  + M_PI);
	point5.y = point3.y + 7 * sin(M_PI + angle  + M_PI);

	// FILL THE POLLY CORRESPONDING TO THE ARROW
	cv::Point *pts = new cv::Point[4];
	pts[0] = point4;
	pts[1] = point2;
	pts[2] = point5;
	cv::fillConvexPoly(large,pts,3,cv::Scalar(100,50,255),8,0);
	delete [] pts;

	// PUT A CIRCLE ON THE CENTER POINT
	cv::circle(large,center,1,cv::Scalar(255,50,0),1,8,0);
	cv::imshow("Latitude",large);
	tmpImg.release();
	large.release();
	tmpImage.release();
}
//==============================================================================
/** Shows how the selected orientation looks on the image.
 */
void AnnotationsHandle::drawOrientation(const cv::Point2f &center,\
unsigned int orient,const std::tr1::shared_ptr<IplImage> im){
	unsigned int length = 60;
	float angle = (M_PI * orient)/180;
	cv::Point point1,point2;
	point1.x = center.x - length * cos(angle);
	point1.y = center.y + length * sin(angle);
	point2.x = center.x - length * cos(angle + M_PI);
	point2.y = center.y + length * sin(angle + M_PI);

	cv::Size imgSize(im->width,im->height);
	cv::clipLine(imgSize,point1,point2);

	cv::Mat tmpImage1(im.get());
	cv::Mat tmpImage(tmpImage1.clone());
	cv::line(tmpImage,point1,point2,cv::Scalar(150,50,0),2,8,0);

	cv::Point2f point3,point4,point5;
	point3.x = center.x - length * 4/5 * cos(angle + M_PI);
	point3.y = center.y + length * 4/5 * sin(angle + M_PI);
	point4.x = point3.x - 7 * cos(M_PI/2.0 + angle);
	point4.y = point3.y + 7 * sin(M_PI/2.0 + angle);
	point5.x = point3.x - 7 * cos(M_PI/2.0 + angle  + M_PI);
	point5.y = point3.y + 7 * sin(M_PI/2.0 + angle  + M_PI);

	cv::Point *pts = new cv::Point[4];
	pts[0] = point4;
	pts[1] = point2;
	pts[2] = point5;
	cv::fillConvexPoly(tmpImage,pts,3,cv::Scalar(255,50,0),8,0);

	delete [] pts;
	cv::circle(tmpImage,center,1,cv::Scalar(255,50,0),1,8,0);
	cv::imshow("image",tmpImage);
	tmpImage.release();
	tmpImage1.release();
}
//==============================================================================
/** Overloaded version for cv::Mat -- shows how the selected orientation
 * looks on the image.
 */
cv::Mat AnnotationsHandle::drawOrientation(const cv::Point2f &center,\
unsigned int orient,const cv::Mat &im,const cv::Scalar &color){
	unsigned int length = 60;
	float angle = (M_PI * orient)/180;
	cv::Point point1,point2;
	point1.x = center.x - length * cos(angle);
	point1.y = center.y + length * sin(angle);
	point2.x = center.x - length * cos(angle + M_PI);
	point2.y = center.y + length * sin(angle + M_PI);

	cv::Size imgSize(im.cols,im.rows);
	cv::clipLine(imgSize,point1,point2);

	cv::Mat tmpImage1(im.clone());
	cv::Mat tmpImage(tmpImage1.clone());
	cv::line(tmpImage,point1,point2,color,2,8,0);

	cv::Point2f point3,point4,point5;
	point3.x = center.x - length * 4/5 * cos(angle + M_PI);
	point3.y = center.y + length * 4/5 * sin(angle + M_PI);
	point4.x = point3.x - 7 * cos(M_PI/2.0 + angle);
	point4.y = point3.y + 7 * sin(M_PI/2.0 + angle);
	point5.x = point3.x - 7 * cos(M_PI/2.0 + angle  + M_PI);
	point5.y = point3.y + 7 * sin(M_PI/2.0 + angle  + M_PI);

	cv::Point *pts = new cv::Point[4];
	pts[0] = point4;
	pts[1] = point2;
	pts[2] = point5;
	cv::fillConvexPoly(tmpImage,pts,3,cv::Scalar(255,50,0),8,0);
	delete [] pts;
	cv::circle(tmpImage,center,1,cv::Scalar(255,50,0),1,8,0);
	return tmpImage;
	tmpImage1.release();
}
//==============================================================================
/** Draws the "menu" of possible poses for the current position.
 */
void AnnotationsHandle::showMenu(const cv::Point2f &center){
	int pose0 = 0;
	int pose1 = 0;
	int pose2 = 0;
	int pose3 = 0;
	int pose4 = 90;
	cvNamedWindow("Latitude",CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Poses",CV_WINDOW_AUTOSIZE);

	IplImage *tmpImg = cvCreateImage(cv::Size(400,1),8,1);
	POSE sit = SITTING,stand = STANDING,bend = BENDING,longi = LONGITUDE,
		lat = LATITUDE;
	for(POSE p=SITTING;p<=LATITUDE;++p){
		switch(p){
			case SITTING:
				if(withPoses_){
					cv::createTrackbar("Sitting","Poses",&pose0,1,\
						AnnotationsHandle::trackbar_callback,&sit);
				}
				break;
			case STANDING:
				if(withPoses_){
					cv::createTrackbar("Standing","Poses",&pose1,1,\
						AnnotationsHandle::trackbar_callback,&stand);
				}
				break;
			case BENDING:
				if(withPoses_){
					cv::createTrackbar("Bending","Poses",&pose2,1,\
						AnnotationsHandle::trackbar_callback,&bend);
				}
				break;
			case LONGITUDE:
				cv::createTrackbar("Longitude","Poses",&pose3,360,\
					AnnotationsHandle::trackbar_callback,&longi);
				break;
			case LATITUDE:
				cv::createTrackbar("Latitude","Poses",&pose4,180,\
					AnnotationsHandle::trackbar_callback,&lat);
				break;
			default:
				//do nothing
				break;
		}
	}
	cvShowImage("Poses",tmpImg);

	cout<<"Press 'c' once the annotation for poses is done."<<endl;
	while(choice_ != 'c' && choice_ != 'C'){
		choice_ = (char)(cv::waitKey(0));
	}
	cvReleaseImage(&tmpImg);
	cvDestroyWindow("Poses");
	cvDestroyWindow("Latitude");
}
//==============================================================================
/** A function that starts a new thread which handles the track-bar event.
 */
void AnnotationsHandle::trackBarHandleFct(int position,void *param){
	unsigned int *ii                       = (unsigned int *)(param);
	AnnotationsHandle::ANNOTATION lastAnno = annotations_.back();
	annotations_.pop_back();
	if(lastAnno.poses_.empty()){
		lastAnno.poses_.assign(poseSize_,0);
		lastAnno.poses_[(int)LATITUDE] = 90;
	}

	// DRAW THE ORIENTATION TO SEE IT
	std::vector<cv::Point2f> points;
	Helpers::genTemplate2(lastAnno.location_,Helpers::persHeight(),\
		Helpers::camHeight(),points);
	cv::Point2f headCenter((points[12].x+points[14].x)/2,\
		(points[12].y+points[14].y)/2);
	if((POSE)(*ii)==LONGITUDE){
		AnnotationsHandle::drawOrientation(headCenter,position,image_);
	}else if((POSE)(*ii)==LATITUDE){
		AnnotationsHandle::drawLatitude(headCenter,lastAnno.location_,position,(POSE)(*ii));
	}

	// FOR ALL CASES STORE THE POSITION
	try{
		lastAnno.poses_.at(*ii) = position;
	}catch (std::exception &e){
		cout<<"Exception "<<e.what()<<endl;
		exit(1);
	}
	annotations_.push_back(lastAnno);
	if((POSE)(*ii) == SITTING){
		int oppPos = cv::getTrackbarPos("Standing","Poses");
		if(oppPos == position){
			cv::setTrackbarPos("Standing","Poses",(1-oppPos));
		}
	}else if((POSE)(*ii) == STANDING){
		int oppPos = cv::getTrackbarPos("Sitting","Poses");
		if(oppPos == position){
			cv::setTrackbarPos("Sitting","Poses",(1-oppPos));
		}
	}
}
//==============================================================================
/** The "on change" handler for the track-bars.
 */
void AnnotationsHandle::trackbar_callback(int position,void *param){
	AnnotationsHandle::trackBarHandleFct(position,param);
}
//==============================================================================
/** Plots the hull indicated by the parameter \c hull on the given image.
 */
void AnnotationsHandle::plotHull(IplImage *img,std::vector<cv::Point2f> &hull){
	hull.push_back(hull.front());
	for(unsigned i=1;i<hull.size();++i){
		cvLine(img,hull[i-1],hull[i],CV_RGB(255,0,0),1);
	}
}
//==============================================================================
/** Starts the annotation process for the images.
 */
int AnnotationsHandle::runAnn(int argc,char **argv,unsigned step,const std::string \
&usedImages,int imgIndex){
	AnnotationsHandle::init();
	if(imgIndex!= -1){
		imgIndex += step;
	}
	choice_ = 'c';
	if(argc != 5){
		cerr<<"usage: ./annotatepos <img_list.txt> <calib.xml> <annotation.txt>\n"<< \
		"<img_directory>   => name of directory containing the images\n"<< \
		"<calib.xml>       => the file contains the calibration data of the camera\n"<< \
		"<prior.txt>       => the file containing the location prior\n"<< \
		"<annotations.txt> => the file in which the annotation data needs to be stored\n"<<endl;
		exit(1);
	} else {
		cout<<"Help info:\n"<< \
		"> press 'q' to quite before all the images are annotated;\n"<< \
		"> press 'n' to skip to the next image without saving the current one;\n"<< \
		"> press 's' to save the annotations for the current image and go to the next one;\n"<<endl;
	}
	unsigned index = 0;
	cerr<<"Loading the images...."<< argv[1] << endl;
	std::deque<std::string> imgs = Helpers::readImages(argv[1],imgIndex);
	cerr<<"Loading the calibration...."<< argv[2] << endl;
	Helpers::loadCalibration(argv[2]);
	std::vector<cv::Point2f> priorHull;
	cerr<<"Loading the location prior...."<< argv[3] << endl;
	Helpers::loadPriorHull(argv[3],priorHull);

	std::cerr<<"LATITUDE: Only looking upwards or downwards matters!"<<std::endl;

	// set the handler of the mouse events to the method: <<mouseHandler>>
	image_ = std::tr1::shared_ptr<IplImage>(cvLoadImage(imgs[index].c_str()),\
		Helpers::releaseImage);
	AnnotationsHandle::plotHull(image_.get(),priorHull);
	std::cerr<<"Annotations for image: "<<imgs[index].substr\
		(imgs[index].rfind("/")+1)<<std::endl;
	cvNamedWindow("image");
	cvSetMouseCallback("image",AnnotationsHandle::mouseHandlerAnn,NULL);
	cvShowImage("image",image_.get());

	// used to write the output stream to a file given in <<argv[3]>>
	std::ofstream annoOut;
	annoOut.open(argv[4],std::ios::out | std::ios::app);
	if(!annoOut){
		errx(1,"Cannot open file %s",argv[4]);
	}
	annoOut.seekp(0,std::ios::end);

	/* while 'q' was not pressed,annotate images and store the info in
	 * the annotation file */
	int key = 0;

int extra = 37;

	while((char)key != 'q' && (char)key != 'Q' && index<imgs.size()) {
		std::cerr<<"Annotations for image: "<<imgs[index].substr\
			(imgs[index].rfind("/")+1)<<std::endl;
		key = cv::waitKey(0);
		/* if the pressed key is 's' stores the annotated positions
		 * for the current image */
		if((char)key == 's'){
//			annoOut<<imgs[index].substr(imgs[index].rfind("/")+1);
std::string tmpName = "japanese_00"+Auxiliary::int2string(index+extra)+".jpg";
annoOut<<tmpName;

			for(unsigned i=0;i!=annotations_.size();++i){
				annoOut <<" ("<<annotations_[i].location_.x<<","\
					<<annotations_[i].location_.y<<")|";
				for(unsigned j=0;j<annotations_[i].poses_.size();++j){
					annoOut<<"("<<poseNames_[j]<<":"<<annotations_[i].poses_[j]<<")";
					if(j<annotations_[i].poses_.size()-1){
						annoOut<<"|";
					}
				}
			}
			annoOut<<endl;
			cout<<"Annotations for image: "<<\
				imgs[index].substr(imgs[index].rfind("/")+1)\
				<<" were successfully saved!"<<endl;
			//clean the annotations_ and release the image
			for(unsigned ind=0;ind<annotations_.size();++ind){
				annotations_[ind].poses_.clear();
			}
			annotations_.clear();
			if(image_){image_.reset();}

			//move image to a different directory to keep track of annotated images
			std::string currLocation = imgs[index].substr(0,imgs[index].rfind("/"));
			std::string newLocation  = currLocation.substr(0,currLocation.rfind("/")) + \
							"/annotated"+usedImages+"/";
			if(!boost::filesystem::is_directory(newLocation)){
				boost::filesystem::create_directory(newLocation);
			}
//			newLocation += imgs[index].substr(imgs[index].rfind("/")+1);
newLocation += tmpName;

			cerr<<"NEW LOCATION >> "<<newLocation<<endl;
			cerr<<"CURR LOCATION >> "<<currLocation<<endl;
			if(rename(imgs[index].c_str(),newLocation.c_str())){
				perror(NULL);
			}

			// load the next image_ or break if it is the last one
			// skip to the next step^th image
			while(index+step>imgs.size() && step>0){
				step = step/10;
			}
			index += step;
			if(index>=imgs.size()){
				break;
			}

			image_ = std::tr1::shared_ptr<IplImage>(cvLoadImage(imgs[index].c_str()),\
				Helpers::releaseImage);
			AnnotationsHandle::plotHull(image_.get(),priorHull);
			cvShowImage("image",image_.get());
		}else if((char)key == 'n'){
			cout<<"Annotations for image: "<<\
				imgs[index].substr(imgs[index].rfind("/")+1)\
				<<" NOT saved!"<<endl;

			//clean the annotations and release the image
			for(unsigned ind=0;ind<annotations_.size();++ind){
				annotations_[ind].poses_.clear();
			}
			annotations_.clear();
			if(image_){image_.reset();}

			// skip to the next step^th image
			while(index+step>imgs.size() && step>0){
				step = step/10;
			}
			index += step;
			if(index>=imgs.size()){
				break;
			}
			image_ = std::tr1::shared_ptr<IplImage>(cvLoadImage(imgs[index].c_str()),\
				Helpers::releaseImage);
			AnnotationsHandle::plotHull(image_.get(),priorHull);
			cvShowImage("image",image_.get());
		}else if(isalpha(key)){
			cout<<"key pressed >>> "<<(char)key<<"["<<key<<"]"<<endl;
		}
	}
	annoOut.close();
	cout<<"Thank you for your time ;)!"<<endl;
	cvDestroyAllWindows();
	return 0;
}
//==============================================================================
/** Starts the annotation of the images on the artificial data (labels in the
 * image name).
 */
int AnnotationsHandle::runAnnArtificial(int argc,char **argv,unsigned step,\
const std::string &usedImages,int imgIndex,int imoffset,unsigned lati,int setoffset){
	AnnotationsHandle::init();
	if(imgIndex!= -1){
		imgIndex += step;
	}
	choice_ = 'c';
	if(argc != 5){
		cerr<<"usage: ./annotatepos <img_list.txt> <calib.xml> <annotation.txt>\n"<< \
		"<img_directory>   => name of directory containing the images\n"<< \
		"<calib.xml>       => the file contains the calibration data of the camera\n"<< \
		"<prior.txt>       => the file containing the location prior\n"<< \
		"<annotations.txt> => the file in which the annotation data needs to be stored\n"<<endl;
		exit(1);
	} else {
		cout<<"Help info:\n"<< \
		"> press 'q' to quite before all the images are annotated;\n"<< \
		"> press 'n' to skip to the next image without saving the current one;\n"<< \
		"> press 's' to save the annotations for the current image and go to the next one;\n"<<endl;
	}
	unsigned index = 0;
	cerr<<"Loading the images...."<< argv[1] << endl;
	std::deque<std::string> imgs = Helpers::readImages(argv[1],imgIndex);
	cerr<<"Loading the calibration...."<< argv[2] << endl;
	Helpers::loadCalibration(argv[2]);
	std::vector<cv::Point2f> priorHull;
	cerr<<"Loading the location prior...."<< argv[3] << endl;
	Helpers::loadPriorHull(argv[3],priorHull);

	std::cerr<<"LATITUDE: Only looking upwards or downwards matters!"<<std::endl;

	// set the handler of the mouse events to the method: <<mouseHandler>>
	image_ = std::tr1::shared_ptr<IplImage>(cvLoadImage(imgs[index].c_str()),\
		Helpers::releaseImage);
	AnnotationsHandle::plotHull(image_.get(),priorHull);
	std::cerr<<"Annotations for image: "<<imgs[index].substr\
		(imgs[index].rfind("/")+1)<<std::endl;
	cvNamedWindow("image");
	cvSetMouseCallback("image",AnnotationsHandle::mouseHandlerAnn,NULL);
	cvShowImage("image",image_.get());

	// used to write the output stream to a file given in <<argv[3]>>
	std::ofstream annoOut;
	annoOut.open(argv[4],std::ios::out | std::ios::app);
	if(!annoOut){
		errx(1,"Cannot open file %s",argv[4]);
	}
	annoOut.seekp(0,std::ios::end);

	/* while 'q' was not pressed,annotate images and store the info in
	 * the annotation file */
	int key = 0;
	while((char)key != 'q' && (char)key != 'Q' && index<imgs.size()) {
		std::string imageName = imgs[index].substr(imgs[index].rfind("/")+1);
		unsigned pos;int picIndx;
		Helpers::imageNumber(imageName,pos,picIndx);
		picIndx = ((picIndx-setoffset)*5)+270;
		while(picIndx>=360){
			picIndx-=360;
		}
		std::cerr<<"Annotations for image: "<<picIndx<<std::endl;

		key = cv::waitKey(0);
		/* if the pressed key is 's' stores the annotated positions
		 * for the current image */
		if((char)key == 's'){
			annoOut<<Auxiliary::int2string(index+imoffset)+"artificial.jpg";
			for(unsigned i=0;i!=annotations_.size();++i){
				annoOut <<" ("<<annotations_[i].location_.x<<","\
					<<annotations_[i].location_.y<<")|";
				annoOut<<"(SITTING:0)|(STANDING:0)|(BENDING:0)|(LONGITUDE:"<<\
					picIndx<<")|(LATITUDE:"<<lati<<")";
			}
			annoOut<<endl;
			cout<<"Annotations for image: "<<\
				imgs[index].substr(imgs[index].rfind("/")+1)\
				<<" were successfully saved!"<<endl;
			//clean the annotations and release the image
			for(unsigned ind=0;ind<annotations_.size();++ind){
				annotations_[ind].poses_.clear();
			}
			annotations_.clear();
			if(image_){image_.reset();}

			//move image to a different directory to keep track of annotated images
			std::string currLocation = imgs[index].substr(0,imgs[index].rfind("/"));
			std::string newLocation  = currLocation.substr(0,currLocation.rfind("/")) + \
							"/annotated"+usedImages+"/";
			if(!boost::filesystem::is_directory(newLocation)){
				boost::filesystem::create_directory(newLocation);
			}
			newLocation += Auxiliary::int2string(index+imoffset)+"artificial.jpg";
			cerr<<"NEW LOCATION >> "<<newLocation<<endl;
			cerr<<"CURR LOCATION >> "<<currLocation<<endl;
			if(rename(imgs[index].c_str(),newLocation.c_str())){
				perror(NULL);
			}

			// load the next image or break if it is the last one
			// skip to the next step^th image
			while(index+step>imgs.size() && step>0){
				step = step/10;
			}
			index += step;
			if(index>=imgs.size()){
				break;
			}

			image_ = std::tr1::shared_ptr<IplImage>(cvLoadImage(imgs[index].c_str()),\
				Helpers::releaseImage);
			AnnotationsHandle::plotHull(image_.get(),priorHull);
			cvShowImage("image",image_.get());
		}else if((char)key == 'n'){
			cout<<"Annotations for image: "<<\
				imgs[index].substr(imgs[index].rfind("/")+1)\
				<<" NOT saved!"<<endl;

			//clean the annotations and release the image
			for(unsigned ind=0;ind<annotations_.size();++ind){
				annotations_[ind].poses_.clear();
			}
			annotations_.clear();
			if(image_){image_.reset();}

			// skip to the next step^th image
			while(index+step>imgs.size() && step>0){
				step = step/10;
			}
			index += step;
			if(index>=imgs.size()){
				break;
			}
			image_ = std::tr1::shared_ptr<IplImage>(cvLoadImage(imgs[index].c_str()),\
				Helpers::releaseImage);
			AnnotationsHandle::plotHull(image_.get(),priorHull);
			cvShowImage("image",image_.get());
		}else if(isalpha(key)){
			cout<<"key pressed >>> "<<(char)key<<"["<<key<<"]"<<endl;
		}
	}
	annoOut.close();
	cout<<"Thank you for your time ;)!"<<endl;
	cvDestroyAllWindows();
	return 0;
}
//==============================================================================
/** Load annotations from file.
 */
void AnnotationsHandle::loadAnnotations(char* filename,\
std::deque<AnnotationsHandle::FULL_ANNOTATIONS> &loadedAnno){
	AnnotationsHandle::init();
	std::ifstream annoFile(filename);

	std::cout<<"Loading annotations of...."<<filename<<std::endl;
	if(annoFile.is_open()){
		AnnotationsHandle::FULL_ANNOTATIONS tmpFullAnno;
		while(annoFile.good()){
			std::string line;
			std::getline(annoFile,line);
			std::deque<std::string> lineVect = Helpers::splitLine(\
				const_cast<char*>(line.c_str()),' ');

			// IF IT IS NOT AN EMPTY FILE
			if(lineVect.size()>0){
				tmpFullAnno.imgFile_ = std::string(lineVect[0]);
				for(unsigned i=1;i<lineVect.size();++i){
					AnnotationsHandle::ANNOTATION tmpAnno;
					std::deque<std::string> subVect = Helpers::splitLine\
						(const_cast<char*>(lineVect[i].c_str()),'|');
					for(unsigned j=0;j<subVect.size();++j){
						std::string temp(subVect[j]);
						if(temp.find("(")!=std::string::npos){
							temp.erase(temp.find("("),1);
						}
						if(temp.find(")")!=std::string::npos){
							temp.erase(temp.find(")"),1);
						}
						subVect[j] = temp;
						if(j==0){
							// location is on the first position
							std::deque<std::string> locVect = Helpers::splitLine\
								(const_cast<char*>(subVect[j].c_str()),',');
							if(locVect.size()==2){
								char *pEndX,*pEndY;
								tmpAnno.location_.x = strtol(locVect[0].c_str(),&pEndX,10);
								tmpAnno.location_.y = strtol(locVect[1].c_str(),&pEndY,10);
							}
						}else{
							if(tmpAnno.poses_.empty()){
								tmpAnno.poses_ = std::deque<unsigned int>(poseSize_,0);
							}
							std::deque<std::string> poseVect = Helpers::splitLine\
								(const_cast<char*>(subVect[j].c_str()),':');
							if(poseVect.size()==2){
								char *pEndP;
								tmpAnno.poses_[(POSE)(j-1)] = \
									strtol(poseVect[1].c_str(),&pEndP,10);
							}
						}
					}
					tmpAnno.id_ = tmpFullAnno.annos_.size();
					tmpFullAnno.annos_.push_back(tmpAnno);
					tmpAnno.poses_.clear();
				}
				loadedAnno.push_back(tmpFullAnno);
				tmpFullAnno.annos_.clear();
			}
			line.clear();
		}
		annoFile.close();
	}
}
//==============================================================================
/** Writes a given FULL_ANNOTATIONS structure into a given file.
 */
void AnnotationsHandle::writeAnnoToFile(\
const std::deque<AnnotationsHandle::FULL_ANNOTATIONS> &fullAnno,\
const std::string &fileName){
	// OPEN THE FILE TO WRITE ANNOTATIONS
	std::ofstream annoOut;
	annoOut.open(fileName.c_str(),std::ios::out | std::ios::app);
	if(!annoOut){
		errx(1,"Cannot open file %s",fileName.c_str());
	}
	annoOut.seekp(0,std::ios::end);

	for(std::size_t k=0;k<fullAnno.size();++k){
		// WRITE THE IMAGE NAME
		annoOut<<fullAnno[k].imgFile_<<" ";

		// FOR EACH ANNOTATION IN THE ANNOTATIONS ARRAY
		for(std::size_t i=0;i<fullAnno[k].annos_.size();++i){

			// WRITE THE LOCATION OF THE DETECTED PERSON
			annoOut<<"("<<fullAnno[k].annos_[i].location_.x<<","<<\
				fullAnno[k].annos_[i].location_.y<<")|";
			for(std::size_t j=0;j<fullAnno[k].annos_[i].poses_.size();++j){
				annoOut<<"("<<(POSE)(j)<<":"<<fullAnno[k].annos_[i].poses_[j]<<")|";
			}
		}
		annoOut<<endl;
		cout<<"Annotations for image: "<<fullAnno[k].imgFile_<<\
			" were successfully saved!"<<endl;
	}
	annoOut.close();
}
//==============================================================================
/** Checks to see if a location can be assigned to a specific ID given the
 * new distance.
 */
bool AnnotationsHandle::canBeAssigned(std::deque<AnnotationsHandle::ASSIGNED>\
&idAssignedTo,short int id,float newDist,short int to){
	bool isHere = false;
	for(unsigned int i=0;i<idAssignedTo.size();++i){
		if(idAssignedTo[i].id_ == id){
			isHere = true;
			if(idAssignedTo[i].dist_>newDist){
				idAssignedTo[i].to_   = to;
				idAssignedTo[i].dist_ = newDist;
				return true;
			}
		}
	}
	if(!isHere){
		bool alreadyTo = false;
		for(unsigned i=0;i<idAssignedTo.size();++i){
			if(idAssignedTo[i].to_ == to){
				alreadyTo = true;
				if(idAssignedTo[i].dist_>newDist){
					//assigned the id to this one and un-assign the old one
					AnnotationsHandle::ASSIGNED temp;
					temp.id_   = id;
					temp.to_   = to;
					temp.dist_ = newDist;
					idAssignedTo.push_back(temp);
					idAssignedTo.erase(idAssignedTo.begin()+i);
					return true;
				}
			}
		}
		if(!alreadyTo){
			AnnotationsHandle::ASSIGNED temp;
			temp.id_   = id;
			temp.to_   = to;
			temp.dist_ = newDist;
			idAssignedTo.push_back(temp);
			return true;
		}
	}
	return false;
}
//==============================================================================
/** Correlate annotations' from locations in \c annoOld to locations in \c
 * annoNew through IDs.
 */
void AnnotationsHandle::correltateLocs(std::deque<AnnotationsHandle::ANNOTATION>\
&annoOld,std::deque<AnnotationsHandle::ANNOTATION> &annoNew,\
std::deque<AnnotationsHandle::ASSIGNED> &idAssignedTo){
	std::deque< std::deque<float> > distMatrix;

	//1. compute the distances between all annotations
	for(unsigned k=0;k<annoNew.size();++k){
		distMatrix.push_back(std::deque<float>());
		for(unsigned l=0;l<annoOld.size();++l){
			distMatrix[k].push_back(0.0);
			distMatrix[k][l] = Helpers::dist(annoNew[k].location_,annoOld[l].location_);
		}
	}

	//2. assign annotations between the 2 groups of annotations
	bool canAssign = true;
	while(canAssign){
		canAssign = false;
		for(unsigned k=0;k<distMatrix.size();++k){ //for each row in new
			float minDist = (float)INFINITY;
			for(unsigned l=0;l<distMatrix[k].size();++l){ // loop over all old
				if(distMatrix[k][l]<minDist && AnnotationsHandle::canBeAssigned\
				(idAssignedTo,annoOld[l].id_,distMatrix[k][l],annoNew[k].id_)){
					minDist   =	distMatrix[k][l];
					canAssign = true;
				}
			}
		}
	}

	//3. update the ids to the new one to correspond to the ids of the old one!!
	for(unsigned i=0;i<annoNew.size();++i){
		for(unsigned n=0;n<idAssignedTo.size();++n){
			if(annoNew[i].id_ == idAssignedTo[n].to_){
				if(idAssignedTo[n].to_ != idAssignedTo[n].id_){
					for(unsigned j=0;j<annoNew.size();++j){
						if(annoNew[j].id_ == idAssignedTo[n].id_){
							annoNew[j].id_ = -1;
						}
					}
				}
				annoNew[i].id_ = idAssignedTo[n].id_;
				break;
			}
		}
	}
}
//==============================================================================
/** Computes the average distance from the predicted location and the annotated
 * one,the number of unpredicted people in each image and the differences in the
 * pose estimation.
 */
void AnnotationsHandle::annoDifferences(std::deque<AnnotationsHandle::FULL_ANNOTATIONS>\
&train,std::deque<AnnotationsHandle::FULL_ANNOTATIONS> &test,float &avgDist,\
float &Ndiff,float ssdLongDiff,float ssdLatDiff,float poseDiff){
	if(train.size() != test.size()) {
		std::cerr<<"Training annotations and test annotations have different sizes";
		exit(1);
	}
	for(unsigned i=0;i<train.size();++i){
		if(train[i].imgFile_ != test[i].imgFile_) {
			errx(1,"Images on positions %i do not correspond",i);
		}

		/* the images might not be in the same ordered so they need to be assigned
		 * to the closest one to them */
		std::deque<AnnotationsHandle::ASSIGNED> idAssignedTo;
		//0. if one of them is 0 then no correlation can be done
		if(train[i].annos_.size()!=0 && test[i].annos_.size()!=0){
			AnnotationsHandle::correltateLocs(train[i].annos_,test[i].annos_,idAssignedTo);
		}
		//1. update average difference for the current image
		for(unsigned l=0;l<idAssignedTo.size();++l){
			cout<<idAssignedTo[l].id_<<" assigned to "<<idAssignedTo[l].to_<<endl;
			avgDist += idAssignedTo[l].dist_;
		}
		if(idAssignedTo.size()>0){
			avgDist /= idAssignedTo.size();
		}

		//2. update the difference in number of people detected
		if(train[i].annos_.size()!=test[i].annos_.size()){
			Ndiff += abs((float)train[i].annos_.size() - \
						(float)test[i].annos_.size());
		}

		//3. update the poses estimation differences
		unsigned int noCorresp = 0;
		for(unsigned int l=0;l<train[i].annos_.size();++l){
			for(unsigned int k=0;k<test[i].annos_.size();++k){
				if(train[i].annos_[l].id_ == test[i].annos_[k].id_){
					// FOR POSES COMPUTE THE DIFFERENCES
					if(withPoses_){
						for(unsigned int m=0;m<train[i].annos_[l].poses_.size()-1;++m){
							poseDiff += abs((float)train[i].annos_[l].poses_[m] - \
								(float)test[i].annos_[k].poses_[m])/2.0;
						}
					}

					// SSD between the predicted values and the correct ones
					int longPos = (int)LONGITUDE,latPos = (int)LATITUDE;
					float angleTrain = ((float)train[i].annos_[l].poses_[longPos]*M_PI/180.0);
					float angleTest  = ((float)test[i].annos_[k].poses_[longPos]*M_PI/180.0);
					ssdLongDiff += pow(cos(angleTrain)-cos(angleTest),2) +\
									pow(sin(angleTrain)-sin(angleTest),2);
					angleTrain = ((float)train[i].annos_[l].poses_[latPos]*M_PI/180.0);
					angleTest  = ((float)test[i].annos_[k].poses_[latPos]*M_PI/180.0);
					ssdLatDiff += pow(cos(angleTrain)-cos(angleTest),2) +\
									pow(sin(angleTrain)-sin(angleTest),2);
				}
			}
		}
		cout<<endl;
	}
	cout<<"avgDist: "<<avgDist<<endl;
	cout<<"Ndiff: "<<Ndiff<<endl;
	cout<<"PoseDiff: "<<poseDiff<<endl;
	cout<<"ssdLongDiff: "<<ssdLongDiff<<endl;
	cout<<"ssdLatDiff: "<<ssdLatDiff<<endl;

	ssdLongDiff /= train.size();
	ssdLatDiff /= train.size();
	avgDist /= train.size();
}
//==============================================================================
/** Displays the complete annotations for all images.
 */
void AnnotationsHandle::displayFullAnns(std::deque<AnnotationsHandle::FULL_ANNOTATIONS>\
&fullAnns){
	for(unsigned int i=0;i<fullAnns.size();++i){
		cout<<"Image name: "<<fullAnns[i].imgFile_<<endl;
		for(unsigned int j=0;j<fullAnns[i].annos_.size();++j){
			cout<<"Annotation Id:"<<fullAnns[i].annos_[j].id_<<\
				"_____________________________________________"<<endl;
			cout<<"Location: ["<<fullAnns[i].annos_[j].location_.x<<","\
				<<fullAnns[i].annos_[j].location_.y<<"]"<<endl;
			for(unsigned l=0;l<poseSize_;++l){
				cout<<"("<<poseNames_[l]<<": "<<fullAnns[i].annos_[j].poses_[l]<<")";
			}
			std::cout<<std::endl;
		}
		cout<<endl;
	}
}
//==============================================================================
/** Evaluates the annotation of the images.
 */
int AnnotationsHandle::runEvaluation(int argc,char **argv){
	AnnotationsHandle::init();
	if(argc != 3){
		cerr<<"usage: cmmd <train_annotations.txt> <train_annotations.txt>\n"<< \
		"<train_annotations.txt> => file containing correct annotations\n"<< \
		"<test_annotations.txt>  => file containing predicted annotations\n"<<endl;
		exit(-1);
	}

	std::deque<AnnotationsHandle::FULL_ANNOTATIONS> allAnnoTrain;
	std::deque<AnnotationsHandle::FULL_ANNOTATIONS> allAnnoTest;
	AnnotationsHandle::loadAnnotations(argv[1],allAnnoTrain);
	AnnotationsHandle::loadAnnotations(argv[2],allAnnoTest);

	AnnotationsHandle::displayFullAnns(allAnnoTrain);

	float avgDist = 0,Ndiff = 0,ssdLatDiff = 0,ssdLongDiff = 0,poseDiff = 0;
	AnnotationsHandle::annoDifferences(allAnnoTrain,allAnnoTest,avgDist,Ndiff,\
		ssdLongDiff,ssdLatDiff,poseDiff);
}
//==============================================================================
/** Check calibration: shows how the projection grows depending on the location
 * of the point.
 */
void AnnotationsHandle::checkCalibration(int argc,char **argv){
	AnnotationsHandle::init();
	if(argc != 3){
		std::cerr<<"run <image_file> <calibration_file>\n"<<std::endl;
		std::abort();
	}

	std::cout<<"Loading the image...."<<argv[1]<<std::endl;
	std::cout<<"Loading the calibration...."<<argv[2]<<std::endl;
	Helpers::loadCalibration(argv[2]);

	// FIND OUT WHERE THE CAMERA LOCATION GETS PROJECTS
	cv::Point2f cam = (*Helpers::proj())(cv::Point3f(Helpers::camPosX(),\
		Helpers::camPosY(),0));
	std::cout<<"Camera position: "<<cam<<std::endl;

	float ratio,step;
	if(cam.x>0 && cam.x<Helpers::width() && cam.y>0 && cam.y<Helpers::height()){
		ratio = 0.30;
		step  = (ratio-0.01)/3.0;
	}else{
		ratio = 0.40;
		step  = (ratio-0.01)/3.0;
	}
	float dimension = std::sqrt(Helpers::width()*Helpers::width()+Helpers::height()*\
		Helpers::height())/2;
	for(float in=0.0;in<ratio;in+=step){
		image_ = std::tr1::shared_ptr<IplImage>(cvLoadImage(argv[1]),Helpers::releaseImage);
		cv::Point2f pt(cam.x,in*dimension+cam.y);
		std::vector<cv::Point2f> points;
		Helpers::genTemplate2(pt,Helpers::persHeight(),Helpers::camHeight(),points);
		for(int i =0;i<points.size();++i){
			points[i].x += float(Helpers::width()/2.0-pt.x);
			points[i].y += float(Helpers::height()/2.0-pt.y);
		}
		Helpers::plotTemplate2(image_.get(),pt,Helpers::persHeight(),\
			Helpers::camHeight(),cv::Scalar(255,0,0),points);
		cvShowImage((std::string("img_")+Auxiliary::int2string\
			(in*Helpers::width())).c_str(),image_.get());
		cvWaitKey(0);
		if(image_){image_.reset();}
	}
}
//==============================================================================
char AnnotationsHandle::choice_ = ' ';
std::deque<std::string> AnnotationsHandle::poseNames_;
bool AnnotationsHandle::withPoses_ = false;
unsigned AnnotationsHandle::poseSize_ = 5;
boost::mutex AnnotationsHandle::trackbarMutex_;
std::tr1::shared_ptr<IplImage> AnnotationsHandle::image_;
std::deque<AnnotationsHandle::ANNOTATION> AnnotationsHandle::annotations_;
//==============================================================================
/*
int main(int argc,char **argv){
	std::string folderSuffix = "_train";
	AnnotationsHandle::runAnn(argc,argv,1,folderSuffix,-1);
//	AnnotationsHandle::runAnnArtificial(argc,argv,1,folderSuffix,-1,653,60,146);
	//AnnotationsHandle::runEvaluation(argc,argv);
}
*/

