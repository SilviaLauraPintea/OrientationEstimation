/* annotationsHandle.cpp
 * Author: Silvia-Laura Pintea
 */
#include "annotationsHandle.h"
//==============================================================================
/** Define a post-fix increment operator for the enum \c POSE.
 */
annotationsHandle::POSE operator++(annotationsHandle::POSE &refPose, int){
	annotationsHandle::POSE oldPose = refPose;
	refPose                         = (annotationsHandle::POSE)(refPose + 1);
	return oldPose;
}
//==============================================================================
/** Mouse handler for annotating people's positions and poses.
 */
void annotationsHandle::mouseHandlerAnn(int event, int x, int y, int flags, void *param){
	cv::Point pt = cv::Point(x,y);
	y            = pt.y;
	x            = pt.x;
	static bool down = false;
	switch (event){
		case CV_EVENT_LBUTTONDOWN:
			if(choice == 'c'){
				down = true;
				cout<<"Left button down at >>> ("<<x<<","<<y<<")"<<endl;
				Annotate::plotAreaTmp(image,(float)x,(float)y);
			}
			break;
		case CV_EVENT_MOUSEMOVE:
			if(down){
				Annotate::plotAreaTmp(image,(float)x,(float)y);
			}
			break;
		case CV_EVENT_LBUTTONUP:
			if(choice == 'c'){
				cout<<"Left button up at >>> ("<<x<<","<<y<<")"<<endl;
				choice = ' ';
				down = false;
				annotationsHandle::ANNOTATION temp;
				temp.location = pt;
				temp.id       = annotations.size();
				temp.poses.assign(4, 0);
				annotations.push_back(temp);
				for(unsigned i=0;i!=annotations.size(); ++i){
					Annotate::plotArea(image, (float)annotations[i].location.x, \
						(float)annotations[i].location.y);
				}
				showMenu(pt);
			}
			break;
	}
}
//==============================================================================
/** Shows how the selected orientation looks on the image.
 */
void annotationsHandle::drawOrientation(cv::Point center, unsigned int orient){
	unsigned int length = 60;
	double angle = (M_PI * orient)/180;
	cv::Point point1;
	point1.x = center.x - length * cos(angle);
	point1.y = center.y + length * sin(angle);

	cv::Point point2;
	point2.x = center.x - length * cos(angle + M_PI);
	point2.y = center.y + length * sin(angle + M_PI);

	cv::Size imgSize(image->width,image->height);
	cv::clipLine(imgSize,point1,point2);

	IplImage *img = cvCloneImage(image);
	cv::Mat tmpImage(img);
	cv::line(tmpImage,point1,point2,cv::Scalar(100,225,0),2,8,0);

	cv::Point point3;
	point3.x = center.x - length * 4/5 * cos(angle + M_PI);
	point3.y = center.y + length * 4/5 * sin(angle + M_PI);

	cv::Point point4;
	point4.x = point3.x - 5 * cos(M_PI/2.0 + angle);
	point4.y = point3.y + 5 * sin(M_PI/2.0 + angle);

	cv::Point point5;
	point5.x = point3.x - 5 * cos(M_PI/2.0 + angle  + M_PI);
	point5.y = point3.y + 5 * sin(M_PI/2.0 + angle  + M_PI);

	cv::Point *pts = new cv::Point[4];
	pts[0] = point4;
	pts[1] = point2;
	pts[2] = point5;
	cv::fillConvexPoly(tmpImage,pts,3,cv::Scalar(255,50,0),8,0);
	delete [] pts;

	cv::circle(tmpImage,center,1,cv::Scalar(255,50,0),1,8,0);
	cv::imshow("image", tmpImage);
	tmpImage.release();
	cvReleaseImage(&img);
}
//==============================================================================
/** Draws the "menu" of possible poses for the current position.
 */
void annotationsHandle::showMenu(cv::Point center){
	int pose0 = 0;
	int pose1 = 0;
	int pose2 = 0;
	int pose3 = 0;
	cv::namedWindow("Poses",CV_WINDOW_AUTOSIZE);
	IplImage *tmpImg = cvCreateImage(cv::Size(300,1),8,1);
	cv::imshow("Poses", tmpImg);

	for(POSE p=SITTING; p<=ORIENTATION;p++){
		unsigned int *pt;
		unsigned int ui_p = (unsigned int)p;
		pt                = &ui_p;
		void *param       = pt;
		switch(p){
			case SITTING:
				cv::createTrackbar("Sitting","Poses", &pose0, 1, \
					trackbar_callback, param);
				break;
			case STANDING:
				cv::createTrackbar("Standing","Poses", &pose1, 1, \
					trackbar_callback, param);
				cv::setTrackbarPos("Standing", "Poses", 1);
				break;
			case BENDING:
				cv::createTrackbar("Bending","Poses", &pose2, 1, \
					trackbar_callback, param);
				break;
			case ORIENTATION:
				cv::createTrackbar("Orientation", "Poses", &pose3, 360, \
					trackbar_callback, param);
				break;
			default:
				//do nothing
				break;
		}
	}
	cout<<"Press 'c' once the annotation for poses is done."<<endl;
	while(choice != 'c' && choice != 'C'){
		choice = (char)(cv::waitKey(0));
	}
	cvReleaseImage(&tmpImg);
	cvDestroyWindow("Poses");
}
//==============================================================================
/** A function that starts a new thread which handles the track-bar event.
 */
void annotationsHandle::trackBarHandleFct(int position,void *param){
	//trackbarMutex.lock();
	unsigned int *ii                       = (unsigned int *)(param);
	annotationsHandle::ANNOTATION lastAnno = annotations.back();
	annotations.pop_back();
	if(lastAnno.poses.size()==0){
		lastAnno.poses.assign(4,0);
	}

	//draw the orientation to see it
	if((POSE)(*ii)==ORIENTATION){
		drawOrientation(lastAnno.location, position);
	}

	try{
		if((POSE)(*ii) == ORIENTATION){
			if(position % 10 != 0){
				position = (int)(position / 10) * 10;
			}
		}
		lastAnno.poses.at(*ii) = position;
	}catch (std::exception &e){
		cout<<"Exception "<<e.what()<<endl;
		exit(1);
	}
	annotations.push_back(lastAnno);
	//trackbarMutex.unlock();

	if((POSE)(*ii) == SITTING){
		cv::setTrackbarPos("Standing", "Poses", (1-position));
	} else if((POSE)(*ii) == STANDING){
		cv::setTrackbarPos("Sitting", "Poses", (1-position));
	}
}
//==============================================================================
/** The "on change" handler for the track-bars.
 */
void annotationsHandle::trackbar_callback(int position,void *param){
	/*boost::thread *trackbarHandle;
	trackbarHandle = new boost::thread(&annotationsHandle::trackBarHandleFct,\
		position, param);
	trackbarHandle->join();
	delete trackbarHandle;*/
	trackBarHandleFct(position, param);
}
//==============================================================================
/** Plots the hull indicated by the parameter \c hull on the given image.
 */
void annotationsHandle::plotHull(IplImage *img, std::vector<CvPoint> &hull){
	hull.push_back(hull.front());
	for(unsigned i=1; i<hull.size(); ++i){
		cvLine(img, hull[i-1],hull[i],CV_RGB(255,0,0),2);
	}
}
//==============================================================================
/** Starts the annotation of the images. The parameters that need to be indicated
 * are:
 *
 * \li argv[1] -- name of directory containing the images
 * \li argv[2] -- the file contains the calibration data of the camera
 * \li argv[3] -- the file in which the annotation data needs to be stored
 */
int annotationsHandle::runAnn(int argc, char **argv){
	choice = 'c';
	if(argc != 5){
		cerr<<"usage: ./annotatepos <img_list.txt> <calib.xml> <annotation.txt>\n"<< \
		"<img_directory>   => name of directory containing the images\n"<< \
		"<calib.xml>       => the file contains the calibration data of the camera\n"<< \
		"<prior.txt>       => the file containing the location prior\n"<< \
		"<annotations.txt> => the file in which the annotation data needs to be stored\n"<<endl;
		exit(-1);
	} else {
		cout<<"Help info:\n"<< \
		"> press 'q' to quite before all the images are annotated;\n"<< \
		"> press 'n' to skip to the next image without saving the current one;\n"<< \
		"> press 's' to save the annotations for the current image and go to the next one;\n"<<endl;
	}
	unsigned index                = 0;
	std::vector<std::string> imgs = readImages(argv[1]);
	loadCalibration(argv[2]);

	// load the priors
	std::vector<CvPoint> priorHull;
	cerr<<"Loading the location prior...."<< argv[3] << endl;
	loadPriorHull(argv[3], priorHull);

	// set the handler of the mouse events to the method: <<mouseHandler>>
	image = cvLoadImage(imgs[index].c_str());
	plotHull(image, priorHull);
	cv::namedWindow("image");
	cvSetMouseCallback("image", mouseHandlerAnn, NULL);
	cv::imshow("image", image);

	// used to write the output stream to a file given in <<argv[3]>>
	ofstream annoOut;
	annoOut.open(argv[4], ios::out | ios::app);
	if(!annoOut){
		errx(1,"Cannot open file %s", argv[4]);
	}
	annoOut.seekp(0, ios::end);

	/* while 'q' was not pressed, annotate images and store the info in
	 * the annotation file */
	int key = 0;
	while((char)key != 'q' && (char)key != 'Q' && index<imgs.size()) {
		key = cv::waitKey(0);
		/* if the pressed key is 's' stores the annotated positions
		 * for the current image */
		if((char)key == 's'){
			annoOut<<imgs[index].substr(imgs[index].rfind("/")+1)<<" ";
			for(unsigned i=0; i!=annotations.size();++i){
				annoOut <<"("<<annotations[i].location.x<<","\
					<<annotations[i].location.y<<")|";
				for(unsigned j=0;j<annotations[i].poses.size();j++){
					switch((POSE)j){
						case SITTING:
							annoOut<<"(SITTING:"<<annotations[i].poses[j]<<")|";
							break;
						case STANDING:
							annoOut<<"(STANDING:"<<annotations[i].poses[j]<<")|";
							break;
						case BENDING:
							annoOut<<"(BENDING:"<<annotations[i].poses[j]<<")|";
							break;
						case ORIENTATION:
							annoOut<<"(ORIENTATION:"<<annotations[i].poses[j]<<") ";
							break;
						default:
							cout<<"Unknown pose ;)";
							break;
					}
				}
			}
			annoOut<<endl;
			cout<<"Annotations for image: "<<\
				imgs[index].substr(imgs[index].rfind("/")+1)\
				<<" were successfully saved!"<<endl;
			//clean the annotations and release the image
			for(unsigned ind=0; ind<annotations.size(); ind++){
				annotations[ind].poses.clear();
			}
			annotations.clear();
			cvReleaseImage(&image);

			//move image to a different directory to keep track of annotated images
			string currLocation = imgs[index].substr(0,imgs[index].rfind("/"));
			string newLocation  = currLocation.substr(0,currLocation.rfind("/")) + \
				"/annotated_images/";
			if(!boost::filesystem::is_directory(newLocation)){
				boost::filesystem::create_directory(newLocation);
			}
			newLocation += imgs[index].substr(imgs[index].rfind("/")+1);
			cerr<<"NEW LOCATION >> "<<newLocation<<endl;
			cerr<<"CURR LOCATION >> "<<currLocation<<endl;
			if(rename(imgs[index].c_str(), newLocation.c_str())){
				perror(NULL);
			}

			// load the next image or break if it is the last one
			index++;
			if(index==imgs.size()){
				break;
			}
			image = cvLoadImage(imgs[index].c_str());
			plotHull(image, priorHull);
			cv::imshow("image", image);
		}else if((char)key == 'n'){
			cout<<"Annotations for image: "<<\
				imgs[index].substr(imgs[index].rfind("/")+1)\
				<<" NOT saved!"<<endl;

			//clean the annotations and release the image
			for(unsigned ind=0; ind<annotations.size(); ind++){
				annotations[ind].poses.clear();
			}
			annotations.clear();
			cvReleaseImage(&image);

			// skip to the next image
			index++;
			if(index==imgs.size()){
				break;
			}
			image = cvLoadImage(imgs[index].c_str());
			plotHull(image, priorHull);
			cv::imshow("image", image);
		}else if(isalpha(key)){
			cout<<"key pressed >>> "<<(char)key<<"["<<key<<"]"<<endl;
		}
	}
	annoOut.close();
	cout<<"Thank you for your time ;)!"<<endl;
	cvReleaseImage(&image);
	cvDestroyWindow("image");
	return 0;
}
//==============================================================================
/** Load annotations from file.
 */
void annotationsHandle::loadAnnotations(char* filename, \
std::vector<annotationsHandle::FULL_ANNOTATIONS> &loadedAnno){
	ifstream annoFile(filename);

	std::cout<<"Loading annotations of...."<<filename<<std::endl;
	if(annoFile.is_open()){
		annotationsHandle::FULL_ANNOTATIONS tmpFullAnno;
		while(annoFile.good()){
			char *line = new char[1024];
			annoFile.getline(line,sizeof(char*)*1024);
			std::vector<std::string> lineVect = splitLine(line,' ');

			// IF IT IS NOT AN EMPTY FILE
			if(lineVect.size()>0){
				tmpFullAnno.imgFile = std::string(lineVect[0]);
				for(unsigned i=1; i<lineVect.size();i++){
					annotationsHandle::ANNOTATION tmpAnno;
					std::vector<std::string> subVect = \
						splitLine(const_cast<char*>(lineVect[i].c_str()),'|');
					for(unsigned j=0; j<subVect.size();j++){
						std::string temp(subVect[j]);
						if(temp.find("(")!=string::npos){
							temp.erase(temp.find("("),1);
						}
						if(temp.find(")")!=string::npos){
							temp.erase(temp.find(")"),1);
						}
						subVect[j] = temp;
						if(j==0){
							// location is on the first position
							std::vector<std::string> locVect = \
								splitLine(const_cast<char*>(subVect[j].c_str()),',');
							if(locVect.size()==2){
								char *pEndX, *pEndY;
								tmpAnno.location.x = strtol(locVect[0].c_str(),&pEndX,10);
								tmpAnno.location.y = strtol(locVect[1].c_str(),&pEndY,10);
							}
						}else{
							if(tmpAnno.poses.empty()){
								tmpAnno.poses = std::vector<unsigned int>(4,0);
							}
							std::vector<std::string> poseVect = \
								splitLine(const_cast<char*>(subVect[j].c_str()),':');
							if(poseVect.size()==2){
								char *pEndP;
								tmpAnno.poses[(POSE)(j-1)] = \
									strtol(poseVect[1].c_str(),&pEndP,10);
							}
						}
					}
					tmpAnno.id = tmpFullAnno.annos.size();
					tmpFullAnno.annos.push_back(tmpAnno);
					tmpAnno.poses.clear();
				}
				loadedAnno.push_back(tmpFullAnno);
				tmpFullAnno.annos.clear();
			}
			delete [] line;
		}
		annoFile.close();
	}
}
//==============================================================================
/** Checks to see if a location can be assigned to a specific ID given the
 * new distance.
 */
bool annotationsHandle::canBeAssigned(std::vector<annotationsHandle::ASSIGNED>\
&idAssignedTo, short int id, double newDist, short int to){
	bool isHere = false;
	for(unsigned int i=0; i<idAssignedTo.size(); i++){
		if(idAssignedTo[i].id == id){
			isHere = true;
			if(idAssignedTo[i].dist>newDist){
				idAssignedTo[i].to   = to;
				idAssignedTo[i].dist = newDist;
				return true;
			}
		}
	}
	if(!isHere){
		bool alreadyTo = false;
		for(unsigned i=0;i<idAssignedTo.size();i++){
			if(idAssignedTo[i].to == to){
				alreadyTo = true;
				if(idAssignedTo[i].dist>newDist){
					//assigned the id to this one and un-assign the old one
					annotationsHandle::ASSIGNED temp;
					temp.id   = id;
					temp.to   = to;
					temp.dist = newDist;
					idAssignedTo.push_back(temp);
					idAssignedTo.erase(idAssignedTo.begin()+i);
					return true;
				}
			}
		}
		if(!alreadyTo){
			annotationsHandle::ASSIGNED temp;
			temp.id   = id;
			temp.to   = to;
			temp.dist = newDist;
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
void annotationsHandle::correltateLocs(std::vector<annotationsHandle::ANNOTATION>\
&annoOld, std::vector<annotationsHandle::ANNOTATION> &annoNew,\
std::vector<annotationsHandle::ASSIGNED> &idAssignedTo){
	std::vector< std::vector<double> > distMatrix;

	//1. compute the distances between all annotations
	for(unsigned k=0;k<annoNew.size();k++){
		distMatrix.push_back(std::vector<double>());
		for(unsigned l=0;l<annoOld.size();l++){
			distMatrix[k].push_back(0.0);
			distMatrix[k][l] = dist(annoNew[k].location, annoOld[l].location);
		}
	}

	//2. assign annotations between the 2 groups of annotations
	bool canAssign = true;
	while(canAssign){
		canAssign = false;
		for(unsigned k=0;k<distMatrix.size();k++){ //for each row in new
			double minDist = (double)INFINITY;
			for(unsigned l=0;l<distMatrix[k].size();l++){ // loop over all old
				if(distMatrix[k][l]<minDist && canBeAssigned(idAssignedTo, \
					annoOld[l].id, distMatrix[k][l], annoNew[k].id)){
					minDist       =	distMatrix[k][l];
					canAssign     = true;
				}
			}
		}
	}

	//3. update the ids to the new one to correspond to the ids of the old one!!
	for(unsigned i=0;i<annoNew.size();i++){
		for(unsigned n=0;n<idAssignedTo.size();n++){
			if(annoNew[i].id == idAssignedTo[n].to){
				if(idAssignedTo[n].to != idAssignedTo[n].id){
					for(unsigned j=0;j<annoNew.size();j++){
						if(annoNew[j].id == idAssignedTo[n].id){
							annoNew[j].id = -1;
						}
					}
				}
				annoNew[i].id = idAssignedTo[n].id;
				break;
			}
		}
	}

}
//==============================================================================
/** Computes the average distance from the predicted location and the annotated
 * one, the number of unpredicted people in each image and the differences in the
 * pose estimation.
 */
void annotationsHandle::annoDifferences(std::vector<annotationsHandle::FULL_ANNOTATIONS>\
&train, std::vector<annotationsHandle::FULL_ANNOTATIONS> &test, double &avgDist,\
double &Ndiff, double ssdOrientDiff, double poseDiff){
	if(train.size() != test.size()) {exit(1);}
	for(unsigned i=0;i<train.size();i++){
		if(train[i].imgFile != test[i].imgFile) {
			errx(1,"Images on positions %i do not correspond",i);
		}

		/* the images might not be in the same ordered so they need to be assigned
		 * to the closest one to them */
		std::vector<annotationsHandle::ASSIGNED> idAssignedTo;
		//0. if one of them is 0 then no correlation can be done
		if(train[i].annos.size()!=0 && test[i].annos.size()!=0){
			correltateLocs(train[i].annos, test[i].annos,idAssignedTo);
		}
		//1. update average difference for the current image
		for(unsigned l=0; l<idAssignedTo.size(); l++){
			cout<<idAssignedTo[l].id<<" assigned to "<<idAssignedTo[l].to<<endl;
			avgDist += idAssignedTo[l].dist;
		}
		if(idAssignedTo.size()>0){
			avgDist /= idAssignedTo.size();
		}

		//2. update the difference in number of people detected
		if(train[i].annos.size()!=test[i].annos.size()){
			Ndiff += abs((double)train[i].annos.size() - \
				(double)test[i].annos.size());
		}

		//3. update the poses estimation differences
		unsigned int noCorresp = 0;
		for(unsigned int l=0;l<train[i].annos.size();l++){
			for(unsigned int k=0;k<test[i].annos.size();k++){
				if(train[i].annos[l].id == test[i].annos[k].id){
					for(unsigned int m=0;m<train[i].annos[l].poses.size()-1;m++){
						if((POSE)m != BENDING){
							poseDiff += abs((double)train[i].annos[l].poses[m] - \
									(double)test[i].annos[k].poses[m])/2.0;
						}else{
							poseDiff += abs((double)train[i].annos[l].poses[m] - \
									(double)test[i].annos[k].poses[m])/2.0;
						}
					}
					// SSD between the predicted values and the correct ones
					double angleTrain = ((double)train[i].annos[l].poses[3]*M_PI/180.0);
					double angleTest  = ((double)test[i].annos[k].poses[3]*M_PI/180.0);
					ssdOrientDiff += pow(cos(angleTrain)-cos(angleTest),2) +\
									pow(sin(angleTrain)-sin(angleTest),2);
					break;
				}
			}
		}
		cout<<endl;
	}
	cout<<"avgDist: "<<avgDist<<endl;
	cout<<"Ndiff: "<<Ndiff<<endl;
	cout<<"PoseDiff: "<<poseDiff<<endl;
	cout<<"ssdOrientDiff: "<<ssdOrientDiff<<endl;

	ssdOrientDiff /= train.size();
	avgDist /= train.size();
}
//==============================================================================
/** Displays the complete annotations for all images.
 */
void annotationsHandle::displayFullAnns(std::vector<annotationsHandle::FULL_ANNOTATIONS>\
&fullAnns){
	for(unsigned int i=0; i<fullAnns.size();i++){
		cout<<"Image name: "<<fullAnns[i].imgFile<<endl;
		for(unsigned int j=0; j<fullAnns[i].annos.size();j++){
			cout<<"Annotation Id:"<<fullAnns[i].annos[j].id<<\
				"_____________________________________________"<<endl;
			cout<<"Location: ["<<fullAnns[i].annos[j].location.x<<","\
				<<fullAnns[i].annos[j].location.y<<"]"<<endl;
			cout<<"Pose: (Sitting: "<<fullAnns[i].annos[j].poses[0]<<\
				"),(Standing: "<<fullAnns[i].annos[j].poses[1]<<\
				"),(Bending: "<<fullAnns[i].annos[j].poses[2]<<\
				"),(Orientation: "<<fullAnns[i].annos[j].poses[3]<<")"<<endl;
		}
		cout<<endl;
	}
}
//==============================================================================
/** Starts the annotation of the images. The parameters that need to be indicated
 * are:
 *
 * \li argv[1] -- train file with the correct annotations;
 * \li argv[2] -- test file with predicted annotations;
 */
int annotationsHandle::runEvaluation(int argc, char **argv){
	if(argc != 3){
		cerr<<"usage: cmmd <train_annotations.txt> <train_annotations.txt>\n"<< \
		"<train_annotations.txt> => file containing correct annotations\n"<< \
		"<test_annotations.txt>  => file containing predicted annotations\n"<<endl;
		exit(-1);
	}

	std::vector<annotationsHandle::FULL_ANNOTATIONS> allAnnoTrain;
	std::vector<annotationsHandle::FULL_ANNOTATIONS> allAnnoTest;
	loadAnnotations(argv[1],allAnnoTrain);
	loadAnnotations(argv[2],allAnnoTest);

	displayFullAnns(allAnnoTrain);

	double avgDist = 0, Ndiff = 0, ssdOrientDiff = 0, poseDiff = 0;
	annoDifferences(allAnnoTrain, allAnnoTest, avgDist, Ndiff, ssdOrientDiff, \
		poseDiff);
}

char annotationsHandle::choice;
boost::mutex annotationsHandle::trackbarMutex;
IplImage *annotationsHandle::image;
std::vector<annotationsHandle::ANNOTATION> annotationsHandle::annotations;
//==============================================================================
/*
int main(int argc, char **argv){
	annotationsHandle::runAnn(argc,argv);
	//annotationsHandle::runEvaluation(argc,argv);
}
*/
