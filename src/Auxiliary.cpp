/* Auxiliary.cpp
 * Author: Silvia-Laura Pintea
 */
#include "Auxiliary.h"
//==============================================================================
/** Reads images from a dir and stores them into a vector of strings.
 */
vector<string> readImages(const char* dirName){
	DIR *dirPoint;
	if((dirPoint = opendir(dirName)) == NULL){
		std::cerr<<"Error opening "<<dirName<<endl;
		exit(1);
	}
	struct dirent *dirEntry;
	vector<string> images;
	//unsigned contor = 0;
	while((dirEntry = readdir(dirPoint)) != NULL){
		string imageName = string(dirEntry->d_name);
		if(imageName.find(".jpg") != string::npos || imageName.find(".JPG") != \
			string::npos || imageName.find(".png") != string::npos || \
			imageName.find(".PNG") != string::npos || imageName.find(".jpeg") !=\
			string::npos || imageName.find(".JPEG") != string::npos || \
			imageName.find(".ppm") != string::npos || imageName.find(".PPM") != \
			string::npos){
			char *tmpDirName = new char[string(dirName).size()+50];
			strcpy(tmpDirName,(char*)dirName);
			images.push_back(string(strcat(tmpDirName,imageName.c_str())));
			delete [] tmpDirName;
		}
	}
	sort(images.begin(),images.end());
	closedir(dirPoint);
	return images;
}
//==============================================================================
/** Converts a pointer to an IplImage to an OpenCV Mat.
 */
cv::Mat ipl2mat(IplImage* ipl_image){
	cv::Mat mat_image(ipl_image);
	return mat_image;
}
//==============================================================================
/** Converts an OpenCV Mat to a pointer to an IplImage.
 */
IplImage* mat2ipl(cv::Mat image){
	IplImage* ipl_image = new IplImage(image);
	return ipl_image;
}
//==============================================================================




