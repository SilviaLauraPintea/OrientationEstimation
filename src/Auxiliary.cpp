/* Auxiliary.cpp
 * Author: Silvia-Laura Pintea
 */
#include "Auxiliary.h"
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
/** Convert the values from a cv::Mat of doubles to be between 0 and 1.
 */
void normalizeMat(cv::Mat &matrix){
	matrix.convertTo(matrix, cv::DataType<double>::type);
	double min = matrix.at<double>(0,0), max = matrix.at<double>(0,0);
	for(int x=0; x<matrix.cols; x++){
		for(int y=0; y<matrix.rows; y++){
			if(min>matrix.at<double>(y,x)){
				min = matrix.at<double>(y,x);
			}
			if(max<matrix.at<double>(y,x)){
				max = matrix.at<double>(y,x);
			}
		}
	}
	matrix -= min;
	matrix /= (max-min);
}
//==============================================================================
/* Changes the values of the matrix to be between [-1,1]
 */
void range1Mat(cv::Mat &matrix){
	normalizeMat(matrix);
	matrix -= 0.5;
	matrix *= 2.0;
}


