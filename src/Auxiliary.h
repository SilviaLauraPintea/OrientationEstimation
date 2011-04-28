/* Auxiliary.h
 * Author: Silvia-Laura Pintea
 */
#ifndef AUXILIARY_H_
#define AUXILIARY_H_
using namespace std;
#include <opencv2/opencv.hpp>

/** Converts a pointer to an IplImage to an OpenCV Mat.
 */
cv::Mat ipl2mat(IplImage* ipl_image);

/** Converts an OpenCV Mat to a pointer to an IplImage.
 */
IplImage* mat2ipl(cv::Mat image);

/** Convert the values from a cv::Mat of doubles to be between 0 and 1.
 */
void normalizeMat(cv::Mat &matrix);

/* Changes the values of the matrix to be between [-1,1].
 */
void range1Mat(cv::Mat &matrix);

/* Write a 2D-matrix to a text file (first row is the dimension of the matrix).
 */
void mat2TxtFile(cv::Mat matrix, char* fileName, bool append = false);

/* Reads a 2D-matrix from a text file (first row is the dimension of the matrix).
 */
void txtFile2Mat(cv::Mat &matrix, char* fileName);

/* Write a 2D-matrix to a binary file (first the dimension of the matrix).
 */
void mat2BinFile(cv::Mat matrix, char* fileName, bool append = false);

/* Reads a 2D-matrix from a binary file (first the dimension of the matrix).
 */
void binFile2mat(cv::Mat &matrix, char* fileName);
/** Convert int to string.
 */
std::string int2string(int i);
/** Changes a given angle in RADIANS to be positive and between [0,2*M_PI).
 */
void angle0to360(float &angle);
/** Changes a given angle in RADIANS to be positive and between [-M_PI,M_PI).
 */
void angle180to180(float &angle);
/** Get perpendicular to a line given by 2 points A, B in point C.
 */
void perpendicularLine(cv::Point2f A, cv::Point2f B, cv::Point2f C, \
	float &m, float &b);
/** Checks to see if a point is on the same side of a line like another given point.
 */
bool sameSubplane(cv::Point2f test,cv::Point2f point, float m, float b);
/** Just displaying an image a bit larger to visualize it better.
 */
void showZoomedImage(cv::Mat image, std::string title="zoomed");
#endif /* AUXILIARY_H_ */
