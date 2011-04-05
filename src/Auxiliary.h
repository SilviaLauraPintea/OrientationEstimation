/* Auxiliary.h
 * Author: Silvia-Laura Pintea
 */
#ifndef AUXILIARY_H_
#define AUXILIARY_H_
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <cstdlib>
#include <stdio.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <data/XmlFile.hh>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/posix_time/time_parsers.hpp>
#include <boost/date_time/posix_time/time_formatters.hpp>
#include "eigenbackground/src/Helpers.hh"
#include "eigenbackground/src/defines.hh"
using namespace std;
using namespace boost;

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
#endif /* AUXILIARY_H_ */
