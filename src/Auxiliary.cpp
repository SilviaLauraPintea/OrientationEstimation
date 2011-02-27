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
/* Changes the values of the matrix to be between [-1,1].
 */
void range1Mat(cv::Mat &matrix){
	normalizeMat(matrix);
	matrix -= 0.5;
	matrix *= 2.0;
}
//==============================================================================
/* Write a matrix to a file (first row is the dimension of the matrix).
 */
void mat2File(cv::Mat matrix, char* fileName, bool append){
	ofstream dictOut;
	try{
		if(append){
			dictOut.open(fileName, ios::out | ios::app);
			dictOut.seekp(0, ios::end);
		}else{
			dictOut.open(fileName, ios::out);
		}
	}catch(std::exception &e){
		cerr<<"Cannot open file: %s"<<e.what()<<endl;
		exit(1);
	}

	matrix.convertTo(matrix, cv::DataType<double>::type);
	dictOut<<matrix.cols<<" "<<matrix.rows<<std::endl;
	for(int y=0; y<matrix.rows; y++){
		for(int x=0; x<matrix.cols; x++){
			dictOut<<matrix.at<double>(y,x)<<" ";
		}
		dictOut<<std::endl;
	}
	dictOut.close();
}
//==============================================================================
/* Read a matrix from a file (first row is the dimension of the matrix).
 */
void file2Mat(cv::Mat &matrix, char* fileName){
	ifstream dictFile(fileName);
	int y=0;
	if(dictFile.is_open()){
		// FIRST LINE IS THE SIZE OF THE MATRIX
		std::string fline;
		std::getline(dictFile, fline);
		std::vector<std::string> flineVect = splitLine(const_cast<char*>\
											(fline.c_str()),' ');
		if(flineVect.size() == 2){
			char *pRows, *pCols;
			int cols = strtol(flineVect[0].c_str(), &pCols, 10);
			int rows = strtol(flineVect[1].c_str(), &pRows, 10);
			matrix   = cv::Mat::zeros(cv::Size(cols,rows),\
						cv::DataType<double>::type);
		}else return;
		fline.clear();
		flineVect.clear();

		// THE REST OF THE LINES ARE READ ONE BY ONE
		while(dictFile.good()){
			std::string line;
			std::getline(dictFile, line);
			std::vector<std::string> lineVect = splitLine(const_cast<char*>\
												(line.c_str()),' ');
			if(lineVect.size()>=1){
				for(std::size_t x=0; x<lineVect.size(); x++){
					char *pValue;
					matrix.at<double>(y,static_cast<int>(x)) = \
						strtod(lineVect[x].c_str(), &pValue);
				}
				y++;
			}
			line.clear();
			lineVect.clear();
		}
		dictFile.close();
	}
	matrix.convertTo(matrix, cv::DataType<double>::type);
}
//==============================================================================



