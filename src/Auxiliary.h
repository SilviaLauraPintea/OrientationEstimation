#ifndef AUXILIARY_H_
#define AUXILIARY_H_

#include <vector>
#include <string>
#include <set>
#include <cstdlib>
#include <stdio.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <data/XmlFile.hh>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/posix_time/time_parsers.hpp>
#include <boost/date_time/posix_time/time_formatters.hpp>
using namespace std;
using namespace boost;

/** Reads images from a dir and stores them into a vector of strings.
 */
vector<string> readImages(const char* dirName);

#endif /* AUXILIARY_H_ */
