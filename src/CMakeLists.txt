#REQUIRED STUFF____________________________________________________________ 
cmake_minimum_required(VERSION 2.6)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

#????????????????????????__________________________________________________ 
INCLUDE(FindPkgConfig)
pkg_check_modules(XML2 REQUIRED libxml-2.0)

#PROJECT TO BE BUILT_______________________________________________________ 
PROJECT(GroupDetection)

#SET VARIABLES' NAMES______________________________________________________
SET(CMAKE_CXX_FLAGS ${XML2_CFLAGS}) #??????????????????

SET(EIGEN_SRC /data/Documents/masterProject/GroupDetection/src/eigenbackground/src CACHE PATH "Location of the eigenbackground src files")
SET(MT_SRC /data/Documents/masterProject/GroupDetection/src CACHE PATH "Location of my src")

SET(EXECUTABLE_OUTPUT_PATH /data/Documents/masterProject/GroupDetection/bin CACHE PATH "Location of the compiled files")

SET(OPENCV2 /usr/local/opencv/include CACHE PATH "Location of the OpenCV headers (version 2.2)")
SET(OPENCV2LIB /usr/local/opencv/lib CACHE PATH "Lib")

SET(VXL_BIN /opt/vxlbin CACHE PATH "Location of the VXL compiled library")
SET(CMN_SRC /data/Documents/masterProject/GroupDetection/src/cmn/src CACHE PATH "Location of the CMN library source code")
SET(CMN_BIN /data/Documents/masterProject/GroupDetection/src/cmn/bin CACHE PATH "Location of the CMN compiled library")

INCLUDE_DIRECTORIES(${OPENCV2} ${VXL_BIN} ${VXL_BIN} ${VXL_BIN}/contrib/tbl ${VXL_BIN}/core ${VXL_BIN}/vcl ${CMN_SRC} ${EIGEN_SRC})
LINK_DIRECTORIES(${VXL_BIN}/lib ${CMN_BIN} ${OPENCV2LIB})

ADD_LIBRARY(helpers
  ${EIGEN_SRC}/Helpers.cc
  ${EIGEN_SRC}/Background.cc
  ${EIGEN_SRC}/ImgProducer.cc
  ${EIGEN_SRC}/Annotate.cc
  ${EIGEN_SRC}/Tracker.cc
  
  ${MT_SRC}/Auxiliary.cpp
  ${MT_SRC}/AnnotationsHandle.cpp
  ${MT_SRC}/FeatureExtractor.cpp  
  ${MT_SRC}/PeopleDetector.cpp  
  ${MT_SRC}/Cholesky.cpp  
  ${MT_SRC}/GaussianProcess.cpp    
)

#ADD_EXECUTABLE(Tracker ${EIGEN_SRC}/Tracker.cc )
#TARGET_LINK_LIBRARIES(Tracker helpers opencv_core opencv_calib3d opencv_imgproc opencv_highgui cmn tools iberty vnl_algo vnl v3p_netlib boost_date_time-mt boost_thread-mt ${XML2_LDFLAGS})

#ADD_EXECUTABLE(Annotate ${EIGEN_SRC}/Annotate.cc )
#TARGET_LINK_LIBRARIES(Annotate helpers opencv_calib3d opencv_core opencv_imgproc opencv_highgui cmn tools iberty vnl_algo vnl v3p_netlib boost_date_time-mt boost_thread-mt ${XML2_LDFLAGS})
#_______________________________________________________________train___________________________________________________________________________
# FROM NOW ON MY FILES WILL COME
#ADD_EXECUTABLE(AnnotationsHandle ${MT_SRC}/AnnotationsHandle.cpp)
#TARGET_LINK_LIBRARIES(AnnotationsHandle helpers opencv_core opencv_imgproc opencv_highgui opencv_calib3d cmn tools iberty vnl_algo vnl v3p_netlib boost_filesystem boost_date_time-mt boost_thread-mt ${XML2_LDFLAGS})

#ADD_EXECUTABLE(PeopleDetector ${MT_SRC}/PeopleDetector.cpp)
#TARGET_LINK_LIBRARIES(PeopleDetector helpers opencv_core opencv_objdetect opencv_calib3d opencv_video opencv_imgproc opencv_highgui opencv_features2d cmn tools iberty vnl_algo vnl v3p_netlib boost_filesystem boost_date_time-mt boost_thread-mt ${XML2_LDFLAGS})

#ADD_EXECUTABLE(GaussianProcess ${MT_SRC}/GaussianProcess.cpp)
#TARGET_LINK_LIBRARIES(GaussianProcess helpers opencv_core opencv_imgproc opencv_highgui cmn tools iberty vnl_algo vnl v3p_netlib ${XML2_LDFLAGS})

#ADD_EXECUTABLE(ClassifyImages ${MT_SRC}/ClassifyImages.cpp)
#TARGET_LINK_LIBRARIES(ClassifyImages helpers opencv_core opencv_ml opencv_objdetect opencv_imgproc opencv_video opencv_highgui opencv_features2d cmn tools iberty vnl_algo vnl v3p_netlib boost_filesystem boost_date_time-mt boost_thread-mt ${XML2_LDFLAGS})



