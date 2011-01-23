#REQUIRED STUFF____________________________________________________________ 
cmake_minimum_required(VERSION 2.6)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

#????????????????????????__________________________________________________ 
INCLUDE(FindPkgConfig)
pkg_check_modules(XML2 REQUIRED libxml-2.0)

#PROJECT TO BE BUILT_______________________________________________________ 
PROJECT(Probabilistic_Tracking_using_Stereo_Cameras_proj)

#OPENCV PACKAGE REQUIRED___________________________________________________
#FIND_PACKAGE(OpenCV REQUIRED)

#SET VARIABLES' NAMES______________________________________________________
SET(CMAKE_CXX_FLAGS ${XML2_CFLAGS}) #??????????????????

SET(EIGEN_SRC /data/Documents/masterProject/Probabilistic_Tracking_using_Stereo_Cameras/src/eigenbackground/src CACHE PATH "Location of the eigenbackground src files")
SET(MT_SRC /data/Documents/masterProject/Probabilistic_Tracking_using_Stereo_Cameras/src CACHE PATH "Location of my src")

SET(EXECUTABLE_OUTPUT_PATH /data/Documents/masterProject/Probabilistic_Tracking_using_Stereo_Cameras/bin CACHE PATH "Location of the compiled files")

SET(OPENCV /usr/local/include/opencv CACHE PATH "Location of the OpenCV headers")
SET(VXL_BIN /opt/vxlbin CACHE PATH "Location of the VXL compiled library")
SET(CMN_SRC /data/Documents/masterProject/Probabilistic_Tracking_using_Stereo_Cameras/src/cmn/src CACHE PATH "Location of the CMN library source code")
SET(CMN_BIN /data/Documents/masterProject/Probabilistic_Tracking_using_Stereo_Cameras/src/cmn/bin CACHE PATH "Location of the CMN compiled library")

INCLUDE_DIRECTORIES(${OPENCV} ${VXL_BIN} ${VXL_BIN} ${VXL_BIN}/contrib/tbl ${VXL_BIN}/core ${VXL_BIN}/vcl ${CMN_SRC} ${EIGEN_SRC})
LINK_DIRECTORIES(${VXL_BIN}/lib ${CMN_BIN})

ADD_LIBRARY(helpers
  ${EIGEN_SRC}/Helpers.cc
  ${EIGEN_SRC}/Background.cc
  ${EIGEN_SRC}/ImgProducer.cc
  ${EIGEN_SRC}/Annotate.cc
  ${EIGEN_SRC}/Tracker.cc

  ${MT_SRC}/Auxiliary.cpp
#  ${MT_SRC}/featureDetector.cpp  
  )

ADD_LIBRARY(calib
  ${EIGEN_SRC}/sample/cameraModel.cpp
  ${EIGEN_SRC}/sample/xmlUtil.cpp
  )

#ADD_EXECUTABLE( Tracker ${EIGEN_SRC}/Tracker.cc )
#TARGET_LINK_LIBRARIES( Tracker helpers highgui  cmn tools iberty vnl_algo vnl v3p_netlib boost_date_time-mt boost_thread-mt ${XML2_LDFLAGS})

#ADD_EXECUTABLE( Annotate ${EIGEN_SRC}/Annotate.cc )
#TARGET_LINK_LIBRARIES( Annotate helpers highgui  cmn tools iberty vnl_algo vnl v3p_netlib boost_date_time-mt boost_thread-mt ${XML2_LDFLAGS})
#_________________________________________________________________________________________________________________________________________
# FROM NOW ON MY FILES WILL COME

ADD_EXECUTABLE(annotationsHandle ${MT_SRC}/annotationsHandle.cpp)
TARGET_LINK_LIBRARIES(annotationsHandle helpers cv highgui cmn tools iberty vnl_algo vnl v3p_netlib boost_filesystem boost_date_time-mt boost_thread-mt ${XML2_LDFLAGS})

ADD_EXECUTABLE(featureDetector ${MT_SRC}/featureDetector.cpp)
TARGET_LINK_LIBRARIES(featureDetector helpers cv highgui cmn tools iberty vnl_algo vnl v3p_netlib boost_filesystem boost_date_time-mt boost_thread-mt ${XML2_LDFLAGS})

#ADD_EXECUTABLE(classifyImages ${MT_SRC}/classifyImages.cpp)
#TARGET_LINK_LIBRARIES(classifyImages helpers cv ml highgui cmn tools iberty vnl_algo vnl v3p_netlib boost_filesystem boost_date_time-mt boost_thread-mt ${XML2_LDFLAGS})
