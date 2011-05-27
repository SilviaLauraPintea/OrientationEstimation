/* gaussianProcess.cpp
 * Original code: Dr Gwenn Englebienne
 * Modified by: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#include <iostream>
#include <fstream>
#include <cmath>
#include <exception>
#include <deque>
#include <opencv2/opencv.hpp>
#include "cholesky.h"
#include "gaussianProcess.h"
#include "Auxiliary.h"
//==============================================================================
/** Initializes or re-initializes a Gaussian Process.
 */
void gaussianProcess::init(gaussianProcess::kernelFunction theKFunction){
	this->chlsky.init();
	this->_norm_fast = false;
	this->_norm_max  = RAND_MAX/2.0;
	this->_norm_next = 0.0;
	this->rand_x     = 0;
	this->rand_y     = 1;
	this->N          = 0;
	this->kFunction  = theKFunction;
	if(!this->alpha.empty()){
		this->alpha.release();
	}
	if(!this->data.empty()){
		this->data.release();
	}
}
//==============================================================================
/** Generates a selected distribution of the functions given the parameters (the
 * mean: mu,the covariance: cov,the data x).
 */
double gaussianProcess::distribution(const cv::Mat &x,\
const gaussianProcess::DISTRIBUTION &distrib,const cv::Mat &mu,const cv::Mat &cov,\
double a,double b,double s){
	assert(cov.type()==cv::DataType<double>::type);
	assert(mu.type()==cv::DataType<double>::type);

	double det2,result;
	cv::Mat diff;
	cv::Mat inv;

	switch(distrib){
		case (gaussianProcess::BETA):
			if(x.cols!=1 || x.rows!=1){
				std::cerr<<"GaussianProcess BETA distribution: size(x) = (1,1)!"<<std::endl;
				goto err;
			}
			result = (gamma(a+b)*(std::pow(x.at<double>(0,0),(a-1.0)))*\
						(std::pow(1.0-x.at<double>(0,0),(b-1.0))))/(gamma(a)+gamma(b));
			break;
		case (gaussianProcess::GAUSS):
			if(x.cols!=1 || x.rows!=1){
				std::cerr<<"GaussianProcess GAUSS distribution: size(x) = (1,1)!"<<std::endl;
				goto err;
			}
			if(mu.cols!=1 || mu.rows!=1){
				std::cerr<<"GaussianProcess GAUSS distribution: size(mu) = (1,1)!(mean)"<<std::endl;
				goto err;
			}
			result = std::exp(-std::pow((x.at<double>(0,0)-mu.at<double>(0,0)),2)/\
						(2.0*std::pow(s,2)))/(std::sqrt(2.0*M_PI)*s);
			break;
		case (gaussianProcess::GAUSS2D):
			if(x.cols!=2 || x.rows!=1){
				std::cerr<<"GaussianProcess GAUSS2D distribution: size(x)=(1,2)!(x.x,x.y)"<<std::endl;
				goto err;
			}
			if(mu.cols!=2 || mu.rows!=1){
				std::cerr<<"GaussianProcess GAUSS2D distribution: size(mu)=(1,2)!(mu.x,mu.y)"<<std::endl;
				goto err;
			}
			if(cov.cols!=2 || cov.rows!=2){
				std::cerr<<"GaussianProcess GAUSS2D distribution: size(cov)=(2,2)!(covariance)"<<std::endl;
				goto err;
			}
			det2   = (cov.at<double>(0,0)*cov.at<double>(1,1) -\
							cov.at<double>(0,1)*cov.at<double>(1,0));
			diff   = x-mu;
			result = 1.0/(2.0*M_PI*std::sqrt(det2))*std::exp(-0.5*\
						diff.dot(diff*cov.inv()));
			break;
		case (gaussianProcess::GAUSSnD):
			if(x.cols<2 || x.rows!=1){
				std::cerr<<"GaussianProcess GAUSSnD distribution: size(x)=(1,n)!(a row)"<<std::endl;
				goto err;
			}
			if(mu.cols<2 || mu.rows!=1){
				std::cerr<<"GaussianProcess GAUSSnD distribution: size(mu)=(1,n)!(a row)"<<std::endl;
				goto err;
			}
			// IF NO DECOMPOSITION WAS DONE,DO THAT
			if(!this->chlsky.checkDecomposition()){
				this->chlsky.decomposeCov(cov);
			}
			this->chlsky.inverse(inv);
			diff   = x-mu;
			result = 1.0/(std::pow((2.0*M_PI),(x.cols/2.0))*\
						exp(0.5*this->chlsky.logDet()))*exp(-0.5*diff.dot(diff*inv));
			break;
		case (gaussianProcess::LOGGAUSSnD):
			if(x.cols<2 || x.rows!=1){
				std::cerr<<"GaussianProcess LOGGAUSS2D distribution: size(x)=(1,n)!(a row)"<<std::endl;
				goto err;
			}
			if(mu.cols<2 || mu.rows!=1){
				std::cerr<<"GaussianProcess LOGGAUSS2D distribution: size(mu)=(1,n)!(a row)"<<std::endl;
				goto err;
			}
			// IF NO DECOMPOSITION WAS DONE,DO THAT
			if(!this->chlsky.checkDecomposition()){
				this->chlsky.decomposeCov(cov);
			}
			this->chlsky.inverse(inv);
			diff   = x-mu;
			result = -0.5*(this->chlsky.n*std::log(2.0*M_PI)+this->chlsky.logDet()+\
						diff.dot(diff*inv));
			break;
	}
	inv.release();
	diff.release();
	return result;
	err:
		diff.release();
		inv.release();
		exit(1);
}
//==============================================================================
/** Trains the Gaussian process.
 */
void gaussianProcess::train(const cv::Mat &X,const cv::Mat &y,\
double (gaussianProcess::*fFunction)(const cv::Mat&,const cv::Mat&,double),\
double sigmasq,double length){
	if(y.rows != X.rows){
		std::cerr<<"In Gaussian Process - train: X and y need to be defined for the"<<\
			" same number of points"<<std::endl;
		return;
	}
	assert(X.type()==cv::DataType<double>::type);
	assert(y.type()==cv::DataType<double>::type);
	this->chlsky.init();
	this->kFunction = fFunction;
	this->N         = X.rows;// NUMBER OF TRAINING DATA POINTS!
	if(!this->data.empty()){
		this->data.release();
	}
	X.copyTo(this->data);
	this->data.convertTo(this->data,cv::DataType<double>::type);

	// BUILD THE KERNEL MARIX K: K(i,j) = k(x[i],x[j])
	cv::Mat K = cv::Mat::zeros(cv::Size(this->N,this->N),cv::DataType<double>::type);
	for(int indy=0;indy<this->N;++indy){
		for(int indx=0;indx<this->N;++indx){
			K.at<double>(indy,indx) = (this->*kFunction)\
				(X.row(indy),X.row(indx),length);
		}
	}

	// ADD sigma^2 TO THE KERNEL MATRIX,K
	for(int indy=0;indy<this->N;++indy){
		K.at<double>(indy,indy) += sigmasq;
	}

    // BUILD THE CHOLESKY DECOMPOSITON IF IT WAS NOT BUILT YET
	if(!this->chlsky.checkDecomposition()){
		if(!this->chlsky.decomposeCov(K)){
			std::cerr<<"Cholesky decomposition failed"<<std::endl;
			exit(1);
		}
	}
	this->chlsky.solve(y,this->alpha);
	this->alpha.convertTo(this->alpha,cv::DataType<double>::type);

	std::cout<<"N: ("<<this->N<<")"<<std::endl;
	std::cout<<"size of alpha: ("<<this->alpha.cols<<","<<this->alpha.rows<<")"<<std::endl;
	std::cout<<"size of data: ("<<this->data.cols<<","<<this->data.rows<<")"<<std::endl;
	K.release();
}
//==============================================================================
/** Returns the prediction for the test data,x (only one test data point).
 */
void gaussianProcess::predict(const cv::Mat &x,gaussianProcess::prediction &predi,\
double length){
	assert(x.type()==cv::DataType<double>::type);
	cv::Mat kstar(cv::Size(1,this->N),cv::DataType<double>::type);
	for(int indy=0;indy<this->N;++indy){
		kstar.at<double>(indy,0) = (this->*kFunction)(this->data.row(indy),x,length);
	}
	for(int i=0;i<this->alpha.cols;++i){
		predi.mean.push_back(kstar.dot(this->alpha.col(i)));
	}
	cv::Mat v;
	this->chlsky.solveL(kstar,v);
	v.convertTo(v,cv::DataType<double>::type);
	predi.variance.push_back((this->*kFunction)(x,x,length) - v.dot(v));
	kstar.release();
	v.release();
}
//==============================================================================
/** Samples the process that generates the inputs.
 */
void gaussianProcess::sample(const cv::Mat &inputs,cv::Mat &smpl){
	cv::Mat Kxstarx(this->N,inputs.cols,cv::DataType<double>::type);
	cv::Mat Kxstarxstar(inputs.cols,inputs.cols,cv::DataType<double>::type);

	for(int indy=0;indy<inputs.cols;++indy){
		for(int indx=0;indx<this->N;++indx){
			Kxstarx.at<double>(indy,indx) = (this->*kFunction)(inputs.row(indy),\
											this->data.row(indx),1.0);
		}
		for(int indx=0;indx<inputs.cols;++indx){
			Kxstarxstar.at<double>(indy,indx) = (this->*kFunction)(inputs.row(indy),\
												inputs.row(indx),1.0);
		}
	}

	cv::Mat Kxxstar = Kxstarx.t();
	cv::Mat mu      = Kxstarx * this->alpha;
	cv::Mat inv;
	this->chlsky.inverse(inv);
	cv::Mat cov     = Kxstarxstar-(Kxstarx*inv)*Kxxstar;

	for(int indy=0;indy<cov.cols;++indy){
		cov.at<double>(indy,indy) += 1.0e-6;
	}
	this->sampleGaussND(mu,cov,smpl);

	Kxstarx.release();
	Kxstarxstar.release();
	Kxxstar.release();
	mu.release();
	cov.release();
	inv.release();
}
//==============================================================================
/** Samples an N-dimensional Gaussian.
 */
void gaussianProcess::sampleGaussND(const cv::Mat &mu,const cv::Mat &cov,cv::Mat &smpl){
	assert(cov.type()==cv::DataType<double>::type);
	assert(mu.type()==cv::DataType<double>::type);
	if(!this->chlsky.checkDecomposition()){
		this->chlsky.decomposeCov(cov);
	}

	for(int indy=0;indy<mu.cols;++indy){
		smpl.at<double>(indy,0) = this->rand_normal();
	}

	smpl = mu + (this->chlsky.covar * smpl);
}
//==============================================================================
/** Returns a random number from the normal distribution.
 */
double gaussianProcess::rand_normal(){
	if(this->_norm_fast){
		this->_norm_fast = false;
		return (this->_norm_next);
	}
	this->_norm_fast = true;

	while(true){
		double u = (std::rand() - _norm_max + 0.5)/_norm_max;
		double v = (std::rand() - _norm_max + 0.5) / _norm_max;

		double x,w =u*u+v*v;
		if (w >= 1) continue;
		x = std::sqrt(-2.0 * std::log(w)/w);
		this->_norm_next = u*x;
		return (v*x);
     }
     return 12345.789;
}
//==============================================================================
/** Samples the Gaussian Process Prior.
 */
void gaussianProcess::sampleGPPrior(double (gaussianProcess::*fFunction)\
(const cv::Mat&,const cv::Mat&,double),const cv::Mat &inputs,cv::Mat &smpl){
	this->kFunction = fFunction;
	cv::Mat mu;
	cv::Mat cov;

	for(int indy=0;indy<inputs.cols;++indy){
		mu.at<double>(indy,0) = 0.0;
		for(int indx=0;indx<inputs.cols;++indx){
			cov.at<double>(indy,indx) = (this->*kFunction)(inputs.row(indy),\
										inputs.row(indx),1.0);
		}
	}

	for(int indy=0;indy<inputs.cols;++indy){
		cov.at<double>(indy,indy) += 1.0e-6;
	}
	this->sampleGaussND(mu,cov,smpl);
	mu.release();
	cov.release();
}
//==============================================================================
// Squared exponential kernel function.
double gaussianProcess::sqexp(const cv::Mat &x1,const cv::Mat &x2,double l){
	assert(x1.type()==cv::DataType<double>::type);
	assert(x2.type()==cv::DataType<double>::type);
	cv::Mat diff = x1-x2;
	diff.convertTo(diff,cv::DataType<double>::type);
	double result = std::exp(-1.0 * diff.dot(diff)/(2.0*l));
	diff.release();
	return result;
}
//==============================================================================
// Matern05 kernel function.
double gaussianProcess::matern05(const cv::Mat &x1,const cv::Mat &x2,double l){
	assert(x1.type()==cv::DataType<double>::type);
	assert(x2.type()==cv::DataType<double>::type);
	cv::Mat diff  = x1-x2;
	diff.convertTo(diff,cv::DataType<double>::type);
	double result = std::sqrt(diff.dot(diff));
	diff.release();
	return std::exp(-1.0 * result/l);
}
//==============================================================================
// Exponential Covariance kernel function.
double gaussianProcess::expCovar(const cv::Mat &x1,const cv::Mat &x2,double l){
	assert(x1.type()==cv::DataType<double>::type);
	assert(x2.type()==cv::DataType<double>::type);
	cv::Mat diff  = x1-x2;
	diff.convertTo(diff,cv::DataType<double>::type);
	double result = std::sqrt(diff.dot(diff));
	diff.release();
	return std::exp(-1.0 * result/l);
}
//==============================================================================
// Matern15 kernel function.
double gaussianProcess::matern15(const cv::Mat &x1,const cv::Mat &x2,double l){
	assert(x1.type()==cv::DataType<double>::type);
	assert(x2.type()==cv::DataType<double>::type);
	cv::Mat diff  = x1-x2;
	diff.convertTo(diff,cv::DataType<double>::type);
	double result = std::sqrt(diff.dot(diff));
	diff.release();
	return (1.0 + std::sqrt(3.0)*result/l) * std::exp(-1.0 * std::sqrt(3.0)*result/l);
}
//==============================================================================
// Matern25 kernel function.
double gaussianProcess::matern25(const cv::Mat &x1,const cv::Mat &x2,double l){
	assert(x1.type()==cv::DataType<double>::type);
	assert(x2.type()==cv::DataType<double>::type);
	cv::Mat diff  = x1-x2;
	diff.convertTo(diff,cv::DataType<double>::type);
	double result = std::sqrt(diff.dot(diff));
	diff.release();
	return (1.0 + std::sqrt(5.0)*result/l + (5.0*result*result)/(3.0*l*l))*\
		std::exp(-1.0 * std::sqrt(5.0)*result/l);
}
//==============================================================================
/** Useful to compute the distance between 2 edges.
 */
double gaussianProcess::matchShapes(const cv::Mat &x1,const cv::Mat &x2,double l){
	assert(x1.type()==cv::DataType<double>::type);
	assert(x2.type()==cv::DataType<double>::type);
	cv::Mat tmpX1,tmpX2;

	// IT NEEDS TO HAVE 2 CHANNELS => NEEDS TO HAVE COLS%2=0
	if(x1.cols%2==1){
		tmpX1 = cv::Mat::zeros(cv::Size(x1.cols+1,x1.rows),x1.type());
		cv::Mat dumm = tmpX1.colRange(0,x1.cols);
		x1.copyTo(dumm);
		dumm.release();

		// ALSO FOR X2 IT SHOULD BE TRUE
		tmpX2 = cv::Mat::zeros(cv::Size(x2.cols+1,x2.rows),x2.type());
		dumm = tmpX2.colRange(0,x2.cols);
		x2.copyTo(dumm);
		dumm.release();
	}else{
		x1.copyTo(tmpX1);
		x2.copyTo(tmpX2);
	}
	tmpX1.convertTo(tmpX1,CV_32FC1);
	tmpX2.convertTo(tmpX2,CV_32FC1);
	tmpX1 = tmpX1.reshape(2,0);
	tmpX2 = tmpX2.reshape(2,0);
	double result = cv::matchShapes(tmpX1,tmpX2,1,0.0);
	tmpX1.release();
	tmpX2.release();
	if(!result){
		result = 0.000000001;
	}

std::cout<<result<<"==>"<<std::exp(-2.0*l/result)<<std::endl;

	return std::exp(-2.0*l/result);
}
//==============================================================================
/*
int main(){
	cv::Mat test(10,5,cv::DataType<double>::type);
	cv::Mat train(100,5,cv::DataType<double>::type);
	cv::Mat targets = cv::Mat::zeros(100,1,cv::DataType<double>::type);
	cv::Mat ttargets = cv::Mat::zeros(10,1,cv::DataType<double>::type);
	train = cv::Mat::zeros(100,5,cv::DataType<double>::type);

	for(unsigned i=0;i<100;++i){
		cv::Mat stupid = train.row(i);
		cv::add(stupid,cv::Scalar(i),stupid);
		targets.at<double>(i,0) = i;
		if(i<10){
			cv::Mat stupid2 = test.row(i);
			cv::add(stupid2,cv::Scalar(i*2.5),stupid2);
			ttargets.at<double>(i,0) = i*2.5;
		}
	}

	std::cout<<"test: "<<test<<std::endl;
	std::cout<<"ttargets: "<<ttargets<<std::endl;

	gaussianProcess gp;
	gp.train(train,targets,&gaussianProcess::sqexp,0.1);

	cv::Mat result;
	for(unsigned i=0;i<test.rows;++i){
		gaussianProcess::prediction predi;
		gp.predict(test.row(i),predi);
		std::cout<<"label: "<<ttargets.at<double>(i,0)<<"\t"<<\
			predi.mean[0]<<" variance:"<<predi.variance[0]<<std::endl;
	}
}
*/
