/* cholesky.cpp
 * Original code: Dr Gwenn Englebienne
 * Modified by: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code, but please retain the above copyright notice.
 */
#include <iostream>
#include <cmath>
#include <err.h>
#include <exception>
#include "cholesky.h"
//==============================================================================
/** Checks to see if the decomposition was already done (returns true if it is
 * done).
 */
bool cholesky::checkDecomposition(){
	return (!this->covar.empty());
}
//==============================================================================
/** (Re)Initializes the class variables so the same instance of the class can be
 * used for multiple decompositions.
 */
void cholesky::init(){
	if(!this->covar.empty()){
		this->covar.release();
		this->n = 0;
	}
}
//==============================================================================
/** Decomposes the (covariance) matrix A into A = LL*.
 */
int cholesky::decomposeCov(cv::Mat a){
	if(a.cols!=a.rows){
		std::cerr<<"For Cholesky decomposeCov: the input matrix needs to be square"<<std::endl;
		exit(1);
	}
	a.convertTo(a, CV_32FC1);
	if(!this->covar.empty()){
		this->covar.release();
	}
	this->n = static_cast<unsigned>(a.rows);
	a.copyTo(this->covar);
    for(int indy=0; indy<this->n; ++indy){
    	for(int indx=0; indx<this->n; ++indx){
        	float sum = this->covar.at<float>(indy, indx);
            for(int k=indy-1; k>=0; --k){
            	sum -= this->covar.at<float>(indy,k)*this->covar.at<float>(indx,k);
            }
            if(indx==indy){
            	if(sum <= 0.0){
            		std::cerr<<"Decomposition failed, not positive defined";
            		return 0;
            	}
            	this->covar.at<float>(indy,indx) = std::sqrt(sum);
            }else{
            	this->covar.at<float>(indx,indy) = sum/this->covar.at<float>(indy,indy);
            }
        }
    }

    // GET THE THE UPPER TRIANGLE
	for(int indy=0; indy<this->n; ++indy){
		for(int indx=0; indx!=indy; ++indx){
			this->covar.at<float>(indx,indy) = 0.0;
		}
	}
	this->covar.convertTo(this->covar, CV_32FC1);
    return 1;
}
//==============================================================================
/** Solves the general linear system: Ax = b and returns x.
 */
void cholesky::solve(cv::Mat b, cv::Mat &x){
	if(b.rows != this->n){
		std::cerr<<"In Cholesky solve: in Ax=b, b has the wrong size"<<std::endl;
		exit(1);
	}
	b.convertTo(b,CV_32FC1);
	x = cv::Mat::zeros(cv::Size(b.cols, this->n), CV_32FC1);
	for(int indx=0; indx<b.cols;  ++indx){ // NOT REALLY NEEDED (JUST 1 COL)
		for(int indy=0; indy<this->n; ++indy){
			float sum = b.at<float>(indy,indx);
			for(int k=indy-1; k>=0; --k){
				sum -= this->covar.at<float>(indy,k)*x.at<float>(k,indx);
			}
			x.at<float>(indy,indx) = sum/this->covar.at<float>(indy,indy);
		}
	}

	for(int indx=0; indx<x.cols; ++indx){ // NOT NEEDED (JUST 1 COL)
		for(int indy=this->n-1; indy>=0; --indy){
			float sum = x.at<float>(indy,indx);
			for(int k=indy+1; k<this->n; ++k){
				sum -= this->covar.at<float>(k,indy)*x.at<float>(k,indx);
			}
			x.at<float>(indy,indx) = sum/this->covar.at<float>(indy,indy);
		}
	}
	x.convertTo(x, CV_32FC1);
}
//==============================================================================
/** Solve the simplified equation Ly = b, and return y (where A=LL*).
 */
void cholesky::solveL(cv::Mat b, cv::Mat &y){
	if(b.rows != this->n){
		std::cerr<<"In Cholesky solveL: in Ly=b, b has the wrong size"<<std::endl;
		exit(1);
	}

	b.convertTo(b,CV_32FC1);
	y = cv::Mat::zeros(cv::Size(1, this->n), CV_32FC1);
	for(int indy=0; indy<this->n; ++indy){
		float sum = b.at<float>(indy,0);
		for(int indx=0; indx<indy; ++indx){
			sum -= this->covar.at<float>(indy,indx) * y.at<float>(indx,0);
		}
		y.at<float>(indy,0) = sum/this->covar.at<float>(indy,indy);
	}
	y.convertTo(y, CV_32FC1);
}
//==============================================================================
/** Solve the simplified equation L'y = b, and return y (where A=LL*).
 */
void cholesky::solveLTranspose(cv::Mat b, cv::Mat &y){
	if(b.rows != this->n){
		std::cerr<<"In Cholesky solveLTranspose: in L'y=b, b has the wrong size"<<std::endl;
		exit(1);
	}

	b.convertTo(b,CV_32FC1);
	y = cv::Mat::zeros(cv::Size(1, this->n), CV_32FC1);
	for(int indy=this->n-1; indy>=0; --indy){
		float sum = b.at<float>(indy,0);
		for(int indx=indy+1; indx<this->n; ++indx){
			sum -= this->covar.at<float>(indx,indy) * y.at<float>(indx,0);
		}
		y.at<float>(indy,0) = sum/this->covar.at<float>(indy,indy);
	}
	y.convertTo(y, CV_32FC1);
}
//==============================================================================
/** Returns the inverse of the covariance: A^{-1}.
 */
void cholesky::inverse(cv::Mat &ainv){
	ainv = cv::Mat::zeros(cv::Size(this->n, this->n),CV_32FC1);
	for(int indy=0; indy<this->n; ++indy){
		for(int indx=0; indx<this->n; ++indx){
			float sum = (indx==indy?1.0:0.0);
			for(int k=indy-1; k>=indx; --k){
				sum -= this->covar.at<float>(indy,k) * ainv.at<float>(indx,k);
			}
			ainv.at<float>(indx,indy) = sum/this->covar.at<float>(indy,indy);
		}
	}
	for(int indy=this->n-1; indy>=0; --indy){
		for(int indx=0; indx<=indy; ++indx){
			float sum = (indy<=indx ? 0.0 : ainv.at<float>(indx,indy));
			for(int k=indy+1; k<this->n; ++k){
				sum -= this->covar.at<float>(k,indy)*ainv.at<float>(indx,k);
			}
			ainv.at<float>(indy,indx) = sum/this->covar.at<float>(indy,indy);
			ainv.at<float>(indx,indy) = sum/this->covar.at<float>(indy,indy);
		}
	}
	ainv.convertTo(ainv, CV_32FC1);
}
//==============================================================================
/** Returns the log of the determiner of the (covariance) matrix, A.
 */
float cholesky::logDet(){
	float sum=0;
	for(int indy=0; indy<this->n; ++indy){
		sum += std::log(this->covar.at<float>(indy,indy));
	}
	return 2.0*sum;
}
//==============================================================================


