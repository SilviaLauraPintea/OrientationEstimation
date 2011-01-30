/* cholesky.cpp
 * Original code: Dr Gwenn Englebienne
 * Modified by: Silvia-Laura Pintea
 */
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
	this->n     = static_cast<unsigned>(a.cols);
	this->covar = a.clone();
    for(unsigned indy=0; indy<this->n; indy++){
    	for(unsigned indx=0; indx<this->n; indx++){
        	double sum = this->covar.at<double>(indy, indx);
            for(int k=indy-1; k>0; --k){
            	sum -= this->covar.at<double>(indy,k)*this->covar.at<double>(indx,k);
            }
            if(indx==indy){
            	if(sum <= 0.0){return 1;}
            	this->covar.at<double>(indy,indx) = std::sqrt(sum);
            }else{
            	this->covar.at<double>(indx,indy) = sum/this->covar.at<double>(indy,indy);
            }
        }
    }

    // GET THE THE UPPER TRIANGLE
	for(unsigned indy=0; indy<this->n; ++indy){
		for(unsigned indx=0; indy<indx; ++indx){
			this->covar.at<double>(indx,indy) = 0.0;
		}
	}
    return 0;
}
//==============================================================================
/** Solves the general linear system: Ax = b and returns x.
 */
void cholesky::solve(cv::Mat b, cv::Mat &x){
	if(b.rows != this->n){
		std::cerr<<"In Cholesky solve: in Ax=b, b has the wrong size"<<std::endl;
		exit(1);
	}
	x = cv::Mat::zeros(cv::Size(b.cols, this->covar.rows), cv::DataType<double>::type);
	for(unsigned indx=0; indx<b.cols; indx++){
		for(unsigned indy=0; indy<this->n; indy++){
			double sum = b.at<double>(indy,indx);
			for(int k=indy-1; k>=0; --k){
				sum -= this->covar.at<double>(indy,k)*x.at<double>(k,indx);
			}
			x.at<double>(indy,indx) = sum/this->covar.at<double>(indy,indy);
		}
	}
	for(unsigned indx=0; indx<x.cols; indx++){
		for(int indy=this->n-1; indy>=0; indy--){
			double sum = x.at<double>(indy,indx);
			for(int k=indy+1; k<this->n; k++){
				sum -= this->covar.at<double>(k,indy)*x.at<double>(k,indx);
			}
			x.at<double>(indy,indx) = sum/this->covar.at<double>(indy,indy);
		}
	}
}
//==============================================================================
/** Solve the simplified equation Ly = b, and return y (where A=LL*).
 */
void cholesky::solveL(cv::Mat b, cv::Mat &y){
	if(b.rows != this->n){
		std::cerr<<"In Cholesky solveL: in Ly=b, b has the wrong size"<<std::endl;
		exit(1);
	}

	y = cv::Mat::zeros(cv::Size(1, this->covar.rows), cv::DataType<double>::type);
	for(unsigned indy=0; indy<this->n; indy++){
		double sum = b.at<double>(indy,0);
		for(unsigned indx=0; indx<indy; indx++){
			sum -= this->covar.at<double>(indy,indx) * y.at<double>(indx,0);
		}
		y.at<double>(indy,0) = sum/this->covar.at<double>(indy,indy);
	}
}
//==============================================================================
/** Solve the simplified equation L'y = b, and return y (where A=LL*).
 */
void cholesky::solveLTranspose(cv::Mat b, cv::Mat &y){
	if(b.rows != this->n){
		std::cerr<<"In Cholesky solveLTranspose: in L'y=b, b has the wrong size"<<std::endl;
		exit(1);
	}

	y = cv::Mat::zeros(cv::Size(1, this->covar.rows), cv::DataType<double>::type);
	for(int indy=this->n-1; indy>=0; --indy){
		double sum = b.at<double>(indy,0);
		for(unsigned indx=indy+1; indx<this->n; indx++){
			sum -= this->covar.at<double>(indx,indy) * y.at<double>(indx,0);
		}
		y.at<double>(indy,0) = sum/this->covar.at<double>(indy,indy);
	}
}
//==============================================================================
/** Returns the inverse of the covariance: A^{-1}.
 */
void cholesky::inverse(cv::Mat &ainv){
	ainv = cv::Mat::zeros(cv::Size(this->covar.cols, this->covar.rows),\
			cv::DataType<double>::type);
	for(unsigned indy=0; indy<this->n; indy++){
		for(unsigned indx=0; indx<this->n; indx++){
			double sum = (indx==indy?1.0:0.0);
			for(int k=indy-1; k>=indx; --k){
				sum -= this->covar.at<double>(indy,k) * ainv.at<double>(indx,k);
			}
			ainv.at<double>(indx,indy) = sum/this->covar.at<double>(indy,indy);
		}
	}

	for(int indy=this->n-1; indy>=0; --indy){
		for(unsigned indx=0; indx<=indy; indx++){
			double sum = (indy<=indx ? 0.0 : ainv.at<double>(indx,indy));
			for(unsigned k=indy+1; k<this->n; k++){
				sum -= this->covar.at<double>(k,indy)*ainv.at<double>(indx,k);
			}
			ainv.at<double>(indy,indx) = sum/this->covar.at<double>(indy,indy);
			ainv.at<double>(indx,indy) = sum/this->covar.at<double>(indy,indy);
		}
	}
}
//==============================================================================
/** Returns the log of the determiner of the (covariance) matrix, A.
 */
double cholesky::logDet(){
	double sum=0;
	for(unsigned indy=0; indy<this->n; indy++){
		sum += std::log(this->covar.at<double>(indy,indy));
	}
	return 2.0*sum;
}
//==============================================================================


