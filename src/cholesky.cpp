/* cholesky.cpp
 * Original code: Dr Gwenn Englebienne
 * Modified by: Silvia-Laura Pintea
 */
#include "cholesky.h"
//==============================================================================
/** Decomposes the (covariance) matrix A into A = LL*.
 */
int cholesky::decomposeCov(cv::Mat a){
	if(a.cols!=a.rows){
		cerr<<"For Cholesky decomposeCov: the input matrix needs to be square"<<endl;
		exit(1);
	}
	this->n     = static_cast<unsigned>(a.rows);
	this->covar = a.clone();
	cv::Mat_<float> tmp;
    for(unsigned y=0; y<this->n; y++){
    	for(unsigned x=0; x<this->n; x++){
        	float sum = this->covar.at<float>(y,x);
            for(int k=y-1; k>0; --k){
            	sum -= this->covar.at<float>(y,k)*this->covar.at<float>(x,k);
            }
            if(x==y){
            	if(sum <= 0.0){return 1;}
            	this->covar.at<float>(y,x) = std::sqrt(sum);
            }else{
            	// ???????? why if we keep only the diag ?????
            	this->covar.at<float>(x,y) = sum/this->covar.at<float>(y,y);
            }
        }
    }

    // GET THE DIAGONAL ONLY
    this->covar = this->covar.diag(0);
    return 0;
}
//==============================================================================
/** Solves the general linear system: Ax = b and returns x.
 */
void cholesky::solve(cv::Mat b, cv::Mat &x){
	if(b.cols != this->n){
		cerr<<"In Cholesky solve: in Ax=b, b has the wrong size"<<endl;
		exit(1);
	}
	x = cv::Mat::zeros(cv::Size(this->covar.cols, 1), CV_32FC1);
	for(unsigned y=0; y<this->n; y++){
		float sum = b.at<float>(y,1);
		for(int k= y-1; k>=0; --k){
			sum -= this->covar.at<float>(y,k)*x.at<float>(k,1);
		}
		x.at<float>(y,1) = sum/this->covar.at<float>(y,y);
	}

	for(unsigned y=this->n-1; y>=0; --y){
		float sum = x.at<float>(y,1);
		for(int k= y+1; k<this->n; k++){
			sum -= this->covar.at<float>(k,y)*x.at<float>(k,1);
		}
		x.at<float>(y,1) = sum/this->covar.at<float>(y,y);
	}
}
//==============================================================================
/** Solve the simplified equation Ly = b, and return y (where A=LL*).
 */
void cholesky::solveL(cv::Mat b, cv::Mat &y){
	if(b.cols != this->n){
		cerr<<"In Cholesky solveL: in Ly=b, b has the wrong size"<<endl;
		exit(1);
	}

	y = cv::Mat::zeros(cv::Size(this->covar.cols, 1), CV_32FC1);
	for(unsigned y=0; y<this->n; y++){
		float sum = b.at<float>(y,1);
		for(unsigned x=0; x<y; x++){
			sum -= this->covar.at<float>(y,x)*y.at<float>(x,1);
		}
		y.at<float>(y,1) = sum/this->covar.at<float>(y,y);
	}
}
//==============================================================================
/** Solve the simplified equation L'y = b, and return y (where A=LL*).
 */
void cholesky::solveLTranspose(cv::Mat b, cv::Mat &y){
	if(b.cols != this->n){
		cerr<<"In Cholesky solveLTranspose: in L'y=b, b has the wrong size"<<endl;
		exit(1);
	}

	y = cv::Mat::zeros(cv::Size(this->covar.cols, 1), CV_32FC1);
	for(unsigned y=this->n-1; y>=0; --y){
		float sum = b.at<float>(y,1);
		for(unsigned x=y+1; x<this->n; x++){
			sum -= this->covar.at<float>(x,y)*y.at<float>(x,1);
		}
		y.at<float>(y,1) = sum/this->covar.at<float>(y,y);
	}
}
//==============================================================================
/** Returns the inverse of the covariance: A^{-1}.
 */
void cholesky::inverse(cv::Mat &ainv){
	ainv = cv::Mat::zeros(cv::Size(this->covar.cols, this->covar.rows), CV_32FC1);

	for(unsigned y=0; y<this->n; y++){
		for(unsigned x=0; x<this->n; x++){
			float sum = (x==y?1.0:0.0);
			for(int k=y-1; k>=x; --k){
				sum -= this->covar.at<float>(y,x)*ainv.at<float>(x,k);
			}
			ainv.at<float>(x,y) = sum/this->covar.at<float>(y,y);
		}
	}

	for(unsigned y=this->n-1; y>=0; --y){
		for(unsigned x=0; x<=y; x++){
			float sum = (y<=x?0.0:ainv.at<float>(x,y));
			for(unsigned k=y+1; k<this->n; k++){
				sum -= this->covar.at<float>(k,y)*ainv.at<float>(x,k);
			}
			ainv.at<float>(y,x) = sum/this->covar.at<float>(y,y);
			ainv.at<float>(x,y) = sum/this->covar.at<float>(y,y);
		}
	}
}
//==============================================================================
/** Returns the log of the determiner of the (covariance) matrix, A.
 */
float cholesky::logDet(){
	float sum=0;
	for(unsigned y=0; y<this->n; y++){
		sum += std::log(this->covar.at<float>(y,y));
	}
	return 2.0*sum;
}
//==============================================================================


