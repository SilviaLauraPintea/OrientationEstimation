/* gaussianProcess.cpp
 * Original code: Dr Gwenn Englebienne
 * Modified by: Silvia-Laura Pintea
 */
#include "gaussianProcess.h"
//==============================================================================
gaussianProcess::gaussianProcess() {
}
//==============================================================================
gaussianProcess::~gaussianProcess() {

}
//==============================================================================
/** Generates a selected distribution of the functions given the parameters (the
 * mean: mu, the covariance: cov, the data x).
 */
void gaussianProcess::distribution(cv::Mat x,gaussianProcess::DISTRIBUTION distrib,
cv::Mat mu = cv::Mat(),cv::Mat cov = cv::Mat(),float a=0,float b=0,float s=0){
	switch(distrib){
		case (gaussianProcess::BETA):
			if(x.cols!=1 || x.rows!=1){
				cerr<<"GaussianProcess BETA distribution: size(x) = (1,1)!"<<endl;
				exit(1);
			}
			return (gamma(a+b)*(std::pow(x.at<float>(0,0),(a-1.0)))*\
				(std::pow(1.0-x.at<float>(0,0),(b-1.0))))/(gamma(a)+gamma(b));
			break;
		case (gaussianProcess::GAUSS):
			if(x.cols!=1 || x.rows!=1){
				cerr<<"GaussianProcess GAUSS distribution: size(x) = (1,1)!"<<endl;
				exit(1);
			}
			if(mu.cols!=1 || mu.rows!=1){
				cerr<<"GaussianProcess GAUSS distribution: size(mu) = (1,1)!(mean)"<<endl;
				exit(1);
			}
			return std::exp(-std::pow((x.at<float>(0,0)-mu.at<float>(0,0)),2)/\
				(2.0*std::pow(s,2)))/(std::sqrt(2.0*M_PI)*s);
			break;
		case (gaussianProcess::DET2):
			break;
		case (gaussianProcess::GAUSS2D):
			if(x.cols!=2 || x.rows!=1){
				cerr<<"GaussianProcess GAUSS2D distribution: size(x)=(1,2)!(x.x,x.y)"<<endl;
				exit(1);
			}
			if(mu.cols!=2 || mu.rows!=1){
				cerr<<"GaussianProcess GAUSS2D distribution: size(mu)=(1,2)!(mu.x,mu.y)"<<endl;
				exit(1);
			}
			if(cov.cols!=2 || cov.rows!=2){
				cerr<<"GaussianProcess GAUSS2D distribution: size(cov)=(2,2)!(covariance)"<<endl;
				exit(1);
			}
			float det2 = (cov.at<float>(0,0)*cov.at<float>(1,1) -\
							cov.at<float>(0,1)*cov.at<float>(1,0));
			cv::Mat diff = x-mu;
			float result = 1.0/(2.0*M_PI*std::sqrt(det2))*std::exp(-0.5*\
				diff.dot(diff*cov.inv()));
			diff.release();
			return result;
			break;
		case (gaussianProcess::GAUSSnD):
			if(x.cols<2 || x.rows!=1){
				cerr<<"GaussianProcess GAUSSnD distribution: size(x)=(1,n)!(a row)"<<endl;
				exit(1);
			}
			if(mu.cols<2 || mu.rows!=1){
				cerr<<"GaussianProcess GAUSSnD distribution: size(mu)=(1,n)!(a row)"<<endl;
				exit(1);
			}
			// IF NO DECOMPOSITION WAS DONE, DO THAT
			if(!this->chlsky.checkDecomposition()){
				this->chlsky.decompose(cov);
			}
			cv::Mat diff = x-mu;
			cv::Mat inv;
			this->chlsky.inverse(inv);
			float result = 1.0/(std::pow((2.0*M_PI),(x.cols/2.0))*\
				exp(0.5*this->chlsky.logDet()))*exp(-0.5*diff.dot(diff*inv));
			inv.release();
			diff.release();
			return result;
			break;
		case (gaussianProcess::LOGGAUSSnD):
			if(x.cols<2 || x.rows!=1){
				cerr<<"GaussianProcess LOGGAUSS2D distribution: size(x)=(1,n)!(a row)"<<endl;
				exit(1);
			}
			if(mu.cols<2 || mu.rows!=1){
				cerr<<"GaussianProcess LOGGAUSS2D distribution: size(mu)=(1,n)!(a row)"<<endl;
				exit(1);
			}
			// IF NO DECOMPOSITION WAS DONE, DO THAT
			if(!this->chlsky.checkDecomposition()){
				this->chlsky.decompose(cov);
			}
			cv::Mat diff = x-mu;
			cv::Mat inv;
			this->chlsky.inverse(inv);
			float result = -0.5*(this->chlsky.n*std::log(2.0*M_PI)+this->chlsky.logDet()+\
				diff.dot(diff*inv));
			inv.release();
			diff.release();
			return result;
			break;
	}
}
//==============================================================================


