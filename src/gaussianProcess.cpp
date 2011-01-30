/* gaussianProcess.cpp
 * Original code: Dr Gwenn Englebienne
 * Modified by: Silvia-Laura Pintea
 */
#include "gaussianProcess.h"
//==============================================================================
/** Generates a selected distribution of the functions given the parameters (the
 * mean: mu, the covariance: cov, the data x).
 */
double gaussianProcess::distribution(cv::Mat x,gaussianProcess::DISTRIBUTION distrib,
cv::Mat mu, cv::Mat cov, double a, double b, double s){
	double det2, result;
	cv::Mat diff;
	cv::Mat inv;

	switch(distrib){
		case (gaussianProcess::BETA):
			if(x.cols!=1 || x.rows!=1){
				std::cerr<<"GaussianProcess BETA distribution: size(x) = (1,1)!"<<std::endl;
				exit(1);
			}
			result = (gamma(a+b)*(std::pow(x.at<double>(0,0),(a-1.0)))*\
						(std::pow(1.0-x.at<double>(0,0),(b-1.0))))/(gamma(a)+gamma(b));
			break;
		case (gaussianProcess::GAUSS):
			if(x.cols!=1 || x.rows!=1){
				std::cerr<<"GaussianProcess GAUSS distribution: size(x) = (1,1)!"<<std::endl;
				exit(1);
			}
			if(mu.cols!=1 || mu.rows!=1){
				std::cerr<<"GaussianProcess GAUSS distribution: size(mu) = (1,1)!(mean)"<<std::endl;
				exit(1);
			}
			result = std::exp(-std::pow((x.at<double>(0,0)-mu.at<double>(0,0)),2)/\
						(2.0*std::pow(s,2)))/(std::sqrt(2.0*M_PI)*s);
			break;
		case (gaussianProcess::GAUSS2D):
			if(x.cols!=2 || x.rows!=1){
				std::cerr<<"GaussianProcess GAUSS2D distribution: size(x)=(1,2)!(x.x,x.y)"<<std::endl;
				exit(1);
			}
			if(mu.cols!=2 || mu.rows!=1){
				std::cerr<<"GaussianProcess GAUSS2D distribution: size(mu)=(1,2)!(mu.x,mu.y)"<<std::endl;
				exit(1);
			}
			if(cov.cols!=2 || cov.rows!=2){
				std::cerr<<"GaussianProcess GAUSS2D distribution: size(cov)=(2,2)!(covariance)"<<std::endl;
				exit(1);
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
				exit(1);
			}
			if(mu.cols<2 || mu.rows!=1){
				std::cerr<<"GaussianProcess GAUSSnD distribution: size(mu)=(1,n)!(a row)"<<std::endl;
				exit(1);
			}
			// IF NO DECOMPOSITION WAS DONE, DO THAT
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
				exit(1);
			}
			if(mu.cols<2 || mu.rows!=1){
				std::cerr<<"GaussianProcess LOGGAUSS2D distribution: size(mu)=(1,n)!(a row)"<<std::endl;
				exit(1);
			}
			// IF NO DECOMPOSITION WAS DONE, DO THAT
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
}
//==============================================================================
/** Trains the Gaussian process.
 */
void gaussianProcess::train(cv::Mat_<double> X, cv::Mat_<double> y,\
double (gaussianProcess::*fFunction)(cv::Mat, cv::Mat, double),double sigmasq){
	if(y.rows != X.rows){
		std::cerr<<"In Gaussian Process - train: X and y need to be defined for the"<<\
			" same number of points"<<std::endl;
		exit(1);
	}

	this->kFunction = fFunction;
	this->N         = X.rows; // NUMBER OF TRAINING DATA POINTS!
	this->data      = X.clone();

	// BUILD THE KERNEL MARIX K: K(i,j) = k(x[i],x[j])
	cv::Mat_<double> K = cv::Mat::zeros(cv::Size(this->N,this->N),\
						 cv::DataType<double>::type);
	for(unsigned indy=0; indy<this->N; indy++){
		for(unsigned indx=0; indx<this->N; indx++){
			K.at<double>(indy,indx) = (this->*kFunction)(X.row(indy),\
										X.row(indx), 1.0);
		}
	}

	// ADD sigma^2 TO THE KERNEL MATRIX, K
	for(unsigned indy=0; indy<this->N; indy++){
		K.at<double>(indy,indy) += sigmasq;
	}

	if(!this->chlsky.checkDecomposition()){
		this->chlsky.decomposeCov(K);
	}
	this->chlsky.solve(y, this->alpha);

	std::cout<<"N: ("<<this->N<<")"<<std::endl;
	std::cout<<"size of alpha: ("<<this->alpha.cols<<","<<this->alpha.rows<<")"<<std::endl;
	std::cout<<"size of data: ("<<this->data.cols<<","<<this->data.rows<<")"<<std::endl;
	K.release();
}
//==============================================================================
/** Returns the prediction for the test data, x (only one test data point).
 */
void gaussianProcess::predict(cv::Mat x, gaussianProcess::prediction &predi){
	cv::Mat kstar(this->data.rows, 1, cv::DataType<double>::type);

	for(unsigned indy=0; indy<this->N; indy++){
		kstar.at<double>(indy,0) = (this->*kFunction)(this->data.row(indy),x,1.0);
	}

	for(unsigned i=0; i<this->alpha.cols; i++){
		predi.mean.push_back(kstar.dot(this->alpha.col(i)));
	}

	cv::Mat v;
	this->chlsky.solveL(kstar,v);
	predi.variance.push_back((this->*kFunction)(x,x,1.0) - v.dot(v));
	kstar.release();
}
//==============================================================================
/** Samples the process that generates the inputs.
 */
void gaussianProcess::sample(cv::Mat inputs, cv::Mat &smpl){
	cv::Mat Kxstarx(this->N, inputs.cols, cv::DataType<double>::type);
	cv::Mat Kxstarxstar(inputs.cols, inputs.cols, cv::DataType<double>::type);

	for(unsigned indy=0; indy<inputs.cols; indy++){
		for(unsigned indx=0; indx<this->N; indx++){
			Kxstarx.at<double>(indy,indx) = (this->*kFunction)(inputs.row(indy),\
											this->data.row(indx),1.0);
		}
		for(unsigned indx=0; indx<inputs.cols; indx++){
			Kxstarxstar.at<double>(indy,indx) = (this->*kFunction)(inputs.row(indy),\
												inputs.row(indx),1.0);
		}
	}

	cv::Mat Kxxstar = Kxstarx.t();
	cv::Mat mu      = Kxstarx * this->alpha;
	cv::Mat inv;
	this->chlsky.inverse(inv);
	cv::Mat cov     = Kxstarxstar-(Kxstarx*inv)*Kxxstar;

	for(unsigned indy=0; indy<cov.cols; indy++){
		cov.at<double>(indy,indy) += 1.0e-6;
	}
	this->sampleGaussND(mu, cov, smpl);

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
void gaussianProcess::sampleGaussND(cv::Mat mu, cv::Mat cov, cv::Mat &smpl){
	if(!this->chlsky.checkDecomposition()){
		this->chlsky.decomposeCov(cov);
	}

	for(unsigned indy=0; indy<mu.cols; indy++){
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

		double x, w =u*u+v*v;
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
void gaussianProcess::sampleGPPrior(double (gaussianProcess::*fFunction)(cv::Mat,\
cv::Mat, double), cv::Mat inputs, cv::Mat &smpl){
	this->kFunction = fFunction;
	cv::Mat mu;
	cv::Mat cov;

	for(unsigned indy=0; indy<inputs.cols; indy++){
		mu.at<double>(indy,0) = 0.0;
		for(unsigned indx=0; indx<inputs.cols; indx++){
			cov.at<double>(indy,indx) = (this->*kFunction)(inputs.row(indy),\
										inputs.row(indx),1.0);
		}
	}

	for(unsigned indy=0; indy<inputs.cols; indy++){
		cov.at<double>(indy,indy) += 1.0e-6;
	}
	this->sampleGaussND(mu, cov, smpl);
	mu.release();
	cov.release();
}
//==============================================================================
// Squared exponential kernel function.
double gaussianProcess::sqexp(cv::Mat x1, cv::Mat x2, double l){
	cv::Mat diff = x1-x2;
	double result = std::exp(-1.0 * diff.dot(diff)/(2.0*l));
	diff.release();
	return result;
}
//==============================================================================
// Matern05 kernel function.
double gaussianProcess::matern05(cv::Mat x1, cv::Mat x2, double l){
	cv::Mat diff = x1-x2;
	double result = std::sqrt(diff.dot(diff));
	diff.release();
	return std::exp(-1.0 * result/l);
}
//==============================================================================
// Exponential Covariance kernel function.
double gaussianProcess::expCovar(cv::Mat x1, cv::Mat x2, double l){
	cv::Mat diff = x1-x2;
	double result = std::sqrt(diff.dot(diff));
	diff.release();
	return std::exp(-1.0 * result/l);
}
//==============================================================================
// Matern15 kernel function.
double gaussianProcess::matern15(cv::Mat x1, cv::Mat x2, double l){
	cv::Mat diff = x1-x2;
	double result = std::sqrt(diff.dot(diff));
	diff.release();
	return (1.0 + std::sqrt(3.0)*result/l) * std::exp(-1.0 * std::sqrt(3.0)*result/l);
}
//==============================================================================
// Matern25 kernel function.
double gaussianProcess::matern25(cv::Mat x1, cv::Mat x2, double l){
	cv::Mat diff = x1-x2;
	double result = std::sqrt(diff.dot(diff));
	diff.release();
	return (1.0 + std::sqrt(5.0)*result/l + (5.0*result*result)/(3.0*l*l))*\
		std::exp(-1.0 * std::sqrt(5.0)*result/l);
}
//==============================================================================
/*
int main(){
	cv::Mat test(10, 100, cv::DataType<double>::type);
	cv::Mat train(100, 100, cv::DataType<double>::type);
	cv::Mat targets = cv::Mat::zeros(100, 2, cv::DataType<double>::type);
	cv::Mat ttargets = cv::Mat::zeros(10, 2, cv::DataType<double>::type);
	train = cv::Mat::zeros(100, 100, cv::DataType<double>::type);

	for(unsigned i=0; i<100; i++){
		cv::Mat stupid = train.row(i);
		cv::add(stupid, cv::Scalar(i*10.0), stupid);
		if(i<10){
			cv::Mat stupid2 = test.row(i);
			cv::add(stupid2, cv::Scalar((99.0-(i*10.0))*10.0), stupid2);
//			ttargets.at<double>(i,0) = (99.0-(i*10.0));
			ttargets.at<double>(i,0) = std::sin((99.0-(i*10.0))*M_PI/180.0);
			ttargets.at<double>(i,1) = std::cos((99.0-(i*10.0))*M_PI/180.0);
		}
//		targets.at<double>(i,0) = i;
		targets.at<double>(i,0) = std::sin(i*M_PI/180.0);
		targets.at<double>(i,1) = std::cos(i*M_PI/180.0);
	}

	gaussianProcess gp;
	gp.train(train, targets, &gaussianProcess::sqexp, 0.1);

	cv::Mat result;
	for(unsigned i=0; i<test.rows; i++){
		gaussianProcess::prediction predi;
		gp.predict(test.row(i), predi);
		//std::cout<<"label: "<<ttargets.at<double>(i,0)<<" "<<\
			predi.mean[0]<<" variance:"<<predi.variance[0]<<std::endl;
		std::cout<<"label: "<<ttargets.at<double>(i,0)<<" "<<\
				ttargets.at<double>(i,1)<<" "<<\
			" mean:"<<predi.mean[0]<<" "<<predi.mean[1]<<" variance:"<<\
			predi.variance[0]<<std::endl;
	}
}
*/
