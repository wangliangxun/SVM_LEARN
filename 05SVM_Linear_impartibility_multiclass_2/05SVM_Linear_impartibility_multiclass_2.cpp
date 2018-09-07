#include ¡°stdafx.h¡±  
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
  
using namespace cv;  
using namespace std;  
  
int main()  
{  
  
    //¡ª¡ª¡ª¡ª¡ª¡ª¡ª 1. Set up training data randomly ¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª  
    Mat trainData(100, 3, CV_32FC1);  
    Mat labels   (100, 1, CV_32FC1);  
  
    RNG rng(100); // Random value generation class  
  
    // Generate random points for the class 1  
    Mat trainClass = trainData.rowRange(0, 40);  
    // The x coordinate of the points is in [0, 0.4)  
    Mat c = trainClass.colRange(0, 1);  
    rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(0.4 * 100));  
    // The y coordinate of the points is in [0, 0.4)  
    c = trainClass.colRange(1, 2);  
    rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(0.4 * 100));  
    // The z coordinate of the points is in [0, 0.4)  
    c = trainClass.colRange(2, 3);  
    rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(0.4 * 100));  
  
    // Generate random points for the class 2  
    trainClass = trainData.rowRange(60, 100);  
    // The x coordinate of the points is in [0.6, 1]  
    c = trainClass.colRange(0, 1);  
    rng.fill(c, RNG::UNIFORM, Scalar(0.6*100), Scalar(100));  
    // The y coordinate of the points is in [0.6, 1)  
    c = trainClass.colRange(1, 2);  
    rng.fill(c, RNG::UNIFORM, Scalar(0.6*100), Scalar(100));  
     // The z coordinate of the points is in [0.6, 1]  
    c = trainClass.colRange(2, 3);  
    rng.fill(c, RNG::UNIFORM, Scalar(0.6*100), Scalar(100));  
  
      
  
    // Generate random points for the classes 3  
    trainClass = trainData.rowRange(  40, 60);  
    // The x coordinate of the points is in [0.4, 0.6)  
    c = trainClass.colRange(0,1);  
    rng.fill(c, RNG::UNIFORM, Scalar(0.4*100), Scalar(0.6*100));  
    // The y coordinate of the points is in [0.4, 0.6)  
    c = trainClass.colRange(1,2);  
    rng.fill(c, RNG::UNIFORM, Scalar(0.4*100), Scalar(0.6*100));  
    // The z coordinate of the points is in [0.4, 0.6)  
    c = trainClass.colRange(2,3);  
    rng.fill(c, RNG::UNIFORM, Scalar(0.4*100), Scalar(0.6*100));  
  
  
  
    //¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª- Set up the labels for the classes ¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª  
    labels.rowRange( 0,  40).setTo(1);  // Class 1  
    labels.rowRange(60, 100).setTo(2);  // Class 2  
    labels.rowRange(40, 60).setTo(3);  // Class 3  
  
  
    //¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª 2. Set up the support vector machines parameters ¡ª¡ª¡ª¡ª¡ª¡ª¨C  
    CvSVMParams params;  
    params.svm_type    = SVM::C_SVC;  
    params.C           = 0.1;  
    params.kernel_type = SVM::LINEAR;  
    params.term_crit   = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);  
  
    //¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª 3. Train the svm ¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª-  
    cout << ¡±Starting training process¡± << endl;  
    CvSVM svm;  
    svm.train(trainData, labels, Mat(), Mat(), params);  
    cout << ¡±Finished training process¡± << endl;  
  
     Mat sampleMat = (Mat_<float>(1,3) << 50, 50,10);  
     float response = svm.predict(sampleMat);  
     cout<<response<<endl;  
  
     sampleMat = (Mat_<float>(1,3) << 50, 50,100);  
     response = svm.predict(sampleMat);  
     cout<<response<<endl;  
  
     sampleMat = (Mat_<float>(1,3) << 50, 50,60);  
     response = svm.predict(sampleMat);  
     cout<<response<<endl;  
      
    waitKey(0);  
}  