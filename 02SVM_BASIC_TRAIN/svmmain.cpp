/*****************************

//湖南长沙·2015-11-8

*************************/

#include <QtCore/QCoreApplication>

#include "opencv2\opencv.hpp"

 

using namespace cv;

 

//训练样本：红点、蓝点。待识别数据：绿点

int main(int argc, char *argv[])

{

    QCoreApplication a(argc, argv);

 

    // 用于保存可视化数据的矩阵

    Mat image = Mat::zeros(512, 512, CV_8UC3);

    for(int i=0;i<image.rows;i++){

        for(int j=0;j<image.cols;j++){

            if(i<image.rows/3)

                image.ptr<Vec3b>(i)[j]=Vec3b(255,0,0);

            else if(i<2*image.rows/3)

                image.ptr<Vec3b>(i)[j]=Vec3b(0,255,0);

            else

                image.ptr<Vec3b>(i)[j]=Vec3b(0,0,255);

        }

    }

 

    float labels[3] = {1.0, 2.0, 3.0};

    Mat labelsMat(3, 1, CV_32FC1, labels);

    float trainingData[3][3] = {{255,0,0}, {0,255,0}, {0,0,255}};

    Mat trainingDataMat(3, 3, CV_32FC1, trainingData);

    // 设置SVM参数

    CvSVMParams params;

    params.svm_type    = CvSVM::C_SVC;//C支持向量分类机,允许用异常值惩罚因子C进行不完全分类

    params.kernel_type = CvSVM::LINEAR;//使用线性内核

    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);//迭代训练过程的中止条件

    // 对SVM进行训练

    CvSVM SVM;

    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

 

    string str1="blue",str2="green",str3="red";

    for (int i = 100; i < image.rows; i+=image.rows/3){

        int j=10;

        int c0=image.ptr<Vec3b>(i)[j][0];

        int c1=image.ptr<Vec3b>(i)[j][1];

        int c2=image.ptr<Vec3b>(i)[j][2];

        Mat sampleMat = (Mat_<float>(1,3) << c0,c1,c2);

        float result = SVM.predict(sampleMat);//进行分类

        //结果处理

        if (result == 1.0)

            putText(image,str1,Point(j,i),FONT_HERSHEY_SIMPLEX,2,Scalar(255,255,255),2,8);

        else if(result == 2.0)

            putText(image,str2,Point(j,i),FONT_HERSHEY_SIMPLEX,2,Scalar(255,255,255),2,8);

        else if(result == 3.0)

            putText(image,str3,Point(j,i),FONT_HERSHEY_SIMPLEX,2,Scalar(255,255,255),2,8);

    }

 

    imshow("SVM分类", image);

    waitKey(0);

    return a.exec();

}