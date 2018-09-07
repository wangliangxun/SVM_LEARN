
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <CTYPE.H>  
 
#define	NTRAINING_SAMPLES	100			// Number of training samples per class
#define FRAC_LINEAR_SEP		0.9f	    // Fraction of samples which compose the linear separable part
 
using namespace cv;
using namespace std;
 
int main(int argc, char* argv[])  
{  
	int size = 400; // height and widht of image  
	const int s = 1000; // number of data  
	int i, j,sv_num;  
	IplImage* img;  
 
	CvSVM svm ;  
	CvSVMParams param;  
	CvTermCriteria criteria; // ֹͣ������׼  
	CvRNG rng = cvRNG();  
	CvPoint pts[s]; // ����1000����  
	float data[s*2]; // �������  
	int res[s]; // ������  
 
	CvMat data_mat, res_mat;  
	CvScalar rcolor;  
 
	const float* support;  
 
	// ͼ������ĳ�ʼ��  
	img = cvCreateImage(cvSize(size,size),IPL_DEPTH_8U,3);  
	cvZero(img);  
 
	// ѧϰ���ݵ�����  
	for (i=0; i<s;++i)  
	{  
		pts[i].x = cvRandInt(&rng)%size;  
		pts[i].y = cvRandInt(&rng)%size;  
 
		if (pts[i].y>50*cos(pts[i].x*CV_PI/100)+200)  
		{  
			cvLine(img,cvPoint(pts[i].x-2,pts[i].y-2),cvPoint(pts[i].x+2,pts[i].y+2),CV_RGB(255,0,0));  
			cvLine(img,cvPoint(pts[i].x+2,pts[i].y-2),cvPoint(pts[i].x-2,pts[i].y+2),CV_RGB(255,0,0));  
			res[i]=1;  
		}  
		else  
		{  
			if (pts[i].x>200)  
			{  
				cvLine(img,cvPoint(pts[i].x-2,pts[i].y-2),cvPoint(pts[i].x+2,pts[i].y+2),CV_RGB(0,255,0));  
				cvLine(img,cvPoint(pts[i].x+2,pts[i].y-2),cvPoint(pts[i].x-2,pts[i].y+2),CV_RGB(0,255,0));  
				res[i]=2;  
			}  
			else  
			{  
				cvLine(img,cvPoint(pts[i].x-2,pts[i].y-2),cvPoint(pts[i].x+2,pts[i].y+2),CV_RGB(0,0,255));  
				cvLine(img,cvPoint(pts[i].x+2,pts[i].y-2),cvPoint(pts[i].x-2,pts[i].y+2),CV_RGB(0,0,255));  
				res[i]=3;  
			}  
		}  
	}  
 
	// ѧϰ���ݵ���ʵ  
	cvNamedWindow("SVMѵ�������ռ估����",CV_WINDOW_AUTOSIZE);  
	cvShowImage("SVMѵ�������ռ估����",img);  
	cvWaitKey(0);  
 
	// ѧϰ����������  
	for (i=0;i<s;++i)  
	{  
		data[i*2] = float(pts[i].x)/size;  
		data[i*2+1] = float(pts[i].y)/size;  
	}  
 
	cvInitMatHeader(&data_mat,s,2,CV_32FC1,data);  
	cvInitMatHeader(&res_mat,s,1,CV_32SC1,res);  
	criteria = cvTermCriteria(CV_TERMCRIT_EPS,1000,FLT_EPSILON);  
	param = CvSVMParams(CvSVM::C_SVC,CvSVM::RBF,10.0,8.0,1.0,10.0,0.5,0.1,NULL,criteria);  
 
	svm.train(&data_mat,&res_mat,NULL,NULL,param);  
 
	// ѧϰ�����ͼ  
	for (i=0;i<size;i++)  
	{  
		for (j=0;j<size;j++)  
		{  
			CvMat m;  
			float ret = 0.0;  
			float a[] = {float(j)/size,float(i)/size};  
			cvInitMatHeader(&m,1,2,CV_32FC1,a);  
			ret = svm.predict(&m);  
 
			switch((int)ret)  
			{  
			case 1:  
				rcolor = CV_RGB(100,0,0);  
				break;  
			case 2:  
				rcolor = CV_RGB(0,100,0);  
				break;  
			case 3:  
				rcolor = CV_RGB(0,0,100);  
				break;  
			}  
			cvSet2D(img,i,j,rcolor);  
		}  
	}  
 
 
	// Ϊ����ʾѧϰ�����ͨ��������ͼ��������������أ��������������з��࣬Ȼ��������������������ɫ�ȼ�����ɫ��ͼ  
	for(i=0;i<s;++i)  
	{  
		CvScalar rcolor;  
		switch(res[i])  
		{  
		case 1:  
			rcolor = CV_RGB(255,0,0);  
			break;  
		case 2:  
			rcolor = CV_RGB(0,255,0);  
			break;  
		case 3:  
			rcolor = CV_RGB(0,0,255);  
			break;  
		}  
		cvLine(img,cvPoint(pts[i].x-2,pts[i].y-2),cvPoint(pts[i].x+2,pts[i].y+2),rcolor);  
		cvLine(img,cvPoint(pts[i].x+2,pts[i].y-2),cvPoint(pts[i].x-2,pts[i].y+2),rcolor);             
	}  
 
	// ֧�������Ļ���  
	sv_num = svm.get_support_vector_count();  
	for (i=0; i<sv_num;++i)  
	{  
		support = svm.get_support_vector(i);  
		cvCircle(img,cvPoint((int)(support[0]*size),(int)(support[i]*size)),5,CV_RGB(200,200,200));  
	}  
 
	cvNamedWindow("SVM",CV_WINDOW_AUTOSIZE);  
	cvShowImage("SVM��������֧������",img);  
	cvWaitKey(0);  
	cvDestroyWindow("SVM");  
	cvReleaseImage(&img);  
	return 0;  
}  