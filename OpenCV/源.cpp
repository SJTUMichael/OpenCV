#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <windows.h>

using namespace std;
using namespace cv;


Point getNextMinLoc(Mat result, Point minLoc, int maxVaule, int templatW, int templatH)
{

	// �Ƚ���һ����Сֵ�㸽������ģ���Ⱥ͸߶ȵĶ�����Ϊ���ֵ��ֹ��������  
	int startX = minLoc.x - templatW;
	int startY = minLoc.y - templatH;
	int endX = minLoc.x + templatW;
	int endY = minLoc.y + templatH;
	if (startX < 0)
	{
		startX = 0;
	}
	if (startY < 0)
	{
		startY = 0;
	}
	if (endX > result.cols - 1)
	{
		endX = result.cols - 1;
	}
	if (endY > result.rows - 1)
	{
		endY = result.rows - 1;
	}
	int y, x;
	float *p;
	for (y = startY; y <= endY; y++)
	{
		p = result.ptr<float>(y);
		for (x = startX; x <= endX; x++)
		{
			p[x] = maxVaule;
		}
	}
	// Ȼ��õ���һ����Сֵ���ҷ���  
	double new_minVaule, new_maxValue;
	Point new_minLoc, new_maxLoc;
	minMaxLoc(result, &new_minVaule, &new_maxValue, &new_minLoc, &new_maxLoc, Mat());
	return new_minLoc;
}

int main()
{	Mat src0, srcResult, templat, src, result; // result������Ž����src0Ϊԭͼ��srcΪ��չ�߽��ͼ��
	char filename[100];

	for (unsigned int i = 0; i < 13; ++i)
	{
		sprintf(filename, "C:\\Users\\Mark\\Desktop\\�����ز�\\data5\\%d.png", i);
		src0 = imread(filename, 0);
		//srcResult = imread("C:\\Users\\Mark\\Desktop\\�����ز�\\data1\\0.png", 1);  //������ʾ 
		templat = imread("C:\\Users\\Mark\\Desktop\\�����ز�\\data5\\mold\\mold.png", 0);

		if (src0.empty() || templat.empty())
		{
			cout << "��ͼƬʧ��" << endl;
			cvWaitKey(0);
			return 0;
		}

		//double t = (double)getTickCount();;  //��������ʱ��

		copyMakeBorder(src0, src, 0, 0, templat.cols, templat.cols, BORDER_CONSTANT, Scalar(0)); //��չ��ƥ���ͼ�񣬱���ͼ�񹤼��������ҽ��룬������չͼ������
		srcResult = src.clone(); //�鿴�����ͼ��

		//t = (double)getTickCount() - t;
		//cout << "The run time is:" << (t * 1000 / getTickFrequency()) << "ms!" << endl;  //�������ʱ��

		int srcW, srcH, templatW, templatH, resultH, resultW;
		srcW = src.cols;
		srcH = src.rows;
		templatW = templat.cols;
		templatH = templat.rows;
		if (srcW < templatW || srcH < templatH) 
		{
			cout << "ģ�岻�ܱ�ԭͼС" << endl;
			cvWaitKey(0);
			return 0;
		}

		resultW = srcW - templatW + 1;
		resultH = srcH - templatH + 1;
		result.create(resultW, resultH, CV_32FC1);    //  ƥ�䷽������Ľ����СֵΪfloat��CV_32FC1��
		matchTemplate(src, templat, result, CV_TM_SQDIFF);  //����ƥ�亯��

		double minValue, maxValue;
		Point minLoc, maxLoc;

		minMaxLoc(result, &minValue, &maxValue, &minLoc, &maxLoc, Mat());
		if (minValue > 2e+8) continue; //����û��ģ���ͼƬ
		cout << result.at<float>(minLoc.y, minLoc.x) << endl;
		rectangle(srcResult, minLoc, Point(minLoc.x + templatW, minLoc.y + templatH), cvScalar(0, 0, 255));

		Point new_minLoc;

		// ������һ����Сֵ  
		new_minLoc = getNextMinLoc(result, minLoc, maxValue, templatW, templatH);
		while (result.at<float>(new_minLoc.y, new_minLoc.x) < 1.5*minValue)
		{
			cout << result.at<float>(new_minLoc.y, new_minLoc.x) << ";" << endl;
			rectangle(srcResult, new_minLoc, Point(new_minLoc.x + templatW, new_minLoc.y + templatH), cvScalar(0, 0, 255));
			new_minLoc = getNextMinLoc(result, new_minLoc, maxValue, templatW, templatH);
		}


		imshow(filename, srcResult);
		imshow("template", templat);
		
	}

	cvWaitKey(0);

	return 0;
}
