#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include<vector>
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
	if (startX < 0) startX = 0;

	if (startY < 0) startY = 0;

	if (endX > result.cols - 1) endX = result.cols - 1;

	if (endY > result.rows - 1)	endY = result.rows - 1;

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

bool AddPoint(vector<Point2i> &TargetPoint, Point newPoint, int templatW, int templatH)
{
	for (int j = TargetPoint.size() - 1; j >= 0; --j)//ȥ�����ڵ㣨Ŀ��ܿ����߽�ʱ,�����Ͳ�����ƥ�䶼����Ӧ��
	{
		if (abs(TargetPoint[j].x - newPoint.x) < templatW && abs(TargetPoint[j].y - newPoint.y) < templatH) //��Ĺ�ϵ
			return 0;
	}

	TargetPoint.push_back(newPoint);
	return 1;
}

bool NewMatchTemplate(Mat src, Mat templat, Mat result)
{

}

bool MyTemplateMatch(Mat src, Mat templat, vector<Point2i> &TargetPoint, Point offset)  //�ҵ�����1��û�ҵ�����0
{
	if (src.cols < templat.cols || src.rows < templat.rows)
	{
		cout << "ģ�岻�ܱ�ԭͼС" << endl;
		cvWaitKey(0);
		return 0;
	}

	Mat result;
	int resultW, resultH;
	resultW = src.cols - templat.cols + 1;
	resultH = src.rows - templat.rows + 1;

	result.create(resultW, resultH, CV_32FC1);    //  ƥ�䷽������Ľ����СֵΪfloat��CV_32FC1��
	matchTemplate(src, templat, result, CV_TM_SQDIFF_NORMED);  //����ƥ�亯��

	double minValue, maxValue;
	Point minLoc, maxLoc;

	minMaxLoc(result, &minValue, &maxValue, &minLoc, &maxLoc, Mat());
	if (minValue > 0.2) return 0; //û�ҵ�

								  //TargetPoint.push_back(minLoc+offset);
	AddPoint(TargetPoint, minLoc + offset, templat.cols, templat.rows);

	Point new_minLoc;

	// ������һ����Сֵ  
	new_minLoc = getNextMinLoc(result, minLoc, maxValue, templat.cols, templat.rows);
	while (result.at<float>(new_minLoc.y, new_minLoc.x) < 0.9*minValue + 0.1*maxValue)
	{
		//TargetPoint.push_back(new_minLoc + offset);
		AddPoint(TargetPoint, new_minLoc + offset, templat.cols, templat.rows);
		new_minLoc = getNextMinLoc(result, new_minLoc, maxValue, templat.cols, templat.rows);
	}

	return 1;
}



int main()
{
	Mat src0, srcResult, templat, src, result; // result������Ž����src0Ϊԭͼ��srcΪ��չ�߽��ͼ��
	char filename[100];
	//srcResult = imread("C:\\Users\\Mark\\Desktop\\�����ز�\\data1\\0.png", 1);  //������ʾ 
	templat = imread("C:\\Users\\Mark\\Desktop\\�����ز�\\data1\\mold\\mold.png", 0);

	for (unsigned int i = 0; i <= 12; ++i)
	{
		sprintf(filename, "C:\\Users\\Mark\\Desktop\\�����ز�\\data1\\%d.png", i);
		src = imread(filename, 0);

		if (src.empty() || templat.empty())
		{
			cout << "��ͼƬʧ��" << endl;
			cvWaitKey(0);
			return 0;
		}

		//double t = (double)getTickCount();;  //��������ʱ��

		//copyMakeBorder(src0, src, 0, 0, templat.cols, templat.cols, BORDER_CONSTANT, Scalar(0,0,0)); //��չ��ƥ���ͼ�񣬱���ͼ�񹤼��������ҽ��룬������չͼ������
		srcResult = src.clone(); //�鿴�����ͼ��


		vector<Point2i> TargetPoint;
		Point minLoc;

		MyTemplateMatch(src, templat, TargetPoint, Point(0, 0));


		int leftTemW, rightTemW;
		leftTemW = rightTemW = templat.cols / 2;

		Rect leftTempRect(0, 0, leftTemW, templat.rows - 1);
		Mat leftTemplat(templat, leftTempRect);
		Rect rightSrcRect(src.cols - templat.cols - 1, 0, templat.cols, src.rows - 1);
		Mat rightSrc(src, rightSrcRect);
		MyTemplateMatch(rightSrc, leftTemplat, TargetPoint, Point(src.cols - templat.cols - 1, 0));

		Rect rightTempRect(templat.cols - rightTemW - 1, 0, rightTemW, templat.rows - 1);
		Mat rightTemplat(templat, rightTempRect);
		Rect leftSrcRect(0, 0, templat.cols, src.rows - 1);
		Mat leftSrc(src, leftSrcRect);
		MyTemplateMatch(leftSrc, rightTemplat, TargetPoint, Point(0 - (templat.cols - rightTemW - 1), 0));


		for (int j = TargetPoint.size() - 1; j >= 0; --j)//����Ŀ����Ӧ�ľ���
		{
			minLoc = TargetPoint[j];
			rectangle(srcResult, minLoc, Point(minLoc.x + templat.cols, minLoc.y + templat.rows), cvScalar(0, 0, 255));
		}

		TargetPoint.clear();

		imshow(filename, srcResult);
		imwrite("a.png", srcResult);
		cvWaitKey(0);
		//imshow("template", templat);

	}



	return 0;
}