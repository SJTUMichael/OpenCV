#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

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
	for (y = startY; y <= endY; y++)
	{
		for (x = startX; x <= endX; x++)
		{
			result.at<float>(y,x) = maxVaule;
		}
	}
	// Ȼ��õ���һ����Сֵ���ҷ���  
	double new_minVaule, new_maxValue;
	Point new_minLoc, new_maxLoc;
	minMaxLoc(result, &new_minVaule, &new_maxValue, &new_minLoc, &new_maxLoc, Mat());
	return new_minLoc;

}
int main()
{
	Mat src = imread("ԭͼ��.png", 0);
	Mat srcResult = imread("ԭͼ��.png", 3);  //������ʾ  
	Mat templat = imread("ģ��.png", 0);
	Mat result;// ������Ž��  

	if (src.empty() || templat.empty())
	{
		cout << "��ͼƬʧ��" << endl;
		return 0;
	}
	int srcW, srcH, templatW, templatH, resultH, resultW;
	srcW = src.cols;
	srcH = src.rows;
	templatW = templat.cols;
	templatH = templat.rows;
	if (srcW < templatW || srcH < templatH)
	{
		cout << "ģ�岻�ܱ�ԭͼС" << endl;
		return 0;
	}
	resultW = srcW - templatW + 1;
	resultH = srcH - templatH + 1;
	result.create(resultW, resultH, CV_32FC1);    //  ƥ�䷽������Ľ����СֵΪfloat  
	matchTemplate(src, templat, result, CV_TM_SQDIFF);
	double minValue, maxValue;
	Point minLoc, maxLoc;
	minMaxLoc(result, &minValue, &maxValue, &minLoc, &maxLoc, Mat());
	rectangle(srcResult, minLoc, cvPoint(minLoc.x + templatW, minLoc.y + templatH), cvScalar(0, 0, 255));
	Point new_minLoc;

	// ������һ����Сֵ  
	new_minLoc = getNextMinLoc(result, minLoc, maxValue, templatW, templatH);
	while(result.at<float>(new_minLoc.y, new_minLoc.x) < 0.5*minValue+0.5*maxValue)
	{
		cout << new_minLoc.y << " , " << new_minLoc.x << endl;
		rectangle(srcResult, new_minLoc, cvPoint(new_minLoc.x + templatW, new_minLoc.y + templatH), cvScalar(0, 0, 255));
		new_minLoc = getNextMinLoc(result, new_minLoc, maxValue, templatW, templatH);
	}

	//cvRectangle(srcResult, new_minLoc, cvPoint(new_minLoc.x + templatW, new_minLoc.y + templatH), cvScalar(0, 0, 255));
	cvNamedWindow("srcResult", 0);
	cvNamedWindow("template", 0);
	imshow("srcResult", srcResult);
	imshow("template", templat);
	cvWaitKey(0);

	return 0;
}
