#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <windows.h>

using namespace std;
using namespace cv;

bool wxyMatchTemplate(Mat src, Mat templat, Mat result);
Point getNextMinLoc(Mat result, Point minLoc, int maxVaule, int templatW, int templatH);
bool AddPoint(vector<Point2i> &TargetPoint, Point newPoint, int templatW, int templatH);
void wxyIntegral(Mat source, Mat integral);
void pickPoints(Mat tempIntegral, vector<Point2i> &calculatePoint, Point center);


float accum(Mat srcIntegral, Mat tempIntegral, Point calPoint) //calPoint为要计算的点,以积分图为基准坐标（多一行列）
{
	int tempW = tempIntegral.cols-1;
	int tempH = tempIntegral.rows-1;

	int *tempLine, *srcLine;
	float *ratioLine;
	int *firstLineSrc = srcIntegral.ptr<int>(calPoint.y - 1);
	int pointInte = 0;
	int orignInte = srcIntegral.at<int>(calPoint.y-1, calPoint.x-1);
	float sum = 0;
	float accum = 0;

	Mat ratio(tempH, tempW, CV_32FC1);

	for (int y = 0; y < tempH; y++) {//点calPoint是包含在待计算图像中的左上角点
		tempLine = tempIntegral.ptr<int>(y + 1);
		srcLine = srcIntegral.ptr<int>(calPoint.y + y);
		ratioLine = ratio.ptr<float>(y);
		for (int x = 0; x < tempW; x++) {
			pointInte = srcLine[calPoint.x + x] + orignInte - firstLineSrc[calPoint.x + x] - srcLine[calPoint.x - 1];
			//if (tempLine[x + 1] == 0) continue;

			ratioLine[x] = (float)(pointInte + 1) / (float)(tempLine[x + 1] + 1);
			sum += ratioLine[x];
		}
	}

	float mean = sum / (tempH*tempW);

	for (int m = 0; m < tempH; m++) {
		ratioLine = ratio.ptr<float>(m);
		for (int n = 0; n < tempW; n++) {
			accum += (ratioLine[n] - mean)*(ratioLine[n] - mean);
		}
	}

	return sqrt(accum / (tempH*tempW - 1));
}



bool wxyMatchTemplate(Mat src, Mat templat, Mat result)
{
	Mat tempIntegral(templat.rows+1, templat.cols+1, CV_32SC1);
	Mat srcIntegral(src.rows + 1, src.cols + 1, CV_32SC1);
	wxyIntegral(templat, tempIntegral);	
	wxyIntegral(src, srcIntegral);
	
	int tempW = templat.cols;
	int tempH = templat.rows;

	int sumX = 0, sumY = 0;
	int count = 0;
	int meanX = 0, meanY = 0;

	for (int a = 0; a < tempH; a++) {
		uchar *p = templat.ptr<uchar>(a);
		for (int b = 0; b < tempW; b++) {
			if (p[b] < 120){
				sumX += b; sumY += a;
				count++;
			}
		}
	}
	meanX = sumX / count;
	meanY = sumY / count;
	
	vector<Point2i> calculatePoint;
	calculatePoint.push_back(Point2i(tempW - 1, tempH - 1));//第一点固定为右下角点。点的坐标以原图为准，不同于积分后扩大1个像素的图

	pickPoints(tempIntegral, calculatePoint,Point(meanX, meanY));
	int pointsNum = calculatePoint.size();

	float tempPoint[50];

	for (int m = 0; m < pointsNum; m++) {
		tempPoint[m] = tempIntegral.at<int>(calculatePoint[m].y + 1, calculatePoint[m].x + 1);
	}


	float *p;
	int resultW = result.cols;
	int resultH = result.rows;
	float sum = 0.0;
	float mean = 0.0;
	bool pass = 0;
	bool findFlag = 0;
	float pointIntegral = 0;
	float ratio[50];//存放选出点对应位置图像和模板像素和的比
	for (int y = 1; y <= resultH; y++)  //遍历src中（M-m）*(N-n)
	{
		p = result.ptr<float>(y-1);
		for (int x = 1; x <= resultW; x++)
		{	
			sum = 0;
			mean = 0;
			pass = 0;
			for (int i = 0; i < pointsNum; i++) {
				pointIntegral = srcIntegral.at<int>(y + calculatePoint[i].y, x + calculatePoint[i].x) + srcIntegral.at<int>(y - 1, x - 1) - srcIntegral.at<int>(y + calculatePoint[i].y, x - 1) - srcIntegral.at<int>(y - 1, x + calculatePoint[i].x);
				ratio[i] = pointIntegral / tempPoint[i];//上式点（x,y）是包含在待计算图像中的左上角点
				if (ratio[i] > 1.2 || ratio[i] < 0.8) {
					pass = 1;
					break;
				}
				sum += ratio[i];
				
				/*if (mean > 1.05 || mean < 0.95){ 
					pass = 1; 
					break; 
				}*/
			}

			if (pass)   continue;
			findFlag = 1;
			p[x - 1] = accum(srcIntegral, tempIntegral, Point(x, y));

			/*mean = sum / pointsNum;
			float accum = 0.0;
			for (int j = 0; j < pointsNum; j++) {
				accum += (ratio[j] - mean)*(ratio[j] - mean);
			}
			p[x-1] = sqrt(accum / (pointsNum - 1));//方差*/
		}
	}
	if (findFlag)
		return 1;
	else
		return 0;
	
}



bool MyTemplateMatch(Mat src, Mat templat, vector<Point2i> &TargetPoint, Point offset)  //找到返回1，没找到返回0
{
	if (src.cols < templat.cols || src.rows < templat.rows)
	{
		cout << "模板不能比原图小" << endl;
		cvWaitKey(0);
		return 0;
	}

	
	int resultW, resultH;
	resultW = src.cols - templat.cols + 1;
	resultH = src.rows - templat.rows + 1;

	Mat result = Mat::ones(resultH, resultW, CV_32FC1);    //  匹配方法计算的结果最小值为float（CV_32FC1）,将result全设为1，以便和标准差比较

	//matchTemplate(src, templat, result, CV_TM_SQDIFF_NORMED);  //核心匹配函数
	if(!wxyMatchTemplate(src, templat, result)) return 0;

	double minValue, maxValue;
	Point minLoc, maxLoc;

	minMaxLoc(result, &minValue, &maxValue, &minLoc, &maxLoc, Mat());

	AddPoint(TargetPoint, minLoc + offset, templat.cols, templat.rows);

	Point new_minLoc;

	// 计算下一个最小值  
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
	Mat src0, srcResult, templat, src, result; // result用来存放结果，src0为原图像，src为扩展边界后图像
	char filename[100];
	//srcResult = imread("C:\\Users\\Mark\\Desktop\\测试素材\\data1\\0.png", 1);  //用来显示 
	templat = imread("C:\\Users\\Mark\\Desktop\\测试素材\\data3\\mold\\mold.png", 0);
	

	for (unsigned int i = 0; i <= 16; ++i)
	{
		sprintf(filename, "C:\\Users\\Mark\\Desktop\\测试素材\\data3\\%d.png", i);//"C:\\Users\\Mark\\Desktop\\原图像.png"
		src = imread(filename, IMREAD_GRAYSCALE);

		if (src.empty() || templat.empty())
		{
			cout << "打开图片失败" << endl;
			cvWaitKey(0);
			return 0;
		}

		//double t = (double)getTickCount();;  //测试运行时间

		//copyMakeBorder(src0, src, 0, 0, templat.cols, templat.cols, BORDER_CONSTANT, Scalar(0,0,0)); //扩展待匹配的图像，本批图像工件从左向右进入，所以扩展图像左右
		srcResult = src.clone(); //查看结果的图像


		vector<Point2i> TargetPoint;
		Point minLoc;


		MyTemplateMatch(src, templat, TargetPoint, Point(0, 0));


		int leftTemW, rightTemW;
		leftTemW = rightTemW = templat.cols / 2;

		/*Rect leftTempRect(0, 0, leftTemW, templat.rows - 1);
		Mat leftTemplat(templat, leftTempRect);
		Rect rightSrcRect(src.cols - templat.cols - 1, 0, templat.cols, src.rows - 1);
		Mat rightSrc(src, rightSrcRect);
		MyTemplateMatch(rightSrc, leftTemplat, TargetPoint, Point(src.cols - templat.cols - 1, 0));

		Rect rightTempRect(templat.cols - rightTemW - 1, 0, rightTemW, templat.rows - 1);
		Mat rightTemplat(templat, rightTempRect);
		Rect leftSrcRect(0, 0, templat.cols, src.rows - 1);
		Mat leftSrc(src, leftSrcRect);
		MyTemplateMatch(leftSrc, rightTemplat, TargetPoint, Point(0 - (templat.cols - rightTemW - 1), 0));*/


		for (int j = TargetPoint.size() - 1; j >= 0; --j)//画出目标点对应的矩形
		{
			minLoc = TargetPoint[j];
			rectangle(srcResult, minLoc, Point(minLoc.x + templat.cols, minLoc.y + templat.rows), cvScalar(0, 0, 255));
		}

		TargetPoint.clear();

		imshow(filename, srcResult);
		imwrite("a.png", srcResult);
		cvWaitKey(0);

	}

	return 0;
}

Point getNextMinLoc(Mat result, Point minLoc, int maxVaule, int templatW, int templatH)
{

	// 先将第一个最小值点附近两倍模板宽度和高度的都设置为最大值防止产生干扰  
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
	// 然后得到下一个最小值并且返回  
	double new_minVaule, new_maxValue;
	Point new_minLoc, new_maxLoc;
	minMaxLoc(result, &new_minVaule, &new_maxValue, &new_minLoc, &new_maxLoc, Mat());
	return new_minLoc;
}

bool AddPoint(vector<Point2i> &TargetPoint, Point newPoint, int templatW, int templatH)
{
	for (int j = TargetPoint.size() - 1; j >= 0; --j)//去掉相邻点（目标很靠近边界时,完整和不完整匹配都会响应）
	{
		if (abs(TargetPoint[j].x - newPoint.x) < templatW && abs(TargetPoint[j].y - newPoint.y) < templatH) //与的关系
			return 0;
	}

	TargetPoint.push_back(newPoint);
	return 1;
}


void wxyIntegral(Mat source, Mat integral) //source和integral都是int型
{
	int sourceW = source.cols;
	int sourceH = source.rows;

	int *p = integral.ptr<int>(0);
	for (int a = 0; a <= sourceW; a++)  //第一行全为0
		p[a] = 0;

	for (int y = 0; y < sourceH; y++) {
		uchar *lineSrc = source.ptr<uchar>(y);
		int *intePre = integral.ptr<int>(y);
		int *inteNow = integral.ptr<int>(y+1);
		inteNow[0] = 0;
		int sum = 0;
		for (int x = 0; x < sourceW; x++) {			
			sum += (lineSrc[x] == 0) ? 1 : 0;
			inteNow[x + 1] = intePre[x + 1] + sum;
		}
	}
}

void pickPoints(Mat tempIntegral, vector<Point2i> &calculatePoint, Point center)//保证取得点，模板的积分图对应点不为零（除数）
{
	int tempW = tempIntegral.cols-1;
	int tempH = tempIntegral.rows-1;
	int pointNum = 2*log2(tempW*tempH);

	srand(GetTickCount());
	int halftempW = tempW / 2;
	int halftempH = tempH / 2;
	int product = center.x*center.y;

	cout << center << endl;

	int x, y;

	for (int i = 0; i < pointNum; i++) {
		do {
			x = rand() % tempW;
			y = rand() % tempH;

		} while (tempIntegral.at<int>(y, x) == 0 || x*y < product);

		calculatePoint.push_back(Point2i(x, y));
	}

}