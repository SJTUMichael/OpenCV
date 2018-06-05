#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <windows.h>

using namespace std;
using namespace cv;

//bool wxyMatchTemplate(Mat src, Mat templat, Mat result);
Point getNextMinLoc(Mat result, Point minLoc, int maxVaule, int templatW, int templatH);
bool AddPoint(vector<Point2i> &TargetPoint, Point newPoint, int templatW, int templatH);
void wxyIntegral(Mat source, Mat integral);
void pickPoints(Mat templat, vector<Point2i> &calculatePoint);


float accum(Mat srcIntegral, Mat tempIntegral, Point calPoint) //calPoint为要计算的点,以积分图为基准坐标（多一行列）
{
	int tempW = tempIntegral.cols-1;
	int tempH = tempIntegral.rows-1;

	int *tempLine, *srcLine; //指向模板和原图像行的指针
	float *ratioLine;//指向比例存储矩阵行的指针
	int *firstLineSrc = srcIntegral.ptr<int>(calPoint.y - 1);//指向待检测点上一行指针
	int pointInte = 0;//该点原图像小矩形积分值
	int orignInte = srcIntegral.at<int>(calPoint.y-1, calPoint.x-1); //待检测点左上角点积分图对应值
	float sum = 0;//所有计入点比例之和
	float accum = 0;//方差

	Mat ratio(tempH, tempW, CV_32FC1);//用来存储计算比例结果

	for (int y = 0; y < tempH; y++) {//点calPoint是包含在待计算图像中的左上角点
		tempLine = tempIntegral.ptr<int>(y + 1);
		srcLine = srcIntegral.ptr<int>(calPoint.y + y);
		ratioLine = ratio.ptr<float>(y);
		for (int x = 0; x < tempW; x++) {
			//if (tempLine[x + 1] == 0) continue;//防止分母为零

			pointInte = srcLine[calPoint.x + x] + orignInte - firstLineSrc[calPoint.x + x] - srcLine[calPoint.x - 1];
			ratioLine[x] = (float)(pointInte + 1) / (float)(tempLine[x + 1] + 1);  //拉普拉斯平滑算法（哈哈）
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

float accumHMV(Mat srcIntegral, Mat tempIntegral, Point calPoint) //大于1/2工件像素值
{//calPoint为要计算的点,以积分图为基准坐标（多一行列）
	int tempW = tempIntegral.cols - 1;
	int tempH = tempIntegral.rows - 1;
	int halfMaxValue = tempIntegral.at<int>(tempH, tempW) / 2;//工件像素和的一半

	int *tempLine, *srcLine; //指向模板和原图像行的指针
	float *ratioLine;//指向比例存储矩阵行的指针
	int *firstLineSrc = srcIntegral.ptr<int>(calPoint.y - 1);//指向待检测点上一行指针
	int pointInte = 0;//该点原图像小矩形积分值
	int orignInte = srcIntegral.at<int>(calPoint.y - 1, calPoint.x - 1); //待检测点左上角点积分图对应值
	float sum = 0;//所有计入点比例之和
	float accum = 0;//方差
	int count = 0;//记录共有多少个点被算入

	Mat ratio(tempH, tempW, CV_32FC1);//用来存储计算比例结果

	for (int y = 0; y < tempH; y++) {//点calPoint是包含在待计算图像中的左上角点
		tempLine = tempIntegral.ptr<int>(y + 1);
		srcLine = srcIntegral.ptr<int>(calPoint.y + y);
		ratioLine = ratio.ptr<float>(y);
		for (int x = 0; x < tempW; x++) {
			if (tempLine[x + 1] < halfMaxValue)//选择出大于工件一半像素值的点参与方差计算
			{ 
				ratioLine[x] = 0;
			}
			else 
			{
				pointInte = srcLine[calPoint.x + x] + orignInte - firstLineSrc[calPoint.x + x] - srcLine[calPoint.x - 1];
				ratioLine[x] = (float)pointInte / (float)tempLine[x + 1];
				sum += ratioLine[x];
				count++;
			}	
			
		}
	}

	float mean = sum / count;

	for (int m = 0; m < tempH; m++) {
		ratioLine = ratio.ptr<float>(m);
		for (int n = 0; n < tempW; n++) {
			if(ratioLine[n] != 0)
				accum += (ratioLine[n] - mean)*(ratioLine[n] - mean);
		}
	}

	return sqrt(accum / (count - 1));
}

float MAD(Mat src, Mat templat, Point calPoint)//point以原图为基准，而不是积分图
{
	int tempW = templat.cols;
	int tempH = templat.rows;
	int *tempLine, *srcLine; //指向模板和原图像行的指针

	int errCount = 0;
	for (int y = 0; y < tempH; y++) {
		tempLine = templat.ptr<int>(y);
		srcLine = src.ptr<int>(calPoint.y + y);
		for (int x = 0; x < tempW; x++) {
			if (srcLine[calPoint.x + x] != tempLine[x])
				errCount++;
		}
	}
	return (float)errCount / (float)(tempW*tempH);
}

float cvMT(Mat src, Mat templat, Point calPoint)//point以原图为基准，而不是积分图
{
	Mat srcPoint(src, Rect(calPoint.x, calPoint.y, templat.cols, templat.rows));
	Mat result;
	result.create(1, 1, CV_32FC1);

	matchTemplate(srcPoint, templat, result, TM_SQDIFF_NORMED);

	return result.at<float>(0, 0);
}

void finalLocate(Mat srcIntegral, vector<Point2i> &TargetPoint, int tempW, int tempH) //TargetPoint是以原图像为基准的坐标值
{
	int maxMoveX = tempW / 20;
	int maxMoveY = tempH / 20;
	int pointNum = TargetPoint.size();
	

	for (int i = 0; i < pointNum; i++)
	{
		int *srcLineS0 = srcIntegral.ptr<int>(TargetPoint[i].y);//up start
		int *srcLineS = srcIntegral.ptr<int>(TargetPoint[i].y + 1);//start
		int *srcLineE = srcIntegral.ptr<int>(TargetPoint[i].y + tempH);//end
		int sumRow = srcLineS[TargetPoint[i].x + tempW] + srcLineS0[TargetPoint[i].x] - srcLineS0[TargetPoint[i].x + tempW] - srcLineS[TargetPoint[i].x];
		int sumCol = srcLineE[TargetPoint[i].x + 1] + srcLineS0[TargetPoint[i].x]- srcLineS0[TargetPoint[i].x + 1]- srcLineE[TargetPoint[i].x];
		int count = 0;

		/*if (sumRow <= 1) {
			int *srcLineSnext = srcIntegral.ptr<int>(TargetPoint[i].y + 2);
			if (srcLineSnext[TargetPoint[i].x + tempW] - srcLineSnext[TargetPoint[i].x + 1] <= 1)
				TargetPoint[i].y += 1;
		}
		else {
			TargetPoint[i].y -= 1;
		}*/
		while (sumRow <= 1) 
		{
			int *S0Next = srcIntegral.ptr<int>(TargetPoint[i].y + 1);
			int *Snext = srcIntegral.ptr<int>(TargetPoint[i].y + 2);
			sumRow = Snext[TargetPoint[i].x + tempW] + S0Next[TargetPoint[i].x] - S0Next[TargetPoint[i].x + tempW] - Snext[TargetPoint[i].x];
			if (sumRow <= 1)
				TargetPoint[i].y += 1;
		}
		while (sumRow > 1)
		{
			TargetPoint[i].y -= 1;
			int *S0Next = srcIntegral.ptr<int>(TargetPoint[i].y);
			int *Snext = srcIntegral.ptr<int>(TargetPoint[i].y + 1);
			sumRow = Snext[TargetPoint[i].x + tempW] + S0Next[TargetPoint[i].x] - S0Next[TargetPoint[i].x + tempW] - Snext[TargetPoint[i].x];
			//if (sumRow > 1)
				
		}

		if (sumCol <= 1){
			if (srcLineE[TargetPoint[i].x + 2] - srcLineS[TargetPoint[i].x + 2] <= 1)
				TargetPoint[i].x += 1;
		}
		else {
			TargetPoint[i].x -= 1;
		}
	}
}

int findSeed(vector<Point2i> &survivePoint, vector<Point2i> &seedPoint, int tempW, int tempH)
{
	int halfTempW = tempW / 2;
	int halfTempH = tempH / 2;

	int seedNum = 0;
	vector<Point2i> groupPoint;


	while (!survivePoint.empty()) 
	{	
		int sumX = 0;
		int sumY = 0;

		groupPoint.push_back(survivePoint[0]);
		survivePoint.erase(survivePoint.begin());

		for (int i = survivePoint.size() - 1; i >= 0; i--) 
		{
			if (abs(survivePoint[i].x - groupPoint[0].x) < halfTempW && abs(survivePoint[i].y - groupPoint[0].y) < halfTempH) 
			{
				groupPoint.push_back(survivePoint[i]);
				survivePoint.erase(survivePoint.begin() + i);
			}
		}

		for (int j = groupPoint.size() - 1; j >= 0; j--) 
		{
			sumX += groupPoint[j].x;
			sumY += groupPoint[j].y;
		}

		seedPoint.push_back(Point2i(sumX / groupPoint.size(), sumY / groupPoint.size()));

		//cout << groupPoint << endl;
		
		seedNum++;
		groupPoint.clear();
	}

	return seedNum;
}

void LSS(Mat src, Mat templat, vector<Point2i> &seedPoint, vector<Point2i> &targetPoint) //线性正方形搜索算法(Line—Square Search，简称LSS)
{
	for (int i = seedPoint.size() - 1; i >= 0; i--) 
	{
		int x = seedPoint[i].x;
		int y = seedPoint[i].y;

		float square[3][3] = { 1.0 };//以种子点为中心的正方形9个点，各自的相关度
		float directionP = 1;//方向点对应的相关度（越小越匹配）
		float min = 1;
		int minX = 1;//最小点坐标
		int minY = 1;
		while (1) 
		{
			min = 1;
			minX = 1;
			minY = 1;
			for (int row = 0; row < 3; row++) 
			{
				for (int col = 0; col < 3; col++) 
				{
					if (x + col - 1 < 0 || y + row - 1 < 0) //越界
					{ 
						square[row][col] = 1; 
						continue; 
					}

					square[row][col] = cvMT(src, templat, Point(x + col - 1, y + row - 1));
					if (square[row][col] - min < 0) 
					{
						min = square[row][col];
						minX = col; minY = row;
					}
				}
			}

			if (minX == 1 && minY == 1) 
				break;//成功找到
			else 
			{
				if (x + 2 * (minX - 1) < 0 || y + 2 * (minY - 1) < 0)  //检测是否超出图片范围
					directionP = 1;
				else
					directionP = cvMT(src, templat, Point(x + 2 * (minX - 1), y + 2 * (minY - 1)));

				if (directionP < min) {
					x = x + 2 * (minX - 1);
					y = y + 2 * (minY - 1);
				}
				else {
					x = x + minX - 1;
					y = y + minY - 1;
				}

			}

		}
		targetPoint.push_back(Point(x, y));
	}
}


bool wxyMatchTemplate(Mat src, Mat templat, vector<Point2i> &TargetPoint)
{
	if (src.cols < templat.cols || src.rows < templat.rows)
	{
		cout << "模板不能比原图小" << endl;
		cvWaitKey(0);
		return 0;
	}

	double t = (double)getTickCount(); //测试运行时间

	int resultW = src.cols - templat.cols + 1;
	int resultH = src.rows - templat.rows + 1;

	Mat result = Mat::ones(resultH, resultW, CV_32FC1);    //  匹配方法计算的结果最小值为float（CV_32FC1）,将result全设为1，以便和标准差比较

	Mat tempIntegral(templat.rows + 1, templat.cols + 1, CV_32SC1);//算积分图
	Mat srcIntegral(src.rows + 1, src.cols + 1, CV_32SC1);
	wxyIntegral(templat, tempIntegral);
	wxyIntegral(src, srcIntegral);

	
	int tempW = templat.cols;
	int tempH = templat.rows;
	
	//下列点坐标都是以原图为准，不同于积分后扩大1个像素的图
	vector<Point2i> calculatePoint;//在每次模板和原图片上点对比时，选中用来计算面积比的点
	vector<Point2i> survivePoint;//经过SSDA+积分图的筛选，剩下来的点
	vector<Point2i> seedPoint;//经过将剩下来点聚类，得到每个目标初步定位的种子点。作为后面cvMT的搜索起始点
	vector<Point2i> targetPoint;//最终经过搜索后得到的目标位置

	pickPoints(templat, calculatePoint);//第一点固定为右下角点。点的坐标以原图为准，不同于积分后扩大1个像素的图
	int pointsNum = calculatePoint.size();

	float tempPoint[20];

	for (int m = 0; m < pointsNum; m++) {//预先将templat中选出点算好，后面不用重复计算，可以直接用
		tempPoint[m] = tempIntegral.at<int>(calculatePoint[m].y + 1, calculatePoint[m].x + 1);
	}


	float *p;
	float sum = 0.0;
	float mean = 0.0;
	bool pass = 0;
	bool findFlag = 0;
	float pointIntegral = 0;
	float ratio[20];//存放选出点对应位置图像和模板像素和的比
	int count = 0;
	for (int y = 1; y <= resultH; y++)  //遍历src中（M-m）*(N-n)
	{
		p = result.ptr<float>(y-1);
		for (int x = 1; x <= resultW; x++)
		{	
			sum = 0;
			mean = 0;
			pass = 0;
			for (int i = 0; i < pointsNum; i++) {//对挑选出来的15个点求比例
				pointIntegral = srcIntegral.at<int>(y + calculatePoint[i].y, x + calculatePoint[i].x) + srcIntegral.at<int>(y - 1, x - 1) - srcIntegral.at<int>(y + calculatePoint[i].y, x - 1) - srcIntegral.at<int>(y - 1, x + calculatePoint[i].x);
				ratio[i] = pointIntegral / tempPoint[i];//上式点（x,y）是包含在待计算图像中的左上角点
				if (ratio[i] > 1.1 || ratio[i] < 0.9) {
					pass = 1;
					break;
				}
				sum += ratio[i];	
			}
			if (pass)   continue;

			mean = sum / pointsNum;
			float accum = 0.0;
			for (int j = 0; j < pointsNum; j++) {
				accum += (ratio[j] - mean)*(ratio[j] - mean);
			}
			float T = (mean - 1)*sqrt(pointsNum*(pointsNum - 1) / accum);
			
			if (T > 2 || T < -2) continue;//T检验法，a = 0.01 -> 2.9768
	
			//cout << T <<Point(x,y) << endl;
			findFlag = 1;
			//p[x - 1] = accumHMV(srcIntegral, tempIntegral, Point(x, y));
			//p[x - 1] = cvMT(src, templat, Point(x - 1, y - 1));

			survivePoint.push_back(Point(x - 1, y - 1));
			
		}
	}

	findSeed(survivePoint, seedPoint, tempW, tempH);

	LSS(src, templat, seedPoint, TargetPoint);

	t = ((double)getTickCount() - t) / getTickFrequency(); //获得时间，单位是秒
	//获取此程序段开始执行时间
	cout << "程序段运行时间：" << 1000*t << "ms!" << endl;

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

	Mat tempIntegral(templat.rows + 1, templat.cols + 1, CV_32SC1);//算积分图
	Mat srcIntegral(src.rows + 1, src.cols + 1, CV_32SC1);
	wxyIntegral(templat, tempIntegral);
	wxyIntegral(src, srcIntegral);

	//matchTemplate(src, templat, result, CV_TM_SQDIFF_NORMED);  //核心匹配函数
	//if(!wxyMatchTemplate(src, templat, srcIntegral, tempIntegral, result)) return 0;

	imshow("点群", result);

	double minValue, maxValue;
	Point minLoc, maxLoc;

	minMaxLoc(result, &minValue, &maxValue, &minLoc, &maxLoc, Mat());

	AddPoint(TargetPoint, minLoc + offset, templat.cols, templat.rows);

	Point new_minLoc;

	// 计算下一个最小值  
	new_minLoc = getNextMinLoc(result, minLoc, maxValue, templat.cols, templat.rows);
	while (result.at<float>(new_minLoc.y, new_minLoc.x) < 0.1*minValue+0.9*maxValue)
	{
		AddPoint(TargetPoint, new_minLoc + offset, templat.cols, templat.rows);
		new_minLoc = getNextMinLoc(result, new_minLoc, maxValue, templat.cols, templat.rows);
	}

	finalLocate(srcIntegral, TargetPoint, templat.cols, templat.rows);

	return 1;
}



int main()
{
	Mat src0, srcResult, templat, src, result; // result用来存放结果，src0为原图像，src为扩展边界后图像
	char filename[100];
	//srcResult = imread("C:\\Users\\Mark\\Desktop\\测试素材\\data1\\0.png", 1);  //用来显示 
	templat = imread("C:\\Users\\Mark\\Desktop\\测试素材\\data2\\mold\\mold.png", 0);
	

	for (unsigned int i = 0; i <= 16; ++i)
	{
		sprintf(filename, "C:\\Users\\Mark\\Desktop\\测试素材\\data2\\%d.png", i);//"C:\\Users\\Mark\\Desktop\\原图像.png"
		src = imread(filename, IMREAD_GRAYSCALE);

		if (src.empty() || templat.empty())
		{
			cout << "打开图片失败" << endl;
			cvWaitKey(0);
			return 0;
		}

		

		//copyMakeBorder(src0, src, 0, 0, templat.cols, templat.cols, BORDER_CONSTANT, Scalar(0,0,0)); //扩展待匹配的图像，本批图像工件从左向右进入，所以扩展图像左右
		srcResult = src.clone(); //查看结果的图像


		vector<Point2i> TargetPoint;
		Point minLoc;


		//MyTemplateMatch(src, templat, TargetPoint, Point(0, 0));
		wxyMatchTemplate(src, templat, TargetPoint);


		/*int leftTemW, rightTemW;
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

void pickPoints(Mat templat, vector<Point2i> &calculatePoint)//保证取得点，模板的积分图对应点不为零（除数）
{
	int tempW = templat.cols;
	int tempH = templat.rows;
	int intervalW = tempW / 5;
	int intervalH = tempH / 5;
	int a = 5;
	for (int y = 0; y < 5; y++) 
	{
		for (int x = 0; x < a; x++) 
		{
			calculatePoint.push_back(Point2i(tempW - 1 - x * intervalW, tempH - 1 - y * intervalH));
		}
		a--;
	}

}