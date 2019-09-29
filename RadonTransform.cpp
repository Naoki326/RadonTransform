// RadonTransform.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "Radon.h"

int main()
{
    //std::cout << "Hello World!\n"; 
	std::string path = "";
	cv::Mat src, dst;
	src = cv::imread(path);
	src.copyTo(dst);

	int w = src.cols;
	int h = src.rows;

	double diagonal = sqrt(2) * max(w, h);
	double rdscale = diagonal / 2048;

	MyAlg::RadonLines rl;
	rl.ChangeSize(src.cols, src.rows, 80, 100, 0.25);
	rl.FeedData(src);
	int numpeaks = 40;
	rl.RadonPeaks(numpeaks, 0.1, 2, 2);
	shared_ptr<double> rhos, thes;
	rhos.reset(new double[numpeaks], std::default_delete<double[]>());
	thes.reset(new double[numpeaks], std::default_delete<double[]>());
	rl.GetResult(numpeaks, rhos, thes);

	//画出直线
	for (int i = 0; i < numpeaks; i++)
	{
		double rho = *(rhos.get()+i);
		rho *= rdscale;
		double the = *(thes.get()+i);
		if (the < 0)
			the = 180 + the;
		double a = cos(the * MyAlg::M_PI / 180.0);
		double b = -sin(the * MyAlg::M_PI / 180.0);
		double x0 = a * rho + (double)w / (2.0);
		double y0 = b * rho + (double)h / (2.0);
		Point p1(x0 + 5000 * b, y0 - 5000 * a);
		Point p2(x0 - 5000 * b, y0 + 5000 * a);
		line(dst, p1, p2, Scalar(0), 6, LINE_8);
	}

}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
