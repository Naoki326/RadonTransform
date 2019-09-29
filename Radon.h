//fft.h
#ifndef NK_RADON
#define NK_RADON

#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include <cmath>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#include "fftw32/include/fftw3.h"
#include "fft.h"
#include "fft2.h"


using std::vector;
using std::shared_ptr;
using namespace cv;

namespace MyAlg
{
	const double M_PI = 3.141592653;

	class RadonLines
	{
	public:
		//��������ֱ��ǲ��ҵ�ֱ�ߵĽǶȵ���ʼֵ����ֵֹ
		//��Χ��(-180, 180]
		RadonLines(double stAng = -179, double edAng = 180, double inc = 1);

		~RadonLines()
		{
			if (ifft1d_ptr_.unique() == true)
				ifft1d_ptr_.reset();
			if (fft2d_ptr_.unique() == true)
				fft2d_ptr_.reset();
		}

		//�������ý��յ�����ͼ�Ĵ�С
		//��Ҫ����Ϊfftw��ҪԤ���С
		void ChangeSize(int width, int height, double stAng = -179, double edAng = 180, double inc = 1);

		//���ò���
		void PadData(const Mat& image, Mat& dst);

		//��������
		void FeedData(const Mat& image);

		//��ֵ��������
		void RadonPeaks(int numpeaks, double threshold, int nhoodx = 0, int nhoody = 0);

		//ȡֱ��
		void GetResult(int &n, shared_ptr<double> rho, shared_ptr<double> the);

	private:
		int width_;
		int height_;

		int padedw_;
		int padedh_;

		Mat trans_x_;
		Mat trans_y_;

		//�������½Ƕ�������س���
		int trans_len_;
		double stAng_;
		double edAng_;
		double inc_;

		shared_ptr<iFFT1D<true> > ifft1d_ptr_;
		shared_ptr<FFT2D<true> > fft2d_ptr_;

		//���Radon�任���
		Mat result_;
		//�������
		vector<double> rho_, the_;

	};
	
}

#endif