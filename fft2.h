//fft.h
#ifndef NK_FFT2
#define NK_FFT2

#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include <math.h>

#include "fft.h"
#include "fftw32/include/fftw3.h"
#define _USE_MATH_DEFINES


#pragma comment(lib, "libfftw3f-3.lib")
#pragma comment(lib, "libfftw3-3.lib")
#pragma comment(lib, "libfftw3l-3.lib")



namespace MyAlg
{
	using std::shared_ptr;
	using std::unique_ptr;

	//fftw_plan fftw_plan_dft_r2c_1d(int n, double *in, fftw_complex *out, unsigned flags);
	//flags: FFTW_ESTIMATE 单次执行速度快 FFTW_MEASURE 对同样大小的数据多次执行速度快
	template<bool flag>
	struct Select2D
	{
		static fftw_plan fft(int n0, int n1, fftw_complex *in, fftw_complex *out)
		{
			return fftw_plan_dft_2d(n0, n1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
		}
	};
	template<>
	struct Select2D<false>
	{
		static fftw_plan fft(int n0, int n1, fftw_complex *in, fftw_complex *out)
		{
			return fftw_plan_dft_2d(n0, n1, in, out, FFTW_FORWARD, FFTW_MEASURE);
		}
	};

	template<bool IsEstimate>
	class FFT2D
	{
	private:
		int size1_;
		int size2_;
		shared_ptr<fftw_complex> in_;
		shared_ptr<fftw_complex> out_;
		fftw_plan fftPlan_;
	public:
		FFT2D(int size1, int size2)
		{
			/*fftw_complex *in, *out;
			double *in2;
			in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)* 4 * 6);
			in2 = (double*)fftw_malloc(sizeof(double)* 4 * 6);
			out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)* 4 * 6);
			string temp;
			temp = "in= \n";
			OutputDebugString(temp.c_str());
			int a[] = { 1, 2, 3, 9, 3, 5, 2, 3, 4, 10, 68, 7, 3, 4, 5, 34, 56, 3, 4, 5, 6, 19, 478, 4, 4, 5, 6, 19, 478, 4, 4, 5, 6, 19, 478, 4 };
			for (int i = 0; i<4; i++)
			{
				for (int j = 0; j<6; j++)
				{
					*(*(in + i * 6 + j) + 0) = a[i*6 + j];
					*(*(in + i * 6 + j) + 1) = 0;
					*(in2 + i * 6 + j) = a[i * 6 + j];
					//cout << *(*(in + i * 3 + j) + 0) << "+" << *(*(in + i * 3 + j) + 1) << "i, ";
					temp = to_string(static_cast<double>(*(*(in + i * 6 + j) + 0))) + "+" + to_string(static_cast<double>(*(*(in + i * 6 + j) + 1))) + "i, ";
					OutputDebugString(temp.c_str());
				}
				OutputDebugString("\n");
			}
			fftw_plan p;
			p = fftw_plan_dft_2d(4, 6, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
			//p = fftw_plan_dft_r2c_2d(6, 6, in2, out, FFTW_ESTIMATE);
			fftw_execute(p);

			temp = "out= ";
			OutputDebugString(temp.c_str());
			double tt = sqrt(4 * 6);
			for (int i = 0; i<4; i++)
			{
				for (int j = 0; j<6; j++)
				{
					temp = to_string(static_cast<double>(*(*(out + i * 6 + j) + 0)) / tt)+"\t";// +"+" + to_string(static_cast<double>(*(*(out + i * 3 + j) + 1)) / tt) + "i, ";
					OutputDebugString(temp.c_str());
				}
				OutputDebugString("\n");
			}
			fftw_destroy_plan(p);
			fftw_free(in);
			fftw_free(out);*/
			
			size1_ = size1;
			size2_ = size2;
			out_.reset(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * size1_ * size2_)), [](fftw_complex* p){ fftw_free(p); });
			in_.reset(new fftw_complex[size1_ * size2_], default_delete<fftw_complex[]>());
			fftPlan_ = Select2D<IsEstimate>::fft(size1_, size2_, in_.get(), out_.get());
		}
		~FFT2D()
		{
			fftw_destroy_plan(fftPlan_);
		}




		void FeedData(const shared_ptr<double> in)
		{
			//memcpy(in_.get(), in.get(), size1_ * size2_ * sizeof(double));
//#pragma omp parallel for
			for (int i = 0; i < size1_*size2_; i++)
			{
				in_.get()[i][0] = in.get()[i];
				in_.get()[i][1] = 0;
			}
		}

		void Execute()
		{
			fftw_execute(fftPlan_);
		}


		void GetInput(Mat & in)
		{
			for (int i = 0; i < size1_; i++)
			{
				for (int j = 0; j < size2_; j++)
				{
					double* mid = in_.get()[i * size2_ + j];
					in.at<char>(i, j) = static_cast<int>(mid[0]);
				}
			}
		}

		void GetResult(shared_ptr<fftw_complex> out)
		{
			memcpy(out.get(), out_.get(), size1_ * size2_ * sizeof(fftw_complex));
		}

		//实部虚部
		void GetResultRC(Mat& out1, Mat& out2)
		{
//#pragma omp parallel for
			for (int i = 0; i < size1_; i++)
			{
				for (int j = 0; j < size2_; j++)
				{
					double* mid = out_.get()[i * size2_ + j];
					out1.at<double>(i, j) = mid[0] / sqrt(size1_*size2_);
					out2.at<double>(i, j) = mid[1] / sqrt(size1_*size2_);
				}
			}
		}
		//实部虚部
		void GetResultR(Mat& out1)
		{
			for (int i = 0; i < size1_; i++)
			{
				for (int j = 0; j < size2_; j++)
				{
					double* mid = out_.get()[i * size2_ + j];
					out1.at<double>(i, j) = mid[0] / sqrt(size1_*size2_);
				}
			}
		}

		//实部虚部
		void GetResultRC(shared_ptr<double> out1, shared_ptr<double> out2)
		{
//#pragma omp parallel for
			for (int i = 0; i < size1_; i++)
			{
				for (int j = 0; j < size2_; j++)
				{
					out1.get()[i * size2_ + j] = out_.get()[i * size2_ + j][0] / sqrt(size1_*size2_);
					out2.get()[i * size2_ + j] = out_.get()[i * size2_ + j][1] / sqrt(size1_*size2_);
				}
			}
		}

		//模、辐角
		void GetResultAT(shared_ptr<double> out1, shared_ptr<double> out2)
		{
			//模和幅值
			for (int i = 0; i < size1_; i++)
			{
				for (int j = 0; j < size2_; j++)
				{
					out1.get()[i * size2_ + j] = sqrt(
						(out_.get()[i * size2_ + j][0] * out_.get()[i * size2_ + j][0] + out_.get()[i * size2_ + j][1] * out_.get()[i * size2_ + j][1]) / (size1_*size2_)
						);
					if (out_.get()[i * size2_ + j][0] != 0)
						out2.get()[i * size2_ + j] = atan(out_.get()[i * size2_ + j][1] / out_.get()[i * size2_ + j][0]);
					else
						out2.get()[i * size2_ + j] = M_PI / 2 * (out_.get()[i * size2_ + j][1]>0 ? 1 : -1);
				}
			}
		}

		//模、辐角
		void GetResultA(shared_ptr<double> out1)
		{
			//模和幅值
			for (int i = 0; i < size1_; i++)
			{
				for (int j = 0; j < size2_; j++)
				{
					out1.get()[i * size2_ + j] = sqrt(
						(out_.get()[i * size2_ + j][0] * out_.get()[i * size2_ + j][0] + out_.get()[i * size2_ + j][1] * out_.get()[i * size2_ + j][1]) / (size1_*size2_)
						);
				}
			}
		}


		//A=[1,2,3,4,5]; B=fftshift(A)=[4,5,1,2,3]; C=ifftshift(A)=[3,4,5,1,2];
		void FFTShift2D(shift_flag flag)
		{
			//向负方向舍入
			int size_floor2 = floor(size2_ / 2.0);
			//向正方向舍入
			int size_ceil2 = ceil(size2_ / 2.0);

			//向负方向舍入
			int size_floor1 = floor(size1_ / 2.0);
			//向正方向舍入
			int size_ceil1 = ceil(size1_ / 2.0);

			if (flag == shift_flag::In)
			{
				shared_ptr<fftw_complex> mid(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)* size_ceil2)), [](fftw_complex* p){ fftw_free(p); });
				for (int i = 0; i < size1_; i++)
				{
					//取出正频率数据
					memmove(mid.get(),								in_.get() + i * size2_,					size_ceil2 * sizeof(fftw_complex));
					//将负频率移动到前面
					memmove(in_.get() + i * size2_,					in_.get() + i * size2_ + size_ceil2,	size_floor2 * sizeof(fftw_complex));
					//正频率放到后面
					memmove(in_.get() + i * size2_ + size_floor2,	mid.get(),								size_ceil2 * sizeof(fftw_complex));
				}
				shared_ptr<fftw_complex> mid2(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)* size_ceil1 * size2_)), [](fftw_complex* p){ fftw_free(p); });
				//取出正频率数据
				memmove(mid2.get(),									in_.get(),								size_ceil1 * size2_ * sizeof(fftw_complex));
				//将负频率移动到前面
				memmove(in_.get(),									in_.get() + size_ceil1 * size2_,		size_floor1 * size2_ * sizeof(fftw_complex));
				//正频率放到后面
				memmove(in_.get() + size_floor1 * size2_,			mid2.get(),								size_ceil1 * size2_ * sizeof(fftw_complex));
			}
			else
			{
				shared_ptr<fftw_complex> mid(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)* size_ceil2)), [](fftw_complex* p){ fftw_free(p); });
				for (int i = 0; i < size1_; i++)
				{
					//取出正频率数据
					memmove(mid.get(),								out_.get() + i * size2_,				size_ceil2 * sizeof(fftw_complex));
					//将负频率移动到前面
					memmove(out_.get() + i * size2_,				out_.get() + i * size2_ + size_ceil2,	size_floor2 * sizeof(fftw_complex));
					//正频率放到后面
					memmove(out_.get() + i * size2_ + size_floor2,	mid.get(),								size_ceil2 * sizeof(fftw_complex));
				}
				shared_ptr<fftw_complex> mid2(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)* size_ceil1 * size2_)), [](fftw_complex* p){ fftw_free(p); });
				//取出正频率数据
				memmove(mid2.get(),									out_.get(),								size_ceil1 * size2_ * sizeof(fftw_complex));
				//将负频率移动到前面
				memmove(out_.get(),									out_.get() + size_ceil1 * size2_,		size_floor1 * size2_ * sizeof(fftw_complex));
				//正频率放到后面
				memmove(out_.get() + size_floor1 * size2_,			mid2.get(),								size_ceil1 * size2_ * sizeof(fftw_complex));
			}
		}

		//A=[1,2,3,4,5]; B=fftshift(A)=[4,5,1,2,3]; C=ifftshift(A)=[3,4,5,1,2];
		void iFFTShift2D(shift_flag flag)
		{
			//向负方向舍入
			int size_floor2 = floor(size2_ / 2.0);
			//向正方向舍入
			int size_ceil2 = ceil(size2_ / 2.0);

			//向负方向舍入
			int size_floor1 = floor(size1_ / 2.0);
			//向正方向舍入
			int size_ceil1 = ceil(size1_ / 2.0);

			if (flag == shift_flag::In)
			{
				shared_ptr<fftw_complex> mid(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)* size_floor2)), [](fftw_complex* p){ fftw_free(p); });
				for (int i = 0; i < size1_; i++)
				{
					//取出正频率数据
					memmove(mid.get(),								in_.get() + i * size2_,					size_floor2 * sizeof(fftw_complex));
					//将负频率移动到前面
					memmove(in_.get() + i * size2_,					in_.get() + i * size2_ + size_floor2,	size_ceil2 * sizeof(fftw_complex));
					//正频率放到后面
					memmove(in_.get() + i * size2_ + size_ceil2,	mid.get(),								size_floor2 * sizeof(fftw_complex));
				}
				shared_ptr<fftw_complex> mid2(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)* size_floor1 * size2_)), [](fftw_complex* p){ fftw_free(p); });
				//取出正频率数据
				memmove(mid2.get(),									in_.get(),								size_floor1 * size2_ * sizeof(fftw_complex));
				//将负频率移动到前面
				memmove(in_.get(),									in_.get() + size_floor1 * size2_,		size_ceil1 * size2_ * sizeof(fftw_complex));
				//正频率放到后面
				memmove(in_.get() + size_ceil1 * size2_,			mid2.get(),								size_floor1 * size2_ * sizeof(fftw_complex));
			}
			else
			{
				shared_ptr<fftw_complex> mid(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)* size_floor2)), [](fftw_complex* p){ fftw_free(p); });
				for (int i = 0; i < size1_; i++)
				{
					//取出正频率数据
					memmove(mid.get(),								out_.get() + i * size2_,				size_floor2 * sizeof(fftw_complex));
					//将负频率移动到前面
					memmove(out_.get() + i * size2_,				out_.get() + i * size2_ + size_floor2,	size_ceil2 * sizeof(fftw_complex));
					//正频率放到后面
					memmove(out_.get() + i * size2_ + size_ceil2,	mid.get(),								size_floor2 * sizeof(fftw_complex));
				}
				shared_ptr<fftw_complex> mid2(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)* size_floor1 * size2_)), [](fftw_complex* p){ fftw_free(p); });
				//取出正频率数据
				memmove(mid2.get(),									out_.get(),								size_floor1 * size2_ * sizeof(fftw_complex));
				//将负频率移动到前面
				memmove(out_.get(),									out_.get() + size_floor1 * size2_,		size_ceil1 * size2_ * sizeof(fftw_complex));
				//正频率放到后面
				memmove(out_.get() + size_ceil1 * size2_,			mid2.get(),								size_floor1 * size2_ * sizeof(fftw_complex));
			}
		}
	};




}


#endif