//fft.h
#ifndef NK_FFT
#define NK_FFT

#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include "fftw32/include/fftw3.h"

#pragma comment(lib, "libfftw3f-3.lib")
#pragma comment(lib, "libfftw3-3.lib")
#pragma comment(lib, "libfftw3l-3.lib")


namespace MyAlg
{
	using std::shared_ptr;
	using std::unique_ptr;

	enum shift_flag
	{
		In,
		Out
	};
	//fftw_plan fftw_plan_dft_r2c_1d(int n, double *in, fftw_complex *out, unsigned flags);
	//flags: FFTW_ESTIMATE 单次执行速度快 FFTW_MEASURE 对同样大小的数据多次执行速度快
	template<bool flag>
	struct Select1D
	{
		static fftw_plan fft(int n, double *in, fftw_complex *out)
		{
			return fftw_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);
		}
	};
	template<>
	struct Select1D<false>
	{
		static fftw_plan fft(int n, double *in, fftw_complex *out)
		{
			return fftw_plan_dft_r2c_1d(n, in, out, FFTW_MEASURE);
		}
	};

	template<bool IsEstimate>
	class FFT1D
	{
	private:
		int size_;
		shared_ptr<double> in_;
		shared_ptr<fftw_complex> out_;
		fftw_plan fftPlan_;
	public:
		FFT1D(int size)
		{
			size_ = size;
			out_.reset(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*size_)), [](fftw_complex* p){ fftw_free(p); });
			in_.reset(new double[size], default_delete<double[]>());
			fftPlan_ = Select1D<IsEstimate>::fft(size_, in_.get(), out_.get());
		}
		~FFT1D()
		{
			fftw_destroy_plan(fftPlan_);
		}

		void FeedData(const shared_ptr<double> in)
		{
			memcpy(in_.get(), in.get(), size_ * sizeof(double));
		}

		void FeedData(const double* in)
		{
			memcpy(in_.get(), in, size_ * sizeof(double));
		}

		void Execute()
		{
			fftw_execute(fftPlan_);
		}

		void GetResult(shared_ptr<fftw_complex> out)
		{
			memcpy(out.get(), out_.get(), size_ * sizeof(fftw_complex));
		}
		//实部虚部
		void GetResultRC(shared_ptr<double> out1, shared_ptr<double> out2)
		{
//#pragma omp parallel for
			for (int i = 0; i < size_; i++)
			{
				out1.get()[i] = out_.get()[i][0] / size_;
				out2.get()[i] = out_.get()[i][1] / size_;
			}
		}
		//模、辐角
		void GetResultAT(shared_ptr<double> out1, shared_ptr<double> out2)
		{
			//模和幅值
//#pragma omp parallel for
			for (int i = 0; i < size_; i++)
			{
				out1.get()[i] = (out_.get()[i][0] * out_.get()[i][0] + out_.get()[i][1] * out_.get()[i][1]) / size_;
				if (out_.get()[i][0] != 0)
					out2.get()[i] = atan(out_.get()[i][1] / out_.get()[i][0]);
				else
					out2.get()[i] = M_PI / 2 * (out_.get()[i][1]>0 ? 1 : -1);
			}
		}
		//模、辐角
		void GetResultA(shared_ptr<double> out1)
		{
			//模和幅值
//#pragma omp parallel for
			for (int i = 0; i < size_; i++)
			{
				out1.get()[i] = (out_.get()[i][0] * out_.get()[i][0] + out_.get()[i][1] * out_.get()[i][1]) / size_;
			}
		}


		//A=[1,2,3,4,5]; B=fftshift(A)=[4,5,1,2,3]; C=ifftshift(A)=[3,4,5,1,2];
		void FFTShift(shift_flag flag)
		{
			if (flag == shift_flag::In)
			{
				//向负方向舍入
				int size_floor2 = floor(size_ / 2);
				//向正方向舍入
				int size_ceil2 = ceil(size_ / 2);

				unique_ptr<double[]> mid(new double[size_ceil2]);
				//取出正频率数据
				memmove(mid, in_.get(), size_ceil2 * sizeof(double));
				//将负频率移动到前面
				memmove(in_.get(), in_.get() + size_ceil2, size_floor2 * sizeof(double));
				//正频率放到后面
				memmove(in_.get() + size_floor2, mid.get(), size_ceil2 * sizeof(double));
			}
			else
			{
				//向负方向舍入
				int size_floor2 = floor(size_ / 2);
				//向正方向舍入
				int size_ceil2 = ceil(size_ / 2);

				unique_ptr<fftw_complex[]> mid(new fftw_complex[size_ceil2]);
				//取出正频率数据
				memmove(mid, out_.get(), size_ceil2 * sizeof(fftw_complex));
				//将负频率移动到前面
				memmove(out_.get(), out_.get() + size_ceil2, size_floor2 * sizeof(fftw_complex));
				//正频率放到后面
				memmove(out_.get() + size_floor2, mid.get(), size_ceil2 * sizeof(fftw_complex));
			}
		}

		//A=[1,2,3,4,5]; B=fftshift(A)=[4,5,1,2,3]; C=ifftshift(A)=[3,4,5,1,2];
		void iFFTShift(shift_flag flag)
		{
			if (flag == shift_flag::In)
			{
				//向负方向舍入
				int size_floor2 = floor(size_ / 2);
				//向正方向舍入
				int size_ceil2 = ceil(size_ / 2);

				unique_ptr<double[]> mid(new double[size_floor2]);
				//取出正频率数据
				memmove(mid, in_.get(), size_floor2 * sizeof(double));
				//将负频率移动到前面
				memmove(in_.get(), in_.get() + size_floor2, size_ceil2 * sizeof(double));
				//正频率放到后面
				memmove(in_.get() + size_ceil2, mid.get(), size_floor2 * sizeof(double));
			}
			else
			{
				//向负方向舍入
				int size_floor2 = floor(size_ / 2);
				//向正方向舍入
				int size_ceil2 = ceil(size_ / 2);

				unique_ptr<fftw_complex[]> mid(new fftw_complex[size_floor2]);
				//取出正频率数据
				memmove(mid, out_.get(), size_floor2 * sizeof(fftw_complex));
				//将负频率移动到前面
				memmove(out_.get(), out_.get() + size_floor2, size_ceil2 * sizeof(fftw_complex));
				//正频率放到后面
				memmove(out_.get() + size_ceil2, mid.get(), size_floor2 * sizeof(fftw_complex));
			}
		}
	};














	//fftw_plan fftw_plan_dft_r2c_1d(int n, double *in, fftw_complex *out, unsigned flags);
	//flags: FFTW_ESTIMATE 单次执行速度快 FFTW_MEASURE 对同样大小的数据多次执行速度快
	template<bool flag>
	struct iSelect1D
	{
		static fftw_plan fft(int n, fftw_complex *in, fftw_complex *out)
		{
			return fftw_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
		}
	};
	template<>
	struct iSelect1D<false>
	{
		static fftw_plan fft(int n, fftw_complex *in, fftw_complex *out)
		{
			return fftw_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_MEASURE);
		}
	};

	template<bool IsEstimate>
	class iFFT1D
	{
	private:
		int size_;
		shared_ptr<fftw_complex> in_;
		shared_ptr<fftw_complex> out_;
		fftw_plan fftPlan_;
	public:
		iFFT1D(int size)
		{
			size_ = size;
			out_.reset(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*size_)), [](fftw_complex* p){ fftw_free(p); });
			in_.reset(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*size_)), [](fftw_complex* p){ fftw_free(p); });
			fftPlan_ = iSelect1D<IsEstimate>::fft(size_, in_.get(), out_.get());
		}
		~iFFT1D()
		{
			fftw_destroy_plan(fftPlan_);
		}

		void FeedData(const shared_ptr<double> in_r, const shared_ptr<double> in_c)
		{
//#pragma omp parallel for
			for (int i = 0; i < size_; i++)
			{
				in_.get()[i][0] = in_r.get()[i];
				in_.get()[i][1] = in_c.get()[i];
			}
		}
		
		void Execute()
		{
			fftw_execute(fftPlan_);
		}

		void GetResult(shared_ptr<fftw_complex> out)
		{
			memcpy(out.get(), out_.get(), size_ * sizeof(fftw_complex));
		}
		//实部虚部
		void GetResultRC(shared_ptr<double> out1, shared_ptr<double> out2)
		{
//#pragma omp parallel for
			for (int i = 0; i < size_; i++)
			{
				out1.get()[i] = out_.get()[i][0] / sqrt(size_);
				out2.get()[i] = out_.get()[i][1] / sqrt(size_);
			}
		}
		//模、辐角
		void GetResultAT(shared_ptr<double> out1, shared_ptr<double> out2)
		{
			//模和幅值
//#pragma omp parallel for
			for (int i = 0; i < size_; i++)
			{
				out1.get()[i] = (out_.get()[i][0] * out_.get()[i][0] + out_.get()[i][1] * out_.get()[i][1]) / sqrt(size_);
				if (out_.get()[i][0] != 0)
					out2.get()[i] = atan(out_.get()[i][1] / out_.get()[i][0]);
				else
					out2.get()[i] = M_PI / 2 * (out_.get()[i][1]>0 ? 1 : -1);
			}
		}
		//模、辐角
		void GetResultA(shared_ptr<double> out1)
		{
			//模和幅值
//#pragma omp parallel for
			for (int i = 0; i < size_; i++)
			{
				out1.get()[i] = (out_.get()[i][0] * out_.get()[i][0] + out_.get()[i][1] * out_.get()[i][1]) / sqrt(size_);
			}
		}


		//A=[1,2,3,4,5]; B=fftshift(A)=[4,5,1,2,3]; C=ifftshift(A)=[3,4,5,1,2];
		void FFTShift(shift_flag flag)
		{
			if (flag == shift_flag::In)
			{
				//向负方向舍入
				int size_floor2 = floor(size_ / 2);
				//向正方向舍入
				int size_ceil2 = ceil(size_ / 2);
				
				shared_ptr<fftw_complex> mid(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)* size_ceil2)), [](fftw_complex* p){ fftw_free(p); });
				//取出正频率数据
				memmove(mid.get(), in_.get(), size_ceil2 * sizeof(fftw_complex));
				//将负频率移动到前面
				memmove(in_.get(), in_.get() + size_ceil2, size_floor2 * sizeof(fftw_complex));
				//正频率放到后面
				memmove(in_.get() + size_floor2, mid.get(), size_ceil2 * sizeof(fftw_complex));
			}
			else
			{
				//向负方向舍入
				int size_floor2 = floor(size_ / 2);
				//向正方向舍入
				int size_ceil2 = ceil(size_ / 2);

				shared_ptr<fftw_complex> mid(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)* size_ceil2)), [](fftw_complex* p){ fftw_free(p); });
				//取出正频率数据
				memmove(mid.get(), out_.get(), size_ceil2 * sizeof(fftw_complex));
				//将负频率移动到前面
				memmove(out_.get(), out_.get() + size_ceil2, size_floor2 * sizeof(fftw_complex));
				//正频率放到后面
				memmove(out_.get() + size_floor2, mid.get(), size_ceil2 * sizeof(fftw_complex));
			}
		}

		//A=[1,2,3,4,5]; B=fftshift(A)=[4,5,1,2,3]; C=ifftshift(A)=[3,4,5,1,2];
		void iFFTShift(shift_flag flag)
		{
			if (flag == shift_flag::In)
			{
				//向负方向舍入
				int size_floor2 = floor(size_ / 2);
				//向正方向舍入
				int size_ceil2 = ceil(size_ / 2);
				
				shared_ptr<fftw_complex> mid(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)* size_floor2)), [](fftw_complex* p){ fftw_free(p); });
				//取出正频率数据
				memmove(mid.get(),					in_.get(),					size_floor2 * sizeof(fftw_complex));
				//将负频率移动到前面
				memmove(in_.get(),					in_.get() + size_floor2,	size_ceil2 * sizeof(fftw_complex));
				//正频率放到后面
				memmove(in_.get() + size_ceil2,		mid.get(),					size_floor2 * sizeof(fftw_complex));
			}
			else
			{
				//向负方向舍入
				int size_floor2 = floor(size_ / 2);
				//向正方向舍入
				int size_ceil2 = ceil(size_ / 2);

				shared_ptr<fftw_complex> mid(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)* size_floor2)), [](fftw_complex* p){ fftw_free(p); });
				//取出正频率数据
				memmove(mid.get(),					out_.get(),					size_floor2 * sizeof(fftw_complex));
				//将负频率移动到前面
				memmove(out_.get(),					out_.get() + size_floor2,	size_ceil2 * sizeof(fftw_complex));
				//正频率放到后面
				memmove(out_.get() + size_ceil2,	mid.get(),					size_floor2 * sizeof(fftw_complex));
			}
		}
	};


}


#endif