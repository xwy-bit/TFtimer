#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <string>
#include <iostream>
#include "time.h"
#include <chrono>
#include <ctime>

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
REGISTER_OP("Timer")
      .Input("input: int32")
      .Output("out1: int32")
      .Output("sec2: int32")
      .Output("millisec: int32")
      .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
      c->set_output(0, c->input(0));
      return Status::OK(); 
      });

class TimerOp : public OpKernel{
public:
    explicit TimerOp(OpKernelConstruction * context) : OpKernel(context){

    }
    void Compute(OpKernelContext* context) override {
        const Tensor& input_tensor = context->input(0);
        Tensor* output_tensor = NULL;
        Tensor* time_sec = NULL;
        Tensor* time_millisec = NULL;  
        time_t time_of_system;
        struct timeval time_now{};
        gettimeofday(&time_now, nullptr);

        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),&output_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1,TensorShape({1}), &time_sec));
        OP_REQUIRES_OK(context, context->allocate_output(2,TensorShape({1}), &time_millisec));                                                       
        auto output_flat = output_tensor->flat<int32>();
        auto input = input_tensor.flat<int32>();
        const int N = input.size();
        for (int i = 0; i < N; i++) {
            output_flat(i) = input(i);
        } 
        auto time_sec_vlaue = time_sec->flat<int32>();
        auto time_millisec_value = time_millisec->flat<int32>();
        time_sec_vlaue(0) = int(time_now.tv_sec);
        time_millisec_value(0) = int(time_now.tv_usec);
    }
};
REGISTER_KERNEL_BUILDER(Name("Timer").Device(DEVICE_CPU), TimerOp);
