#include <cnnl.h>
#include <cnrt.h>
#include <memory>

namespace {
struct Operation {
  cnnlHandle_t handle = nullptr;
  cnnlTensorDescriptor_t tensor = nullptr;
  cnnlActivationDescriptor_t activation = nullptr;
  int kind;

  ~Operation() {
    if (activation) cnnlDestroyActivationDescriptor(activation);
    if (tensor) cnnlDestroyTensorDescriptor(tensor);
    if (handle) cnnlDestroy(handle);
  }
};

#define CNNL_CHECK(call) do { auto status = (call); if (status != CNNL_STATUS_SUCCESS) return status; } while (false)
}

extern "C" int source_create(void **result, cnrtQueue_t queue, int kind,
                              int rows, int columns) {
  if (!result || !queue || rows <= 0 || columns <= 0 || (kind != 0 && kind != 1))
    return CNNL_STATUS_BAD_PARAM;
  auto operation = std::make_unique<Operation>();
  operation->kind = kind;
  CNNL_CHECK(cnnlCreate(&operation->handle));
  CNNL_CHECK(cnnlSetQueue(operation->handle, queue));
  CNNL_CHECK(cnnlCreateTensorDescriptor(&operation->tensor));
  // A descriptor-only reshape: the original last axis is the reduction axis.
  int dimensions[] = {rows, 1, columns};
  CNNL_CHECK(cnnlSetTensorDescriptor(operation->tensor, CNNL_LAYOUT_ARRAY,
                                    CNNL_DTYPE_HALF, 3, dimensions));
  if (kind == 0) {
    CNNL_CHECK(cnnlCreateActivationDescriptor(&operation->activation));
    CNNL_CHECK(cnnlSetActivationDescriptor_v6(operation->activation,
        CNNL_ACTIVATION_RELU, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_PROPAGATE_NAN,
        0.0f, 0, 1.0f, 1.0f, true, false));
  }
  *result = operation.release();
  return CNNL_STATUS_SUCCESS;
}

extern "C" int source_run(void *context, const void *input, void *output) {
  auto &operation = *static_cast<Operation *>(context);
  if (operation.kind == 0)
    return cnnlActivationForward(operation.handle, operation.activation, nullptr,
        operation.tensor, input, nullptr, operation.tensor, output);
  return cnnlSoftmaxForward(operation.handle, CNNL_SOFTMAX_ACCURATE,
      CNNL_SOFTMAX_MODE_LOW_DIMENSION, nullptr, operation.tensor, input,
      nullptr, operation.tensor, output);
}

extern "C" void source_destroy(void *context) {
  delete static_cast<Operation *>(context);
}
