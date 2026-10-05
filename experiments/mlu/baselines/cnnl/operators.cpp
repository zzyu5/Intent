#include <cnnl.h>
#include <cnrt.h>
#include <cstdio>
#include <memory>

namespace {
struct Operation {
  cnnlHandle_t handle = nullptr;
  cnnlTensorDescriptor_t tensor = nullptr;
  cnnlTensorDescriptor_t lhs = nullptr, rhs = nullptr;
  cnnlActivationDescriptor_t activation = nullptr;
  cnnlMatMulDescriptor_t matrix = nullptr;
  cnnlMatMulAlgo_t algorithm = nullptr;
  cnnlMatMulHeuristicResult_t heuristic = nullptr;
  void *workspace = nullptr;
  size_t workspaceBytes = 0;
  int kind;

  ~Operation() {
    if (workspace) cnrtFree(workspace);
    if (heuristic) cnnlDestroyMatMulHeuristicResult(heuristic);
    if (algorithm) cnnlMatMulAlgoDestroy(algorithm);
    if (matrix) cnnlMatMulDescDestroy(matrix);
    if (rhs) cnnlDestroyTensorDescriptor(rhs);
    if (lhs) cnnlDestroyTensorDescriptor(lhs);
    if (activation) cnnlDestroyActivationDescriptor(activation);
    if (tensor) cnnlDestroyTensorDescriptor(tensor);
    if (handle) cnnlDestroy(handle);
  }
};

#define CNNL_CHECK(call) do { auto status = (call); if (status != CNNL_STATUS_SUCCESS) { \
  std::fprintf(stderr, "CNNL %s failed: %s\n", #call, cnnlGetErrorString(status)); return status; \
} } while (false)
}

extern "C" int source_create(void **result, cnrtQueue_t queue, int kind,
                              int rows, int columns, int depth) {
  if (!result || !queue || rows <= 0 || columns <= 0 || kind < 0 || kind > 2 ||
      (kind == 2 && depth <= 0))
    return CNNL_STATUS_BAD_PARAM;
  auto operation = std::make_unique<Operation>();
  operation->kind = kind;
  CNNL_CHECK(cnnlCreate(&operation->handle));
  CNNL_CHECK(cnnlSetQueue(operation->handle, queue));
  CNNL_CHECK(cnnlCreateTensorDescriptor(&operation->tensor));
  if (kind == 2) {
    CNNL_CHECK(cnnlCreateTensorDescriptor(&operation->lhs));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&operation->rhs));
    int aShape[] = {rows, depth}, bShape[] = {depth, columns}, cShape[] = {rows, columns};
    CNNL_CHECK(cnnlSetTensorDescriptor(operation->lhs, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF, 2, aShape));
    CNNL_CHECK(cnnlSetTensorDescriptor(operation->rhs, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF, 2, bShape));
    CNNL_CHECK(cnnlSetTensorDescriptor(operation->tensor, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF, 2, cShape));
    CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(operation->tensor, CNNL_DTYPE_FLOAT));
    CNNL_CHECK(cnnlMatMulDescCreate(&operation->matrix));
    cnnlDataType_t compute = CNNL_DTYPE_FLOAT;
    int no = 0;
    CNNL_CHECK(cnnlSetMatMulDescAttr(operation->matrix, CNNL_MATMUL_DESC_COMPUTE_TYPE, &compute, sizeof(compute)));
    CNNL_CHECK(cnnlSetMatMulDescAttr(operation->matrix, CNNL_MATMUL_DESC_TRANSA, &no, sizeof(no)));
    CNNL_CHECK(cnnlSetMatMulDescAttr(operation->matrix, CNNL_MATMUL_DESC_TRANSB, &no, sizeof(no)));
    CNNL_CHECK(cnnlSetMatMulDescAttr(operation->matrix, CNNL_MATMUL_USE_BETA, &no, sizeof(no)));
    CNNL_CHECK(cnnlMatMulAlgoCreate(&operation->algorithm));
    CNNL_CHECK(cnnlCreateMatMulHeuristicResult(&operation->heuristic));
    int returned = 0;
    CNNL_CHECK(cnnlGetMatMulAlgoHeuristic(operation->handle, operation->matrix,
        operation->lhs, operation->rhs, operation->tensor, operation->tensor, nullptr,
        1, &operation->heuristic, &returned));
    if (returned != 1) return CNNL_STATUS_NOT_SUPPORTED;
    CNNL_CHECK(cnnlGetMatMulHeuristicResult(operation->heuristic, operation->algorithm, &operation->workspaceBytes));
    if (operation->workspaceBytes && cnrtMalloc(&operation->workspace, operation->workspaceBytes) != 0)
      return CNNL_STATUS_ALLOC_FAILED;
    *result = operation.release();
    return CNNL_STATUS_SUCCESS;
  }
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

extern "C" int source_run(void *context, const void *input, const void *rhs, void *output) {
  auto &operation = *static_cast<Operation *>(context);
  if (operation.kind == 2) {
    float alpha = 1.0f, beta = 0.0f;
    return cnnlMatMul_v2(operation.handle, operation.matrix, operation.algorithm,
        &alpha, operation.lhs, input, operation.rhs, rhs, &beta,
        operation.tensor, output, operation.workspace, operation.workspaceBytes,
        operation.tensor, output);
  }
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
