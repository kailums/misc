
#include <unordered_map>

struct TritonKernelMetaData {
  const char* name_start;
  const char* func_name;
  const char* group_name;
  const char* name;
  int num_warps;
  int shared;
  std::unordered_map<std::string, int> constants;
};

const TritonKernelMetaData metadata[] = {
  { "_binary_softmax_fp32_1024_hsaco_start", "softmax_kernel_01234", "softmax_fp32", "softmax_fp32_1024", 4, 1024, { { "BLOCK_SIZE", 1024} } }, { "_binary_softmax_fp32_2048_hsaco_start", "softmax_kernel_01234", "softmax_fp32", "softmax_fp32_2048", 8, 2048, { { "BLOCK_SIZE", 2048} } }, { "_binary_softmax_fp32_4096_hsaco_start", "softmax_kernel_01234", "softmax_fp32", "softmax_fp32_4096", 16, 4096, { { "BLOCK_SIZE", 4096} } }, { "_binary_softmax_fp32_8192_hsaco_start", "softmax_kernel_01234", "softmax_fp32", "softmax_fp32_8192", 16, 4096, { { "BLOCK_SIZE", 8192} } }, { "_binary_softmax_fp32_16384_hsaco_start", "softmax_kernel_01234", "softmax_fp32", "softmax_fp32_16384", 16, 4096, { { "BLOCK_SIZE", 16384} } }, { "_binary_softmax_fp16_1024_hsaco_start", "softmax_kernel_01234", "softmax_fp16", "softmax_fp16_1024", 4, 512, { { "BLOCK_SIZE", 1024} } }, { "_binary_softmax_fp16_2048_hsaco_start", "softmax_kernel_01234", "softmax_fp16", "softmax_fp16_2048", 8, 1024, { { "BLOCK_SIZE", 2048} } }, { "_binary_softmax_fp16_4096_hsaco_start", "softmax_kernel_01234", "softmax_fp16", "softmax_fp16_4096", 16, 2048, { { "BLOCK_SIZE", 4096} } }, { "_binary_softmax_fp16_8192_hsaco_start", "softmax_kernel_01234", "softmax_fp16", "softmax_fp16_8192", 16, 2048, { { "BLOCK_SIZE", 8192} } }, { "_binary_softmax_fp16_16384_hsaco_start", "softmax_kernel_01234", "softmax_fp16", "softmax_fp16_16384", 16, 2048, { { "BLOCK_SIZE", 16384} } },
};
    