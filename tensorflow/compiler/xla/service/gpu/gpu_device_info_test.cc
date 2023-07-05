/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/gpu_device_info.h"

#include <string>

#include "absl/strings/string_view.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "tsl/platform/test.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"

#endif

//[YG]


#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "xla/service/hlo_cost_analysis.h"
#include "xla/tools/hlo_module_loader.h"
#include "tsl/platform/init_main.h"
#include "xla/service/gpu/backend_configs.pb.h"

// [YG]
// For GPU backend
#include "xla/service/gpu/gpu_hlo_schedule.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/gpu_device_info.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace stream_executor {
namespace gpu {
namespace {

namespace se = stream_executor;

TEST(DeviceInfoTest, DeviceInfoIsCorrect) {
  std::string test_platform = "cuda";
  #if TENSORFLOW_USE_ROCM
    test_platform = "rocm";
  #endif
  // GPU 0
  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName(test_platform).value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  const xla::gpu::GpuDeviceInfo dev_info = xla::gpu::GetGpuDeviceInfo(executor);
  absl::string_view name(dev_info.name);
  std::cout<<"gpu NAME: "<<name<<"\n"; // NVIDIA A100-SXM4-80GB
  std::cout<<"gpu threads_per_block_limit: "<<dev_info.threads_per_block_limit<<"\n"; //1024
  
  // GPU 1
  se::StreamExecutor* executor_ = platform->ExecutorForDevice(1).value();
  const xla::gpu::GpuDeviceInfo dev_info_ = xla::gpu::GetGpuDeviceInfo(executor);
  absl::string_view name_(dev_info_.name);
  std::cout<<"gpu NAME: "<<name_<<"\n"; // NVIDIA A100-SXM4-80GB
  std::cout<<"gpu threads_per_block_limit: "<<dev_info_.threads_per_block_limit<<"\n"; //1024

}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor

namespace {
const char* const kUsage = R"(
This tool prints the compute cost (flops and memory traffic) of an HLO module.

The input file can be obtained from XProf graph viewer by clicking
"Download as short text".

Usage:

  bazel run compute_cost -- -input=path/to/hlo_module -format=[hlo|pb|pbtxt]
)";
}  // namespace

int main(int argc, char** argv) {
  std::string input, format;
  input = "/root/yg/xla/xla/tools/data/add.hlo";
  format = "hlo";
  bool bool_profile = true;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input", &input, "input file"),
      tsl::Flag("format", &format, "hlo|pb|pbtxt")};
      tsl::Flag("xla_hlo_profile", &bool_profile, "bool value for real profile");
  xla::AppendDebugOptionsFlags(&flag_list);
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);
  if (!parse_ok) {
    LOG(QFATAL) << kUsageString;
  }

  xla::HloCostAnalysis analysis([](const xla::Shape& shape) {
    return xla::ShapeUtil::ByteSizeOf(shape, 8);
  });

  TF_CHECK_OK(xla::LoadModuleFromFile(input, {}, format)
                  .value()
                  ->entry_computation()
                  ->root_instruction()
                  ->Accept(&analysis));

  std::cout << std::setw(5) << std::setprecision(4)
            << analysis.flop_count() / (1e9) << " GFLOPS. "
            << analysis.bytes_accessed() / (1e6) << " MiB." << std::endl;
  

  // Multi GPUs

  /*auto computation = xla::LoadModuleFromFile(input, {}, format)
                      .value()
                      ->entry_computation();
  XlaBuilder builder("add");
  TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build());
  std::string profile_output;*/
  //ExecuteAndFetchProfile(&profile_output, client, computation, lhs_shape,
  //                       rhs_shape);
  namespace se = stream_executor;
  std::string test_platform = "cuda";
  #if TENSORFLOW_USE_ROCM
    test_platform = "rocm";
  #endif

  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName(test_platform).value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value(); // GPU 0
  const xla::gpu::GpuDeviceInfo dev_info = xla::gpu::GetGpuDeviceInfo(executor);
  absl::string_view name(dev_info.name);
  std::cout<<"gpu NAME: "<<name<<"\n"; // NVIDIA A100-SXM4-80GB
  std::cout<<"gpu threads_per_block_limit: "<<dev_info.threads_per_block_limit<<"\n"; //1024

  //auto test_backend = xla::gpu::backend();
  //auto a = xla::gpu::backend()
  //std::cout<<"a: "<<type(a)<<"\n";
  //::testing::InitGoogleTest(&argc, argv);
  
  //return RUN_ALL_TESTS();

  return 0;
}
