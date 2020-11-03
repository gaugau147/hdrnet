// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "third_party/array/array.h"
#include "third_party/tensorflow/core/framework/op.h"
#include "third_party/tensorflow/core/framework/op_kernel.h"
#include "third_party/tensorflow/core/framework/shape_inference.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "third_party/tensorflow/core/framework/tensor_types.h"

#include <algorithm>
#include <cmath>

#include "numerics.h"

using CpuDevice = ::Eigen::ThreadPoolDevice;
using GpuDevice = ::Eigen::GpuDevice;

using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

namespace hdrnet {



void BilateralSlice(nda::array_ref_of_rank<const float, 5> grid,
                    nda::array_ref_of_rank<const float, 3> guide,
                    nda::array_ref_of_rank<float, 4> out) {
  // - Samples centered at 0.5f.
  // - Repeating boundary conditions.
  const int grid_depth = grid.dim<1>().extent();
  const int grid_width = grid.dim<2>().extent();
  const int grid_height = grid.dim<3>().extent();
  const float scale_x = static_cast<float>(grid_width) / guide.width();
  const float scale_y = static_cast<float>(grid_height) / guide.height();

  nda::for_all_indices(out.shape(), [&](int c, int x, int y, int b) {
    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5f as well.
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    // Grid trilinear interpolation to retrieve grid(gxf, gyf, gzf, i, j).
    float value = 0.0f;
    for (int gy = gy0; gy < gy0 + 2; ++gy) {
      const int gyc = std::clamp(gy, 0, grid_height - 1);
      const float wy = LerpWeight(gy + 0.5f, gyf);

      for (int gx = gx0; gx < gx0 + 2; ++gx) {
        const int gxc = std::clamp(gx, 0, grid_width - 1);
        const float wx = LerpWeight(gx + 0.5f, gxf);

        for (int gz = gz0; gz < gz0 + 2; ++gz) {
          const int gzc = std::clamp(gz, 0, grid_depth - 1);
          const float wz = SmoothedLerpWeight(gz + 0.5f, gzf);

          value += wx * wy * wz * grid(c, gzc, gxc, gyc, b);
        }
      }
    }
    // Grid trilinear interpolation.

    out(c, x, y, b) = value;
  });
}

void BilateralSliceGridGrad(
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 5> grid_vjp_out) {
  const int grid_depth = grid_vjp_out.dim<1>().extent();
  const int grid_width = grid_vjp_out.dim<2>().extent();
  const int grid_height = grid_vjp_out.dim<3>().extent();
  const int guide_width = guide.width();
  const int guide_height = guide.height();
  const float scale_x = static_cast<float>(guide.width()) / grid_width;
  const float scale_y = static_cast<float>(guide.height()) / grid_height;

  nda::for_all_indices(grid_vjp_out.shape(), [&](int gc, int gz, int gx, int gy,
                                                 int b) {
    const int x0 = static_cast<int>(std::floor(scale_x * (gx + 0.5f - 1.0f)));
    const int x1_exclusive =
        static_cast<int>(std::ceil(scale_x * (gx + 0.5f + 1.0f)));
    const int y0 = static_cast<int>(std::floor(scale_y * (gy + 0.5f - 1.0f)));
    const int y1_exclusive =
        static_cast<int>(std::ceil(scale_y * (gy + 0.5f + 1.0f)));

    float vjp_value = 0.0f;
    for (int y = y0; y < y1_exclusive; ++y) {
      const int y_mirror = MirrorBoundary(y, guide_height);
      const float gyf = (y + 0.5f) / scale_y;
      const float wy = LerpWeight(gy + 0.5f, gyf);

      for (int x = x0; x < x1_exclusive; ++x) {
        // TODO(jiawen): Consider using clamp boundary.
        const int x_mirror = MirrorBoundary(x, guide_width);
        const float gxf = (x + 0.5f) / scale_x;
        const float wx = LerpWeight(gx + 0.5f, gxf);

        // TODO(jiawen): Offset gz by 0.5 as well.
        const float gzf = guide(x_mirror, y_mirror, b) * grid_depth;
        float wz = SmoothedLerpWeight(gz + 0.5f, gzf);
        if ((gz == 0 && gzf < 0.5f) ||
            (gz == grid_depth - 1 && gzf > grid_depth - 0.5f)) {
          wz = 1.0f;
        }

        vjp_value += wz * wx * wy * codomain_tangent(gc, x_mirror, y_mirror, b);
      }  // y
    }    // x

    grid_vjp_out(gc, gz, gx, gy, b) = vjp_value;
  });
}

void BilateralSliceGuideGrad(
    nda::array_ref_of_rank<const float, 5> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 3> guide_vjp_out) {
  const int grid_channels = grid.dim<0>().extent();
  const int grid_depth = grid.dim<1>().extent();
  const int grid_width = grid.dim<2>().extent();
  const int grid_height = grid.dim<3>().extent();
  const float scale_x = static_cast<float>(grid_width) / guide.width();
  const float scale_y = static_cast<float>(grid_height) / guide.height();

  nda::for_all_indices(guide_vjp_out.shape(), [&](int x, int y, int b) {
    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5f as well.
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    float vjp_value = 0.0f;
    for (int c = 0; c < grid_channels; ++c) {
      float grid_sample = 0.0f;

      // Grid trilinear interpolation to retrieve grid(c, gzf, gxf, gyf, gzf).
      for (int gy = gy0; gy < gy0 + 2; ++gy) {
        const int gyc = std::clamp(gy, 0, grid_height - 1);
        const float wy = LerpWeight(gy + 0.5f, gyf);

        for (int gx = gx0; gx < gx0 + 2; ++gx) {
          const int gxc = std::clamp(gx, 0, grid_width - 1);
          const float wx = LerpWeight(gx + 0.5f, gxf);

          for (int gz = gz0; gz < gz0 + 2; ++gz) {
            const int gzc = std::clamp(gz, 0, grid_depth - 1);
            // TODO(jiawen): Offset gz by 0.5 as well?
            const float dwz =
                grid_depth * SmoothedLerpWeightGrad(gz + 0.5f, gzf);

            grid_sample += wx * wy * dwz * grid(c, gzc, gxc, gyc, b);
          }
        }
      }
      vjp_value += grid_sample * codomain_tangent(c, x, y, b);
    }  // Sum over c.

    guide_vjp_out(x, y, b) = vjp_value;
  });
}


void BilateralSliceApply(nda::array_ref_of_rank<const float, 6> grid,
                         nda::array_ref_of_rank<const float, 3> guide,
                         nda::array_ref_of_rank<const float, 4> input,
                         nda::array_ref_of_rank<float, 4> out) {
  // - Samples centered at 0.5.
  // - Repeating boundary conditions.
  const int grid_input_channels = grid.dim<0>().extent();
  const int grid_depth = grid.dim<2>().extent();
  const int grid_width = grid.dim<3>().extent();
  const int grid_height = grid.dim<4>().extent();
  const int input_channels = input.dim<0>().extent();
  const int input_width = input.dim<1>().extent();
  const int input_height = input.dim<2>().extent();
  const float scale_x = static_cast<float>(grid_width) / input_width;
  const float scale_y = static_cast<float>(grid_height) / input_height;

  nda::for_all_indices(out.shape(), [&](int i, int x, int y, int b) {
    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5 as well.
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    float value = 0.0f;
    for (int j = 0; j < grid_input_channels; ++j) {
      // Grid trilinear interpolation to retrieve grid(gxf, gyf, gzf, i, j).
      float grid_sample = 0.0f;
      for (int gy = gy0; gy < gy0 + 2; ++gy) {
        const int gyc = std::clamp(gy, 0, grid_height - 1);
        const float wy = LerpWeight(gy + 0.5f, gyf);

        for (int gx = gx0; gx < gx0 + 2; ++gx) {
          const int gxc = std::clamp(gx, 0, grid_width - 1);
          const float wx = LerpWeight(gx + 0.5f, gxf);

          for (int gz = gz0; gz < gz0 + 2; ++gz) {
            const int gzc = std::clamp(gz, 0, grid_depth - 1);
            const float wz = SmoothedLerpWeight(gz + 0.5f, gzf);

            grid_sample += wx * wy * wz * grid(j, i, gzc, gxc, gyc, b);
          }  // gz
        }    // gy
      }      // gx
      // Grid trilinear interpolation.

      // Matrix multiply.
      if (j < input_channels) {
        value += grid_sample * input(j, x, y, b);
      } else {  // Offset term
        value += grid_sample;
      }
    }  // j

    out(i, x, y, b) = value;
  });
}

void BilateralSliceApplyGridGrad(
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 6> vjp_out) {
  const int grid_depth = vjp_out.dim<2>().extent();
  const int grid_width = vjp_out.dim<3>().extent();
  const int grid_height = vjp_out.dim<4>().extent();
  const int input_channels = input.dim<0>().extent();
  const int input_width = input.dim<1>().extent();
  const int input_height = input.dim<2>().extent();
  const float scale_x = static_cast<float>(input_width) / grid_width;
  const float scale_y = static_cast<float>(input_height) / grid_height;

  nda::for_all_indices(vjp_out.shape(), [&](int j, int i, int gz, int gx,
                                            int gy, int b) {
    const int x0 = static_cast<int>(std::floor(scale_x * (gx + 0.5f - 1.0f)));
    const int x1_exclusive =
        static_cast<int>(std::ceil(scale_x * (gx + 0.5f + 1.0f)));
    const int y0 = static_cast<int>(std::floor(scale_y * (gy + 0.5f - 1.0f)));
    const int y1_exclusive =
        static_cast<int>(std::ceil(scale_y * (gy + 0.5f + 1.0f)));

    float vjp_value = 0.0f;
    for (int y = y0; y < y1_exclusive; ++y) {
      const int y_mirror = MirrorBoundary(y, input_height);
      const float gyf = (y + 0.5f) / scale_y;
      const float wy = LerpWeight(gy + 0.5f, gyf);

      for (int x = x0; x < x1_exclusive; ++x) {
        // TODO(jiawen): Consider using clamp boundary.
        const int x_mirror = MirrorBoundary(x, input_width);
        const float gxf = (x + 0.5f) / scale_x;
        const float wx = LerpWeight(gx + 0.5f, gxf);

        // TODO(jiawen): Offset gz by 0.5 as well.
        const float gzf = guide(x_mirror, y_mirror, b) * grid_depth;
        float wz = SmoothedLerpWeight(gz + 0.5f, gzf);
        if ((gz == 0 && gzf < 0.5f) ||
            (gz == grid_depth - 1 && gzf > grid_depth - 0.5f)) {
          wz = 1.0f;
        }

        // Index `input` accounting for optional offset.
        const float input_value =
            (j < input_channels) ? input(j, x_mirror, y_mirror, b) : 1.0f;
        const float grad_value = wx * wy * wz * input_value;

        vjp_value += grad_value * codomain_tangent(i, x_mirror, y_mirror, b);
      }  // y
    }    // x

    vjp_out(j, i, gz, gx, gy, b) = vjp_value;
  });
}

void BilateralSliceApplyGuideGrad(
    nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 3> vjp_out) {
  const int grid_input_channels = grid.dim<0>().extent();
  const int output_channels = grid.dim<1>().extent();
  const int grid_depth = grid.dim<2>().extent();
  const int grid_width = grid.dim<3>().extent();
  const int grid_height = grid.dim<4>().extent();
  const int input_channels = input.dim<0>().extent();
  const int input_width = input.dim<1>().extent();
  const int input_height = input.dim<2>().extent();
  const float scale_x = static_cast<float>(grid_width) / input_width;
  const float scale_y = static_cast<float>(grid_height) / input_height;

  nda::for_all_indices(vjp_out.shape(), [&](int x, int y, int b) {
    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5 as well.
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    float vjp_value = 0.0f;
    for (int i = 0; i < output_channels; ++i) {
      float grad_value = 0.0f;

      for (int j = 0; j < grid_input_channels; ++j) {
        float grid_sample = 0.0f;

        // Grid trilinear interpolation to retrieve grid(gxf, gyf, gzf, i, j).
        for (int gy = gy0; gy < gy0 + 2; ++gy) {
          const int gyc = std::clamp(gy, 0, grid_height - 1);
          const float wy = LerpWeight(gy + 0.5f, gyf);

          for (int gx = gx0; gx < gx0 + 2; ++gx) {
            const int gxc = std::clamp(gx, 0, grid_width - 1);
            const float wx = LerpWeight(gx + 0.5f, gxf);

            for (int gz = gz0; gz < gz0 + 2; ++gz) {
              const int gzc = std::clamp(gz, 0, grid_depth - 1);
              // TODO(jiawen): Offset gz by 0.5 as well?
              const float dwz =
                  grid_depth * SmoothedLerpWeightGrad(gz + 0.5f, gzf);

              grid_sample += wx * wy * dwz * grid(j, i, gzc, gxc, gyc, b);
            }  // gz
          }    // gy
        }      // gx
        // Grid trilinear interpolation.

        // Index `input` accounting for optional offset.
        const float input_value =
            (j < input_channels) ? input(j, x, y, b) : 1.0f;
        grad_value += grid_sample * input_value;
      }  // Sum over j.

      vjp_value += grad_value * codomain_tangent(i, x, y, b);
    }  // Sum over i.

    vjp_out(x, y, b) = vjp_value;
  });
}

void BilateralSliceApplyInputGrad(
    nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 4> vjp_out) {
  const int output_channels = grid.dim<1>().extent();
  const int grid_depth = grid.dim<2>().extent();
  const int grid_width = grid.dim<3>().extent();
  const int grid_height = grid.dim<4>().extent();
  const int guide_width = guide.dim<0>().extent();
  const int guide_height = guide.dim<1>().extent();
  const float scale_x = static_cast<float>(grid_width) / guide_width;
  const float scale_y = static_cast<float>(grid_height) / guide_height;

  nda::for_all_indices(vjp_out.shape(), [&](int j, int x, int y, int b) {
    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5 as well.
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    float vjp_value = 0.0f;
    for (int i = 0; i < output_channels; ++i) {
      float grad_value = 0.0f;
      // Grid trilinear interpolation to retrieve grid(gxf, gyf, gzf, i, j).
      for (int gy = gy0; gy < gy0 + 2; ++gy) {
        const int gyc = std::clamp(gy, 0, grid_height - 1);
        const float wy = LerpWeight(gy + 0.5f, gyf);

        for (int gx = gx0; gx < gx0 + 2; ++gx) {
          const int gxc = std::clamp(gx, 0, grid_width - 1);
          const float wx = LerpWeight(gx + 0.5f, gxf);

          for (int gz = gz0; gz < gz0 + 2; ++gz) {
            const int gzc = std::clamp(gz, 0, grid_depth - 1);
            const float wz = SmoothedLerpWeight(gz + 0.5f, gzf);

            grad_value += wx * wy * wz * grid(j, i, gzc, gxc, gyc, b);
          }  // gz
        }    // gy
      }      // gx
      // Grid trilinear interpolation.

      vjp_value += grad_value * codomain_tangent(i, x, y, b);
    }  // Sum over i.

    vjp_out(j, x, y, b) = vjp_value;
  });
}


template <typename Device>
bool BilateralSlice(const Device& device,
                    nda::array_ref_of_rank<const float, 5> grid,
                    nda::array_ref_of_rank<const float, 3> guide,
                    nda::array_ref_of_rank<float, 4> out);

template <typename Device>
bool BilateralSliceGrad(const Device& device,
                        nda::array_ref_of_rank<const float, 5> grid,
                        nda::array_ref_of_rank<const float, 3> guide,
                        nda::array_ref_of_rank<const float, 4> codomain_tangent,
                        nda::array_ref_of_rank<float, 5> grid_vjp_out,
                        nda::array_ref_of_rank<float, 3> guide_vjp_out);

// Specialize for the CPU (ignoring the device).
template <>
bool BilateralSlice<CpuDevice>(const CpuDevice& device,
                               nda::array_ref_of_rank<const float, 5> grid,
                               nda::array_ref_of_rank<const float, 3> guide,
                               nda::array_ref_of_rank<float, 4> out) {
  BilateralSlice(grid, guide, out);
  return true;
}

template <>
bool BilateralSliceGrad<CpuDevice>(
    const CpuDevice& device, nda::array_ref_of_rank<const float, 5> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 5> grid_vjp_out,
    nda::array_ref_of_rank<float, 3> guide_vjp_out) {
  BilateralSliceGridGrad(guide, codomain_tangent, grid_vjp_out);
  BilateralSliceGuideGrad(grid, guide, codomain_tangent, guide_vjp_out);
  return true;
}

// Specialize for the GPU.
#if GOOGLE_CUDA

// Forward declare CUDA launchers since #includes are messy.
bool BilateralSliceCudaLauncher(const GpuDevice& device,
                                nda::array_ref_of_rank<const float, 5> grid,
                                nda::array_ref_of_rank<const float, 3> guide,
                                nda::array_ref_of_rank<float, 4> out);

bool BilateralSliceGradCudaLauncher(
    const GpuDevice& device, nda::array_ref_of_rank<const float, 5> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 5> grid_vjp_out,
    nda::array_ref_of_rank<float, 3> guide_vjp_out);

template <>
bool BilateralSlice<GpuDevice>(const GpuDevice& device,
                               nda::array_ref_of_rank<const float, 5> grid,
                               nda::array_ref_of_rank<const float, 3> guide,
                               nda::array_ref_of_rank<float, 4> out) {
  return BilateralSliceCudaLauncher(device, grid, guide, out);
}

template <>
bool BilateralSliceGrad<GpuDevice>(
    const GpuDevice& device, nda::array_ref_of_rank<const float, 5> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 5> grid_vjp_out,
    nda::array_ref_of_rank<float, 3> guide_vjp_out) {
  return BilateralSliceGradCudaLauncher(device, grid, guide, codomain_tangent,
                                        grid_vjp_out, guide_vjp_out);
}

#endif  // GOOGLE_CUDA

template <typename Device>
class BilateralSliceOp : public OpKernel {
 public:
  explicit BilateralSliceOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grabs the inputs.
    const Tensor& grid = context->input(0);
    const Tensor& guide = context->input(1);

    // Check tensor dims.
    OP_REQUIRES(context, grid.dims() == 5,
                tensorflow::errors::InvalidArgument(
                    "Grid should be 5D (batch_size, grid_height, grid_width, "
                    "grid_depth, grid_channels)."));
    OP_REQUIRES(context, guide.dims() == 3,
                tensorflow::errors::InvalidArgument(
                    "Guide image should be 3D (batch_size, height, width)."));

    // Input shapes.
    const int batch_size = grid.dim_size(0);
    const int grid_height = grid.dim_size(1);
    const int grid_width = grid.dim_size(2);
    const int grid_depth = grid.dim_size(3);
    const int grid_channels = grid.dim_size(4);
    const int guide_height = guide.dim_size(1);
    const int guide_width = guide.dim_size(2);

    // Allocate output tensor.
    const TensorShape output_shape(
        {batch_size, guide_height, guide_width, grid_channels});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    // TF: (b, h, w, d, c), c changes fastest.
    // nda: (c, d, w, h, b), c changes fastest.
    auto grid_ref = nda::make_array_ref(
        grid.flat<float>().data(),
        nda::shape_of_rank<5>(grid_channels, grid_depth, grid_width,
                              grid_height, batch_size));
    // TF: (b, h, w), w changes fastest.
    // nda: (w, h, b), w changes fastest.
    auto guide_ref = nda::make_array_ref(
        guide.flat<float>().data(),
        nda::shape_of_rank<3>(guide_width, guide_height, batch_size));

    // TF: (b, h, w, c), c changes fastest.
    // nda: (c, w, h, b), c changes fastest.
    auto output_ref =
        nda::make_array_ref(output->flat<float>().data(),
                            nda::shape_of_rank<4>(grid_channels, guide_width,
                                                  guide_height, batch_size));

    const bool status = BilateralSlice(context->eigen_device<Device>(),
                                       grid_ref, guide_ref, output_ref);
    if (!status) {
      context->SetStatus(
          tensorflow::errors::Internal("BilateralSlice kernel failed."));
    }
  }
};

template <typename Device>
class BilateralSliceGradOp : public OpKernel {
 public:
  explicit BilateralSliceGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the inputs.
    const Tensor& grid = context->input(0);
    const Tensor& guide = context->input(1);
    const Tensor& codomain_tangent = context->input(2);

    // Check tensor dims.
    OP_REQUIRES(context, grid.dims() == 5,
                tensorflow::errors::InvalidArgument(
                    "Grid should be 5D with shape (batch_size, grid_height, "
                    "grid_width, grid_depth, grid_channels)."));
    OP_REQUIRES(context, guide.dims() == 3,
                tensorflow::errors::InvalidArgument(
                    "Guide image should be 3D (batch_size, height, width)."));
    OP_REQUIRES(context, codomain_tangent.dims() == 4,
                tensorflow::errors::InvalidArgument(
                    "Codomain tangent should be 4D (batch, height, width, "
                    "nchannels))."));

    // Allocate vjp buffers, which have the same shape as the primals.
    Tensor* grid_vjp = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, grid.shape(), &grid_vjp));
    Tensor* guide_vjp = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, guide.shape(), &guide_vjp));

    // Input shapes.
    const int batch_size = grid.dim_size(0);
    const int grid_height = grid.dim_size(1);
    const int grid_width = grid.dim_size(2);
    const int grid_depth = grid.dim_size(3);
    const int grid_channels = grid.dim_size(4);
    const int guide_height = guide.dim_size(1);
    const int guide_width = guide.dim_size(2);

    // `grid` and `grid_vjp`:
    //
    // TF: (b, h, w, d, c), c changes fastest.
    // nda: (c, d, w, h, b), c changes fastest.
    auto grid_ref = nda::make_array_ref(
        grid.flat<float>().data(),
        nda::shape_of_rank<5>(grid_channels, grid_depth, grid_width,
                              grid_height, batch_size));
    auto grid_vjp_ref =
        nda::make_array_ref(grid_vjp->flat<float>().data(), grid_ref.shape());

    // `guide` and `guide_vjp`:
    //
    // TF: (b, h, w), w changes fastest.
    // nda: (w, h, b), w changes fastest.
    auto guide_ref = nda::make_array_ref(
        guide.flat<float>().data(),
        nda::shape_of_rank<3>(guide_width, guide_height, batch_size));
    auto guide_vjp_ref =
        nda::make_array_ref(guide_vjp->flat<float>().data(), guide_ref.shape());

    // `codomain_tangent`:
    //
    // TF: (b, h, w, c), c changes fastest.
    // nda: (c, w, h, b), c changes fastest.
    auto codomain_tangent_ref =
        nda::make_array_ref(codomain_tangent.flat<float>().data(),
                            nda::shape_of_rank<4>(grid_channels, guide_width,
                                                  guide_height, batch_size));

    const bool status =
        BilateralSliceGrad(context->eigen_device<Device>(), grid_ref, guide_ref,
                           codomain_tangent_ref, grid_vjp_ref, guide_vjp_ref);
    if (!status) {
      context->SetStatus(
          tensorflow::errors::Internal("BilateralSliceGrad kernel failed."));
    }
  }
};


// Declare BilateralSlice and BilateralSliceGrad templated on the device. They
// will be specialized for each device.
template <typename Device>
bool BilateralSliceApply(const Device& device,
                         nda::array_ref_of_rank<const float, 6> grid,
                         nda::array_ref_of_rank<const float, 3> guide,
                         nda::array_ref_of_rank<const float, 4> input,
                         nda::array_ref_of_rank<float, 4> out);

template <typename Device>
bool BilateralSliceApplyGrad(
    const Device& device, nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 6> grid_vjp_out,
    nda::array_ref_of_rank<float, 3> guide_vjp_out,
    nda::array_ref_of_rank<float, 4> input_vjp_out);

// Specialize for the CPU (ignoring the device).
template <>
bool BilateralSliceApply<CpuDevice>(
    const CpuDevice& device, nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<float, 4> out) {
  BilateralSliceApply(grid, guide, input, out);
  return true;
}

template <>
bool BilateralSliceApplyGrad<CpuDevice>(
    const CpuDevice& device, nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 6> grid_vjp_out,
    nda::array_ref_of_rank<float, 3> guide_vjp_out,
    nda::array_ref_of_rank<float, 4> input_vjp_out) {
  BilateralSliceApplyGridGrad(guide, input, codomain_tangent, grid_vjp_out);
  BilateralSliceApplyGuideGrad(grid, guide, input, codomain_tangent,
                               guide_vjp_out);
  BilateralSliceApplyInputGrad(grid, guide, codomain_tangent, input_vjp_out);
  return true;
}

// Specialize for the GPU.
#if GOOGLE_CUDA

// Forward declare CUDA launchers since #includes are messy.
bool BilateralSliceApplyCudaLauncher(
    const GpuDevice& device, nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<float, 4> out);

bool BilateralSliceApplyGradCudaLauncher(
    const GpuDevice& device, nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 6> grid_vjp_out,
    nda::array_ref_of_rank<float, 3> guide_vjp_out,
    nda::array_ref_of_rank<float, 4> input_vjp_out);

template <>
bool BilateralSliceApply<GpuDevice>(
    const GpuDevice& device, nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<float, 4> out) {
  return BilateralSliceApplyCudaLauncher(device, grid, guide, input, out);
}

template <>
bool BilateralSliceApplyGrad<GpuDevice>(
    const GpuDevice& device, nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 6> grid_vjp_out,
    nda::array_ref_of_rank<float, 3> guide_vjp_out,
    nda::array_ref_of_rank<float, 4> input_vjp_out) {
  return BilateralSliceApplyGradCudaLauncher(device, grid, guide, input,
                                             codomain_tangent, grid_vjp_out,
                                             guide_vjp_out, input_vjp_out);
}

#endif  // GOOGLE_CUDA

template <typename Device>
class BilateralSliceApplyOp : public OpKernel {
 private:
  bool has_offset_;

 public:
  explicit BilateralSliceApplyOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("has_offset", &has_offset_));
  }

  void Compute(OpKernelContext* context) override {
    // Grabs the inputs.
    const Tensor& grid = context->input(0);
    const Tensor& guide = context->input(1);
    const Tensor& input = context->input(2);

    // Check tensor dims.
    OP_REQUIRES(context, grid.dims() == 5,
                tensorflow::errors::InvalidArgument(
                    "Input grid should be 5D (batch_size, height, width, "
                    "depth, output_channels * input_channels)"));
    OP_REQUIRES(context, guide.dims() == 3,
                tensorflow::errors::InvalidArgument(
                    "Guide image should be 3D (batch_size, height, width)"));
    OP_REQUIRES(context, input.dims() == 4,
                tensorflow::errors::InvalidArgument(
                    "Input image should be 4D (batch_size, height, width, "
                    "input_channels)"));

    // Input shapes.
    const int batch_size = grid.dim_size(0);
    const int grid_height = grid.dim_size(1);
    const int grid_width = grid.dim_size(2);
    const int grid_depth = grid.dim_size(3);
    const int grid_channels = grid.dim_size(4);
    const int guide_height = guide.dim_size(1);
    const int guide_width = guide.dim_size(2);
    const int input_channels = input.dim_size(3);

    OP_REQUIRES(context,
                (input.dim_size(0) == guide.dim_size(0)) &&
                    input.dim_size(1) == guide_height &&
                    input.dim_size(2) == guide_width,
                tensorflow::errors::InvalidArgument(
                    "Input and guide size should match."));
    OP_REQUIRES(
        context, guide.dim_size(0) == batch_size,
        tensorflow::errors::InvalidArgument("Batch sizes should match."));

    // Check grid and input shape compatibility.
    const int grid_input_channels =
        has_offset_ ? input_channels + 1 : input_channels;
    const int output_channels = grid_channels / grid_input_channels;
    if (has_offset_) {
      OP_REQUIRES(context, grid_channels % grid_input_channels == 0,
                  tensorflow::errors::InvalidArgument(
                      "Slicing with affine offset, grid should have "
                      "output_channels * (input_channels + 1) channels."));
    } else {
      OP_REQUIRES(context, grid_channels % grid_input_channels == 0,
                  tensorflow::errors::InvalidArgument(
                      "Slicing without affine offset, grid should have "
                      "output_channels * input_channels channels."));
    }

    // Allocate output tensor.
    const TensorShape output_shape(
        {batch_size, guide_height, guide_width, output_channels});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    // TF: (b, h, w, d, c), c changes fastest.
    // nda: (c, d, w, h, b), c changes fastest.
    // reinterpret in nda as (j, i, d, w, h, b), j changes fastest, then i.
    auto grid_ref = nda::make_array_ref(
        grid.flat<float>().data(),
        nda::shape_of_rank<6>(grid_input_channels, output_channels, grid_depth,
                              grid_width, grid_height, batch_size));

    // TF: (b, h, w), w changes fastest.
    // nda: (w, h, b), w changes fastest.
    auto guide_ref = nda::make_array_ref(
        guide.flat<float>().data(),
        nda::shape_of_rank<3>(guide_width, guide_height, batch_size));

    // TF: (b, h, w, j), w changes fastest.
    // nda: (j, w, h, b), j changes fastest.
    auto input_ref =
        nda::make_array_ref(input.flat<float>().data(),
                            nda::shape_of_rank<4>(input_channels, guide_width,
                                                  guide_height, batch_size));

    // TF: (b, h, w, i), w changes fastest.
    // nda: (i, w, h, b), i changes fastest.
    auto output_ref =
        nda::make_array_ref(output->flat<float>().data(),
                            nda::shape_of_rank<4>(output_channels, guide_width,
                                                  guide_height, batch_size));
    const bool status =
        BilateralSliceApply(context->eigen_device<Device>(), grid_ref,
                            guide_ref, input_ref, output_ref);
    if (!status) {
      context->SetStatus(
          tensorflow::errors::Internal("BilateralSliceApply kernel failed."));
    }
  }
};

template <typename Device>
class BilateralSliceApplyGradOp : public OpKernel {
 private:
  bool has_offset_;

 public:
  explicit BilateralSliceApplyGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("has_offset", &has_offset_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the inputs.
    const Tensor& grid = context->input(0);
    const Tensor& guide = context->input(1);
    const Tensor& input = context->input(2);
    const Tensor& codomain_tangent = context->input(3);

    // Check tensor dims.
    OP_REQUIRES(context, grid.dims() == 5,
                tensorflow::errors::InvalidArgument(
                    "Grid should be 5D (batch, h, w, depth, output_channels * "
                    "input_channels)"));
    OP_REQUIRES(context, guide.dims() == 3,
                tensorflow::errors::InvalidArgument(
                    "Guide image should be 3D (batches, height, width)"));
    OP_REQUIRES(context, input.dims() == 4,
                tensorflow::errors::InvalidArgument(
                    "Input image should be 4D (batches, height, width, "
                    "input_channels)"));

    // Input shapes.
    const int batch_size = guide.dim_size(0);
    const int guide_height = guide.dim_size(1);
    const int guide_width = guide.dim_size(2);
    const int grid_height = grid.dim_size(1);
    const int grid_width = grid.dim_size(2);
    const int grid_depth = grid.dim_size(3);
    const int grid_channels = grid.dim_size(4);
    const int input_channels = input.dim_size(3);

    // Allocate vjp buffers, which have the same shape as the primals.
    Tensor* grid_vjp = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, grid.shape(), &grid_vjp));
    Tensor* guide_vjp = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, guide.shape(), &guide_vjp));
    Tensor* input_vjp = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, input.shape(), &input_vjp));

    // TODO(jiawen): Do extra shape validation here, or maybe shape inference
    // will take care of it.

    // Check grid and input shape compatibility.
    const int grid_input_channels =
        has_offset_ ? input_channels + 1 : input_channels;
    const int output_channels = grid_channels / grid_input_channels;
    if (has_offset_) {
      OP_REQUIRES(context, grid_channels % grid_input_channels == 0,
                  tensorflow::errors::InvalidArgument(
                      "Slicing with affine offset, grid should have "
                      "output_channels * (input_channels + 1) channels."));
    } else {
      OP_REQUIRES(context, grid_channels % grid_input_channels == 0,
                  tensorflow::errors::InvalidArgument(
                      "Slicing without affine offset, grid should have "
                      "output_channels * input_channels channels."));
    }

    // `grid` and `grid_vjp`:
    //
    // TF: (b, h, w, d, c), c changes fastest.
    // nda: (c, d, w, h, b), c changes fastest.
    // reinterpret in nda as (j, i, d, w, h, b), j changes fastest, then i.
    auto grid_ref = nda::make_array_ref(
        grid.flat<float>().data(),
        nda::shape_of_rank<6>(grid_input_channels, output_channels, grid_depth,
                              grid_width, grid_height, batch_size));
    auto grid_vjp_ref = nda::make_array_ref(
        grid_vjp->flat<float>().data(),
        nda::shape_of_rank<6>(grid_input_channels, output_channels, grid_depth,
                              grid_width, grid_height, batch_size));

    // `guide` and `guide_vjp`:
    //
    // TF: (b, h, w), w changes fastest.
    // nda: (w, h, b), w changes fastest.
    auto guide_ref = nda::make_array_ref(
        guide.flat<float>().data(),
        nda::shape_of_rank<3>(guide_width, guide_height, batch_size));
    auto guide_vjp_ref = nda::make_array_ref(
        guide_vjp->flat<float>().data(),
        nda::shape_of_rank<3>(guide_width, guide_height, batch_size));

    // `input` and `input_vjp`:
    //
    // TF: (b, h, w, j), w changes fastest.
    // nda: (j, w, h, b), j changes fastest.
    auto input_ref =
        nda::make_array_ref(input.flat<float>().data(),
                            nda::shape_of_rank<4>(input_channels, guide_width,
                                                  guide_height, batch_size));
    auto input_vjp_ref =
        nda::make_array_ref(input_vjp->flat<float>().data(),
                            nda::shape_of_rank<4>(input_channels, guide_width,
                                                  guide_height, batch_size));
    // `codomain_tangent`:
    //
    // TF: (b, h, w, i), i changes fastest.
    // nda: (i, w, h, b), i changes fastest.
    auto codomain_tangent_ref =
        nda::make_array_ref(codomain_tangent.flat<float>().data(),
                            nda::shape_of_rank<4>(output_channels, guide_width,
                                                  guide_height, batch_size));

    const bool status = BilateralSliceApplyGrad(
        context->eigen_device<Device>(), grid_ref, guide_ref, input_ref,
        codomain_tangent_ref, grid_vjp_ref, guide_vjp_ref, input_vjp_ref);
    if (!status) {
      context->SetStatus(tensorflow::errors::Internal(
          "BilateralSliceApplyGrad kernel failed."));
    }
  }
};
}  // namespace hdrnet

REGISTER_KERNEL_BUILDER(Name("BilateralSlice").Device(tensorflow::DEVICE_CPU),
                        hdrnet::BilateralSliceOp<CpuDevice>);
REGISTER_KERNEL_BUILDER(
    Name("BilateralSliceGrad").Device(tensorflow::DEVICE_CPU),
    hdrnet::BilateralSliceGradOp<CpuDevice>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("BilateralSlice").Device(tensorflow::DEVICE_GPU),
                        hdrnet::BilateralSliceOp<GpuDevice>);
REGISTER_KERNEL_BUILDER(
    Name("BilateralSliceGrad").Device(tensorflow::DEVICE_GPU),
    hdrnet::BilateralSliceGradOp<GpuDevice>);
#endif  // GOOGLE_CUDA

REGISTER_OP("BilateralSlice")
    .Input("grid: float")
    .Input("guide: float")
    .Output("out: float")
    .Doc("Slices grid at the location defined by guide to produce output.")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle grid;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &grid));
      ShapeHandle guide;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &guide));
      const DimensionHandle batch_size = c->Dim(grid, 0);
      const DimensionHandle h = c->Dim(guide, 1);
      const DimensionHandle w = c->Dim(guide, 2);
      const DimensionHandle grid_channels = c->Dim(grid, 4);
      c->set_output(0, c->MakeShape({batch_size, h, w, grid_channels}));
      return Status::OK();
    });

// TODO(jiawen): Investigate whether we need to set a shape function for
// gradient ops or if they're automatic.
REGISTER_OP("BilateralSliceGrad")
    .Input("grid: float")
    .Input("guide: float")
    .Input("backprop: float")
    .Output("grid_grad: float")
    .Output("guide_grad: float");


REGISTER_KERNEL_BUILDER(
    Name("BilateralSliceApply").Device(tensorflow::DEVICE_CPU),
    hdrnet::BilateralSliceApplyOp<CpuDevice>);
REGISTER_KERNEL_BUILDER(
    Name("BilateralSliceApplyGrad").Device(tensorflow::DEVICE_CPU),
    hdrnet::BilateralSliceApplyGradOp<CpuDevice>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("BilateralSliceApply").Device(tensorflow::DEVICE_GPU),
    hdrnet::BilateralSliceApplyOp<GpuDevice>);
REGISTER_KERNEL_BUILDER(
    Name("BilateralSliceApplyGrad").Device(tensorflow::DEVICE_GPU),
    hdrnet::BilateralSliceApplyGradOp<GpuDevice>);
#endif  // GOOGLE_CUDA

REGISTER_OP("BilateralSliceApply")
    .Input("grid: float")
    .Input("guide: float")
    .Input("input: float")
    .Attr("has_offset: bool")
    .Output("out: float")
    .Doc(
        "Slices grid at the location defined by guide and applies it to input.")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle grid;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &grid));
      ShapeHandle guide;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &guide));
      ShapeHandle input_image;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &input_image));
      const DimensionHandle batch_size = c->Dim(grid, 0);
      const DimensionHandle h = c->Dim(input_image, 1);
      const DimensionHandle w = c->Dim(input_image, 2);
      DimensionHandle output_channels;
      bool has_offset;
      TF_RETURN_IF_ERROR(c->GetAttr("has_offset", &has_offset));
      if (has_offset) {
        // With affine offset:
        // output_channels = grid_channels / (input_channels + 1).
        DimensionHandle input_channels_offset;
        TF_RETURN_IF_ERROR(
            c->Add(c->Dim(input_image, 3), 1, &input_channels_offset));
        TF_RETURN_IF_ERROR(c->Divide(c->Dim(grid, 4), input_channels_offset,
                                     true, &output_channels));
      } else {
        // Without affine offset:
        // output_channels = grid_channels / channels_in.
        TF_RETURN_IF_ERROR(c->Divide(c->Dim(grid, 4), c->Dim(input_image, 3),
                                     true, &output_channels));
      }
      c->set_output(0, c->MakeShape({batch_size, h, w, output_channels}));
      return Status::OK();
    });

// TODO(jiawen): Investigate whether we need to set a shape function for
// gradient ops or if they're automatic.
REGISTER_OP("BilateralSliceApplyGrad")
    .Input("grid: float")
    .Input("guide: float")
    .Input("input: float")
    .Input("backprop: float")
    .Attr("has_offset: bool")
    .Output("grid_grad: float")
    .Output("guide_grad: float")
    .Output("input_grad: float");
