#include "VSHelper4.h"
#include "VapourSynth4.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <numbers>
#include <vector>

static constexpr auto DEFAULT_BLOCKSIZE = 32;
static constexpr auto DEFAULT_THRESHOLD_W = 2.0f;
static constexpr auto DEFAULT_ANGLE_TOLERANCE = 8.0f;
static constexpr auto DEFAULT_W_JNB = 3.0f;
static constexpr auto DEFAULT_PERCENTILE = 22.0f;
static constexpr auto DEFAULT_SOBEL_THRESHOLD = 0.1f;

typedef enum {
    OUTPUT_MODE_ORIGINAL = 0,
    OUTPUT_MODE_SHARPNESS_MAP = 1
} OutputMode;

typedef struct {
    VSNode* node;
    const VSVideoInfo* vi;
    float percentile;
    int blocksize;
    float threshold_w;
    float angle_tolerance;
    float tan_angle_tol;
    float w_jnb;
    float sobel_threshold;
    OutputMode output_mode;
} PSIData;

template <typename T, auto NeedSharpnessMap = false>
static inline auto
calculatePSI(const T* VS_RESTRICT
                 src, // clang handles `auto __restrict` but gcc & msvc don't
             auto width, auto height, auto stride, auto percentile,
             auto max_val, auto blocksize, auto threshold_w, auto tan_angle_tol,
             auto w_jnb, auto sobel_threshold,
             auto sharpness_map = nullptr) noexcept {
    using namespace Eigen;

    const auto stride_elements = stride / sizeof(T);

    const auto sobel_threshold_scaled =
        (std::is_same_v<T, float>)
            ? sobel_threshold
            : sobel_threshold * static_cast<float>(max_val);

    Map<const Matrix<T, Dynamic, Dynamic, RowMajor>, 0, OuterStride<>>
        src_matrix(src, height, width, OuterStride<>(stride_elements));

    MatrixXf image_for_sobel = src_matrix.template cast<float>();

    MatrixXf image_float;
    if constexpr (std::is_same_v<T, float>) {
        image_float = src_matrix.template cast<float>();
    } else {
        image_float =
            src_matrix.template cast<float>() / static_cast<float>(max_val);
    }

    MatrixXf Ix = MatrixXf::Zero(height, width);
    MatrixXf Iy = MatrixXf::Zero(height, width);

    for (auto y = 1; y < height - 1; ++y) {
        for (auto x = 1; x < width - 1; ++x) {
            const auto h_grad_up =
                image_for_sobel(y - 1, x + 1) - image_for_sobel(y - 1, x - 1);
            const auto h_grad_mid =
                image_for_sobel(y, x + 1) - image_for_sobel(y, x - 1);
            const auto h_grad_down =
                image_for_sobel(y + 1, x + 1) - image_for_sobel(y + 1, x - 1);
            Ix(y, x) = h_grad_up + 2.0f * h_grad_mid + h_grad_down;

            const auto v_grad_left =
                image_for_sobel(y + 1, x - 1) - image_for_sobel(y - 1, x - 1);
            const auto v_grad_mid =
                image_for_sobel(y + 1, x) - image_for_sobel(y - 1, x);
            const auto v_grad_right =
                image_for_sobel(y + 1, x + 1) - image_for_sobel(y - 1, x + 1);
            Iy(y, x) = v_grad_left + 2.0f * v_grad_mid + v_grad_right;
        }
    }

    auto magnitude_sq = Ix.cwiseProduct(Ix) + Iy.cwiseProduct(Iy);

    const auto sobel_threshold_sq =
        sobel_threshold_scaled * sobel_threshold_scaled;
    Array<bool, Dynamic, Dynamic> edges =
        (magnitude_sq.array() > sobel_threshold_sq);

    MatrixXf edge_widths = MatrixXf::Zero(height, width);
    auto widths_count = 0;

    for (auto y = 0; y < height; ++y) {
        for (auto x = 0; x < width; ++x) {
            if (!edges(y, x)) {
                continue;
            }

            const auto ix_val = Ix(y, x);
            const auto iy_val = Iy(y, x);

            if (ix_val == 0.0f && iy_val == 0.0f) {
                continue;
            }

            if (std::abs(ix_val) < std::abs(iy_val) * tan_angle_tol) {
                auto width_up = 0;
                auto width_down = 0;
                auto valid_width = false;
                auto min_val = 0.0f;
                auto max_val_local = 0.0f;

                if (iy_val < 0.0f) { // Angle near -90 deg
                    auto prev_val_up = image_float(y, x);
                    for (auto d = 1; d < height; d++) {
                        auto up = y - d;
                        if (up < 0) {
                            width_up = -1;
                            break;
                        }
                        const auto curr_val_up = image_float(up, x);
                        if (curr_val_up <= prev_val_up) {
                            width_up = d - 1;
                            max_val_local = prev_val_up;
                            break;
                        }
                        prev_val_up = curr_val_up;
                    }

                    auto prev_val_down = image_float(y, x);
                    for (auto d = 1; d < height; d++) {
                        auto down = y + d;
                        if (down >= height) {
                            width_down = -1;
                            break;
                        }
                        const auto curr_val_down = image_float(down, x);
                        if (curr_val_down >= prev_val_down) {
                            width_down = d - 1;
                            min_val = prev_val_down;
                            break;
                        }
                        prev_val_down = curr_val_down;
                    }

                    if (width_up != -1 && width_down != -1) {
                        valid_width = true;
                        const auto mag = std::sqrt(magnitude_sq(y, x));
                        const auto cos_phi2 = std::abs(iy_val) / mag;
                        edge_widths(y, x) = (width_up + width_down) / cos_phi2;
                        const auto slope =
                            (max_val_local - min_val) / edge_widths(y, x);
                        if (edge_widths(y, x) >= w_jnb) {
                            edge_widths(y, x) -= slope;
                        }
                    }
                } else { // Angle near 90 deg (iy_val > 0)
                    auto prev_val_up = image_float(y, x);
                    for (auto d = 1; d < height; d++) {
                        auto up = y - d;
                        if (up < 0) {
                            width_up = -1;
                            break;
                        }
                        const auto curr_val_up = image_float(up, x);
                        if (curr_val_up >= prev_val_up) {
                            width_up = d - 1;
                            min_val = prev_val_up;
                            break;
                        }
                        prev_val_up = curr_val_up;
                    }

                    auto prev_val_down = image_float(y, x);
                    for (auto d = 1; d < height; d++) {
                        auto down = y + d;
                        if (down >= height) {
                            width_down = -1;
                            break;
                        }
                        const auto curr_val_down = image_float(down, x);
                        if (curr_val_down <= prev_val_down) {
                            width_down = d - 1;
                            max_val_local = prev_val_down;
                            break;
                        }
                        prev_val_down = curr_val_down;
                    }

                    if (width_up != -1 && width_down != -1) {
                        valid_width = true;
                        const auto mag = std::sqrt(magnitude_sq(y, x));
                        const auto cos_phi2 = std::abs(iy_val) / mag;
                        edge_widths(y, x) = (width_up + width_down) / cos_phi2;
                        const auto slope =
                            (max_val_local - min_val) / edge_widths(y, x);
                        if (edge_widths(y, x) >= w_jnb) {
                            edge_widths(y, x) -= slope;
                        }
                    }
                }

                if (valid_width) {
                    ++widths_count;
                }
            }
        }
    }

    if (widths_count == 0) {
        return 0.0f;
    }

    const auto row_blocks = height / blocksize;
    const auto col_blocks = width / blocksize;
    std::vector<float> avg_w;
    avg_w.reserve((row_blocks - 2) * (col_blocks - 2));

    for (auto i = 1; i < row_blocks - 1; ++i) {
        std::vector<float> local_avg_w;
        for (auto j = 1; j < col_blocks - 1; ++j) {
            const auto block_row = i * blocksize;
            const auto block_col = j * blocksize;

            const auto block_end_y = std::min(block_row + blocksize, height);
            const auto block_end_x = std::min(block_col + blocksize, width);

            auto block =
                edge_widths.block(block_row, block_col, block_end_y - block_row,
                                  block_end_x - block_col);

            const auto w_sum = block.sum();
            const auto non_zero_count = (block.array() != 0.0f).count();

            if (w_sum >= threshold_w && non_zero_count > 0) {
                local_avg_w.push_back(w_sum / non_zero_count);
            }
        }
        avg_w.insert(avg_w.end(), local_avg_w.begin(), local_avg_w.end());
    }

    if (avg_w.empty()) {
        return 0.0f;
    }

    std::sort(avg_w.begin(), avg_w.end());

    const auto quota_w = percentile * 0.01f;
    const auto nr_of_used_blocks =
        static_cast<int>(std::ceil(avg_w.size() * quota_w));

    if (nr_of_used_blocks == 0) {
        return 0.0f;
    }

    const auto blocks_to_sum =
        std::min(nr_of_used_blocks, static_cast<int>(avg_w.size()));

    Eigen::Map<const Eigen::VectorXf> avg_w_vec(avg_w.data(), blocks_to_sum);
    const auto sum_sharpest = avg_w_vec.sum();

    if constexpr (NeedSharpnessMap) {
        Map<Matrix<float, Dynamic, Dynamic, RowMajor>> output_map(
            sharpness_map, height, width);
        output_map = edge_widths;
    }

    return (sum_sharpest > 0.0f)
               ? (static_cast<float>(blocks_to_sum) / sum_sharpest)
               : 0.0f;
}

template <OutputMode Mode>
static inline const VSFrame* VS_CC psiGetFrame(auto n, auto activationReason,
                                               auto instanceData,
                                               [[maybe_unused]] auto frameData,
                                               auto frameCtx, auto core,
                                               auto vsapi) noexcept {
    auto d = static_cast<PSIData*>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const auto src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const auto fi = vsapi->getVideoFrameFormat(src);
        const auto height = vsapi->getFrameHeight(src, 0);
        const auto width = vsapi->getFrameWidth(src, 0);

        VSFrame* dst;
        const auto max_val =
            (d->vi->format.sampleType == stFloat)
                ? 1.0f
                : static_cast<float>((1 << d->vi->format.bitsPerSample) - 1);

        if constexpr (Mode == OUTPUT_MODE_ORIGINAL) {
            const auto numPlanes = fi->numPlanes;
            std::vector<const VSFrame*> planeSrc(numPlanes, src);
            std::vector<int> planes(numPlanes);
            for (auto i = 0; i < numPlanes; ++i) {
                planes[i] = i;
            }

            dst = vsapi->newVideoFrame2(fi, width, height, planeSrc.data(),
                                        planes.data(), src, core);
        } else {
            VSVideoFormat format;
            vsapi->queryVideoFormat(&format, cfGray, stFloat, 32, 0, 0, core);
            dst = vsapi->newVideoFrame(&format, width, height, src, core);
        }

        const void* VS_RESTRICT srcp = vsapi->getReadPtr(src, 0);
        const auto src_stride = vsapi->getStride(src, 0);

        auto psi_score = 0.0f;
        std::vector<float> sharpness_map;
        float* sharpness_ptr = nullptr;

        if constexpr (Mode == OUTPUT_MODE_SHARPNESS_MAP) {
            sharpness_map.resize(width * height);
            sharpness_ptr = sharpness_map.data();
        }

        if (d->vi->format.sampleType == stInteger) {
            if (d->vi->format.bitsPerSample == 8) {
                psi_score =
                    calculatePSI<uint8_t, (Mode == OUTPUT_MODE_SHARPNESS_MAP)>(
                        static_cast<const uint8_t*>(srcp), width, height,
                        src_stride, d->percentile, max_val, d->blocksize,
                        d->threshold_w, d->tan_angle_tol, d->w_jnb,
                        d->sobel_threshold, sharpness_ptr);
            } else {
                psi_score =
                    calculatePSI<uint16_t, (Mode == OUTPUT_MODE_SHARPNESS_MAP)>(
                        static_cast<const uint16_t*>(srcp), width, height,
                        src_stride, d->percentile, max_val, d->blocksize,
                        d->threshold_w, d->tan_angle_tol, d->w_jnb,
                        d->sobel_threshold, sharpness_ptr);
            }
        } else {
            psi_score =
                calculatePSI<float, (Mode == OUTPUT_MODE_SHARPNESS_MAP)>(
                    static_cast<const float*>(srcp), width, height, src_stride,
                    d->percentile, max_val, d->blocksize, d->threshold_w,
                    d->tan_angle_tol, d->w_jnb, d->sobel_threshold,
                    sharpness_ptr);
        }

        if constexpr (Mode == OUTPUT_MODE_SHARPNESS_MAP) {
            float* VS_RESTRICT dstp =
                reinterpret_cast<float*>(vsapi->getWritePtr(dst, 0));
            const auto dst_stride = vsapi->getStride(dst, 0) / sizeof(float);

            for (auto y = 0; y < height; ++y) {
                const auto src_row = sharpness_map.data() + y * width;
                auto dst_row = dstp + y * dst_stride;
                std::copy(src_row, src_row + width, dst_row);
            }
        }

        vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "PSI", psi_score,
                           maReplace);

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static inline auto VS_CC psiFilterFree(auto instanceData,
                                       [[maybe_unused]] auto core,
                                       auto vsapi) noexcept {
    auto d = static_cast<PSIData*>(instanceData);
    vsapi->freeNode(d->node);
    free(d);
}

template <typename T, typename Validator>
static inline bool getAndValidateParam(auto in, auto vsapi, auto param_name,
                                       T default_value, auto& result,
                                       auto filter_name, auto out,
                                       Validator validator, auto error_msg) {
    auto err = 0;
    if constexpr (std::is_same_v<T, float>) {
        result = static_cast<T>(vsapi->mapGetFloat(in, param_name, 0, &err));
    } else if constexpr (std::is_same_v<T, int>) {
        result = static_cast<T>(vsapi->mapGetInt(in, param_name, 0, &err));
    } else if constexpr (std::is_enum_v<T>) {
        result = static_cast<T>(vsapi->mapGetInt(in, param_name, 0, &err));
    }

    if (err) {
        result = default_value;
        return true;
    }

    if (!validator(result)) {
        vsapi->mapSetError(
            out, std::format("{}: {}", filter_name, error_msg).c_str());
        return false;
    }

    return true;
}

static inline auto VS_CC psiCreate(auto in, auto out,
                                   [[maybe_unused]] auto userData, auto core,
                                   auto vsapi) noexcept {
    PSIData d;
    PSIData* data;

    constexpr auto filter_name = "PSI";

    d.node = vsapi->mapGetNode(in, "clip", 0, 0);
    const auto vi = vsapi->getVideoInfo(d.node);

    d.vi = vi;

    if (!vsh::isConstantVideoFormat(vi)) {
        vsapi->mapSetError(
            out, std::format("{}: only clips with constant format are accepted",
                             filter_name)
                     .c_str());
        vsapi->freeNode(d.node);
        return;
    }

    if (!(((vi->format.bitsPerSample >= 8 && vi->format.bitsPerSample <= 16) &&
           vi->format.sampleType == stInteger) ||
          (vi->format.bitsPerSample == 32 &&
           vi->format.sampleType == stFloat))) {
        vsapi->mapSetError(
            out, std::format("{}: only 8-16 bit integer or 32 bit float input "
                             "are accepted, got {} bit {}",
                             filter_name, vi->format.bitsPerSample,
                             vi->format.sampleType == stInteger ? "integer"
                                                                : "float")
                     .c_str());
        vsapi->freeNode(d.node);
        return;
    }

    if (!getAndValidateParam(
            in, vsapi, "percentile", DEFAULT_PERCENTILE, d.percentile,
            filter_name, out,
            [](auto val) { return val > 0.0f && val <= 100.0f; },
            std::format("percentile must be in the range (0, 100], got {}",
                        d.percentile)
                .c_str())) {
        vsapi->freeNode(d.node);
        return;
    }

    if (!getAndValidateParam(
            in, vsapi, "blocksize", DEFAULT_BLOCKSIZE, d.blocksize, filter_name,
            out, [](auto val) { return val > 0; },
            std::format("blocksize must be greater than 0, got {}", d.blocksize)
                .c_str())) {
        vsapi->freeNode(d.node);
        return;
    }

    if (!getAndValidateParam(
            in, vsapi, "threshold_w", DEFAULT_THRESHOLD_W, d.threshold_w,
            filter_name, out, [](auto val) { return val >= 0.0f; },
            std::format("threshold_w must be non-negative, got {}",
                        d.threshold_w)
                .c_str())) {
        vsapi->freeNode(d.node);
        return;
    }

    if (!getAndValidateParam(
            in, vsapi, "angle_tolerance", DEFAULT_ANGLE_TOLERANCE,
            d.angle_tolerance, filter_name, out,
            [](auto val) { return val > 0.0f && val <= 90.0f; },
            std::format("angle_tolerance must be in the range (0, 90], got {}",
                        d.angle_tolerance)
                .c_str())) {
        vsapi->freeNode(d.node);
        return;
    }

    if (!getAndValidateParam(
            in, vsapi, "w_jnb", DEFAULT_W_JNB, d.w_jnb, filter_name, out,
            [](auto val) { return val >= 0.0f; },
            std::format("w_jnb must be non-negative, got {}", d.w_jnb)
                .c_str())) {
        vsapi->freeNode(d.node);
        return;
    }

    if (!getAndValidateParam(
            in, vsapi, "sobel_threshold", DEFAULT_SOBEL_THRESHOLD,
            d.sobel_threshold, filter_name, out,
            [](auto val) { return val >= 0.0f; },
            std::format("sobel_threshold must be non-negative, got {}",
                        d.sobel_threshold)
                .c_str())) {
        vsapi->freeNode(d.node);
        return;
    }

    auto err = 0;
    auto raw_output_mode = vsapi->mapGetInt(in, "output_mode", 0, &err);
    if (err) {
        d.output_mode = OUTPUT_MODE_ORIGINAL;
    } else {
        if (raw_output_mode != OUTPUT_MODE_ORIGINAL &&
            raw_output_mode != OUTPUT_MODE_SHARPNESS_MAP) {
            vsapi->mapSetError(
                out,
                std::format("{}: output_mode must be 0 (original frame) or 1 "
                            "(sharpness map), got {}",
                            filter_name, raw_output_mode)
                    .c_str());
            vsapi->freeNode(d.node);
            return;
        }
        d.output_mode = static_cast<OutputMode>(raw_output_mode);
    }

    d.tan_angle_tol =
        std::tan(d.angle_tolerance * std::numbers::pi_v<float> / 180.0f);

    data = static_cast<PSIData*>(malloc(sizeof(d)));
    *data = d;

    VSVideoInfo out_vi = *vi;
    if (d.output_mode == OUTPUT_MODE_SHARPNESS_MAP) {
        vsapi->queryVideoFormat(&out_vi.format, cfGray, stFloat, 32, 0, 0,
                                core);
    }

    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};

    if (d.output_mode == OUTPUT_MODE_ORIGINAL) {
        vsapi->createVideoFilter(
            out, filter_name, &out_vi, psiGetFrame<OUTPUT_MODE_ORIGINAL>,
            psiFilterFree, fmParallel, deps, 1, data, core);
    } else {
        vsapi->createVideoFilter(
            out, filter_name, &out_vi, psiGetFrame<OUTPUT_MODE_SHARPNESS_MAP>,
            psiFilterFree, fmParallel, deps, 1, data, core);
    }
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.yuygfgg.psi", "psi",
                         "VapourSynth PSI (Perceptual Sharpness Index) Plugin",
                         VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0,
                         plugin);
    vspapi->registerFunction("PSI",
                             "clip:vnode;"
                             "percentile:float:opt;"
                             "blocksize:int:opt;"
                             "threshold_w:float:opt;"
                             "angle_tolerance:float:opt;"
                             "w_jnb:float:opt;"
                             "sobel_threshold:float:opt;"
                             "output_mode:int:opt;",
                             "clip:vnode;", psiCreate, NULL, plugin);
}
