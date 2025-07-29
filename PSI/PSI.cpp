#include "VSHelper4.h"
#include "VapourSynth4.h"
#include <algorithm>
#include <cmath>
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

typedef struct {
    VSNode* node;
    const VSVideoInfo* vi;
    float percentile;
    int bits_per_sample;
    VSSampleType sample_type;
    uint16_t max_value;
    int blocksize;
    float threshold_w;
    float angle_tolerance;
    float w_jnb;
    float sobel_threshold;
} PSIData;

template <typename T>
static inline auto
calculatePSI(const auto src, auto width, auto height, auto stride,
             auto percentile, auto max_val, auto blocksize, auto threshold_w,
             auto angle_tolerance, auto w_jnb, auto sobel_threshold) noexcept {
    const auto stride_elements = stride / sizeof(T);

    const auto total_pixels = width * height;

    const auto sobel_threshold_scaled =
        (std::is_same_v<T, float>)
            ? sobel_threshold
            : sobel_threshold * static_cast<float>(max_val);

    std::vector<bool> edges(total_pixels, false);
    std::vector<float> Ix(total_pixels, 0.0f);
    std::vector<float> Iy(total_pixels, 0.0f);

    const auto sobel_threshold_sq =
        sobel_threshold_scaled * sobel_threshold_scaled;

    auto get = [&](int r, int c) {
        return static_cast<float>((src + r * stride_elements)[c]);
    };

    for (auto y = 1; y < height - 1; ++y) {
        auto ix_row = Ix.data() + y * width;
        auto iy_row = Iy.data() + y * width;

        for (auto x = 1; x < width - 1; ++x) {
            const auto p_tl = get(y - 1, x - 1);
            const auto p_tc = get(y - 1, x);
            const auto p_tr = get(y - 1, x + 1);
            const auto p_ml = get(y, x - 1);
            const auto p_mr = get(y, x + 1);
            const auto p_bl = get(y + 1, x - 1);
            const auto p_bc = get(y + 1, x);
            const auto p_br = get(y + 1, x + 1);

            const auto gx = -p_tl + p_tr + -2 * p_ml + 2 * p_mr + -p_bl + p_br;

            const auto gy = -p_tl - 2 * p_tc - p_tr + p_bl + 2 * p_bc + p_br;

            const auto magnitude_sq = gx * gx + gy * gy;
            edges[y * width + x] = (magnitude_sq > sobel_threshold_sq);

            ix_row[x] = gx;
            iy_row[x] = gy;
        }
    }

    std::vector<float> phi(total_pixels);
    constexpr auto rad_to_deg = 180.0f / std::numbers::pi;

    for (auto i = 0; i < total_pixels; ++i) {
        phi[i] = std::atan2(Iy[i], Ix[i]) * rad_to_deg;
    }

    std::vector<float> edge_widths(total_pixels, 0.0f);
    auto widths_count = 0;

    constexpr auto deg_to_rad = std::numbers::pi / 180.0f;

    auto get_normalized = [&](int r, int c) {
        if constexpr (std::is_same_v<T, float>) {
            return (src + r * stride_elements)[c];
        } else {
            return static_cast<float>((src + r * stride_elements)[c]) /
                   static_cast<float>(max_val);
        }
    };

    for (auto y = 0; y < height; ++y) {
        for (auto x = 0; x < width; ++x) {
            const auto idx = y * width + x;
            if (!edges[idx] || (Ix[idx] == 0.0f && Iy[idx] == 0.0f)) {
                continue;
            }

            const auto angle = phi[idx];
            auto width_up = 0;
            auto width_down = 0;
            auto valid_width = false;
            auto min_val = 0.0f;
            auto max_val = 0.0f;

            // Check for horizontal edge, gradient pointing upwards (~90°)
            if (std::abs(angle + 90.0f) < angle_tolerance) {
                // Search upward
                auto prev_val_up = get_normalized(y, x);
                for (auto d = 1; d < height; d++) {
                    auto up = y - d;
                    if (up < 0) {
                        width_up = -1;
                        break;
                    }
                    const auto curr_val_up = get_normalized(up, x);
                    if (curr_val_up <= prev_val_up) {
                        width_up = d - 1;
                        max_val = prev_val_up;
                        break;
                    }
                    prev_val_up = curr_val_up;
                }

                // Search downward
                auto prev_val_down = get_normalized(y, x);
                for (auto d = 1; d < height; d++) {
                    auto down = y + d;
                    if (down >= height) {
                        width_down = -1;
                        break;
                    }
                    const auto curr_val_down = get_normalized(down, x);
                    if (curr_val_down >= prev_val_down) {
                        width_down = d - 1;
                        min_val = prev_val_down;
                        break;
                    }
                    prev_val_down = curr_val_down;
                }

                if (width_up != -1 && width_down != -1) {
                    valid_width = true;
                    const auto phi2 = (angle + 90.0f) * deg_to_rad;
                    const auto cos_phi2 = std::cos(phi2);
                    edge_widths[idx] = (width_up + width_down) / cos_phi2;
                    const auto slope =
                        (max_val - min_val) / edge_widths[idx];
                    if (edge_widths[idx] >= w_jnb) {
                        edge_widths[idx] -= slope;
                    }
                }
            }

            // Check for horizontal edge, gradient pointing downwards (~-90°)
            if (std::abs(angle - 90.0f) < angle_tolerance) {
                // Search upward
                auto prev_val_up = get_normalized(y, x);
                for (auto d = 1; d < height; d++) {
                    auto up = y - d;
                    if (up < 0) {
                        width_up = -1;
                        break;
                    }
                    const auto curr_val_up = get_normalized(up, x);
                    if (curr_val_up >= prev_val_up) {
                        width_up = d - 1;
                        min_val = prev_val_up;
                        break;
                    }
                    prev_val_up = curr_val_up;
                }

                // Search downward
                auto prev_val_down = get_normalized(y, x);
                for (auto d = 1; d < height; d++) {
                    auto down = y + d;
                    if (down >= height) {
                        width_down = -1;
                        break;
                    }
                    const auto curr_val_down = get_normalized(down, x);
                    if (curr_val_down <= prev_val_down) {
                        width_down = d - 1;
                        max_val = prev_val_down;
                        break;
                    }
                    prev_val_down = curr_val_down;
                }

                if (width_up != -1 && width_down != -1) {
                    valid_width = true;
                    const auto phi2 = (angle - 90.0f) * deg_to_rad;
                    const auto cos_phi2 = std::cos(phi2);
                    edge_widths[idx] = (width_up + width_down) / cos_phi2;
                    const auto slope =
                        (max_val - min_val) / edge_widths[idx];
                    if (edge_widths[idx] >= w_jnb) {
                        edge_widths[idx] -= slope;
                    }
                }
            }

            if (valid_width) {
                ++widths_count;
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
        for (auto j = 1; j < col_blocks - 1; ++j) {
            const auto block_row = i * blocksize;
            const auto block_col = j * blocksize;

            auto w_sum = 0.0f;
            auto non_zero_count = 0;

            const auto block_end_y = std::min(block_row + blocksize, height);
            const auto block_end_x = std::min(block_col + blocksize, width);

            for (auto y = block_row; y < block_end_y; ++y) {
                const auto edge_row = edge_widths.data() + y * width;
                for (auto x = block_col; x < block_end_x; ++x) {
                    const auto w = edge_row[x];
                    w_sum += w;
                    if (w != 0.0f) {
                        ++non_zero_count;
                    }
                }
            }

            if (w_sum >= threshold_w && non_zero_count > 0) {
                avg_w.push_back(w_sum / non_zero_count);
            }
        }
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

    auto sum_sharpest = 0.0f;
    const auto blocks_to_sum =
        std::min(nr_of_used_blocks, static_cast<int>(avg_w.size()));
    for (auto i = 0; i < blocks_to_sum; ++i) {
        sum_sharpest += avg_w[i];
    }

    return (sum_sharpest > 0.0f)
               ? (static_cast<float>(blocks_to_sum) / sum_sharpest)
               : 0.0f;
}

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

        const auto numPlanes = fi->numPlanes;
        std::vector<const VSFrame*> planeSrc(numPlanes, src);
        std::vector<int> planes(numPlanes);
        for (auto i = 0; i < numPlanes; ++i) {
            planes[i] = i;
        }

        const auto dst = vsapi->newVideoFrame2(
            fi, width, height, planeSrc.data(), planes.data(), src, core);

        const void* srcp = vsapi->getReadPtr(src, 0);
        const auto src_stride = vsapi->getStride(src, 0);

        auto psi_score = 0.0f;
        const auto max_val = (d->sample_type == stFloat)
                                 ? 1.0f
                                 : static_cast<float>(d->max_value);

        if (d->sample_type == stInteger) {
            if (d->bits_per_sample == 8) {
                psi_score = calculatePSI<uint8_t>(
                    static_cast<const uint8_t*>(srcp), width, height,
                    src_stride, d->percentile, max_val, d->blocksize,
                    d->threshold_w, d->angle_tolerance, d->w_jnb,
                    d->sobel_threshold);
            } else {
                psi_score = calculatePSI<uint16_t>(
                    static_cast<const uint16_t*>(srcp), width, height,
                    src_stride, d->percentile, max_val, d->blocksize,
                    d->threshold_w, d->angle_tolerance, d->w_jnb,
                    d->sobel_threshold);
            }
        } else {
            psi_score = calculatePSI<float>(
                static_cast<const float*>(srcp), width, height, src_stride,
                d->percentile, max_val, d->blocksize, d->threshold_w,
                d->angle_tolerance, d->w_jnb, d->sobel_threshold);
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

static inline auto VS_CC psiCreate(auto in, auto out,
                                   [[maybe_unused]] auto userData, auto core,
                                   auto vsapi) noexcept {
    PSIData d;
    PSIData* data;
    auto err = 0;

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

    if (!(((vi->format.bitsPerSample == 8 || vi->format.bitsPerSample == 16) &&
           vi->format.sampleType == stInteger) ||
          (vi->format.bitsPerSample == 32 &&
           vi->format.sampleType == stFloat))) {
        vsapi->mapSetError(
            out,
            std::format(
                "{}: only 8-16 bit integer or 32 bit float input are accepted",
                filter_name)
                .c_str());
        vsapi->freeNode(d.node);
        return;
    }

    d.percentile =
        static_cast<float>(vsapi->mapGetFloat(in, "percentile", 0, &err));
    if (err) {
        d.percentile = DEFAULT_PERCENTILE;
    }

    if (d.percentile <= 0.0f || d.percentile > 100.0f) {
        vsapi->mapSetError(
            out, std::format("{}: percentile must be in the range (0, 100]",
                             filter_name)
                     .c_str());
        vsapi->freeNode(d.node);
        return;
    }

    d.blocksize = static_cast<int>(vsapi->mapGetInt(in, "blocksize", 0, &err));
    if (err) {
        d.blocksize = DEFAULT_BLOCKSIZE;
    }
    if (d.blocksize <= 0) {
        vsapi->mapSetError(
            out,
            std::format("{}: blocksize must be greater than 0", filter_name)
                .c_str());
        vsapi->freeNode(d.node);
        return;
    }

    d.threshold_w =
        static_cast<float>(vsapi->mapGetFloat(in, "threshold_w", 0, &err));
    if (err) {
        d.threshold_w = DEFAULT_THRESHOLD_W;
    }
    if (d.threshold_w < 0.0f) {
        vsapi->mapSetError(
            out,
            std::format("{}: threshold_w must be non-negative", filter_name)
                .c_str());
        vsapi->freeNode(d.node);
        return;
    }

    d.angle_tolerance =
        static_cast<float>(vsapi->mapGetFloat(in, "angle_tolerance", 0, &err));
    if (err) {
        d.angle_tolerance = DEFAULT_ANGLE_TOLERANCE;
    }
    if (d.angle_tolerance <= 0.0f || d.angle_tolerance > 90.0f) {
        vsapi->mapSetError(
            out, std::format("{}: angle_tolerance must be in the range (0, 90]",
                             filter_name)
                     .c_str());
        vsapi->freeNode(d.node);
        return;
    }

    d.w_jnb = static_cast<float>(vsapi->mapGetFloat(in, "w_jnb", 0, &err));
    if (err) {
        d.w_jnb = DEFAULT_W_JNB;
    }
    if (d.w_jnb < 0.0f) {
        vsapi->mapSetError(
            out,
            std::format("{}: w_jnb must be non-negative", filter_name).c_str());
        vsapi->freeNode(d.node);
        return;
    }

    d.sobel_threshold =
        static_cast<float>(vsapi->mapGetFloat(in, "sobel_threshold", 0, &err));
    if (err) {
        d.sobel_threshold = DEFAULT_SOBEL_THRESHOLD;
    }
    if (d.sobel_threshold < 0.0f) {
        vsapi->mapSetError(
            out,
            std::format("{}: sobel_threshold must be non-negative", filter_name)
                .c_str());
        vsapi->freeNode(d.node);
        return;
    }

    d.sample_type = static_cast<VSSampleType>(vi->format.sampleType);
    d.bits_per_sample = vi->format.bitsPerSample;

    if (d.sample_type == stInteger) {
        d.max_value = static_cast<uint16_t>((1 << d.bits_per_sample) - 1);
    } else {
        d.max_value = 0;
    }

    data = static_cast<PSIData*>(malloc(sizeof(d)));
    *data = d;

    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, filter_name, vi, psiGetFrame, psiFilterFree,
                             fmParallel, deps, 1, data, core);
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
                             "sobel_threshold:float:opt;",
                             "clip:vnode;", psiCreate, NULL, plugin);
}
