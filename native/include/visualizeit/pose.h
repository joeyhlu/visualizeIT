#ifndef VISUALIZEIT_POSE_H
#define VISUALIZEIT_POSE_H

#if defined(_WIN32) && defined(VI_SHARED)
#define VI_API __declspec(dllexport)
#else
#define VI_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum vi_status { VI_OK = 0, VI_INVALID_ARGUMENT = 1 };

/* Row-major 4x4 matrices multiplying column vectors; translation is metres.
 * Input: OpenCV camera <- right-handed glTF object (x right, y up, z back).
 * Output: Unity camera <- Unity object (x right, y up, z forward).
 * Distinct input/output buffers may overlap. Rejects non-rigid/non-finite input.
 */
VI_API int vi_cv_gltf_pose_to_unity(const float input[16], float output[16]);
VI_API unsigned int vi_pose_abi_version(void);

#ifdef __cplusplus
}
#endif
#endif
