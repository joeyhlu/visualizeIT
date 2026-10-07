#include "visualizeit/pose.h"
#include <cmath>
#include <cstring>

namespace {
bool rigid(const float* m) {
    for (int i=0; i<16; ++i) if (!std::isfinite(m[i])) return false;
    if (std::fabs(m[12])>1e-4f || std::fabs(m[13])>1e-4f || std::fabs(m[14])>1e-4f || std::fabs(m[15]-1)>1e-4f) return false;
    for (int a=0; a<3; ++a) for (int b=0; b<3; ++b) {
        float dot=0;
        for (int row=0; row<3; ++row) dot += m[row*4+a]*m[row*4+b];
        if (std::fabs(dot-(a==b ? 1.f : 0.f))>1e-3f) return false;
    }
    float det=m[0]*(m[5]*m[10]-m[6]*m[9])-m[1]*(m[4]*m[10]-m[6]*m[8])+m[2]*(m[4]*m[9]-m[5]*m[8]);
    return std::fabs(det-1)>1e-3f ? false : true;
}
}

int vi_cv_gltf_pose_to_unity(const float input[16],float output[16]) {
    if (!input || !output || !rigid(input)) return VI_INVALID_ARGUMENT;
    const float cameraSigns[4]={1,-1,1,1};
    const float objectSigns[4]={1,1,-1,1};
    float converted[16];
    for (int row=0; row<4; ++row) for (int col=0; col<4; ++col)
        converted[row*4+col]=cameraSigns[row]*input[row*4+col]*objectSigns[col];
    std::memcpy(output,converted,sizeof(converted));
    return VI_OK;
}
unsigned int vi_pose_abi_version(void) { return 1; }
