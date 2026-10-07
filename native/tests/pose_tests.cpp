#include "visualizeit/pose.h"
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    float input[16]={1,0,0,0.12f, 0,1,0,-0.08f, 0,0,1,0.6f, 0,0,0,1};
    float output[16];
    if (vi_cv_gltf_pose_to_unity(input,output)!=VI_OK) return 1;
    // Verify the coordinate conversion preserves a physical point in camera space.
    float gltf[4]={0.02f,0.03f,-0.04f,1};
    float unityObject[4]={gltf[0],gltf[1],-gltf[2],1};
    float cvPoint[4]={},unityPoint[4]={};
    for (int row=0; row<4; ++row) for (int col=0; col<4; ++col) {
        cvPoint[row]+=input[row*4+col]*gltf[col];
        unityPoint[row]+=output[row*4+col]*unityObject[col];
    }
    const float cameraSigns[4]={1,-1,1,1};
    for (int i=0; i<4; ++i) if (std::fabs(unityPoint[i]-cameraSigns[i]*cvPoint[i])>1e-6f) return 2;
    if (vi_cv_gltf_pose_to_unity(output,output)!=VI_OK) return 3;
    for (int i=0; i<16; ++i) if (std::fabs(output[i]-input[i])>1e-6f) return 4;
    input[0]=2;
    if (vi_cv_gltf_pose_to_unity(input,output)!=VI_INVALID_ARGUMENT) return 5;
    input[0]=std::numeric_limits<float>::quiet_NaN();
    if (vi_cv_gltf_pose_to_unity(input,output)!=VI_INVALID_ARGUMENT) return 6;
    if (vi_cv_gltf_pose_to_unity(nullptr,output)!=VI_INVALID_ARGUMENT) return 7;
    if (vi_pose_abi_version()!=1) return 8;
    std::cout << "PASS: pose coordinates, in-place round trip, invalid scale, NaN, null, ABI\n";
    return 0;
}
