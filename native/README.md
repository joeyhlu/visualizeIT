# Native vision boundary

The implemented C ABI currently normalizes rigid object poses between OpenCV/glTF and Unity coordinates. It is independently testable and is **not an object tracker**. The fixture app uses native ARKit/ARCore world tracking through AR Foundation and does not load this library yet.

Build on a machine with CMake and a C++17 toolchain:

```sh
cmake -S native -B native/build
cmake --build native/build
ctest --test-dir native/build --output-on-failure
```

For Android use the Unity editor's NDK toolchain, then package an ARM64 shared library; for iOS compile the same source as a static library in the Xcode build. No prebuilt binary is committed.

After the physical AR gate passes, the next native module will use OpenCV for reference matching, PnP, optical flow, and pose confidence. Deformable modes will have separate vertex-update outputs. These algorithms are intentionally not represented by placeholder successful tracking responses.
