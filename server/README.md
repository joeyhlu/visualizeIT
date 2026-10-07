# Reconstruction milestone

The reconstruction server has **not been implemented or deployed**. It is gated by successful physical validation of the phone AR app, as required by the implementation plan.

Its agreed boundary is an asynchronous scan job accepting calibrated images and optional synchronized depth, with create/upload/start/status/artifact/delete operations. A completed job will supply a metric GLB object mesh, UV coordinates, confidence, and tracking references. SAM 3.1 segmentation and COLMAP reconstruction belong in a GPU worker, outside the live rendering path.

Do not expose a success endpoint or synthetic object asset while this milestone is gated. See [device validation](../docs/validation.md) for the required first-milestone evidence.
