# Remote comparison preview

The user approved sharing the comparison page and its three public-dataset
clips through a temporary Cloudflare URL on 2026-10-02. The link is recorded
in `.cache/phone-preview/state.json`. Anyone with it can view the selected
comparison assets. The computer and both preview/tunnel processes must remain
running; this is a temporary preview, not a deployed mobile AR application.

`tools/serve_quality_preview.py` listens only on loopback port 8877 and serves
an explicit read-only route list: the comparison HTML, per-object player data,
measured reports, three mesh buffers and three benchmark-window videos. It
does not serve the workspace, raw model cache, annotation editor or directory
listings. HTTP byte ranges support video seeking. ETags allow clients to check
for changed results without downloading unchanged multi-megabyte JSON files.

The cloudflared executable is workspace-local, downloaded from the official
Cloudflare GitHub release `2026.9.3`, with SHA-256 verified as
`f096265ec2fcbe9bb6e2d64268db167ced3fcbb83d894bdb9e2fcdb26f2ea7e2`.
The tunnel makes an outbound connection; no incoming firewall or router rule
was added. No system service or startup entry was installed. Logs are in
`.cache/phone-preview/`. The original localhost server remains on port 8876.

On narrow screens the viewer defaults to a larger single mapping panel;
controls switch to the mask, source image, old tracker or all four views.
It checks for saved result updates every 20 seconds without restarting playback.
Clips are fully fetched before playback. Asset requests retry up to three times
with bounded timeouts and show a retry message on connection loss. Completed
bottle experiments and incremental keyboard render experiments can be selected
from the mapping control; unprocessed poses are explicitly marked.
Mobile-size rendering, public HTTP/video access and seeking were verified in
the browser. A physical iPhone/Android playback test remains unverified.

Protocol/route checks: `python -m unittest tools.test_quality_preview`.
