"""Phone/workbench connection controls. Fake metadata is never model or phone accuracy."""
import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
PIN='27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311'

def upstream():
 return {'service':'visualizeit-workbench','schema':'workbench_service_v2','registered_clip_count':9,
  'primary_model':{'export_sha256':PIN,'installed_head_sha256':None},
  'capabilities':{'pipeline':True,'neural_blocked':['torch_runtime','serving_admission']},
  'admission':{'admitted':False},'defaults':{'head_configured':True,'metric_bundle_configured':False},
  'secret':'DO_NOT_PUBLISH','private_path':'C:/DO_NOT_PUBLISH'}

class ProjectionTests(unittest.TestCase):
 def project(self,body):
  from tools.phone_pipeline_bridge import sanitize_tracking_status
  return sanitize_tracking_status(body)
 def test_realistic_blocked_metadata_never_becomes_live_or_installed(self):
  d=self.project(upstream());self.assertEqual(d['state'],'reachable_blocked');self.assertFalse(d['live_supported']);self.assertFalse(d['quality_claim']);self.assertEqual(d['requested_head_sha256'],PIN);self.assertIsNone(d['installed_head_sha256']);self.assertNotIn('DO_NOT_PUBLISH',json.dumps(d))
 def test_older_export_is_rejected(self):
  d=upstream();d['primary_model']['export_sha256']='5'*64;r=self.project(d);self.assertIn('model_mismatch',r['blockers']);self.assertEqual(r['state'],'reachable_blocked')
 def test_recorded_metadata_cannot_enable_live(self):
  d=upstream();d['capabilities']['neural_blocked']=[];d['admission']['admitted']=True;d['defaults']['metric_bundle_configured']=True;r=self.project(d);self.assertEqual(r['state'],'recorded_configured');self.assertFalse(r['live_supported']);self.assertIsNone(r['installed_head_sha256'])
 def test_missing_metadata_and_truthy_nonbooleans_fail_closed(self):
  for d in [{},None,[],{'service':'prototype-service/1'},upstream()]:
   if isinstance(d,dict) and 'defaults' in d:d['defaults']={'head_configured':1,'metric_bundle_configured':'true'};d['admission']={'admitted':1}
   with self.subTest(d=d):self.assertEqual(self.project(d)['state'],'reachable_blocked')
 def test_private_blockers_and_path_strings_are_redacted(self):
  d=upstream();d['capabilities']['neural_blocked']=['C:/DO_NOT_PUBLISH','Bearer SECRET',{'private_path':'DO_NOT_PUBLISH'},'torch_runtime'];r=self.project(d);self.assertNotIn('DO_NOT_PUBLISH',json.dumps(r));self.assertNotIn('SECRET',json.dumps(r));self.assertIn('torch_runtime',r['upstream_blockers'])
 def test_legacy_service_cannot_be_admitted_even_with_forged_ready_fields(self):
  d=upstream();d.update(service='prototype-service/1');d['capabilities']['neural_blocked']=[];d['admission']['admitted']=True;d['defaults']['metric_bundle_configured']=True;self.assertEqual(self.project(d)['state'],'reachable_blocked')
 def test_installed_head_is_reported_without_manufacturing(self):
  d=upstream();d['primary_model']['installed_head_sha256']=PIN;self.assertEqual(self.project(d)['installed_head_sha256'],PIN)

 def test_installed_older_head_blocks_ready_metadata(self):
  d=upstream();d['capabilities']['neural_blocked']=[];d['admission']['admitted']=True;d['defaults']['metric_bundle_configured']=True;d['primary_model']['installed_head_sha256']='5'*64;r=self.project(d);self.assertEqual(r['state'],'reachable_blocked');self.assertIn('model_mismatch',r['blockers'])
 def test_disabled_execution_cannot_be_reported_recorded_supported(self):
  d=upstream();d['capabilities']['neural_blocked']=[];d['admission']['admitted']=True;d['defaults']['metric_bundle_configured']=True;d['capabilities']['pipeline']=False;r=self.project(d);self.assertEqual(r['state'],'reachable_blocked');self.assertFalse(r['recorded_supported'])

class HTTPConnectionTests(unittest.TestCase):
 def setUp(self):
  from tools.serve_design_studio import create_server
  self.temp=tempfile.TemporaryDirectory(prefix='phone-connect-test-',dir=ROOT/'.cache');self.addCleanup(self.temp.cleanup)
  self.server=create_server('127.0.0.1',0,storage_root=Path(self.temp.name)/'studio');self.body=upstream();self.requests=[]
  def fake_get(path):self.requests.append(path);return self.body if path=='/api/status' else {'clips':[]}
  self.server.bridge._get=fake_get
  self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.addCleanup(self.shutdown)
 def shutdown(self):self.server.shutdown();self.server.server_close();self.thread.join(timeout=3)
 def request(self,method,path,*,auth=True,body=None,headers=None):
  h={'Origin':f'http://127.0.0.1:{self.server.server_port}'}
  if auth:h['Authorization']='Bearer '+self.server.token
  h.update(headers or {});c=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
  try:c.request(method,path,body=body,headers=h);r=c.getresponse();return r.status,dict(r.getheaders()),r.read()
  finally:c.close()
 def test_pairing_and_origin_guards_precede_upstream_access(self):
  self.assertEqual(self.request('GET','/api/studio/tracking-status',auth=False)[0],401);self.assertEqual(self.requests,[])
  self.assertEqual(self.request('GET','/api/studio/tracking-status',headers={'Origin':'https://wrong.invalid'})[0],403);self.assertEqual(self.requests,[])
 def test_sanitized_status_only_reads_original_workbench(self):
  status,h,b=self.request('GET','/api/studio/tracking-status');self.assertEqual(status,200);self.assertEqual(self.requests,['/api/status']);self.assertNotIn(b'DO_NOT_PUBLISH',b);self.assertFalse(json.loads(b)['live_supported']);self.assertIn('no-store',h['Cache-Control'])
 def test_disabled_live_route_rejects_without_reading_or_forwarding_body(self):
  from tools.serve_design_studio import Handler
  with patch.object(Handler,'_body',side_effect=AssertionError('Frame body was read')):
   status,_,b=self.request('POST','/api/studio/tracking-frames',body=b'not-a-camera-frame',headers={'Content-Type':'application/json'})
  self.assertEqual(status,501,b);self.assertEqual(self.requests,[])
 def test_upstream_disconnect_has_no_legacy_fallback(self):
  self.server.bridge._get=lambda path:(_ for _ in ()).throw(OSError('DO_NOT_PUBLISH'))
  status,_,b=self.request('GET','/api/studio/tracking-status');self.assertEqual(status,200);self.assertEqual(json.loads(b)['state'],'disconnected');self.assertNotIn(b'DO_NOT_PUBLISH',b)
 def test_new_module_has_fixed_static_route_and_no_query_bypass(self):
  self.assertEqual(self.request('GET','/phone-pipeline.js',auth=False)[0],200)
  self.assertEqual(self.request('GET','/api/studio/tracking-status?target=evil')[0],400)

class TransportDeadlineTests(unittest.TestCase):
 def setUp(self):
  self.mode='slow';owner=self
  class Upstream(BaseHTTPRequestHandler):
   def do_GET(inner):
    try:
     if owner.mode=='slow':
      time.sleep(1.1);body=b'{"ok":true}';inner.send_response(200);inner.send_header('Content-Type','application/json');inner.send_header('Content-Length',str(len(body)));inner.end_headers();inner.wfile.write(body)
     else:
      inner.send_response(200);inner.send_header('Content-Type','application/json');inner.send_header('Content-Length','100');inner.end_headers()
      for _ in range(20):inner.wfile.write(b' ');inner.wfile.flush();time.sleep(.25)
    except OSError:pass
   def log_message(inner,*args):pass
  self.upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream);self.thread=threading.Thread(target=self.upstream.serve_forever,daemon=True);self.thread.start();self.addCleanup(self.stop)
 def stop(self):self.upstream.shutdown();self.upstream.server_close();self.thread.join(timeout=3)
 def test_cold_read_under_hard_deadline_is_not_false_disconnect(self):
  from tools.serve_design_studio import WorkbenchBridge
  self.assertEqual(WorkbenchBridge(self.upstream.server_port)._get('/api/status'),{'ok':True})
 def test_drip_feed_still_stops_at_hard_deadline(self):
  from tools.serve_design_studio import WorkbenchBridge
  self.mode='drip';start=time.perf_counter()
  with self.assertRaises((OSError,ValueError,http.client.HTTPException)):
   WorkbenchBridge(self.upstream.server_port)._get('/api/status')
  self.assertLess(time.perf_counter()-start,3.5)

if __name__=='__main__':unittest.main()
