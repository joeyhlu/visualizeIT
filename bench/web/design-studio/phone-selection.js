/* Frame identity and selection transitions; usable without a browser. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.PhoneSelection=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';
  const fields=['epoch','frameId','timestamp','width','height','orientation','revision'];
  function sameFrame(a,b){return !!a&&!!b&&fields.every(k=>a[k]===b[k]);}
  function pointInCanvas(clientX,clientY,rect){if(!rect.width||!rect.height)return null;const x=(clientX-rect.left)/rect.width,y=(clientY-rect.top)/rect.height;return x<0||y<0||x>1||y>1?null:{x,y};}
  function boundedSize(width,height,max=640){const ratio=Math.min(1,max/Math.max(width,height));return {width:Math.max(1,Math.round(width*ratio)),height:Math.max(1,Math.round(height*ratio))};}
  // Canvas pixels use object-fit:cover; visible taps must account for its crop.
  function coverPointInCanvas(clientX,clientY,rect,width,height){if(!width||!height||!rect.width||!rect.height)return null;const p=pointInCanvas(clientX,clientY,rect);if(!p)return null;const scale=Math.max(rect.width/width,rect.height/height);return {x:(clientX-rect.left+(width*scale-rect.width)/2)/(width*scale),y:(clientY-rect.top+(height*scale-rect.height)/2)/(height*scale)};}
  class Session {
    constructor(){this.epoch=0;this.phase='idle';this.frame=null;this.strokes=[];this.mask=null;this.manual=false;this.serial=0;}
    reset(phase='idle'){this.epoch++;this.phase=phase;this.frame=null;this.mask=null;this.strokes=[];this.manual=false;return this.epoch;}
    begin(){return this.reset('starting');}
    permission(epoch){if(epoch!==this.epoch||this.phase!=='starting')return false;this.phase='live';return true;}
    select(point,width,height,orientation,timestamp){if(this.phase!=='live')return null;this.reset('live');const tag=this.capture(width,height,orientation,timestamp);this.strokes=[{mode:'positive',points:[{x:point.x,y:point.y}]}];return tag;}
    capture(width,height,orientation,timestamp){if(this.phase!=='live')throw Error('Camera is not live.');this.frame={epoch:this.epoch,frameId:++this.serial,timestamp,width,height,orientation,revision:0};this.mask=null;this.strokes=[];return {...this.frame};}
    accepts(tag){return (this.phase==='live'||this.phase==='locked')&&sameFrame(this.frame,tag);}
    accept(tag,mask){if(!this.accepts(tag))return false;this.mask=mask;return true;}
    lock(point,manual=false){if(this.phase!=='live'||!this.frame)return false;this.phase='locked';this.manual=manual;this.strokes=manual?[]:[{mode:'positive',points:[point]}];this.frame.revision++;this.mask=null;return true;}
    correct(mode,points){if(this.phase!=='locked'||!['positive','negative'].includes(mode)||!points.length)return false;this.strokes.push({mode,points:points.map(p=>({x:p.x,y:p.y}))});this.frame.revision++;this.mask=null;return true;}
    undo(){const minimum=this.manual?0:1;if(this.phase!=='locked'||this.strokes.length<=minimum)return false;this.strokes.pop();this.frame.revision++;this.mask=null;return true;}
    clear(){if(this.phase!=='locked')return false;this.strokes=this.manual?[]:this.strokes.slice(0,1);this.frame.revision++;this.mask=null;return true;}
    useManual(){if(this.phase!=='locked')return false;this.manual=true;this.strokes=[];this.mask=null;this.frame.revision++;return true;}
    tag(){return this.frame?{...this.frame}:null;}
  }
  return {Session,sameFrame,pointInCanvas,coverPointInCanvas,boundedSize};
});
