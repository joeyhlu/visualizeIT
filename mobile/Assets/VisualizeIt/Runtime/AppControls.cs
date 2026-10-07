using System;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.UI;
using UnityEngine.UI;
using VisualizeIt.Core;

namespace VisualizeIt
{
    public sealed class AppControls : MonoBehaviour
    {
        private VisualizeItApp app;
        private Canvas canvas;
        private RectTransform safeArea;
        private RectTransform aim;
        public Vector2 PlacementScreenPoint => aim != null
            ? RectTransformUtility.WorldToScreenPoint(null,aim.position)
            : new Vector2(Screen.width*0.5f,Screen.height*0.5f);
        private Text message,capabilities;
        private Transform content;
        private Font font;
        private readonly Color ink = new Color(0.91f,0.96f,0.97f);
        private readonly Color panel = new Color(0.035f,0.065f,0.09f,0.96f);
        private readonly Color accent = new Color(0.08f,0.72f,0.62f);
        private Rect lastSafeArea;

        public void Initialize(VisualizeItApp application)
        {
            app = application;
            font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            var root = new GameObject("Phone controls",typeof(RectTransform),typeof(Canvas),typeof(CanvasScaler),typeof(GraphicRaycaster));
            root.transform.SetParent(transform,false);
            canvas = root.GetComponent<Canvas>(); canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = root.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1080,1920);
            scaler.matchWidthOrHeight = 0.5f;
            var events = new GameObject("Event System",typeof(EventSystem),typeof(InputSystemUIInputModule));
            events.transform.SetParent(transform,false);
            events.GetComponent<InputSystemUIInputModule>().AssignDefaultActions();
            safeArea = Rect("Safe area",root.transform);
            FitSafeArea();
            BuildHeader();
            BuildCrosshair();
            BuildControls();
            app.SettingsChanged += RebuildControls;
        }

        private void BuildHeader()
        {
            var header = Rect("Status",safeArea);
            header.anchorMin = new Vector2(0,1); header.anchorMax = Vector2.one;
            header.pivot = new Vector2(0.5f,1); header.sizeDelta = new Vector2(0,240);
            header.gameObject.AddComponent<Image>().color = panel;
            var layout = header.gameObject.AddComponent<VerticalLayoutGroup>();
            layout.padding = new RectOffset(36,36,20,20); layout.spacing = 8;
            layout.childControlHeight = true; layout.childForceExpandHeight = false;
            Label(header,"VisualizeIt",44,60);
            capabilities = Label(header,"Checking camera…",23,62);
            message = Label(header,app.Message,25,75);
        }

        private void BuildCrosshair()
        {
            var cross = Rect("Aim",safeArea);
            aim = cross;
            cross.anchorMin = cross.anchorMax = new Vector2(0.5f,0.5f);
            cross.sizeDelta = new Vector2(64,64);
            var label = cross.gameObject.AddComponent<Text>();
            label.font = font; label.fontSize = 48; label.text = "+"; label.color = ink;
            label.alignment = TextAnchor.MiddleCenter; label.raycastTarget = false;
        }

        private void BuildControls()
        {
            var controls = Rect("Controls",safeArea);
            controls.anchorMin = Vector2.zero; controls.anchorMax = new Vector2(1,0.43f);
            controls.offsetMin = controls.offsetMax = Vector2.zero;
            controls.gameObject.AddComponent<Image>().color = panel;
            var scroll = controls.gameObject.AddComponent<ScrollRect>();
            scroll.horizontal = false; scroll.movementType = ScrollRect.MovementType.Clamped;
            var viewport = Rect("Viewport",controls); Stretch(viewport);
            viewport.gameObject.AddComponent<Image>().color = Color.clear;
            viewport.gameObject.AddComponent<RectMask2D>();
            var body = Rect("Content",viewport);
            body.anchorMin = new Vector2(0,1); body.anchorMax = Vector2.one;
            body.pivot = new Vector2(0.5f,1); body.sizeDelta = Vector2.zero;
            var layout = body.gameObject.AddComponent<VerticalLayoutGroup>();
            layout.padding = new RectOffset(28,28,20,28); layout.spacing = 12;
            layout.childControlHeight = true; layout.childForceExpandHeight = false;
            layout.childControlWidth = true; layout.childForceExpandWidth = true;
            body.gameObject.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
            scroll.viewport = viewport; scroll.content = body;
            content = body;
            Populate();
        }

        private void RebuildControls()
        {
            // Disable old controls immediately; destruction is deferred until end-of-frame.
            foreach (Transform child in content) { child.gameObject.SetActive(false); Destroy(child.gameObject); }
            Populate();
        }

        private void Populate()
        {
            var settings = app.Settings;
            Label(content,"FIXTURE • match a stationary physical object",23,40);
            var shapeRow = Row(content);
            foreach (SurfaceShape shape in Enum.GetValues(typeof(SurfaceShape)))
            {
                SurfaceShape selected = shape;
                Button(shapeRow,(settings.shape == shape ? "● " : "") + shape,() => app.ChangeShape(selected));
            }
            var placement = Row(content);
            Button(placement,app.EditorPreview ? "Preview fixture" : "Place at +",app.PlaceAtCrosshair,true);
            Button(placement,"Clear",app.ResetPlacement);
            Dimension("Width / diameter (cm)",0,settings.dimensions.x);
            Dimension("Height (cm)",1,settings.dimensions.y);
            Dimension("Depth: plane / box (cm)",2,settings.dimensions.z);
            Slider("Repeat width (cm)",0.5f,100,settings.patternMetres*100,v => settings.patternMetres=v/100);
            Slider("Pattern rotation (°)",-180,180,settings.patternRotation,v => settings.patternRotation=v);
            if (settings.shape == SurfaceShape.Cylinder)
            {
                Slider("Wrap seam position (°)",-180,180,settings.seamDegrees,app.SetSeamAngle);
                Button(Row(content),"Fit cylinder repeats at 0°",app.FitCylinderWrap);
            }
            Slider("Roughness",0,1,settings.roughness,v => settings.roughness=v);
            Slider("Object yaw (°)",-180,180,settings.yawDegrees,v => { settings.yawDegrees=v; app.UpdateAlignment(); });
            Slider("Move sideways (cm)",-20,20,settings.alignmentOffset.x*100,v => { var p=settings.alignmentOffset; p.x=v/100; settings.alignmentOffset=p; app.UpdateAlignment(); });
            Slider("Move forward (cm)",-20,20,settings.alignmentOffset.z*100,v => { var p=settings.alignmentOffset; p.z=v/100; settings.alignmentOffset=p; app.UpdateAlignment(); });
            Slider("Lift from support (cm)",0,30,settings.alignmentOffset.y*100,v => { var p=settings.alignmentOffset; p.y=v/100; settings.alignmentOffset=p; app.UpdateAlignment(); });
            Slider("Surface lift (mm)",0,10,settings.shellMetres*1000,v => settings.shellMetres=v/1000);
            var designs = Row(content);
            Button(designs,"Import design",app.Importer.Pick,true);
            Button(designs,"Checkerboard",app.UseCheckerboard);
            var persistence = Row(content);
            Button(persistence,"Save",app.SaveProject); Button(persistence,"Load",app.LoadProject);
            Button(persistence,"Photo",app.SaveScreenshot);
            Button(Row(content),"Export session report",app.ExportDiagnostics);
            Label(content,"Scanning and moving-object modes unlock after phone AR validation.",21,65);
        }

        private void Dimension(string title, int axis, float metres)
        {
            var row = Row(content);
            Label(row,title,24,64);
            var rect = Rect("Centimetres",row);
            rect.gameObject.AddComponent<LayoutElement>().preferredWidth = 240;
            rect.gameObject.AddComponent<Image>().color = new Color(0.12f,0.18f,0.22f);
            var input = rect.gameObject.AddComponent<InputField>();
            var textRect = Rect("Value",rect); Stretch(textRect); textRect.offsetMin = new Vector2(12,0); textRect.offsetMax = new Vector2(-12,0);
            var text = textRect.gameObject.AddComponent<Text>(); text.font=font; text.fontSize=27; text.color=ink; text.alignment=TextAnchor.MiddleLeft;
            input.textComponent=text; input.contentType=InputField.ContentType.DecimalNumber;
            input.text=(metres*100).ToString("F1",System.Globalization.CultureInfo.InvariantCulture);
            input.onEndEdit.AddListener(value =>
            {
                if (float.TryParse(value,System.Globalization.NumberStyles.Float,System.Globalization.CultureInfo.InvariantCulture,out float cm)
                    && !float.IsNaN(cm) && !float.IsInfinity(cm) && cm >= 0.5f && cm <= 500)
                    app.SetDimension(axis,cm/100);
                else app.SetMessage("Enter a size from 0.5 to 500 cm.");
                input.text=(app.Settings.dimensions[axis]*100).ToString("F1",System.Globalization.CultureInfo.InvariantCulture);
            });
        }

        private void Slider(string title, float min, float max, float initial, Action<float> change)
        {
            var group = Rect(title,content);
            group.gameObject.AddComponent<LayoutElement>().preferredHeight=90;
            var titleRect = Rect("Label",group); titleRect.anchorMin=new Vector2(0,0.55f); titleRect.anchorMax=Vector2.one;
            titleRect.offsetMin=titleRect.offsetMax=Vector2.zero;
            var label=titleRect.gameObject.AddComponent<Text>(); label.font=font; label.fontSize=24; label.color=ink;
            label.text=title + "  " + initial.ToString("F2"); label.raycastTarget=false;
            var track=Rect("Slider",group); track.anchorMin=Vector2.zero; track.anchorMax=new Vector2(1,0.48f); track.offsetMin=new Vector2(10,0); track.offsetMax=new Vector2(-10,0);
            var background=track.gameObject.AddComponent<Image>(); background.color=new Color(0.17f,0.23f,0.27f);
            var slider=track.gameObject.AddComponent<UnityEngine.UI.Slider>();
            var handleArea=Rect("Handle area",track); Stretch(handleArea); handleArea.offsetMin=new Vector2(18,0); handleArea.offsetMax=new Vector2(-18,0);
            var handle=Rect("Handle",handleArea); handle.sizeDelta=new Vector2(38,48);
            var handleImage=handle.gameObject.AddComponent<Image>(); handleImage.color=accent;
            slider.handleRect=handle; slider.targetGraphic=handleImage; slider.minValue=min; slider.maxValue=max;
            slider.value=Mathf.Clamp(initial,min,max);
            slider.onValueChanged.AddListener(value => { label.text=title+"  "+value.ToString("F2"); change(value); });
        }

        private Transform Row(Transform parent)
        {
            var row=Rect("Row",parent);
            row.gameObject.AddComponent<LayoutElement>().preferredHeight=72;
            var layout=row.gameObject.AddComponent<HorizontalLayoutGroup>();
            layout.spacing=12; layout.childControlWidth=true; layout.childForceExpandWidth=true;
            layout.childControlHeight=true; layout.childForceExpandHeight=true; return row;
        }

        private void Button(Transform parent,string title,UnityEngine.Events.UnityAction action,bool primary=false)
        {
            var rect=Rect(title,parent);
            rect.gameObject.AddComponent<LayoutElement>().flexibleWidth=1;
            var image=rect.gameObject.AddComponent<Image>(); image.color=primary ? accent : new Color(0.13f,0.2f,0.24f);
            var button=rect.gameObject.AddComponent<UnityEngine.UI.Button>(); button.targetGraphic=image; button.onClick.AddListener(action);
            var child=Rect("Text",rect); Stretch(child);
            var label=child.gameObject.AddComponent<Text>(); label.font=font; label.fontSize=25; label.text=title;
            label.color=primary ? new Color(0.015f,0.09f,0.08f) : ink; label.alignment=TextAnchor.MiddleCenter; label.raycastTarget=false;
        }

        private Text Label(Transform parent,string value,int size,float height)
        {
            var rect=Rect("Label",parent);
            rect.gameObject.AddComponent<LayoutElement>().preferredHeight=height;
            var label=rect.gameObject.AddComponent<Text>(); label.font=font; label.fontSize=size; label.color=ink;
            label.text=value; label.raycastTarget=false; label.alignment=TextAnchor.MiddleLeft; return label;
        }
        private static RectTransform Rect(string name,Transform parent)
        {
            var result=new GameObject(name,typeof(RectTransform)).GetComponent<RectTransform>();
            result.SetParent(parent,false); return result;
        }
        private static void Stretch(RectTransform rect) { rect.anchorMin=Vector2.zero; rect.anchorMax=Vector2.one; rect.offsetMin=rect.offsetMax=Vector2.zero; }
        private void FitSafeArea()
        {
            lastSafeArea=Screen.safeArea;
            safeArea.anchorMin=new Vector2(lastSafeArea.xMin/Mathf.Max(Screen.width,1),lastSafeArea.yMin/Mathf.Max(Screen.height,1));
            safeArea.anchorMax=new Vector2(lastSafeArea.xMax/Mathf.Max(Screen.width,1),lastSafeArea.yMax/Mathf.Max(Screen.height,1));
            safeArea.offsetMin=safeArea.offsetMax=Vector2.zero;
        }
        private void Update()
        {
            if (app == null) return;
            message.text=app.Message; capabilities.text=app.Capabilities;
            if (Screen.safeArea != lastSafeArea) FitSafeArea();
        }
        public void SetVisible(bool visible) { if (canvas != null) canvas.enabled=visible; }
        private void OnDestroy() { if (app != null) app.SettingsChanged -= RebuildControls; }
    }
}
