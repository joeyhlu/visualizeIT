Shader "VisualizeIt/SurfacePattern"
{
    Properties
    {
        _BaseMap("Design", 2D) = "white" {}
        _BaseColor("Surface color", Color) = (1,1,1,1)
        _PatternMetres("Tile width in metres", Float) = 0.05
        _PatternRotation("Rotation in radians", Float) = 0
        _Smoothness("Smoothness", Range(0,1)) = 0.4
        _Visibility("Tracking visibility", Range(0,1)) = 1
        _ShellMetres("Surface lift in metres", Range(0,0.01)) = 0.002
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" "RenderPipeline"="UniversalPipeline" }
        Cull Back
        ZWrite On
        HLSLINCLUDE
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
        TEXTURE2D(_BaseMap); SAMPLER(sampler_BaseMap);
        CBUFFER_START(UnityPerMaterial)
            float4 _BaseMap_TexelSize;
            half4 _BaseColor;
            float _PatternMetres;
            float _PatternRotation;
            half _Smoothness;
            half _Visibility;
            float _ShellMetres;
        CBUFFER_END
        struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; float2 uv : TEXCOORD0; UNITY_VERTEX_INPUT_INSTANCE_ID };
        struct Varyings
        {
            float4 positionCS : SV_POSITION;
            float3 positionWS : TEXCOORD0;
            half3 normalWS : TEXCOORD1;
            float2 uv : TEXCOORD2;
            UNITY_VERTEX_OUTPUT_STEREO
        };
        Varyings SurfaceVertex(Attributes input)
        {
            Varyings output = (Varyings)0;
            UNITY_SETUP_INSTANCE_ID(input);
            UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
            float3 positionOS = input.positionOS.xyz + input.normalOS * _ShellMetres;
            VertexPositionInputs position = GetVertexPositionInputs(positionOS);
            output.positionCS = position.positionCS;
            output.positionWS = position.positionWS;
            output.normalWS = TransformObjectToWorldNormal(input.normalOS);
            output.uv = input.uv;
            return output;
        }
        void TrackingClip(float2 screenPixel)
        {
            // Identical coverage in color and depth passes keeps AR background occlusion consistent.
            float threshold = frac(52.9829189 * frac(dot(floor(screenPixel),float2(0.06711056,0.00583715))));
            clip(_Visibility - threshold - 0.00001);
        }
        ENDHLSL
        Pass
        {
            Name "SurfaceForward"
            Tags { "LightMode"="UniversalForward" }
            HLSLPROGRAM
            #pragma target 3.0
            #pragma vertex SurfaceVertex
            #pragma fragment SurfaceFragment
            #pragma multi_compile_instancing
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            #pragma multi_compile_fragment _ _REFLECTION_PROBE_BLENDING
            #pragma multi_compile_fragment _ _REFLECTION_PROBE_BOX_PROJECTION
            #pragma multi_compile_fragment _ _REFLECTION_PROBE_ATLAS
            half4 SurfaceFragment(Varyings input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
                TrackingClip(input.positionCS.xy);
                float sine = sin(_PatternRotation), cosine = cos(_PatternRotation);
                float2 metricUV = float2(cosine*input.uv.x - sine*input.uv.y, sine*input.uv.x + cosine*input.uv.y);
                float aspect = max(_BaseMap_TexelSize.w,1) / max(_BaseMap_TexelSize.z,1);
                // Only tile width has a user minimum. Clamping height to 5 mm would stretch wide images.
                float2 uv = metricUV / float2(max(_PatternMetres,0.005),max(_PatternMetres*aspect,0.000001));
                half4 design = SAMPLE_TEXTURE2D(_BaseMap,sampler_BaseMap,uv);
                SurfaceData surface = (SurfaceData)0;
                surface.albedo = lerp(_BaseColor.rgb,design.rgb,design.a);
                surface.metallic = 0;
                surface.smoothness = _Smoothness;
                surface.normalTS = half3(0,0,1);
                surface.occlusion = 1;
                surface.alpha = 1;
                InputData data = (InputData)0;
                data.positionWS = input.positionWS;
                data.normalWS = NormalizeNormalPerPixel(input.normalWS);
                data.viewDirectionWS = GetWorldSpaceNormalizeViewDir(input.positionWS);
                data.shadowCoord = TransformWorldToShadowCoord(input.positionWS);
                data.bakedGI = SampleSH(data.normalWS);
                data.normalizedScreenSpaceUV = GetNormalizedScreenSpaceUV(input.positionCS);
                data.shadowMask = half4(1,1,1,1);
                return UniversalFragmentPBR(data,surface);
            }
            ENDHLSL
        }
        Pass
        {
            Name "SurfaceDepth"
            Tags { "LightMode"="DepthOnly" }
            ColorMask R
            HLSLPROGRAM
            #pragma target 3.0
            #pragma vertex SurfaceVertex
            #pragma fragment DepthFragment
            #pragma multi_compile_instancing
            half4 DepthFragment(Varyings input) : SV_Target
            {
                TrackingClip(input.positionCS.xy);
                return input.positionCS.z;
            }
            ENDHLSL
        }
    }
    Fallback Off
}
