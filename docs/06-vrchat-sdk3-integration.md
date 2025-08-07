# VRChat SDK3 Integration Implementation Guide

## Executive Summary

This document provides comprehensive implementation guidance for seamless VRChat SDK3 integration, covering avatar descriptor automation, viseme mapping, FX animator generation, and expression menu creation. The implementation ensures full compatibility with VRChat's Avatar 3.0 system and performance standards.

## VRChat SDK3 Architecture Overview

VRChat SDK3 introduces a sophisticated avatar system built around:
- **Avatar Descriptor**: Central configuration component
- **Playable Layers**: Multiple animator layers for different functionalities
- **Expression System**: User-controllable avatar expressions via radial menus
- **Parameters**: Sync'd values for cross-client avatar state
- **Performance Ranking**: Optimization guidelines for platform compatibility

### SDK3 Component Hierarchy

```
VRCAvatarDescriptor
├── View Position & Eye Look
├── Lip Sync Configuration
├── Playable Layers (5 total)
│   ├── Base Layer (locomotion)
│   ├── Additive Layer
│   ├── Gesture Layer
│   ├── Action Layer
│   └── FX Layer (expressions)
├── Expression Parameters
├── Expression Menu
└── Colliders & Contacts
```

## Core Integration Components

### VRChat Standards and Constants

```csharp
using UnityEngine;
using System.Collections.Generic;

namespace VRChatBlendShapeTools.VRChat
{
    public static class VRChatStandards
    {
        // VRChat Viseme Names (Oculus Standard)
        public static readonly string[] VISEME_NAMES = {
            "sil",  // 0 - Silence
            "pp",   // 1 - p, b, m
            "ff",   // 2 - f, v
            "th",   // 3 - th, dh
            "dd",   // 4 - t, d, n, l
            "kk",   // 5 - k, g, ng, h, r
            "ch",   // 6 - ch, jh, sh, zh
            "ss",   // 7 - s, z
            "nn",   // 8 - n, ng
            "rr",   // 9 - r
            "aa",   // 10 - aa, ae, ah, ao, aw, ay, ey, oy
            "e",    // 11 - eh, er, uh, uw
            "ih",   // 12 - ih, iy
            "oh",   // 13 - oh, ow
            "ou"    // 14 - uw, ow, oy
        };
        
        public static readonly string[] VRC_VISEME_KEYS = {
            "vrc.v_sil", "vrc.v_pp", "vrc.v_ff", "vrc.v_th", "vrc.v_dd",
            "vrc.v_kk", "vrc.v_ch", "vrc.v_ss", "vrc.v_nn", "vrc.v_rr",
            "vrc.v_aa", "vrc.v_e", "vrc.v_ih", "vrc.v_oh", "vrc.v_ou"
        };
        
        // Standard Expression Names
        public static readonly string[] EMOTION_NAMES = {
            "Angry", "Fun", "Joy", "Sorrow", "Surprised"
        };
        
        // Performance Limits
        public static readonly Dictionary<string, int> PERFORMANCE_LIMITS = new Dictionary<string, int>
        {
            {"Excellent_PolyCount", 7500},
            {"Good_PolyCount", 20000},
            {"Medium_PolyCount", 50000},
            {"Poor_PolyCount", 70000},
            {"Excellent_MaterialSlots", 1},
            {"Good_MaterialSlots", 4},
            {"Medium_MaterialSlots", 8},
            {"Poor_MaterialSlots", 16},
            {"Excellent_MeshRenderers", 1},
            {"Good_MeshRenderers", 4},
            {"Medium_MeshRenderers", 8},
            {"Poor_MeshRenderers", 16},
            {"MaxBones", 200},
            {"MaxBlendShapes", 32},
            {"MaxTextureSize", 2048}
        };
        
        // Parameter Names for Expressions
        public static readonly Dictionary<string, string> EXPRESSION_PARAMETERS = new Dictionary<string, string>
        {
            {"Angry", "Expr_Angry"},
            {"Fun", "Expr_Fun"},
            {"Joy", "Expr_Joy"},
            {"Sorrow", "Expr_Sorrow"},
            {"Surprised", "Expr_Surprised"}
        };
    }
}
```

### Avatar Descriptor Auto-Configuration

```csharp
using UnityEngine;
using UnityEditor;
using VRC.SDK3.Avatars.Components;
using VRC.SDK3.Avatars.ScriptableObjects;
using System.Collections.Generic;
using System.Linq;

namespace VRChatBlendShapeTools.VRChat
{
    public class AvatarDescriptorConfigurator
    {
        public static void ConfigureAvatarDescriptor(GameObject avatar, AvatarConfigurationOptions options)
        {
            var descriptor = EnsureAvatarDescriptor(avatar);
            
            // Configure basic settings
            ConfigureViewPosition(descriptor);
            ConfigureEyeLook(descriptor, options.EnableEyeLook);
            ConfigureLipSync(descriptor, options);
            
            // Configure playable layers
            ConfigurePlayableLayers(descriptor, options);
            
            // Configure expression system
            if (options.SetupExpressions)
            {
                ConfigureExpressionSystem(descriptor, options);
            }
            
            EditorUtility.SetDirty(descriptor);
            
            Debug.Log($"Avatar descriptor configured for {avatar.name}");
        }
        
        private static VRCAvatarDescriptor EnsureAvatarDescriptor(GameObject avatar)
        {
            var descriptor = avatar.GetComponent<VRCAvatarDescriptor>();
            
            if (descriptor == null)
            {
                descriptor = avatar.AddComponent<VRCAvatarDescriptor>();
                Debug.Log("Added VRCAvatarDescriptor component");
            }
            
            return descriptor;
        }
        
        private static void ConfigureViewPosition(VRCAvatarDescriptor descriptor)
        {
            var animator = descriptor.GetComponent<Animator>();
            
            if (animator != null && animator.isHuman)
            {
                // Use head bone for accurate positioning
                var headBone = animator.GetBoneTransform(HumanBodyBones.Head);
                
                if (headBone != null)
                {
                    var localHeadPos = descriptor.transform.InverseTransformPoint(headBone.position);
                    
                    // Position slightly forward and up from head center
                    descriptor.ViewPosition = new Vector3(
                        localHeadPos.x,
                        localHeadPos.y + 0.06f,  // 6cm up from head center
                        localHeadPos.z + 0.1f    // 10cm forward
                    );
                    
                    Debug.Log($"View position set using head bone: {descriptor.ViewPosition}");
                    return;
                }
            }
            
            // Fallback: use mesh bounds
            var renderer = descriptor.GetComponentInChildren<SkinnedMeshRenderer>();
            if (renderer != null)
            {
                var bounds = renderer.bounds;
                var center = descriptor.transform.InverseTransformPoint(bounds.center);
                
                descriptor.ViewPosition = new Vector3(
                    0f,
                    center.y + bounds.size.y * 0.4f,  // Near top of avatar
                    center.z + 0.1f                   // Slightly forward
                );
                
                Debug.Log($"View position set using mesh bounds: {descriptor.ViewPosition}");
            }
        }
        
        private static void ConfigureEyeLook(VRCAvatarDescriptor descriptor, bool enableEyeLook)
        {
            if (!enableEyeLook)
            {
                descriptor.enableEyeLook = false;
                return;
            }
            
            var animator = descriptor.GetComponent<Animator>();
            
            if (animator != null && animator.isHuman)
            {
                var leftEye = animator.GetBoneTransform(HumanBodyBones.LeftEye);
                var rightEye = animator.GetBoneTransform(HumanBodyBones.RightEye);
                
                if (leftEye != null && rightEye != null)
                {
                    descriptor.enableEyeLook = true;
                    descriptor.customEyeLookSettings.leftEye = leftEye;
                    descriptor.customEyeLookSettings.rightEye = rightEye;
                    
                    // Configure eye movement ranges
                    descriptor.customEyeLookSettings.eyesLookingUp.max = 15f;
                    descriptor.customEyeLookSettings.eyesLookingDown.max = 12f;
                    descriptor.customEyeLookSettings.eyesLookingStraight.max = 10f;
                    
                    Debug.Log("Eye look configured with custom eye bones");
                    return;
                }
            }
            
            Debug.LogWarning("Eye look requested but no eye bones found - disabling eye look");
            descriptor.enableEyeLook = false;
        }
        
        private static void ConfigureLipSync(VRCAvatarDescriptor descriptor, AvatarConfigurationOptions options)
        {
            if (!options.SetupVisemes)
            {
                descriptor.lipSync = VRC.SDKBase.VRC_AvatarDescriptor.LipSyncStyle.Default;
                return;
            }
            
            var renderer = descriptor.GetComponentInChildren<SkinnedMeshRenderer>();
            
            if (renderer == null)
            {
                Debug.LogWarning("No SkinnedMeshRenderer found for viseme setup");
                return;
            }
            
            var mesh = renderer.sharedMesh;
            var visemeMapping = AutoMapVisemes(mesh);
            
            if (visemeMapping.Count >= 15)
            {
                descriptor.lipSync = VRC.SDKBase.VRC_AvatarDescriptor.LipSyncStyle.VisemeBlendShape;
                descriptor.VisemeSkinnedMesh = renderer;
                descriptor.VisemeBlendShapes = visemeMapping.ToArray();
                
                Debug.Log($"Visemes configured: {visemeMapping.Count(s => !string.IsNullOrEmpty(s))} mapped");
            }
            else
            {
                Debug.LogWarning($"Insufficient viseme blend shapes found ({visemeMapping.Count}/15)");
                descriptor.lipSync = VRC.SDKBase.VRC_AvatarDescriptor.LipSyncStyle.Default;
            }
        }
        
        private static List<string> AutoMapVisemes(Mesh mesh)
        {
            var mapping = new List<string>(new string[15]);
            var blendShapeNames = new List<string>();
            
            // Get all blend shape names
            for (int i = 0; i < mesh.blendShapeCount; i++)
            {
                blendShapeNames.Add(mesh.GetBlendShapeName(i));
            }
            
            // Map each viseme using priority system
            for (int i = 0; i < VRChatStandards.VISEME_NAMES.Length; i++)
            {
                var visemeName = VRChatStandards.VISEME_NAMES[i];
                var vrcName = VRChatStandards.VRC_VISEME_KEYS[i];
                
                // Priority 1: Exact VRC naming
                var exactMatch = blendShapeNames.FirstOrDefault(name =>
                    string.Equals(name, vrcName, System.StringComparison.OrdinalIgnoreCase));
                
                if (exactMatch != null)
                {
                    mapping[i] = exactMatch;
                    continue;
                }
                
                // Priority 2: Standard viseme name
                var standardMatch = blendShapeNames.FirstOrDefault(name =>
                    string.Equals(name, visemeName, System.StringComparison.OrdinalIgnoreCase));
                
                if (standardMatch != null)
                {
                    mapping[i] = standardMatch;
                    continue;
                }
                
                // Priority 3: Partial match
                var partialMatch = blendShapeNames.FirstOrDefault(name =>
                    name.ToLower().Contains(visemeName.ToLower()));
                
                if (partialMatch != null)
                {
                    mapping[i] = partialMatch;
                    continue;
                }
                
                // No match found
                mapping[i] = "";
                Debug.LogWarning($"No blend shape found for viseme: {visemeName}");
            }
            
            return mapping;
        }
        
        private static void ConfigurePlayableLayers(VRCAvatarDescriptor descriptor, AvatarConfigurationOptions options)
        {
            // Initialize playable layers array
            if (descriptor.baseAnimationLayers == null || descriptor.baseAnimationLayers.Length != 5)
            {
                descriptor.baseAnimationLayers = new VRCAvatarDescriptor.CustomAnimLayer[5];
                
                // Initialize with default values
                for (int i = 0; i < 5; i++)
                {
                    descriptor.baseAnimationLayers[i] = new VRCAvatarDescriptor.CustomAnimLayer
                    {
                        type = (VRCAvatarDescriptor.AnimLayerType)i,
                        animatorController = null,
                        isDefault = true
                    };
                }
            }
            
            // Configure FX layer if requested
            if (options.CreateFXController && options.FXController != null)
            {
                descriptor.baseAnimationLayers[4] = new VRCAvatarDescriptor.CustomAnimLayer
                {
                    type = VRCAvatarDescriptor.AnimLayerType.FX,
                    animatorController = options.FXController,
                    isDefault = false
                };
                
                Debug.Log("FX layer configured with custom controller");
            }
        }
        
        private static void ConfigureExpressionSystem(VRCAvatarDescriptor descriptor, AvatarConfigurationOptions options)
        {
            // Configure expression parameters
            if (options.ExpressionParameters != null)
            {
                descriptor.expressionParameters = options.ExpressionParameters;
            }
            
            // Configure expression menu
            if (options.ExpressionMenu != null)
            {
                descriptor.expressionsMenu = options.ExpressionMenu;
            }
            
            Debug.Log("Expression system configured");
        }
    }
    
    [System.Serializable]
    public class AvatarConfigurationOptions
    {
        public bool EnableEyeLook = true;
        public bool SetupVisemes = true;
        public bool SetupExpressions = true;
        public bool CreateFXController = true;
        
        public RuntimeAnimatorController FXController;
        public VRCExpressionParameters ExpressionParameters;
        public VRCExpressionsMenu ExpressionMenu;
    }
}
```

### FX Animator Controller Generation

```csharp
using UnityEngine;
using UnityEditor;
using UnityEditor.Animations;
using VRC.SDK3.Avatars.ScriptableObjects;
using System.Collections.Generic;
using System.IO;

namespace VRChatBlendShapeTools.VRChat
{
    public class FXAnimatorGenerator
    {
        public static GeneratedFXAssets GenerateFXController(GameObject avatar, Mesh mesh)
        {
            var avatarName = avatar.name;
            var assetsDir = $"Assets/Generated/VRChat/{avatarName}";
            
            // Ensure directories exist
            Directory.CreateDirectory(assetsDir);
            Directory.CreateDirectory($"{assetsDir}/Animations");
            Directory.CreateDirectory($"{assetsDir}/Controllers");
            
            var generatedAssets = new GeneratedFXAssets();
            
            // Generate Expression Parameters
            generatedAssets.ExpressionParameters = GenerateExpressionParameters(mesh, $"{assetsDir}/{avatarName}_Parameters.asset");
            
            // Generate Expression Menu
            generatedAssets.ExpressionMenu = GenerateExpressionMenu(mesh, $"{assetsDir}/{avatarName}_Menu.asset");
            
            // Generate FX Controller
            generatedAssets.FXController = GenerateFXAnimatorController(avatar, mesh, $"{assetsDir}/Controllers/{avatarName}_FX.controller");
            
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            
            Debug.Log($"Generated FX assets for {avatarName}");
            return generatedAssets;
        }
        
        private static VRCExpressionParameters GenerateExpressionParameters(Mesh mesh, string assetPath)
        {
            var parameters = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            var parameterList = new List<VRCExpressionParameters.Parameter>();
            
            // Add standard VRChat parameters (these are reserved)
            parameterList.Add(new VRCExpressionParameters.Parameter
            {
                name = "VRCEmote",
                valueType = VRCExpressionParameters.ValueType.Int,
                defaultValue = 0,
                saved = false
            });
            
            parameterList.Add(new VRCExpressionParameters.Parameter
            {
                name = "VRCFaceBlendH",
                valueType = VRCExpressionParameters.ValueType.Float,
                defaultValue = 0,
                saved = false
            });
            
            parameterList.Add(new VRCExpressionParameters.Parameter
            {
                name = "VRCFaceBlendV",
                valueType = VRCExpressionParameters.ValueType.Float,
                defaultValue = 0,
                saved = false
            });
            
            // Add expression parameters
            foreach (var emotionName in VRChatStandards.EMOTION_NAMES)
            {
                if (HasBlendShapeForExpression(mesh, emotionName))
                {
                    var parameterName = VRChatStandards.EXPRESSION_PARAMETERS.GetValueOrDefault(emotionName, emotionName);
                    
                    parameterList.Add(new VRCExpressionParameters.Parameter
                    {
                        name = parameterName,
                        valueType = VRCExpressionParameters.ValueType.Float,
                        defaultValue = 0f,
                        saved = true
                    });
                }
            }
            
            // Add custom blend shape parameters
            for (int i = 0; i < mesh.blendShapeCount; i++)
            {
                var blendShapeName = mesh.GetBlendShapeName(i);
                
                // Skip visemes and emotions already handled
                if (IsVisemeBlendShape(blendShapeName) || VRChatStandards.EMOTION_NAMES.Contains(blendShapeName))
                    continue;
                
                // Add parameter for custom blend shape
                var parameterName = SanitizeParameterName(blendShapeName);
                
                if (parameterList.Count < 256) // VRChat parameter limit
                {
                    parameterList.Add(new VRCExpressionParameters.Parameter
                    {
                        name = parameterName,
                        valueType = VRCExpressionParameters.ValueType.Float,
                        defaultValue = 0f,
                        saved = true
                    });
                }
            }
            
            parameters.parameters = parameterList.ToArray();
            
            AssetDatabase.CreateAsset(parameters, assetPath);
            return parameters;
        }
        
        private static VRCExpressionsMenu GenerateExpressionMenu(Mesh mesh, string assetPath)
        {
            var menu = ScriptableObject.CreateInstance<VRCExpressionsMenu>();
            var controls = new List<VRCExpressionsMenu.Control>();
            
            // Create emotion controls
            foreach (var emotionName in VRChatStandards.EMOTION_NAMES)
            {
                if (HasBlendShapeForExpression(mesh, emotionName))
                {
                    var parameterName = VRChatStandards.EXPRESSION_PARAMETERS.GetValueOrDefault(emotionName, emotionName);
                    
                    controls.Add(new VRCExpressionsMenu.Control
                    {
                        name = emotionName,
                        type = VRCExpressionsMenu.Control.ControlType.RadialPuppet,
                        parameter = new VRCExpressionsMenu.Control.Parameter
                        {
                            name = parameterName
                        },
                        value = 1f,
                        icon = LoadEmotionIcon(emotionName)
                    });
                }
            }
            
            // Create submenu for additional blend shapes if needed
            var customBlendShapes = GetCustomBlendShapes(mesh);
            if (customBlendShapes.Count > 0)
            {
                var submenu = CreateCustomBlendShapeSubmenu(customBlendShapes, $"{Path.GetDirectoryName(assetPath)}/{Path.GetFileNameWithoutExtension(assetPath)}_Custom.asset");
                
                controls.Add(new VRCExpressionsMenu.Control
                {
                    name = "Custom Shapes",
                    type = VRCExpressionsMenu.Control.ControlType.SubMenu,
                    subMenu = submenu,
                    icon = LoadDefaultIcon()
                });
            }
            
            menu.controls = controls.ToArray();
            
            AssetDatabase.CreateAsset(menu, assetPath);
            return menu;
        }
        
        private static VRCExpressionsMenu CreateCustomBlendShapeSubmenu(List<string> blendShapeNames, string assetPath)
        {
            var submenu = ScriptableObject.CreateInstance<VRCExpressionsMenu>();
            var controls = new List<VRCExpressionsMenu.Control>();
            
            foreach (var shapeName in blendShapeNames.Take(8)) // VRChat menu limit
            {
                var parameterName = SanitizeParameterName(shapeName);
                
                controls.Add(new VRCExpressionsMenu.Control
                {
                    name = shapeName,
                    type = VRCExpressionsMenu.Control.ControlType.RadialPuppet,
                    parameter = new VRCExpressionsMenu.Control.Parameter
                    {
                        name = parameterName
                    },
                    value = 1f
                });
            }
            
            submenu.controls = controls.ToArray();
            
            AssetDatabase.CreateAsset(submenu, assetPath);
            return submenu;
        }
        
        private static AnimatorController GenerateFXAnimatorController(GameObject avatar, Mesh mesh, string controllerPath)
        {
            var controller = AnimatorController.CreateAnimatorControllerAtPath(controllerPath);
            
            // Get or create default layer
            var baseLayer = controller.layers[0];
            baseLayer.name = "BlendShape Controls";
            var stateMachine = baseLayer.stateMachine;
            
            // Create parameters and blend trees for each relevant blend shape
            var createdParameters = new HashSet<string>();
            
            // Process emotion blend shapes
            foreach (var emotionName in VRChatStandards.EMOTION_NAMES)
            {
                if (HasBlendShapeForExpression(mesh, emotionName))
                {
                    var parameterName = VRChatStandards.EXPRESSION_PARAMETERS.GetValueOrDefault(emotionName, emotionName);
                    
                    if (!createdParameters.Contains(parameterName))
                    {
                        CreateBlendShapeControl(controller, stateMachine, avatar, emotionName, parameterName);
                        createdParameters.Add(parameterName);
                    }
                }
            }
            
            // Process custom blend shapes
            var customShapes = GetCustomBlendShapes(mesh);
            foreach (var shapeName in customShapes)
            {
                var parameterName = SanitizeParameterName(shapeName);
                
                if (!createdParameters.Contains(parameterName) && controller.parameters.Length < 256)
                {
                    CreateBlendShapeControl(controller, stateMachine, avatar, shapeName, parameterName);
                    createdParameters.Add(parameterName);
                }
            }
            
            return controller;
        }
        
        private static void CreateBlendShapeControl(
            AnimatorController controller,
            AnimatorStateMachine stateMachine,
            GameObject avatar,
            string blendShapeName,
            string parameterName)
        {
            // Add parameter
            controller.AddParameter(new AnimatorControllerParameter
            {
                name = parameterName,
                type = AnimatorControllerParameterType.Float,
                defaultFloat = 0f
            });
            
            // Create animation clips
            var offClip = CreateBlendShapeAnimation(avatar, blendShapeName, 0f);
            var onClip = CreateBlendShapeAnimation(avatar, blendShapeName, 100f);
            
            // Save animation assets
            var animationsDir = Path.GetDirectoryName(AssetDatabase.GetAssetPath(controller)) + "/../Animations";
            var offClipPath = $"{animationsDir}/{blendShapeName}_Off.anim";
            var onClipPath = $"{animationsDir}/{blendShapeName}_On.anim";
            
            AssetDatabase.CreateAsset(offClip, offClipPath);
            AssetDatabase.CreateAsset(onClip, onClipPath);
            
            // Create blend tree
            var blendTree = new BlendTree
            {
                name = $"{blendShapeName}_BlendTree",
                blendType = BlendTreeType.Simple1D,
                blendParameter = parameterName,
                useAutomaticThresholds = false,
                minThreshold = 0f,
                maxThreshold = 1f
            };
            
            // Add motions to blend tree
            blendTree.children = new ChildMotion[]
            {
                new ChildMotion { motion = offClip, threshold = 0f, timeScale = 1f },
                new ChildMotion { motion = onClip, threshold = 1f, timeScale = 1f }
            };
            
            // Create state
            var state = stateMachine.AddState($"{blendShapeName}_State");
            state.motion = blendTree;
            state.writeDefaultValues = false;
            
            // Set as default state if it's the first one
            if (stateMachine.states.Length == 1)
            {
                stateMachine.defaultState = state;
            }
        }
        
        private static AnimationClip CreateBlendShapeAnimation(GameObject avatar, string blendShapeName, float value)
        {
            var clip = new AnimationClip
            {
                name = $"{blendShapeName}_{(value > 0 ? "On" : "Off")}"
            };
            
            // Find the skinned mesh renderer
            var renderer = avatar.GetComponentInChildren<SkinnedMeshRenderer>();
            if (renderer == null)
            {
                Debug.LogError("No SkinnedMeshRenderer found for animation creation");
                return clip;
            }
            
            // Get relative path to the renderer
            var rendererPath = GetRelativePath(avatar.transform, renderer.transform);
            
            // Create animation curve
            var curve = new AnimationCurve();
            curve.AddKey(0f, value);
            curve.AddKey(1f / 60f, value); // One frame duration
            
            // Apply curve to clip
            var binding = new EditorCurveBinding
            {
                path = rendererPath,
                type = typeof(SkinnedMeshRenderer),
                propertyName = $"blendShape.{blendShapeName}"
            };
            
            AnimationUtility.SetEditorCurve(clip, binding, curve);
            
            return clip;
        }
        
        // Helper methods
        private static bool HasBlendShapeForExpression(Mesh mesh, string expressionName)
        {
            for (int i = 0; i < mesh.blendShapeCount; i++)
            {
                var shapeName = mesh.GetBlendShapeName(i);
                if (string.Equals(shapeName, expressionName, System.StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }
            return false;
        }
        
        private static bool IsVisemeBlendShape(string blendShapeName)
        {
            return VRChatStandards.VRC_VISEME_KEYS.Contains(blendShapeName) ||
                   VRChatStandards.VISEME_NAMES.Any(v => blendShapeName.ToLower().Contains($"v_{v}"));
        }
        
        private static List<string> GetCustomBlendShapes(Mesh mesh)
        {
            var customShapes = new List<string>();
            
            for (int i = 0; i < mesh.blendShapeCount; i++)
            {
                var shapeName = mesh.GetBlendShapeName(i);
                
                if (!IsVisemeBlendShape(shapeName) && !VRChatStandards.EMOTION_NAMES.Contains(shapeName))
                {
                    customShapes.Add(shapeName);
                }
            }
            
            return customShapes;
        }
        
        private static string SanitizeParameterName(string name)
        {
            return name.Replace(" ", "_")
                      .Replace(".", "_")
                      .Replace("-", "_")
                      .Replace("(", "")
                      .Replace(")", "");
        }
        
        private static string GetRelativePath(Transform root, Transform target)
        {
            if (target == root)
                return "";
            
            var path = new List<string>();
            var current = target;
            
            while (current != root && current.parent != null)
            {
                path.Insert(0, current.name);
                current = current.parent;
            }
            
            return string.Join("/", path);
        }
        
        private static Texture2D LoadEmotionIcon(string emotionName)
        {
            // Load emotion-specific icon if available
            var iconPath = $"Assets/VRChatBlendShapeTools/Icons/{emotionName}.png";
            var icon = AssetDatabase.LoadAssetAtPath<Texture2D>(iconPath);
            
            return icon ?? LoadDefaultIcon();
        }
        
        private static Texture2D LoadDefaultIcon()
        {
            // Return a default icon texture
            return EditorGUIUtility.FindTexture("d_BlendTree Icon");
        }
    }
    
    [System.Serializable]
    public class GeneratedFXAssets
    {
        public AnimatorController FXController;
        public VRCExpressionParameters ExpressionParameters;
        public VRCExpressionsMenu ExpressionMenu;
    }
}
```

### Performance Validation and Optimization

```csharp
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;

namespace VRChatBlendShapeTools.VRChat
{
    public class VRChatPerformanceValidator
    {
        public static VRChatPerformanceReport ValidateAvatarPerformance(GameObject avatar)
        {
            var report = new VRChatPerformanceReport();
            
            // Analyze mesh statistics
            AnalyzeMeshPerformance(avatar, report);
            
            // Analyze material usage
            AnalyzeMaterialPerformance(avatar, report);
            
            // Analyze bone structure
            AnalyzeBonePerformance(avatar, report);
            
            // Analyze blend shapes
            AnalyzeBlendShapePerformance(avatar, report);
            
            // Calculate overall performance rating
            report.OverallRating = CalculatePerformanceRating(report);
            
            // Generate optimization recommendations
            GenerateOptimizationRecommendations(report);
            
            return report;
        }
        
        private static void AnalyzeMeshPerformance(GameObject avatar, VRChatPerformanceReport report)
        {
            var renderers = avatar.GetComponentsInChildren<Renderer>();
            
            int totalPolygons = 0;
            int skinnedMeshCount = 0;
            int meshRendererCount = 0;
            
            foreach (var renderer in renderers)
            {
                if (renderer is SkinnedMeshRenderer skinnedMesh)
                {
                    if (skinnedMesh.sharedMesh != null)
                    {
                        totalPolygons += skinnedMesh.sharedMesh.triangles.Length / 3;
                        skinnedMeshCount++;
                    }
                }
                else if (renderer is MeshRenderer meshRenderer)
                {
                    var meshFilter = renderer.GetComponent<MeshFilter>();
                    if (meshFilter?.sharedMesh != null)
                    {
                        totalPolygons += meshFilter.sharedMesh.triangles.Length / 3;
                        meshRendererCount++;
                    }
                }
            }
            
            report.PolygonCount = totalPolygons;
            report.SkinnedMeshRendererCount = skinnedMeshCount;
            report.MeshRendererCount = meshRendererCount;
            
            // Evaluate polygon performance
            if (totalPolygons <= VRChatStandards.PERFORMANCE_LIMITS["Excellent_PolyCount"])
                report.PolygonRating = PerformanceRating.Excellent;
            else if (totalPolygons <= VRChatStandards.PERFORMANCE_LIMITS["Good_PolyCount"])
                report.PolygonRating = PerformanceRating.Good;
            else if (totalPolygons <= VRChatStandards.PERFORMANCE_LIMITS["Medium_PolyCount"])
                report.PolygonRating = PerformanceRating.Medium;
            else if (totalPolygons <= VRChatStandards.PERFORMANCE_LIMITS["Poor_PolyCount"])
                report.PolygonRating = PerformanceRating.Poor;
            else
                report.PolygonRating = PerformanceRating.VeryPoor;
        }
        
        private static void AnalyzeMaterialPerformance(GameObject avatar, VRChatPerformanceReport report)
        {
            var renderers = avatar.GetComponentsInChildren<Renderer>();
            var materials = new HashSet<Material>();
            var textureMemory = 0L;
            
            foreach (var renderer in renderers)
            {
                if (renderer.sharedMaterials != null)
                {
                    foreach (var material in renderer.sharedMaterials)
                    {
                        if (material != null)
                        {
                            materials.Add(material);
                        }
                    }
                }
            }
            
            report.MaterialSlotCount = materials.Count;
            
            // Analyze textures
            foreach (var material in materials)
            {
                var shader = material.shader;
                
                for (int i = 0; i < ShaderUtil.GetPropertyCount(shader); i++)
                {
                    if (ShaderUtil.GetPropertyType(shader, i) == ShaderUtil.ShaderPropertyType.TexEnv)
                    {
                        var propertyName = ShaderUtil.GetPropertyName(shader, i);
                        var texture = material.GetTexture(propertyName) as Texture2D;
                        
                        if (texture != null)
                        {
                            textureMemory += GetTextureMemoryUsage(texture);
                            
                            if (texture.width > VRChatStandards.PERFORMANCE_LIMITS["MaxTextureSize"] ||
                                texture.height > VRChatStandards.PERFORMANCE_LIMITS["MaxTextureSize"])
                            {
                                report.Issues.Add($"Large texture detected: {texture.name} ({texture.width}x{texture.height})");
                            }
                        }
                    }
                }
            }
            
            report.TextureMemoryMB = textureMemory / (1024 * 1024);
            
            // Evaluate material performance
            if (report.MaterialSlotCount <= VRChatStandards.PERFORMANCE_LIMITS["Excellent_MaterialSlots"])
                report.MaterialRating = PerformanceRating.Excellent;
            else if (report.MaterialSlotCount <= VRChatStandards.PERFORMANCE_LIMITS["Good_MaterialSlots"])
                report.MaterialRating = PerformanceRating.Good;
            else if (report.MaterialSlotCount <= VRChatStandards.PERFORMANCE_LIMITS["Medium_MaterialSlots"])
                report.MaterialRating = PerformanceRating.Medium;
            else if (report.MaterialSlotCount <= VRChatStandards.PERFORMANCE_LIMITS["Poor_MaterialSlots"])
                report.MaterialRating = PerformanceRating.Poor;
            else
                report.MaterialRating = PerformanceRating.VeryPoor;
        }
        
        private static void AnalyzeBonePerformance(GameObject avatar, VRChatPerformanceReport report)
        {
            var animator = avatar.GetComponent<Animator>();
            
            if (animator?.avatar?.humanDescription?.human != null)
            {
                report.BoneCount = animator.avatar.humanDescription.human.Length;
            }
            else
            {
                // Count bones manually
                var allTransforms = avatar.GetComponentsInChildren<Transform>();
                report.BoneCount = allTransforms.Length;
            }
            
            if (report.BoneCount > VRChatStandards.PERFORMANCE_LIMITS["MaxBones"])
            {
                report.Issues.Add($"Too many bones: {report.BoneCount} (max recommended: {VRChatStandards.PERFORMANCE_LIMITS["MaxBones"]})");
                report.BoneRating = PerformanceRating.Poor;
            }
            else
            {
                report.BoneRating = PerformanceRating.Good;
            }
        }
        
        private static void AnalyzeBlendShapePerformance(GameObject avatar, VRChatPerformanceReport report)
        {
            var renderers = avatar.GetComponentsInChildren<SkinnedMeshRenderer>();
            int totalBlendShapes = 0;
            
            foreach (var renderer in renderers)
            {
                if (renderer.sharedMesh != null)
                {
                    totalBlendShapes += renderer.sharedMesh.blendShapeCount;
                }
            }
            
            report.BlendShapeCount = totalBlendShapes;
            
            if (totalBlendShapes > VRChatStandards.PERFORMANCE_LIMITS["MaxBlendShapes"])
            {
                report.Issues.Add($"Many blend shapes: {totalBlendShapes} (recommended: <{VRChatStandards.PERFORMANCE_LIMITS["MaxBlendShapes"]})");
                report.BlendShapeRating = PerformanceRating.Medium;
            }
            else
            {
                report.BlendShapeRating = PerformanceRating.Good;
            }
        }
        
        private static PerformanceRating CalculatePerformanceRating(VRChatPerformanceReport report)
        {
            var ratings = new[]
            {
                report.PolygonRating,
                report.MaterialRating,
                report.BoneRating,
                report.BlendShapeRating
            };
            
            var averageScore = ratings.Select(r => (int)r).Average();
            
            return (PerformanceRating)Mathf.RoundToInt((float)averageScore);
        }
        
        private static void GenerateOptimizationRecommendations(VRChatPerformanceReport report)
        {
            var recommendations = new List<string>();
            
            if (report.PolygonRating < PerformanceRating.Good)
            {
                recommendations.Add("Consider reducing polygon count through decimation or LOD system");
            }
            
            if (report.MaterialRating < PerformanceRating.Good)
            {
                recommendations.Add("Combine materials using texture atlasing to reduce draw calls");
            }
            
            if (report.BoneCount > 150)
            {
                recommendations.Add("Remove unnecessary bones or consider bone merging");
            }
            
            if (report.TextureMemoryMB > 50)
            {
                recommendations.Add("Compress textures or reduce resolution to save VRAM");
            }
            
            report.OptimizationRecommendations = recommendations;
        }
        
        private static long GetTextureMemoryUsage(Texture2D texture)
        {
            // Approximate memory usage calculation
            var format = texture.format;
            var bytesPerPixel = 4; // Default RGBA32
            
            switch (format)
            {
                case TextureFormat.DXT1:
                case TextureFormat.BC4:
                    bytesPerPixel = 1;
                    break;
                case TextureFormat.DXT5:
                case TextureFormat.BC5:
                    bytesPerPixel = 1;
                    break;
                case TextureFormat.RGB24:
                    bytesPerPixel = 3;
                    break;
                case TextureFormat.RGBA32:
                case TextureFormat.ARGB32:
                    bytesPerPixel = 4;
                    break;
            }
            
            return texture.width * texture.height * bytesPerPixel;
        }
    }
    
    public enum PerformanceRating
    {
        Excellent = 4,
        Good = 3,
        Medium = 2,
        Poor = 1,
        VeryPoor = 0
    }
    
    [System.Serializable]
    public class VRChatPerformanceReport
    {
        public PerformanceRating OverallRating;
        public PerformanceRating PolygonRating;
        public PerformanceRating MaterialRating;
        public PerformanceRating BoneRating;
        public PerformanceRating BlendShapeRating;
        
        public int PolygonCount;
        public int MaterialSlotCount;
        public int BoneCount;
        public int BlendShapeCount;
        public int SkinnedMeshRendererCount;
        public int MeshRendererCount;
        public long TextureMemoryMB;
        
        public List<string> Issues = new List<string>();
        public List<string> OptimizationRecommendations = new List<string>();
        
        public string GetRatingString()
        {
            return OverallRating switch
            {
                PerformanceRating.Excellent => "Excellent",
                PerformanceRating.Good => "Good",
                PerformanceRating.Medium => "Medium",
                PerformanceRating.Poor => "Poor",
                PerformanceRating.VeryPoor => "Very Poor",
                _ => "Unknown"
            };
        }
        
        public Color GetRatingColor()
        {
            return OverallRating switch
            {
                PerformanceRating.Excellent => Color.green,
                PerformanceRating.Good => Color.yellow,
                PerformanceRating.Medium => new Color(1f, 0.65f, 0f), // Orange
                PerformanceRating.Poor => Color.red,
                PerformanceRating.VeryPoor => Color.magenta,
                _ => Color.white
            };
        }
    }
}
```

## Automated Upload Validation

```csharp
using UnityEngine;
using UnityEditor;
using VRC.SDK3.Avatars.Components;

namespace VRChatBlendShapeTools.VRChat
{
    public class VRChatUploadValidator
    {
        public static UploadValidationResult ValidateAvatarForUpload(GameObject avatar)
        {
            var result = new UploadValidationResult();
            
            // Check for VRCAvatarDescriptor
            var descriptor = avatar.GetComponent<VRCAvatarDescriptor>();
            if (descriptor == null)
            {
                result.AddError("VRCAvatarDescriptor component is required");
                return result;
            }
            
            // Validate descriptor settings
            ValidateDescriptorSettings(descriptor, result);
            
            // Validate performance
            ValidatePerformanceRequirements(avatar, result);
            
            // Validate mesh integrity
            ValidateMeshIntegrity(avatar, result);
            
            // Validate animation setup
            ValidateAnimationSetup(descriptor, result);
            
            return result;
        }
        
        private static void ValidateDescriptorSettings(VRCAvatarDescriptor descriptor, UploadValidationResult result)
        {
            // Check view position
            if (descriptor.ViewPosition == Vector3.zero)
            {
                result.AddWarning("View position not set - using default");
            }
            
            // Validate lip sync setup
            if (descriptor.lipSync == VRC.SDKBase.VRC_AvatarDescriptor.LipSyncStyle.VisemeBlendShape)
            {
                if (descriptor.VisemeSkinnedMesh == null)
                {
                    result.AddError("Viseme blend shape mode selected but no mesh assigned");
                }
                else if (descriptor.VisemeBlendShapes == null || descriptor.VisemeBlendShapes.Length != 15)
                {
                    result.AddError("Incomplete viseme blend shape mapping");
                }
            }
            
            // Check animation layers
            if (descriptor.baseAnimationLayers != null)
            {
                for (int i = 0; i < descriptor.baseAnimationLayers.Length; i++)
                {
                    var layer = descriptor.baseAnimationLayers[i];
                    if (!layer.isDefault && layer.animatorController == null)
                    {
                        result.AddWarning($"Custom {layer.type} layer specified but no controller assigned");
                    }
                }
            }
        }
        
        private static void ValidatePerformanceRequirements(GameObject avatar, UploadValidationResult result)
        {
            var performanceReport = VRChatPerformanceValidator.ValidateAvatarPerformance(avatar);
            
            // Add performance-related issues
            foreach (var issue in performanceReport.Issues)
            {
                result.AddWarning($"Performance: {issue}");
            }
            
            // Check critical performance limits
            if (performanceReport.PolygonCount > VRChatStandards.PERFORMANCE_LIMITS["Poor_PolyCount"])
            {
                result.AddError($"Polygon count ({performanceReport.PolygonCount:N0}) exceeds maximum limit");
            }
            
            if (performanceReport.BoneCount > VRChatStandards.PERFORMANCE_LIMITS["MaxBones"])
            {
                result.AddError($"Bone count ({performanceReport.BoneCount}) exceeds maximum limit");
            }
        }
        
        private static void ValidateMeshIntegrity(GameObject avatar, UploadValidationResult result)
        {
            var renderers = avatar.GetComponentsInChildren<SkinnedMeshRenderer>();
            
            foreach (var renderer in renderers)
            {
                if (renderer.sharedMesh == null)
                {
                    result.AddError($"SkinnedMeshRenderer '{renderer.name}' has no mesh assigned");
                    continue;
                }
                
                var mesh = renderer.sharedMesh;
                
                // Check for valid UV mapping
                if (mesh.uv.Length == 0)
                {
                    result.AddWarning($"Mesh '{mesh.name}' has no UV coordinates");
                }
                
                // Check for degenerate triangles
                var vertices = mesh.vertices;
                var triangles = mesh.triangles;
                
                for (int i = 0; i < triangles.Length; i += 3)
                {
                    var v1 = vertices[triangles[i]];
                    var v2 = vertices[triangles[i + 1]];
                    var v3 = vertices[triangles[i + 2]];
                    
                    if (Vector3.Cross(v2 - v1, v3 - v1).magnitude < 0.001f)
                    {
                        result.AddWarning($"Mesh '{mesh.name}' contains degenerate triangles");
                        break;
                    }
                }
            }
        }
        
        private static void ValidateAnimationSetup(VRCAvatarDescriptor descriptor, UploadValidationResult result)
        {
            // Validate expression parameters
            if (descriptor.expressionParameters != null)
            {
                var parameters = descriptor.expressionParameters.parameters;
                
                if (parameters.Length > 256)
                {
                    result.AddError("Too many expression parameters (max 256)");
                }
                
                var memoryUsage = 0;
                foreach (var param in parameters)
                {
                    switch (param.valueType)
                    {
                        case VRCExpressionParameters.ValueType.Bool:
                            memoryUsage += 1;
                            break;
                        case VRCExpressionParameters.ValueType.Int:
                        case VRCExpressionParameters.ValueType.Float:
                            memoryUsage += 8;
                            break;
                    }
                }
                
                if (memoryUsage > 256)
                {
                    result.AddError($"Expression parameters exceed memory limit: {memoryUsage}/256 bits");
                }
            }
            
            // Validate expression menu
            if (descriptor.expressionsMenu != null)
            {
                ValidateExpressionMenu(descriptor.expressionsMenu, result, 0);
            }
        }
        
        private static void ValidateExpressionMenu(VRCExpressionsMenu menu, UploadValidationResult result, int depth)
        {
            if (depth > 5)
            {
                result.AddError("Expression menu depth exceeds maximum (5 levels)");
                return;
            }
            
            if (menu.controls.Length > 8)
            {
                result.AddError($"Expression menu has too many controls: {menu.controls.Length}/8");
            }
            
            foreach (var control in menu.controls)
            {
                if (control.type == VRCExpressionsMenu.Control.ControlType.SubMenu && control.subMenu != null)
                {
                    ValidateExpressionMenu(control.subMenu, result, depth + 1);
                }
            }
        }
    }
    
    [System.Serializable]
    public class UploadValidationResult
    {
        public List<string> Errors = new List<string>();
        public List<string> Warnings = new List<string>();
        
        public bool IsValid => Errors.Count == 0;
        
        public void AddError(string message)
        {
            Errors.Add(message);
            Debug.LogError($"Upload Validation Error: {message}");
        }
        
        public void AddWarning(string message)
        {
            Warnings.Add(message);
            Debug.LogWarning($"Upload Validation Warning: {message}");
        }
    }
}
```

## Implementation Timeline

### Week 1: Core SDK3 Integration
- VRChat standards and constants
- Avatar descriptor auto-configuration
- Basic viseme mapping system

### Week 2: FX System Generation
- FX animator controller generation
- Expression parameters creation
- Animation clip generation

### Week 3: Performance and Validation
- Performance validation system
- Upload readiness checking
- Optimization recommendations

### Week 4: Advanced Features
- Expression menu generation
- Advanced parameter management
- Integration testing with VRChat SDK

## Performance Targets

- **Configuration Speed**: <30 seconds for complete avatar setup
- **Accuracy**: 95%+ successful viseme auto-detection
- **VRChat Compliance**: 100% validation against VRChat standards  
- **Performance Rating**: Maintain "Good" or better performance rating
- **Compatibility**: Support VRChat SDK 3.4+ through latest versions

This comprehensive VRChat SDK3 integration implementation provides professional-grade automation while ensuring full compatibility with VRChat's Avatar 3.0 system requirements and performance standards.