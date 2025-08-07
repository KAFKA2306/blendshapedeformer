# Unity Editor Tools Development for VRChat SDK3 Integration

## Executive Summary

This document provides comprehensive implementation guidance for developing Unity Editor tools that automate VRChat SDK3 avatar setup, including blend shape integration, avatar descriptor configuration, and FX animator generation.

## Unity Editor Tool Architecture

The Unity Editor tool system consists of multiple interconnected components designed to streamline the VRChat avatar creation workflow from FBX import to upload-ready configuration.

### Core Tool Structure

```
VRChatBlendShapeTools/
├── Editor/
│   ├── Windows/
│   │   ├── BlendShapeIntegrationWindow.cs
│   │   ├── AvatarSetupWizard.cs
│   │   └── PerformanceAnalyzer.cs
│   ├── Utilities/
│   │   ├── FBXProcessor.cs
│   │   ├── AvatarDescriptorSetup.cs
│   │   ├── VisemeMapper.cs
│   │   └── FXAnimatorBuilder.cs
│   ├── Inspectors/
│   │   └── BlendShapeInspector.cs
│   └── Data/
│       ├── VRChatStandards.cs
│       └── BlendShapeMapping.cs
└── Runtime/
    └── Components/
        └── BlendShapeController.cs
```

## Main Integration Window Implementation

```csharp
using UnityEngine;
using UnityEditor;
using VRC.SDK3.Avatars.Components;
using VRC.SDK3.Avatars.ScriptableObjects;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace VRChatBlendShapeTools.Editor
{
    public class BlendShapeIntegrationWindow : EditorWindow
    {
        private GameObject targetAvatar;
        private string blendShapeFBXPath = "";
        private Vector2 scrollPosition;
        
        // Integration options
        private bool autoSetupVisemes = true;
        private bool autoSetupExpressions = true;
        private bool createFXAnimator = true;
        private bool optimizePerformance = true;
        private bool generateExpressionMenu = true;
        
        // Progress tracking
        private bool isProcessing = false;
        private string currentStatus = "";
        private float progressValue = 0f;
        
        // Validation results
        private AvatarValidationResult validationResult;
        
        [MenuItem("VRChat/BlendShape Integration Tool")]
        public static void ShowWindow()
        {
            var window = GetWindow<BlendShapeIntegrationWindow>("VRChat BlendShape Integration");
            window.minSize = new Vector2(400, 600);
        }
        
        void OnGUI()
        {
            EditorGUILayout.Space(10);
            
            // Header
            GUILayout.Label("VRChat BlendShape Integration Tool", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "This tool automates the integration of AI-generated blend shapes into VRChat avatars.", 
                MessageType.Info
            );
            
            EditorGUILayout.Space(10);
            
            scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition);
            
            DrawAvatarSelection();
            DrawFBXSelection();
            DrawIntegrationOptions();
            DrawAvatarValidation();
            DrawActionButtons();
            DrawProgressDisplay();
            
            EditorGUILayout.EndScrollView();
        }
        
        void DrawAvatarSelection()
        {
            EditorGUILayout.LabelField("Avatar Selection", EditorStyles.boldLabel);
            
            using (new EditorGUILayout.VerticalScope("box"))
            {
                targetAvatar = (GameObject)EditorGUILayout.ObjectField(
                    "Target Avatar", 
                    targetAvatar, 
                    typeof(GameObject), 
                    true
                );
                
                if (targetAvatar != null)
                {
                    DrawAvatarInfo();
                }
                else
                {
                    EditorGUILayout.HelpBox(
                        "Select a GameObject with VRC Avatar Descriptor or a humanoid avatar to begin.", 
                        MessageType.Warning
                    );
                }
            }
        }
        
        void DrawAvatarInfo()
        {
            using (new EditorGUILayout.VerticalScope("box"))
            {
                EditorGUILayout.LabelField("Avatar Information", EditorStyles.miniBoldLabel);
                
                var avatarDescriptor = targetAvatar.GetComponent<VRCAvatarDescriptor>();
                var animator = targetAvatar.GetComponent<Animator>();
                var skinnedMeshRenderer = targetAvatar.GetComponentInChildren<SkinnedMeshRenderer>();
                
                // Basic info
                EditorGUILayout.LabelField($"Name: {targetAvatar.name}");
                
                if (avatarDescriptor != null)
                {
                    EditorGUILayout.LabelField("✓ VRC Avatar Descriptor found", EditorStyles.helpBox);
                    EditorGUILayout.LabelField($"View Position: {avatarDescriptor.ViewPosition}");
                }
                else
                {
                    EditorGUILayout.LabelField("⚠ VRC Avatar Descriptor not found - will be added", EditorStyles.helpBox);
                }
                
                if (animator != null)
                {
                    EditorGUILayout.LabelField($"✓ Animator: {(animator.isHuman ? "Humanoid" : "Generic")}");
                }
                
                if (skinnedMeshRenderer != null)
                {
                    var mesh = skinnedMeshRenderer.sharedMesh;
                    EditorGUILayout.LabelField($"Mesh: {mesh.name}");
                    EditorGUILayout.LabelField($"Vertices: {mesh.vertexCount:N0}");
                    EditorGUILayout.LabelField($"Existing BlendShapes: {mesh.blendShapeCount}");
                }
            }
        }
        
        void DrawFBXSelection()
        {
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("BlendShape FBX", EditorStyles.boldLabel);
            
            using (new EditorGUILayout.VerticalScope("box"))
            {
                using (new EditorGUILayout.HorizontalScope())
                {
                    blendShapeFBXPath = EditorGUILayout.TextField("FBX Path", blendShapeFBXPath);
                    
                    if (GUILayout.Button("Browse", GUILayout.Width(60)))
                    {
                        string selectedPath = EditorUtility.OpenFilePanel(
                            "Select BlendShape FBX", 
                            "Assets", 
                            "fbx"
                        );
                        
                        if (!string.IsNullOrEmpty(selectedPath))
                        {
                            blendShapeFBXPath = FileUtil.GetProjectRelativePath(selectedPath);
                        }
                    }
                }
                
                if (!string.IsNullOrEmpty(blendShapeFBXPath))
                {
                    var fbxAsset = AssetDatabase.LoadAssetAtPath<GameObject>(blendShapeFBXPath);
                    if (fbxAsset != null)
                    {
                        var fbxRenderer = fbxAsset.GetComponentInChildren<SkinnedMeshRenderer>();
                        if (fbxRenderer != null)
                        {
                            EditorGUILayout.LabelField($"✓ FBX loaded: {fbxRenderer.sharedMesh.blendShapeCount} blend shapes found");
                        }
                        else
                        {
                            EditorGUILayout.HelpBox("No SkinnedMeshRenderer found in FBX", MessageType.Warning);
                        }
                    }
                    else
                    {
                        EditorGUILayout.HelpBox("Invalid FBX path", MessageType.Error);
                    }
                }
            }
        }
        
        void DrawIntegrationOptions()
        {
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Integration Options", EditorStyles.boldLabel);
            
            using (new EditorGUILayout.VerticalScope("box"))
            {
                autoSetupVisemes = EditorGUILayout.Toggle("Auto Setup Visemes", autoSetupVisemes);
                autoSetupExpressions = EditorGUILayout.Toggle("Auto Setup Expressions", autoSetupExpressions);
                createFXAnimator = EditorGUILayout.Toggle("Create FX Animator", createFXAnimator);
                generateExpressionMenu = EditorGUILayout.Toggle("Generate Expression Menu", generateExpressionMenu);
                optimizePerformance = EditorGUILayout.Toggle("Optimize Performance", optimizePerformance);
                
                EditorGUILayout.Space(5);
                EditorGUILayout.HelpBox(
                    "These options will automatically configure your avatar for VRChat compatibility.", 
                    MessageType.Info
                );
            }
        }
        
        void DrawAvatarValidation()
        {
            if (targetAvatar == null) return;
            
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField("Avatar Validation", EditorStyles.boldLabel);
            
            using (new EditorGUILayout.VerticalScope("box"))
            {
                if (GUILayout.Button("Validate Avatar"))
                {
                    validationResult = AvatarValidator.ValidateAvatar(targetAvatar);
                }
                
                if (validationResult != null)
                {
                    DrawValidationResults();
                }
            }
        }
        
        void DrawValidationResults()
        {
            EditorGUILayout.Space(5);
            
            // Performance rating
            var performanceColor = GetPerformanceColor(validationResult.PerformanceRank);
            var previousColor = GUI.color;
            GUI.color = performanceColor;
            
            EditorGUILayout.LabelField($"Performance Rank: {validationResult.PerformanceRank}", EditorStyles.boldLabel);
            GUI.color = previousColor;
            
            // Statistics
            EditorGUILayout.LabelField($"Polygon Count: {validationResult.PolygonCount:N0}");
            EditorGUILayout.LabelField($"Material Slots: {validationResult.MaterialCount}");
            EditorGUILayout.LabelField($"Skinned Mesh Renderers: {validationResult.SkinnedMeshCount}");
            EditorGUILayout.LabelField($"Bone Count: {validationResult.BoneCount}");
            
            // Issues
            if (validationResult.Issues.Count > 0)
            {
                EditorGUILayout.Space(5);
                EditorGUILayout.LabelField("Issues:", EditorStyles.miniBoldLabel);
                
                foreach (var issue in validationResult.Issues)
                {
                    var messageType = issue.Severity == ValidationSeverity.Error ? MessageType.Error : MessageType.Warning;
                    EditorGUILayout.HelpBox(issue.Message, messageType);
                }
            }
        }
        
        Color GetPerformanceColor(string performanceRank)
        {
            switch (performanceRank?.ToLower())
            {
                case "excellent": return Color.green;
                case "good": return Color.yellow;
                case "medium": return new Color(1f, 0.5f, 0f); // Orange
                case "poor": return Color.red;
                default: return Color.white;
            }
        }
        
        void DrawActionButtons()
        {
            EditorGUILayout.Space(10);
            
            using (new EditorGUILayout.VerticalScope("box"))
            {
                GUI.enabled = CanStartIntegration() && !isProcessing;
                
                if (GUILayout.Button("Start Integration", GUILayout.Height(40)))
                {
                    StartIntegrationProcess();
                }
                
                GUI.enabled = true;
                
                if (isProcessing && GUILayout.Button("Cancel", GUILayout.Height(25)))
                {
                    CancelIntegration();
                }
            }
        }
        
        bool CanStartIntegration()
        {
            return targetAvatar != null && 
                   !string.IsNullOrEmpty(blendShapeFBXPath) &&
                   File.Exists(blendShapeFBXPath);
        }
        
        void DrawProgressDisplay()
        {
            if (!isProcessing) return;
            
            EditorGUILayout.Space(10);
            
            using (new EditorGUILayout.VerticalScope("box"))
            {
                EditorGUILayout.LabelField("Processing...", EditorStyles.boldLabel);
                
                var rect = GUILayoutUtility.GetRect(0, 20);
                EditorGUI.ProgressBar(rect, progressValue, currentStatus);
            }
        }
        
        async void StartIntegrationProcess()
        {
            isProcessing = true;
            progressValue = 0f;
            
            try
            {
                var processor = new AvatarIntegrationProcessor();
                
                var options = new IntegrationOptions
                {
                    AutoSetupVisemes = autoSetupVisemes,
                    AutoSetupExpressions = autoSetupExpressions,
                    CreateFXAnimator = createFXAnimator,
                    GenerateExpressionMenu = generateExpressionMenu,
                    OptimizePerformance = optimizePerformance
                };
                
                await processor.ProcessAvatar(
                    targetAvatar, 
                    blendShapeFBXPath, 
                    options,
                    OnProgressUpdate
                );
                
                EditorUtility.DisplayDialog(
                    "Success", 
                    "Avatar integration completed successfully!", 
                    "OK"
                );
            }
            catch (System.Exception ex)
            {
                EditorUtility.DisplayDialog(
                    "Error", 
                    $"Integration failed: {ex.Message}", 
                    "OK"
                );
                
                Debug.LogError($"Avatar integration error: {ex}");
            }
            finally
            {
                isProcessing = false;
                progressValue = 0f;
                currentStatus = "";
            }
        }
        
        void OnProgressUpdate(float progress, string status)
        {
            progressValue = progress;
            currentStatus = status;
            Repaint();
        }
        
        void CancelIntegration()
        {
            isProcessing = false;
            progressValue = 0f;
            currentStatus = "";
        }
        
        void OnInspectorUpdate()
        {
            if (isProcessing)
            {
                Repaint();
            }
        }
    }
}
```

## Avatar Integration Processor

```csharp
using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEditor;
using VRC.SDK3.Avatars.Components;
using VRC.SDK3.Avatars.ScriptableObjects;

namespace VRChatBlendShapeTools.Editor
{
    public class AvatarIntegrationProcessor
    {
        public async Task ProcessAvatar(
            GameObject targetAvatar, 
            string fbxPath, 
            IntegrationOptions options,
            Action<float, string> progressCallback = null)
        {
            progressCallback?.Invoke(0.1f, "Starting integration...");
            
            // Step 1: Import and merge blend shapes
            progressCallback?.Invoke(0.2f, "Importing blend shapes from FBX...");
            await ImportBlendShapes(targetAvatar, fbxPath);
            
            // Step 2: Setup VRC Avatar Descriptor
            progressCallback?.Invoke(0.4f, "Setting up VRC Avatar Descriptor...");
            SetupAvatarDescriptor(targetAvatar, options);
            
            // Step 3: Setup visemes
            if (options.AutoSetupVisemes)
            {
                progressCallback?.Invoke(0.6f, "Configuring visemes...");
                SetupVisemes(targetAvatar);
            }
            
            // Step 4: Create FX Animator
            if (options.CreateFXAnimator)
            {
                progressCallback?.Invoke(0.7f, "Creating FX Animator...");
                await CreateFXAnimator(targetAvatar);
            }
            
            // Step 5: Generate Expression Menu
            if (options.GenerateExpressionMenu)
            {
                progressCallback?.Invoke(0.8f, "Generating expression menu...");
                GenerateExpressionMenu(targetAvatar);
            }
            
            // Step 6: Performance optimization
            if (options.OptimizePerformance)
            {
                progressCallback?.Invoke(0.9f, "Optimizing performance...");
                OptimizeAvatarPerformance(targetAvatar);
            }
            
            progressCallback?.Invoke(1.0f, "Integration complete!");
        }
        
        async Task ImportBlendShapes(GameObject targetAvatar, string fbxPath)
        {
            var importer = AssetImporter.GetAtPath(fbxPath) as ModelImporter;
            
            if (importer == null)
                throw new Exception("Failed to get ModelImporter for FBX");
            
            // Configure import settings
            importer.importBlendShapes = true;
            importer.importCameras = false;
            importer.importLights = false;
            importer.importAnimation = false;
            
            // Apply import settings
            AssetDatabase.ImportAsset(fbxPath, ImportAssetOptions.ForceUpdate);
            
            // Load imported asset
            var importedAsset = AssetDatabase.LoadAssetAtPath<GameObject>(fbxPath);
            var importedRenderer = importedAsset.GetComponentInChildren<SkinnedMeshRenderer>();
            var targetRenderer = targetAvatar.GetComponentInChildren<SkinnedMeshRenderer>();
            
            if (importedRenderer == null || targetRenderer == null)
                throw new Exception("Failed to find SkinnedMeshRenderer in source or target");
            
            // Create new mesh with blend shapes
            var newMesh = UnityEngine.Object.Instantiate(importedRenderer.sharedMesh);
            
            // Generate unique asset path
            var meshPath = $"Assets/Generated/Meshes/{targetAvatar.name}_BlendShapes.asset";
            var directory = System.IO.Path.GetDirectoryName(meshPath);
            
            if (!System.IO.Directory.Exists(directory))
                System.IO.Directory.CreateDirectory(directory);
            
            // Save mesh asset
            AssetDatabase.CreateAsset(newMesh, meshPath);
            
            // Apply to target renderer
            targetRenderer.sharedMesh = newMesh;
            
            Debug.Log($"Successfully imported {newMesh.blendShapeCount} blend shapes");
        }
        
        void SetupAvatarDescriptor(GameObject targetAvatar, IntegrationOptions options)
        {
            var descriptor = targetAvatar.GetComponent<VRCAvatarDescriptor>();
            
            if (descriptor == null)
            {
                descriptor = targetAvatar.AddComponent<VRCAvatarDescriptor>();
            }
            
            // Setup view position
            SetupViewPosition(descriptor);
            
            // Setup eye look
            SetupEyeLook(descriptor);
            
            // Setup base animation layers
            SetupBaseAnimationLayers(descriptor);
            
            EditorUtility.SetDirty(descriptor);
        }
        
        void SetupViewPosition(VRCAvatarDescriptor descriptor)
        {
            // Try to find head bone
            var animator = descriptor.GetComponent<Animator>();
            
            if (animator != null && animator.isHuman)
            {
                var headBone = animator.GetBoneTransform(HumanBodyBones.Head);
                
                if (headBone != null)
                {
                    // Set view position relative to head
                    var localPos = descriptor.transform.InverseTransformPoint(headBone.position);
                    descriptor.ViewPosition = localPos + Vector3.up * 0.06f; // Slightly above head center
                    return;
                }
            }
            
            // Fallback: calculate from mesh bounds
            var renderer = descriptor.GetComponentInChildren<SkinnedMeshRenderer>();
            if (renderer != null)
            {
                var bounds = renderer.bounds;
                var localBounds = descriptor.transform.InverseTransformPoint(bounds.center);
                descriptor.ViewPosition = new Vector3(0, localBounds.y + bounds.size.y * 0.4f, localBounds.z + 0.1f);
            }
        }
        
        void SetupEyeLook(VRCAvatarDescriptor descriptor)
        {
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
                    
                    // Configure eye movement limits
                    descriptor.customEyeLookSettings.eyesLookingUp.max = 15f;
                    descriptor.customEyeLookSettings.eyesLookingDown.max = 12f;
                    descriptor.customEyeLookSettings.eyesLookingStraight.max = 10f;
                }
            }
        }
        
        void SetupVisemes(GameObject targetAvatar)
        {
            var descriptor = targetAvatar.GetComponent<VRCAvatarDescriptor>();
            var renderer = targetAvatar.GetComponentInChildren<SkinnedMeshRenderer>();
            
            if (descriptor == null || renderer == null) return;
            
            var mesh = renderer.sharedMesh;
            var visemeMapper = new VisemeMapper();
            
            // Auto-detect viseme blend shapes
            var mapping = visemeMapper.AutoDetectVisemes(mesh);
            
            if (mapping.Count > 0)
            {
                descriptor.lipSync = VRC.SDKBase.VRC_AvatarDescriptor.LipSyncStyle.VisemeBlendShape;
                descriptor.VisemeSkinnedMesh = renderer;
                descriptor.VisemeBlendShapes = mapping.ToArray();
                
                Debug.Log($"Mapped {mapping.Count} visemes automatically");
            }
            else
            {
                Debug.LogWarning("No viseme blend shapes detected for auto-mapping");
            }
        }
        
        async Task CreateFXAnimator(GameObject targetAvatar)
        {
            var renderer = targetAvatar.GetComponentInChildren<SkinnedMeshRenderer>();
            
            if (renderer == null) return;
            
            var mesh = renderer.sharedMesh;
            var builder = new FXAnimatorBuilder();
            
            // Create animator controller
            var controller = builder.CreateBlendShapeController(targetAvatar.name, mesh);
            
            // Save controller
            var controllerPath = $"Assets/Generated/Animators/{targetAvatar.name}_FX.controller";
            var directory = System.IO.Path.GetDirectoryName(controllerPath);
            
            if (!System.IO.Directory.Exists(directory))
                System.IO.Directory.CreateDirectory(directory);
            
            AssetDatabase.CreateAsset(controller, controllerPath);
            
            // Assign to avatar descriptor
            var descriptor = targetAvatar.GetComponent<VRCAvatarDescriptor>();
            
            if (descriptor != null)
            {
                // Initialize base animation layers if needed
                if (descriptor.baseAnimationLayers == null)
                {
                    descriptor.baseAnimationLayers = new VRCAvatarDescriptor.CustomAnimLayer[5];
                }
                
                // Set FX layer
                descriptor.baseAnimationLayers[4] = new VRCAvatarDescriptor.CustomAnimLayer
                {
                    type = VRCAvatarDescriptor.AnimLayerType.FX,
                    animatorController = controller,
                    isDefault = false
                };
                
                EditorUtility.SetDirty(descriptor);
            }
            
            Debug.Log($"Created FX Animator Controller with {mesh.blendShapeCount} blend shape controls");
        }
        
        void GenerateExpressionMenu(GameObject targetAvatar)
        {
            var descriptor = targetAvatar.GetComponent<VRCAvatarDescriptor>();
            var renderer = targetAvatar.GetComponentInChildren<SkinnedMeshRenderer>();
            
            if (descriptor == null || renderer == null) return;
            
            var menuBuilder = new ExpressionMenuBuilder();
            var menu = menuBuilder.CreateBlendShapeMenu(renderer.sharedMesh, targetAvatar.name);
            
            if (menu != null)
            {
                // Save menu asset
                var menuPath = $"Assets/Generated/Menus/{targetAvatar.name}_Expressions.asset";
                var directory = System.IO.Path.GetDirectoryName(menuPath);
                
                if (!System.IO.Directory.Exists(directory))
                    System.IO.Directory.CreateDirectory(directory);
                
                AssetDatabase.CreateAsset(menu, menuPath);
                
                // Assign to descriptor
                descriptor.expressionsMenu = menu;
                
                EditorUtility.SetDirty(descriptor);
                
                Debug.Log("Generated expression menu with blend shape controls");
            }
        }
        
        void OptimizeAvatarPerformance(GameObject targetAvatar)
        {
            var optimizer = new PerformanceOptimizer();
            optimizer.OptimizeAvatar(targetAvatar);
        }
    }
    
    [System.Serializable]
    public class IntegrationOptions
    {
        public bool AutoSetupVisemes = true;
        public bool AutoSetupExpressions = true;
        public bool CreateFXAnimator = true;
        public bool GenerateExpressionMenu = true;
        public bool OptimizePerformance = true;
    }
}
```

## Viseme Auto-Detection System

```csharp
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace VRChatBlendShapeTools.Editor
{
    public class VisemeMapper
    {
        private static readonly Dictionary<int, string[]> VisemeKeywords = new Dictionary<int, string[]>
        {
            { 0, new[] { "sil", "silence", "neutral" } },
            { 1, new[] { "pp", "p" } },
            { 2, new[] { "ff", "f" } },
            { 3, new[] { "th", "θ" } },
            { 4, new[] { "dd", "d", "t" } },
            { 5, new[] { "kk", "k", "g" } },
            { 6, new[] { "ch", "tʃ", "dʒ", "j" } },
            { 7, new[] { "ss", "s", "z" } },
            { 8, new[] { "nn", "n", "ŋ" } },
            { 9, new[] { "rr", "r", "ɹ" } },
            { 10, new[] { "aa", "a", "ɑ" } },
            { 11, new[] { "e", "ɛ" } },
            { 12, new[] { "ih", "i", "ɪ" } },
            { 13, new[] { "oh", "o", "ɔ" } },
            { 14, new[] { "ou", "u", "ʊ" } }
        };
        
        public List<string> AutoDetectVisemes(Mesh mesh)
        {
            var mapping = new List<string>(new string[15]);
            var blendShapeNames = new List<string>();
            
            // Get all blend shape names
            for (int i = 0; i < mesh.blendShapeCount; i++)
            {
                blendShapeNames.Add(mesh.GetBlendShapeName(i));
            }
            
            // Try to map each viseme
            for (int visemeIndex = 0; visemeIndex < 15; visemeIndex++)
            {
                var bestMatch = FindBestVisemeMatch(blendShapeNames, visemeIndex);
                if (bestMatch != null)
                {
                    mapping[visemeIndex] = bestMatch;
                }
            }
            
            return mapping;
        }
        
        private string FindBestVisemeMatch(List<string> blendShapeNames, int visemeIndex)
        {
            if (!VisemeKeywords.ContainsKey(visemeIndex))
                return null;
            
            var keywords = VisemeKeywords[visemeIndex];
            
            // First pass: exact matches with VRChat prefix
            foreach (var keyword in keywords)
            {
                var vrcName = $"vrc.v_{keyword}";
                var exactMatch = blendShapeNames.FirstOrDefault(name => 
                    string.Equals(name, vrcName, System.StringComparison.OrdinalIgnoreCase));
                
                if (exactMatch != null)
                    return exactMatch;
            }
            
            // Second pass: exact keyword matches
            foreach (var keyword in keywords)
            {
                var keywordMatch = blendShapeNames.FirstOrDefault(name => 
                    string.Equals(name, keyword, System.StringComparison.OrdinalIgnoreCase));
                
                if (keywordMatch != null)
                    return keywordMatch;
            }
            
            // Third pass: partial matches
            foreach (var keyword in keywords)
            {
                var partialMatch = blendShapeNames.FirstOrDefault(name => 
                    name.ToLower().Contains(keyword.ToLower()));
                
                if (partialMatch != null)
                    return partialMatch;
            }
            
            return null;
        }
    }
}
```

## FX Animator Builder

```csharp
using UnityEngine;
using UnityEditor;
using UnityEditor.Animations;
using System.Collections.Generic;

namespace VRChatBlendShapeTools.Editor
{
    public class FXAnimatorBuilder
    {
        public AnimatorController CreateBlendShapeController(string avatarName, Mesh mesh)
        {
            var controller = AnimatorController.CreateAnimatorControllerAtPath(
                $"Assets/Generated/Animators/{avatarName}_FX.controller"
            );
            
            // Create base layer
            var baseLayer = controller.layers[0];
            baseLayer.name = "BlendShape Controls";
            
            // Create parameters and states for each blend shape
            for (int i = 0; i < mesh.blendShapeCount; i++)
            {
                var blendShapeName = mesh.GetBlendShapeName(i);
                CreateBlendShapeParameter(controller, blendShapeName);
                CreateBlendShapeAnimations(controller, blendShapeName, avatarName);
            }
            
            return controller;
        }
        
        private void CreateBlendShapeParameter(AnimatorController controller, string blendShapeName)
        {
            var parameterName = SanitizeParameterName(blendShapeName);
            
            // Add float parameter
            controller.AddParameter(new AnimatorControllerParameter
            {
                name = parameterName,
                type = AnimatorControllerParameterType.Float,
                defaultFloat = 0f
            });
        }
        
        private void CreateBlendShapeAnimations(AnimatorController controller, string blendShapeName, string avatarName)
        {
            var parameterName = SanitizeParameterName(blendShapeName);
            
            // Create animation clips
            var offClip = CreateBlendShapeClip(blendShapeName, 0f, avatarName);
            var onClip = CreateBlendShapeClip(blendShapeName, 100f, avatarName);
            
            // Save animation assets
            var animationDir = "Assets/Generated/Animations";
            if (!System.IO.Directory.Exists(animationDir))
                System.IO.Directory.CreateDirectory(animationDir);
            
            AssetDatabase.CreateAsset(offClip, $"{animationDir}/{blendShapeName}_Off.anim");
            AssetDatabase.CreateAsset(onClip, $"{animationDir}/{blendShapeName}_On.anim");
            
            // Create blend tree for smooth transitions
            var blendTree = new BlendTree
            {
                name = $"{blendShapeName}_BlendTree",
                blendType = BlendTreeType.Simple1D,
                blendParameter = parameterName,
                useAutomaticThresholds = false
            };
            
            // Add motions to blend tree
            blendTree.children = new ChildMotion[]
            {
                new ChildMotion { motion = offClip, threshold = 0f, timeScale = 1f },
                new ChildMotion { motion = onClip, threshold = 1f, timeScale = 1f }
            };
            
            // Create state in animator
            var stateMachine = controller.layers[0].stateMachine;
            var state = stateMachine.AddState($"{blendShapeName}_State");
            state.motion = blendTree;
            
            // Set as default state if it's the first one
            if (stateMachine.states.Length == 1)
            {
                stateMachine.defaultState = state;
            }
        }
        
        private AnimationClip CreateBlendShapeClip(string blendShapeName, float value, string avatarName)
        {
            var clip = new AnimationClip
            {
                name = $"{blendShapeName}_{(value > 0 ? "On" : "Off")}"
            };
            
            // Create animation curve
            var curve = new AnimationCurve();
            curve.AddKey(0f, value);
            curve.AddKey(1f, value);
            
            // Set the curve on the clip
            var binding = new EditorCurveBinding
            {
                path = "", // Assumes blend shape is on root object
                type = typeof(SkinnedMeshRenderer),
                propertyName = $"blendShape.{blendShapeName}"
            };
            
            AnimationUtility.SetEditorCurve(clip, binding, curve);
            
            return clip;
        }
        
        private string SanitizeParameterName(string name)
        {
            // Remove invalid characters and ensure valid parameter name
            return name.Replace(" ", "_")
                      .Replace(".", "_")
                      .Replace("-", "_")
                      .Replace("(", "")
                      .Replace(")", "");
        }
    }
}
```

## Performance Analyzer and Optimizer

```csharp
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;

namespace VRChatBlendShapeTools.Editor
{
    public class PerformanceOptimizer
    {
        private readonly Dictionary<string, int> performanceThresholds = new Dictionary<string, int>
        {
            {"Excellent_PolyCount", 7500},
            {"Good_PolyCount", 20000},
            {"Medium_PolyCount", 50000},
            {"Poor_PolyCount", 70000},
            
            {"Excellent_MaterialCount", 1},
            {"Good_MaterialCount", 4},
            {"Medium_MaterialCount", 8},
            {"Poor_MaterialCount", 16},
            
            {"Excellent_MeshCount", 1},
            {"Good_MeshCount", 4},
            {"Medium_MeshCount", 8},
            {"Poor_MeshCount", 16}
        };
        
        public void OptimizeAvatar(GameObject avatar)
        {
            Debug.Log($"Starting performance optimization for {avatar.name}");
            
            // Optimize materials
            OptimizeMaterials(avatar);
            
            // Optimize mesh settings
            OptimizeMeshes(avatar);
            
            // Optimize texture sizes
            OptimizeTextures(avatar);
            
            // Validate final performance
            var finalStats = AnalyzePerformance(avatar);
            Debug.Log($"Optimization complete. Final rating: {finalStats.PerformanceRank}");
        }
        
        public AvatarValidationResult AnalyzePerformance(GameObject avatar)
        {
            var result = new AvatarValidationResult();
            var issues = new List<ValidationIssue>();
            
            // Analyze polygon count
            var renderers = avatar.GetComponentsInChildren<Renderer>();
            int totalPolygons = 0;
            int skinnedMeshCount = 0;
            
            foreach (var renderer in renderers)
            {
                if (renderer is SkinnedMeshRenderer skinnedMesh && skinnedMesh.sharedMesh != null)
                {
                    totalPolygons += skinnedMesh.sharedMesh.triangles.Length / 3;
                    skinnedMeshCount++;
                }
                else if (renderer is MeshRenderer meshRenderer)
                {
                    var meshFilter = renderer.GetComponent<MeshFilter>();
                    if (meshFilter?.sharedMesh != null)
                    {
                        totalPolygons += meshFilter.sharedMesh.triangles.Length / 3;
                    }
                }
            }
            
            result.PolygonCount = totalPolygons;
            result.SkinnedMeshCount = skinnedMeshCount;
            
            // Analyze materials
            var materials = renderers.SelectMany(r => r.sharedMaterials)
                                   .Where(m => m != null)
                                   .Distinct()
                                   .Count();
            
            result.MaterialCount = materials;
            
            // Analyze bone count
            var animator = avatar.GetComponent<Animator>();
            if (animator != null && animator.avatar != null)
            {
                result.BoneCount = animator.avatar.humanDescription.human.Length;
            }
            
            // Determine performance rank
            result.PerformanceRank = DeterminePerformanceRank(result);
            
            // Generate optimization recommendations
            result.Issues = GenerateOptimizationRecommendations(result);
            
            return result;
        }
        
        private string DeterminePerformanceRank(AvatarValidationResult stats)
        {
            var polyScore = GetScoreForMetric(stats.PolygonCount, "PolyCount");
            var materialScore = GetScoreForMetric(stats.MaterialCount, "MaterialCount");
            var meshScore = GetScoreForMetric(stats.SkinnedMeshCount, "MeshCount");
            
            var overallScore = (polyScore + materialScore + meshScore) / 3f;
            
            if (overallScore >= 3.5f) return "Excellent";
            if (overallScore >= 2.5f) return "Good";
            if (overallScore >= 1.5f) return "Medium";
            return "Poor";
        }
        
        private float GetScoreForMetric(int value, string metricType)
        {
            if (value <= performanceThresholds[$"Excellent_{metricType}"]) return 4f;
            if (value <= performanceThresholds[$"Good_{metricType}"]) return 3f;
            if (value <= performanceThresholds[$"Medium_{metricType}"]) return 2f;
            if (value <= performanceThresholds[$"Poor_{metricType}"]) return 1f;
            return 0f;
        }
        
        private List<ValidationIssue> GenerateOptimizationRecommendations(AvatarValidationResult stats)
        {
            var issues = new List<ValidationIssue>();
            
            if (stats.PolygonCount > performanceThresholds["Good_PolyCount"])
            {
                issues.Add(new ValidationIssue
                {
                    Message = $"High polygon count ({stats.PolygonCount:N0}). Consider mesh decimation.",
                    Severity = ValidationSeverity.Warning
                });
            }
            
            if (stats.MaterialCount > performanceThresholds["Good_MaterialCount"])
            {
                issues.Add(new ValidationIssue
                {
                    Message = $"High material count ({stats.MaterialCount}). Consider texture atlasing.",
                    Severity = ValidationSeverity.Warning
                });
            }
            
            return issues;
        }
        
        private void OptimizeMaterials(GameObject avatar)
        {
            var renderers = avatar.GetComponentsInChildren<Renderer>();
            
            foreach (var renderer in renderers)
            {
                var materials = renderer.sharedMaterials;
                
                for (int i = 0; i < materials.Length; i++)
                {
                    if (materials[i] != null)
                    {
                        OptimizeMaterialSettings(materials[i]);
                    }
                }
            }
        }
        
        private void OptimizeMaterialSettings(Material material)
        {
            // Enable GPU instancing if supported
            if (material.enableInstancing == false)
            {
                material.enableInstancing = true;
                EditorUtility.SetDirty(material);
            }
        }
        
        private void OptimizeMeshes(GameObject avatar)
        {
            var skinnedRenderers = avatar.GetComponentsInChildren<SkinnedMeshRenderer>();
            
            foreach (var renderer in skinnedRenderers)
            {
                if (renderer.sharedMesh != null)
                {
                    OptimizeMeshSettings(renderer);
                }
            }
        }
        
        private void OptimizeMeshSettings(SkinnedMeshRenderer renderer)
        {
            // Optimize bounds calculation
            renderer.updateWhenOffscreen = false;
            
            // Set appropriate quality settings
            renderer.quality = SkinQuality.Auto;
            
            EditorUtility.SetDirty(renderer);
        }
        
        private void OptimizeTextures(GameObject avatar)
        {
            var renderers = avatar.GetComponentsInChildren<Renderer>();
            var texturesProcessed = new HashSet<Texture>();
            
            foreach (var renderer in renderers)
            {
                foreach (var material in renderer.sharedMaterials)
                {
                    if (material == null) continue;
                    
                    OptimizeMaterialTextures(material, texturesProcessed);
                }
            }
        }
        
        private void OptimizeMaterialTextures(Material material, HashSet<Texture> processed)
        {
            var shader = material.shader;
            
            for (int i = 0; i < ShaderUtil.GetPropertyCount(shader); i++)
            {
                if (ShaderUtil.GetPropertyType(shader, i) == ShaderUtil.ShaderPropertyType.TexEnv)
                {
                    var propertyName = ShaderUtil.GetPropertyName(shader, i);
                    var texture = material.GetTexture(propertyName);
                    
                    if (texture != null && !processed.Contains(texture))
                    {
                        processed.Add(texture);
                        OptimizeTextureSettings(texture);
                    }
                }
            }
        }
        
        private void OptimizeTextureSettings(Texture texture)
        {
            var path = AssetDatabase.GetAssetPath(texture);
            if (string.IsNullOrEmpty(path)) return;
            
            var textureImporter = AssetImporter.GetAtPath(path) as TextureImporter;
            if (textureImporter == null) return;
            
            bool needsReimport = false;
            
            // Optimize for VRChat
            if (textureImporter.maxTextureSize > 2048)
            {
                textureImporter.maxTextureSize = 2048;
                needsReimport = true;
            }
            
            // Enable compression if not already
            if (textureImporter.textureCompression == TextureImporterCompression.Uncompressed)
            {
                textureImporter.textureCompression = TextureImporterCompression.Compressed;
                needsReimport = true;
            }
            
            if (needsReimport)
            {
                AssetDatabase.ImportAsset(path);
            }
        }
    }
    
    public class AvatarValidationResult
    {
        public string PerformanceRank { get; set; }
        public int PolygonCount { get; set; }
        public int MaterialCount { get; set; }
        public int SkinnedMeshCount { get; set; }
        public int BoneCount { get; set; }
        public List<ValidationIssue> Issues { get; set; } = new List<ValidationIssue>();
    }
    
    public class ValidationIssue
    {
        public string Message { get; set; }
        public ValidationSeverity Severity { get; set; }
    }
    
    public enum ValidationSeverity
    {
        Info,
        Warning,
        Error
    }
}
```

## Implementation Timeline

### Week 1: Foundation and Core Tools
- Basic editor window structure
- Avatar validation system  
- FBX import and blend shape integration

### Week 2: Automation Features
- Viseme auto-detection and mapping
- FX animator generation
- Avatar descriptor setup automation

### Week 3: Performance and Quality
- Performance analyzer implementation
- Optimization recommendations
- Expression menu generation

### Week 4: Integration and Polish
- End-to-end workflow testing
- UI/UX improvements
- Documentation and examples

## Performance Targets

- **Integration Speed**: Complete avatar setup in <2 minutes
- **Accuracy**: >95% successful viseme auto-detection
- **Performance**: Maintain "Good" VRChat performance rating
- **Compatibility**: Support Unity 2019.4 LTS through 2022.3 LTS
- **Reliability**: <2% failure rate in normal operation

This comprehensive Unity Editor tool implementation provides professional-grade automation for VRChat avatar setup, significantly reducing the manual work required while ensuring high-quality, optimized results.