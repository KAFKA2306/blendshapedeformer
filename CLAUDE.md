# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Unity blendshape deformer project for VRChat avatar BlendShape auto-generation powered by MLDeformer. The project aims to automatically generate VRChat-compatible Viseme and expression blend shapes using machine learning, reducing manual work from 30 hours to ~3 hours (90% reduction).

## Architecture

The system consists of three main components:

1. **Core Inference Server** (Python 3.11)
   - ONNX Runtime with TensorRT/OpenVINO optimization
   - Handles ML model loading and vertex offset prediction
   - Runs locally on `127.0.0.1:50051` (security-restricted)

2. **Blender Add-on** 
   - UI panels for blend shape generation
   - Communicates with inference server via gRPC
   - Handles mesh validation and quality assurance
   - Supports Blender 3.0+

3. **Unity Editor Tool**
   - Integrates generated blend shapes into VRChat avatars
   - Auto-configures Avatar Descriptor and Viseme mapping
   - Creates FX Animator controllers
   - Supports Unity 2019.4LTS/2022.3LTS with VRChat SDK3

## Key Components

### Backend (`/backend/`)
- `model_manager.py` - ONNX model loading and caching
- `predictor.py` - Batch inference and quantization
- `postprocess.py` - Vertex clamping and Laplacian smoothing
- `server.py` - gRPC API server

### Blender Add-on (`/blender_addon/`)
- `panel.py` - UI panels and controls
- `operator_generate.py` - BlendShape generation operators
- `mesh_validator.py` - Mesh validation (vertex count, bones, UV)
- `qa_autofix.py` - Quality assurance and auto-fix

### Unity Tool (`/unity_tool/Editor/`)
- `FbxBlendShapeImporter.cs` - FBX import with blend shapes
- `VisemeAutoMapper.cs` - Automatic viseme mapping (15 VRChat standard)
- `AvatarDescriptorUtil.cs` - Avatar Descriptor configuration
- `FxAnimatorBuilder.cs` - FX layer animator generation
- `PerformanceAudit.cs` - VRChat performance optimization

## Development Commands

No specific build/test commands are documented yet as this appears to be in the specification phase. Key development areas:

- Python components use PEP8 + mypy strict, Black formatting, pytest
- C# components follow Unity Coding Standard
- 1 module = 1 PR development approach

## Performance Targets

- Single Viseme inference: ≤3 seconds
- Average vertex error: ≤5mm  
- GPU memory usage: ≤2GB
- BlendShape corruption rate: ≤2%
- Support for up to 70,000 polygons (VRChat Good rating)

## Supported Platforms

- Windows 10/11, macOS 12+, Ubuntu 20.04+
- NVIDIA GPU preferred (CUDA), CPU fallback available
- VRChat SDK3 compatible

## Security Considerations

- Inference server binds only to localhost
- ONNX models encrypted with AES256
- Auto-updates use signed ZIP with SHA-256 verification
- Sandboxed inference environment

## Data Flow

1. User clicks "Generate BlendShapes" in Blender
2. gRPC request sent to Core Server
3. ML model predicts vertex offsets
4. Blender creates Shape Keys and runs quality checks
5. FBX exported automatically
6. Unity tool imports and configures VRChat components
7. Ready for VRChat SDK3 upload

## VRChat Integration

Generates standard VRChat components:
- 15 Viseme blend shapes (vrc.v_sil, vrc.v_pp, etc.)
- 5 emotion expressions (Angry, Fun, Joy, Sorrow, Surprised)
- Automatic Avatar Descriptor configuration
- ViewPosition and EyeLook setup
- FX Animator with BlendShape parameters