# VRChat BlendShape Auto-Generator Installation & Setup Guide

> **Complete Installation Guide for Unity BlendShape Deformer VRChat Project**
>
> This guide covers installation and setup for the Core Inference Server (Python 3.11), Blender Add-on, and Unity Editor Tool across Windows, macOS, and Linux platforms.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Security Considerations](#security-considerations)
3. [Platform-Specific Installation](#platform-specific-installation)
4. [Component Setup](#component-setup)
5. [Verification & Testing](#verification--testing)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)

## System Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Operating System** | Windows 10, macOS 12+, Ubuntu 20.04+ | Windows 11, macOS 13+, Ubuntu 22.04+ |
| **CPU** | Intel i5-8400 / AMD Ryzen 5 2600 | Intel i7-10700K / AMD Ryzen 7 3700X |
| **RAM** | 8GB | 16GB+ |
| **GPU** | GTX 1060 6GB / RX 580 8GB | RTX 3060 8GB+ (CUDA preferred) |
| **Storage** | 10GB free space | 20GB+ SSD |
| **Python** | Python 3.11+ | Python 3.11.5+ |
| **Blender** | Blender 3.0+ | Blender 4.0+ |
| **Unity** | Unity 2019.4LTS | Unity 2022.3LTS |

### VRChat Integration Requirements

- **VRChat SDK3** (Avatars 3.0)
- **Avatar polygon count**: ≤70,000 (VRChat "Good" rating)
- **Bone count**: ≤200 bones
- **UV mapping**: Required for all meshes

### Performance Targets

- **Single Viseme inference**: ≤3 seconds
- **Average vertex error**: ≤5mm
- **GPU memory usage**: ≤2GB
- **BlendShape corruption rate**: ≤2%
- **System availability**: 99% uptime

## Security Considerations

> ⚠️ **Important Security Information**

This system implements several security measures:

- **Local-only operation**: Inference server binds only to `127.0.0.1:50051` (localhost)
- **Encrypted models**: ONNX models are AES256-encrypted and decrypted during runtime only
- **Signed updates**: Auto-updates use SHA-256 verification with signed ZIP packages
- **Sandboxed inference**: Inference operations run in isolated environment

### Network Security

- **No external connections**: System operates entirely offline after installation
- **No data transmission**: Avatar data never leaves your local machine
- **Firewall compatibility**: No inbound connections required

## Platform-Specific Installation

### Windows 10/11 Installation

#### Prerequisites Check

```powershell
# Check Windows version
winver

# Check Python installation
python --version
# Should show Python 3.11.x

# Check CUDA availability (if NVIDIA GPU)
nvcc --version
nvidia-smi
```

#### Automated Installation (Recommended)

1. **Download the Windows installer**:
   - Download `VRChatBlendShapeGenerator_Windows_v1.0.exe`
   - Verify SHA-256 checksum against provided hash

2. **Run installer as Administrator**:
   ```cmd
   # Right-click installer → "Run as administrator"
   VRChatBlendShapeGenerator_Windows_v1.0.exe
   ```

3. **Installation steps**:
   - Accept license agreement
   - Choose installation directory (default: `C:\Program Files\VRChatBlendShapeGenerator`)
   - Select components:
     - ✅ Core Inference Server
     - ✅ Blender Add-on
     - ✅ Unity Editor Tools
   - Configure GPU acceleration (CUDA/DirectML)
   - Wait for installation (5-10 minutes)

#### Manual Installation (Advanced)

1. **Install Python dependencies**:
   ```cmd
   pip install --upgrade pip
   pip install onnxruntime-gpu==1.16.0  # For NVIDIA GPU
   # OR
   pip install onnxruntime==1.16.0      # For CPU-only
   
   pip install numpy==1.24.3
   pip install scipy==1.10.1
   pip install grpcio==1.59.0
   pip install grpcio-tools==1.59.0
   ```

2. **Install system dependencies**:
   ```cmd
   # Visual C++ Redistributable (if not installed)
   # Download from Microsoft official site
   
   # CUDA Toolkit 11.8+ (for GPU acceleration)
   # Download from NVIDIA official site
   ```

### macOS 12+ Installation

#### Prerequisites Check

```bash
# Check macOS version
sw_vers

# Check Python installation
python3 --version

# Check available compute resources
system_profiler SPHardwareDataType
```

#### Automated Installation

1. **Download the macOS installer**:
   - Download `VRChatBlendShapeGenerator_macOS_v1.0.dmg`
   - Verify signature: `codesign -dv --verbose=4 VRChatBlendShapeGenerator_macOS_v1.0.dmg`

2. **Install**:
   ```bash
   # Mount DMG
   hdiutil attach VRChatBlendShapeGenerator_macOS_v1.0.dmg
   
   # Run installer
   sudo installer -pkg "/Volumes/VRChatBlendShapeGenerator/Install.pkg" -target /
   ```

3. **Grant permissions**:
   - **System Settings** → **Privacy & Security** → **Developer Tools**
   - Add Blender and Unity to allowed applications

#### Manual Installation

1. **Install Python dependencies**:
   ```bash
   # Install via Homebrew (recommended)
   brew install python@3.11
   
   # Create virtual environment
   python3 -m venv ~/vrchat-blendshape-env
   source ~/vrchat-blendshape-env/bin/activate
   
   # Install packages
   pip install onnxruntime==1.16.0
   pip install numpy==1.24.3
   pip install scipy==1.10.1
   pip install grpcio==1.59.0
   ```

2. **Configure Metal Performance Shaders** (Apple Silicon):
   ```bash
   # Enable Metal acceleration
   export ONNXRUNTIME_EXECUTION_PROVIDERS="CoreMLExecutionProvider,CPUExecutionProvider"
   ```

### Linux (Ubuntu 20.04+) Installation

#### Prerequisites Check

```bash
# Check Ubuntu version
lsb_release -a

# Check Python installation
python3 --version

# Check GPU availability
lspci | grep -i nvidia  # For NVIDIA
lspci | grep -i amd     # For AMD
```

#### Automated Installation

1. **Download and install**:
   ```bash
   # Download package
   wget https://releases.vrchat-blendshape.com/linux/vrchat-blendshape-generator_1.0_amd64.deb
   
   # Verify signature
   gpg --verify vrchat-blendshape-generator_1.0_amd64.deb.sig
   
   # Install
   sudo dpkg -i vrchat-blendshape-generator_1.0_amd64.deb
   sudo apt-get install -f  # Fix dependencies if needed
   ```

#### Manual Installation

1. **System dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3-pip
   sudo apt install build-essential cmake
   
   # For NVIDIA GPU support
   sudo apt install nvidia-driver-525 nvidia-cuda-toolkit
   
   # For AMD GPU support (ROCm)
   sudo apt install rocm-dev hip-runtime-amd
   ```

2. **Python environment**:
   ```bash
   python3.11 -m venv ~/vrchat-blendshape-env
   source ~/vrchat-blendshape-env/bin/activate
   
   pip install --upgrade pip
   pip install onnxruntime-gpu==1.16.0  # NVIDIA
   # OR
   pip install onnxruntime-rocm==1.16.0  # AMD
   # OR 
   pip install onnxruntime==1.16.0       # CPU-only
   
   pip install numpy==1.24.3 scipy==1.10.1 grpcio==1.59.0
   ```

## Component Setup

### 1. Core Inference Server Setup

The inference server handles ML model loading and vertex offset prediction.

#### Server Configuration

1. **Create configuration file**:

**Windows**: `%APPDATA%\VRChatBlendShapeGenerator\config.json`
**macOS**: `~/Library/Application Support/VRChatBlendShapeGenerator/config.json`
**Linux**: `~/.config/VRChatBlendShapeGenerator/config.json`

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 50051,
    "max_workers": 4,
    "timeout_seconds": 30
  },
  "inference": {
    "device": "auto",
    "batch_size": 8,
    "precision": "fp16",
    "cache_size_mb": 512
  },
  "security": {
    "model_encryption": true,
    "localhost_only": true,
    "auto_update": true
  },
  "performance": {
    "gpu_memory_limit_gb": 2.0,
    "cpu_threads": -1,
    "optimization_level": "all"
  }
}
```

2. **Start the server**:

```bash
# Automatic startup (recommended)
vrchat-blendshape-server --start --daemon

# Manual startup for debugging
vrchat-blendshape-server --verbose --no-daemon
```

3. **Verify server status**:

```bash
# Check if server is running
curl http://127.0.0.1:50051/health

# Expected response:
# {"status": "healthy", "version": "1.0.0", "gpu_available": true}
```

#### GPU Acceleration Setup

**NVIDIA CUDA Setup**:
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Test CUDA with ONNX Runtime
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should include 'CUDAExecutionProvider'
```

**AMD ROCm Setup (Linux)**:
```bash
# Verify ROCm installation
rocm-smi

# Test ROCm with ONNX Runtime
python -c "import onnxruntime as ort; print('ROCMExecutionProvider' in ort.get_available_providers())"
```

**Apple Metal Setup (macOS)**:
```bash
# Test Metal acceleration
python -c "import onnxruntime as ort; print('CoreMLExecutionProvider' in ort.get_available_providers())"
```

### 2. Blender Add-on Installation

#### Install Add-on

1. **Download add-on**:
   - File: `vrchat_blendshape_generator.zip` (do not extract)

2. **Install in Blender**:
   ```
   Blender → Edit → Preferences → Add-ons → Install...
   → Select vrchat_blendshape_generator.zip
   → Enable "VRChat: BlendShape Generator"
   ```

3. **Configure add-on settings**:
   ```
   Blender Preferences → Add-ons → VRChat: BlendShape Generator → Preferences
   ```

   Set the following paths:
   - **Server URL**: `http://127.0.0.1:50051`
   - **Model Path**: Auto-detected (or manually set)
   - **Output Directory**: Choose your preferred FBX export location
   - **Quality Settings**: "High" (recommended)

#### Verify Add-on Installation

1. **Check panel availability**:
   - Open Blender with a mesh object selected
   - Look for **VRChat** tab in the 3D Viewport sidebar (press `N`)
   - Expand **BlendShape Generator** panel

2. **Test server connection**:
   - Click **"Test Connection"** in the add-on panel
   - Should show: ✅ **"Server connection successful"**

### 3. Unity Editor Tool Setup

#### Install Unity Package

1. **Unity version compatibility**:
   - **Supported**: Unity 2019.4LTS, 2022.3LTS, 2023.3LTS
   - **VRChat SDK**: Ensure latest VRChat SDK3 Avatars is installed

2. **Install via Package Manager**:
   ```
   Unity → Window → Package Manager
   → "+" → "Add package from tarball..."
   → Select "com.vrchat.blendshape-generator-1.0.0.tgz"
   ```

   Or manual installation:
   ```
   Unity → Assets → Import Package → Custom Package...
   → Select "VRChatBlendShapeGenerator_Unity.unitypackage"
   ```

3. **Verify installation**:
   - Check menu: **VRChat → BlendShape Generator**
   - Tool window should open with integration options

#### Configure Unity Tool

1. **Set preferences**:
   ```
   Unity → VRChat → BlendShape Generator → Settings
   ```

   Configure:
   - **Default Import Path**: Your FBX storage location
   - **Avatar Template**: VRChat Avatar 3.0
   - **Performance Target**: "Good" (≤70k polygons)
   - **Auto-create Animator**: ✅ Enabled
   - **Auto-setup Visemes**: ✅ Enabled

2. **Validate VRChat SDK**:
   - Ensure **VRChat SDK3 - Avatars** is installed and updated
   - Check **VRChat SDK → Show Control Panel** is accessible

## Verification & Testing

### System Health Check

Run the comprehensive system test to verify all components:

```bash
# Windows
vrchat-blendshape-test.exe --full-system-check

# macOS/Linux
./vrchat-blendshape-test --full-system-check
```

Expected output:
```
✅ Core Inference Server: Running (127.0.0.1:50051)
✅ GPU Acceleration: CUDA 11.8 detected
✅ Model Files: 3/3 models loaded successfully
✅ Blender Add-on: Installed and enabled
✅ Unity Integration: Package installed
✅ VRChat SDK: Version 3.4.2 compatible

🎯 Performance Benchmark:
   Single inference: 2.1s (Target: ≤3.0s) ✅
   Memory usage: 1.4GB (Target: ≤2.0GB) ✅
   Vertex accuracy: 3.2mm avg (Target: ≤5.0mm) ✅

System Status: ALL TESTS PASSED ✅
```

### End-to-End Workflow Test

Test the complete pipeline with a sample avatar:

1. **Download test avatar**:
   ```bash
   # Download sample VRChat avatar for testing
   wget https://samples.vrchat-blendshape.com/test_avatar.fbx
   ```

2. **Blender workflow test**:
   - Import `test_avatar.fbx` into Blender
   - Select the avatar mesh
   - Navigate to **VRChat** tab → **BlendShape Generator**
   - Click **"Generate All BlendShapes"**
   - Wait ~3 minutes for completion
   - Verify 15 Viseme + 5 Expression shape keys created

3. **Unity integration test**:
   - Export FBX from Blender with generated BlendShapes
   - Import FBX into Unity project
   - Use **VRChat → Auto BlendShape Integration**
   - Select target avatar and imported FBX
   - Click **"Start Integration"**
   - Verify Avatar Descriptor setup and Animator creation

4. **VRChat upload test**:
   - Open **VRChat SDK → Show Control Panel**
   - Build & Test the avatar
   - Should pass all validations with "Good" performance rating

### Performance Validation

Verify system meets performance targets:

```python
# Run performance test script (included with installation)
python performance_test.py --avatar test_avatar.fbx --iterations 5

# Expected results:
# Average inference time: 2.1s ± 0.3s
# Peak GPU memory: 1.4GB
# Vertex accuracy: 3.2mm ± 1.1mm
# BlendShape quality score: 94.2%
```

## Troubleshooting

### Common Installation Issues

#### Issue: "ONNX Runtime not found"
**Symptoms**: Import error when starting inference server
**Solutions**:
1. Verify Python environment activation
2. Reinstall ONNX Runtime: `pip install --force-reinstall onnxruntime-gpu`
3. Check CUDA version compatibility
4. Try CPU-only version: `pip install onnxruntime`

#### Issue: "Server connection failed"
**Symptoms**: Blender add-on cannot connect to inference server
**Solutions**:
1. Verify server is running: `curl http://127.0.0.1:50051/health`
2. Check firewall settings (allow localhost connections)
3. Restart server with verbose logging: `vrchat-blendshape-server --verbose`
4. Verify port 50051 is not in use: `netstat -an | grep 50051`

#### Issue: "GPU out of memory"
**Symptoms**: CUDA allocation failed during inference
**Solutions**:
1. Reduce batch size in config.json: `"batch_size": 4` or `"batch_size": 2`
2. Close other GPU-intensive applications
3. Switch to CPU inference: `"device": "cpu"` in config
4. Reduce avatar polygon count below 50,000

#### Issue: "Blender add-on not visible"
**Symptoms**: VRChat tab not appearing in Blender
**Solutions**:
1. Verify add-on is enabled: **Preferences → Add-ons** → Search "VRChat"
2. Check Blender version compatibility (3.0+ required)
3. Restart Blender after installation
4. Check error console: **Window → Toggle System Console**

#### Issue: "Unity package import failed"
**Symptoms**: Error during Unity package installation
**Solutions**:
1. Verify Unity version compatibility (2019.4LTS+)
2. Ensure VRChat SDK3 is installed first
3. Clear Unity cache: **Assets → Refresh**
4. Reimport package: **Assets → Reimport All**

### Performance Issues

#### Issue: Slow inference (>5 seconds)
**Solutions**:
1. Enable GPU acceleration (check CUDA/ROCm installation)
2. Update GPU drivers to latest version
3. Increase GPU memory allocation in config
4. Reduce model precision: `"precision": "fp16"` or `"precision": "int8"`

#### Issue: Poor BlendShape quality
**Solutions**:
1. Enable **Auto Quality Check** in Blender add-on
2. Ensure avatar is in T-pose before generation
3. Verify UV mapping integrity
4. Check mesh for duplicate vertices or non-manifold geometry
5. Increase inference precision: `"precision": "fp32"`

#### Issue: High memory usage
**Solutions**:
1. Reduce batch size: `"batch_size": 2`
2. Enable memory optimization: `"optimization_level": "memory"`
3. Clear model cache regularly
4. Process avatars individually rather than in batches

### Platform-Specific Issues

#### Windows-specific
- **Missing Visual C++ Redistributable**: Install from Microsoft official site
- **CUDA driver conflicts**: Uninstall old CUDA versions before installing new
- **Antivirus blocking**: Add installation directory to antivirus exclusions

#### macOS-specific
- **Gatekeeper blocking**: Right-click → "Open" for unsigned binaries
- **Metal performance issues**: Ensure macOS 12+ and latest Xcode Command Line Tools
- **Permissions denied**: Grant Full Disk Access to Terminal and Blender

#### Linux-specific
- **Missing shared libraries**: Install dev packages for missing .so files
- **GPU permissions**: Add user to `video` group: `sudo usermod -a -G video $USER`
- **AppImage execution**: Make executable: `chmod +x *.AppImage`

### Advanced Diagnostics

#### Enable debug logging

1. **Server debug mode**:
   ```bash
   vrchat-blendshape-server --log-level DEBUG --log-file debug.log
   ```

2. **Blender add-on debugging**:
   - Enable **Developer Extras** in Blender Preferences
   - Check **Window → Toggle System Console** for Python errors
   - Set add-on **Log Level** to "Debug" in preferences

3. **Unity debugging**:
   - Open **Console** window (Window → General → Console)
   - Enable **Collapse**, **Clear on Play**, and **Error Pause**
   - Check for VRChatBlendShape-related warnings/errors

#### System information collection

Run diagnostics collector for support:

```bash
# Generate comprehensive system report
vrchat-blendshape-diagnostics --output system_report.zip

# Includes:
# - System specifications
# - Installed software versions  
# - Configuration files
# - Recent log files
# - Performance benchmarks
```

## Performance Optimization

### GPU Acceleration Optimization

#### NVIDIA CUDA Optimization
```json
{
  "inference": {
    "device": "cuda:0",
    "precision": "fp16",
    "optimization_level": "all"
  },
  "cuda_settings": {
    "memory_growth": true,
    "memory_limit_gb": 6.0,
    "allow_soft_placement": true
  }
}
```

#### AMD ROCm Optimization (Linux)
```json
{
  "inference": {
    "device": "rocm:0", 
    "precision": "fp16"
  },
  "rocm_settings": {
    "memory_limit_gb": 8.0,
    "optimization_passes": "all"
  }
}
```

#### Apple Metal Optimization (macOS)
```json
{
  "inference": {
    "device": "coreml",
    "precision": "fp16"
  },
  "coreml_settings": {
    "compute_units": "all",
    "model_format": "mlpackage"
  }
}
```

### Memory Usage Optimization

For systems with limited RAM/VRAM:

```json
{
  "performance": {
    "low_memory_mode": true,
    "batch_size": 1,
    "model_caching": false,
    "gradient_checkpointing": true
  }
}
```

### Quality vs Speed Trade-offs

**Maximum Quality** (slower):
```json
{
  "inference": {
    "precision": "fp32",
    "batch_size": 1,
    "optimization_level": "none"
  },
  "quality": {
    "vertex_refinement_iterations": 5,
    "smoothing_strength": "high",
    "anatomical_constraints": true
  }
}
```

**Maximum Speed** (lower quality):
```json
{
  "inference": {
    "precision": "int8",
    "batch_size": 16,
    "optimization_level": "all"
  },
  "quality": {
    "vertex_refinement_iterations": 1,
    "smoothing_strength": "low",
    "anatomical_constraints": false
  }
}
```

---

## Next Steps

After successful installation:

1. **Complete the [User Manual](USER_MANUAL.md)** for detailed usage instructions
2. **Join the [Discord Community](https://discord.gg/vrchat-blendshape)** for support and updates
3. **Star the [GitHub Repository](https://github.com/vrchat/blendshape-generator)** to stay informed of releases
4. **Read the [Best Practices Guide](BEST_PRACTICES.md)** for optimal results

---

## Support & Community

- **Technical Support**: [GitHub Issues](https://github.com/vrchat/blendshape-generator/issues)
- **Community Discord**: [Join Server](https://discord.gg/vrchat-blendshape)  
- **Documentation**: [Full Documentation](https://docs.vrchat-blendshape.com)
- **Video Tutorials**: [YouTube Channel](https://youtube.com/@vrchat-blendshape)

---

*Installation Guide Version 1.0 | Updated: August 2025*
*System Requirements Subject to Change | Check Latest Documentation*