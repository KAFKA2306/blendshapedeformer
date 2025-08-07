# gRPC Communication System Implementation Guide

## Executive Summary

This document provides comprehensive implementation guidance for the gRPC communication system that enables secure, high-performance communication between Blender add-ons and the ML inference server. The system is designed for localhost-only operation with thread-safe patterns optimized for Blender's bpy API constraints.

## gRPC Architecture Overview

The communication system follows a client-server architecture where:
- **Server**: Python-based ML inference engine running locally
- **Client**: Blender add-on components making inference requests
- **Protocol**: gRPC with Protocol Buffers for efficient serialization
- **Security**: Localhost-only binding with process isolation

### System Architecture Diagram

```
┌─────────────────────┐    gRPC/HTTP2     ┌──────────────────────┐
│   Blender Add-on    │◄─────────────────►│  Inference Server    │
│   (gRPC Client)     │  localhost:50051   │   (gRPC Server)      │
├─────────────────────┤                    ├──────────────────────┤
│ • Request Manager   │                    │ • Model Manager      │
│ • Progress Tracker  │                    │ • ONNX Runtime       │
│ • Error Handler     │                    │ • Resource Pool      │
│ • Thread Manager    │                    │ • Performance Stats  │
└─────────────────────┘                    └──────────────────────┘
```

## Protocol Buffer Definitions

```protobuf
// inference.proto
syntax = "proto3";

package vrchat.blendshape.inference;

// Main inference service
service InferenceService {
    // Health check
    rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
    
    // Single vertex offset prediction
    rpc PredictVertexOffsets(VertexOffsetRequest) returns (VertexOffsetResponse);
    
    // Batch prediction for multiple expressions
    rpc PredictBatch(BatchPredictionRequest) returns (BatchPredictionResponse);
    
    // Get server performance statistics
    rpc GetPerformanceStats(PerformanceStatsRequest) returns (PerformanceStatsResponse);
    
    // Model management
    rpc LoadModel(LoadModelRequest) returns (LoadModelResponse);
    rpc UnloadModel(UnloadModelRequest) returns (UnloadModelResponse);
}

// Health check messages
message HealthCheckRequest {
    string service = 1;
}

message HealthCheckResponse {
    enum ServingStatus {
        UNKNOWN = 0;
        SERVING = 1;
        NOT_SERVING = 2;
        SERVICE_UNKNOWN = 3;
    }
    ServingStatus status = 1;
    string message = 2;
    ServerInfo server_info = 3;
}

message ServerInfo {
    string version = 1;
    string model_version = 2;
    bool gpu_available = 3;
    string device_info = 4;
    int32 max_concurrent_requests = 5;
}

// Vertex offset prediction messages
message VertexOffsetRequest {
    // Vertex positions as flat array [x1,y1,z1,x2,y2,z2,...]
    repeated float vertices = 1 [packed=true];
    int32 vertex_count = 2;
    
    // Condition encoding
    ExpressionCondition condition = 3;
    
    // Request options
    PredictionOptions options = 4;
}

message ExpressionCondition {
    enum ExpressionType {
        VISEME = 0;
        EMOTION = 1;
        CUSTOM = 2;
    }
    
    ExpressionType type = 1;
    string name = 2;        // e.g., "aa", "Joy", "Custom_Expression"
    float intensity = 3;    // 0.0 to 1.0
    
    // Extended features
    repeated float features = 4 [packed=true];  // Additional condition features
    map<string, float> parameters = 5;          // Named parameters
}

message PredictionOptions {
    bool apply_smoothing = 1;
    int32 smoothing_iterations = 2;
    float max_displacement = 3;     // Maximum vertex displacement in meters
    bool clamp_extreme_values = 4;
    int32 timeout_seconds = 5;
}

message VertexOffsetResponse {
    bool success = 1;
    string error_message = 2;
    
    // Vertex offsets as flat array [dx1,dy1,dz1,dx2,dy2,dz2,...]
    repeated float offsets = 3 [packed=true];
    
    // Metadata
    PredictionMetadata metadata = 4;
}

message PredictionMetadata {
    float inference_time_ms = 1;
    float preprocessing_time_ms = 2;
    float postprocessing_time_ms = 3;
    int32 vertices_processed = 4;
    string model_version = 5;
    QualityMetrics quality = 6;
}

message QualityMetrics {
    float avg_displacement = 1;
    float max_displacement = 2;
    int32 extreme_vertex_count = 3;
    float smoothness_score = 4;
}

// Batch prediction messages
message BatchPredictionRequest {
    repeated VertexOffsetRequest requests = 1;
    int32 max_parallel_requests = 2;
}

message BatchPredictionResponse {
    repeated VertexOffsetResponse responses = 1;
    float total_processing_time_ms = 2;
    bool all_successful = 3;
}

// Performance statistics
message PerformanceStatsRequest {
    bool reset_stats = 1;
}

message PerformanceStatsResponse {
    int64 total_requests = 1;
    float avg_inference_time_ms = 2;
    float max_inference_time_ms = 3;
    float min_inference_time_ms = 4;
    int64 successful_requests = 5;
    int64 failed_requests = 6;
    float uptime_seconds = 7;
    ResourceUsage resource_usage = 8;
}

message ResourceUsage {
    float cpu_usage_percent = 1;
    float memory_usage_mb = 2;
    float gpu_memory_usage_mb = 3;
    float gpu_utilization_percent = 4;
}

// Model management messages
message LoadModelRequest {
    string model_path = 1;
    string model_name = 2;
    map<string, string> options = 3;
}

message LoadModelResponse {
    bool success = 1;
    string error_message = 2;
    string model_version = 3;
}

message UnloadModelRequest {
    string model_name = 1;
}

message UnloadModelResponse {
    bool success = 1;
    string error_message = 2;
}
```

## Server Implementation

```python
import grpc
from concurrent import futures
import threading
import time
import psutil
import logging
import numpy as np
from typing import Dict, List, Optional
import asyncio
import signal
import sys
from pathlib import Path

# Generated protobuf classes
import inference_pb2
import inference_pb2_grpc

# ML inference components
from onnx_inference_engine import ONNXInferenceEngine
from model_manager import ModelManager
from performance_monitor import PerformanceMonitor

class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    """
    High-performance gRPC inference service for VRChat BlendShape generation
    """
    
    def __init__(self, model_path: str = None, max_workers: int = 4):
        self.model_manager = ModelManager()
        self.performance_monitor = PerformanceMonitor()
        self.max_workers = max_workers
        self.shutdown_event = threading.Event()
        
        # Load default model if provided
        if model_path:
            self.model_manager.load_model(model_path, "default")
        
        # Server info
        self.server_info = inference_pb2.ServerInfo(
            version="1.0.0",
            model_version="vrchat_blendshape_v1",
            gpu_available=self.model_manager.has_gpu_support(),
            device_info=self.model_manager.get_device_info(),
            max_concurrent_requests=max_workers
        )
        
        logging.info(f"InferenceServicer initialized with {max_workers} workers")
    
    def HealthCheck(self, request, context):
        """Health check endpoint"""
        
        try:
            # Test model availability
            model_available = self.model_manager.has_loaded_models()
            
            if model_available:
                status = inference_pb2.HealthCheckResponse.SERVING
                message = "Service is healthy and ready"
            else:
                status = inference_pb2.HealthCheckResponse.NOT_SERVING
                message = "No models loaded"
            
            return inference_pb2.HealthCheckResponse(
                status=status,
                message=message,
                server_info=self.server_info
            )
            
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            
            return inference_pb2.HealthCheckResponse(
                status=inference_pb2.HealthCheckResponse.NOT_SERVING,
                message=f"Health check failed: {e}"
            )
    
    def PredictVertexOffsets(self, request, context):
        """Single vertex offset prediction"""
        
        start_time = time.time()
        
        try:
            # Validate request
            if not request.vertices or request.vertex_count <= 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Invalid vertex data")
                return inference_pb2.VertexOffsetResponse(success=False, error_message="Invalid vertex data")
            
            # Convert protobuf data to numpy arrays
            vertices = np.array(request.vertices, dtype=np.float32)
            vertices = vertices.reshape(1, request.vertex_count, 3)  # Add batch dimension
            
            # Encode condition
            condition = self._encode_condition(request.condition)
            
            # Get model
            model = self.model_manager.get_default_model()
            if model is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("No model loaded")
                return inference_pb2.VertexOffsetResponse(success=False, error_message="No model loaded")
            
            # Perform inference
            preprocessing_start = time.time()
            # Preprocessing could include normalization, feature extraction, etc.
            preprocessing_time = (time.time() - preprocessing_start) * 1000
            
            inference_start = time.time()
            offsets = model.predict_vertex_offsets(vertices, condition)
            inference_time = (time.time() - inference_start) * 1000
            
            postprocessing_start = time.time()
            
            # Apply post-processing if requested
            if request.options.apply_smoothing:
                offsets = self._apply_smoothing(
                    offsets, 
                    vertices, 
                    request.options.smoothing_iterations
                )
            
            if request.options.clamp_extreme_values:
                offsets = self._clamp_extreme_displacements(
                    offsets, 
                    request.options.max_displacement
                )
            
            postprocessing_time = (time.time() - postprocessing_start) * 1000
            
            # Calculate quality metrics
            quality = self._calculate_quality_metrics(offsets)
            
            # Create metadata
            metadata = inference_pb2.PredictionMetadata(
                inference_time_ms=inference_time,
                preprocessing_time_ms=preprocessing_time,
                postprocessing_time_ms=postprocessing_time,
                vertices_processed=request.vertex_count,
                model_version=self.server_info.model_version,
                quality=quality
            )
            
            # Update performance statistics
            total_time = time.time() - start_time
            self.performance_monitor.record_request(total_time, success=True)
            
            # Return response
            return inference_pb2.VertexOffsetResponse(
                success=True,
                offsets=offsets.flatten().tolist(),
                metadata=metadata
            )
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            total_time = time.time() - start_time
            self.performance_monitor.record_request(total_time, success=False)
            
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            
            return inference_pb2.VertexOffsetResponse(
                success=False,
                error_message=str(e)
            )
    
    def PredictBatch(self, request, context):
        """Batch prediction for multiple expressions"""
        
        start_time = time.time()
        responses = []
        successful_count = 0
        
        # Limit concurrent requests
        max_parallel = min(request.max_parallel_requests or 4, self.max_workers)
        
        try:
            # Process requests in batches
            for i in range(0, len(request.requests), max_parallel):
                batch = request.requests[i:i + max_parallel]
                
                # Process batch in parallel
                with futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
                    batch_futures = [
                        executor.submit(self._process_single_request, req, context)
                        for req in batch
                    ]
                    
                    for future in futures.as_completed(batch_futures):
                        try:
                            response = future.result()
                            responses.append(response)
                            if response.success:
                                successful_count += 1
                        except Exception as e:
                            logging.error(f"Batch request failed: {e}")
                            responses.append(inference_pb2.VertexOffsetResponse(
                                success=False,
                                error_message=str(e)
                            ))
            
            total_time = (time.time() - start_time) * 1000
            all_successful = successful_count == len(request.requests)
            
            return inference_pb2.BatchPredictionResponse(
                responses=responses,
                total_processing_time_ms=total_time,
                all_successful=all_successful
            )
            
        except Exception as e:
            logging.error(f"Batch prediction failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            
            return inference_pb2.BatchPredictionResponse(
                responses=[],
                total_processing_time_ms=0,
                all_successful=False
            )
    
    def GetPerformanceStats(self, request, context):
        """Get server performance statistics"""
        
        try:
            stats = self.performance_monitor.get_stats()
            resource_usage = self._get_resource_usage()
            
            if request.reset_stats:
                self.performance_monitor.reset()
            
            return inference_pb2.PerformanceStatsResponse(
                total_requests=stats['total_requests'],
                avg_inference_time_ms=stats['avg_time'] * 1000,
                max_inference_time_ms=stats['max_time'] * 1000,
                min_inference_time_ms=stats['min_time'] * 1000,
                successful_requests=stats['successful_requests'],
                failed_requests=stats['failed_requests'],
                uptime_seconds=stats['uptime'],
                resource_usage=resource_usage
            )
            
        except Exception as e:
            logging.error(f"Failed to get performance stats: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.PerformanceStatsResponse()
    
    def LoadModel(self, request, context):
        """Load a new model"""
        
        try:
            success = self.model_manager.load_model(
                request.model_path, 
                request.model_name or "default"
            )
            
            if success:
                return inference_pb2.LoadModelResponse(
                    success=True,
                    model_version=self.server_info.model_version
                )
            else:
                return inference_pb2.LoadModelResponse(
                    success=False,
                    error_message="Failed to load model"
                )
                
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            
            return inference_pb2.LoadModelResponse(
                success=False,
                error_message=str(e)
            )
    
    def UnloadModel(self, request, context):
        """Unload a model"""
        
        try:
            success = self.model_manager.unload_model(request.model_name)
            
            return inference_pb2.UnloadModelResponse(success=success)
            
        except Exception as e:
            logging.error(f"Model unloading failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            
            return inference_pb2.UnloadModelResponse(
                success=False,
                error_message=str(e)
            )
    
    def _process_single_request(self, request, context):
        """Process a single request (used for batch processing)"""
        # This would call PredictVertexOffsets internally
        # but without the gRPC context overhead
        return self.PredictVertexOffsets(request, context)
    
    def _encode_condition(self, condition_proto):
        """Convert protobuf condition to model input format"""
        
        # Create condition vector (this would match the model's expected input format)
        condition_vector = np.zeros(64, dtype=np.float32)  # Example: 64-dimensional
        
        # Encode expression type
        if condition_proto.type == inference_pb2.ExpressionCondition.VISEME:
            condition_vector[0] = 1.0
        elif condition_proto.type == inference_pb2.ExpressionCondition.EMOTION:
            condition_vector[1] = 1.0
        
        # Encode intensity
        condition_vector[2] = condition_proto.intensity
        
        # Add expression-specific encoding
        # This would be customized based on the actual model requirements
        
        return condition_vector.reshape(1, -1)  # Add batch dimension
    
    def _apply_smoothing(self, offsets, vertices, iterations):
        """Apply Laplacian smoothing to vertex offsets"""
        # Implement smoothing algorithm
        return offsets
    
    def _clamp_extreme_displacements(self, offsets, max_displacement):
        """Clamp extreme vertex displacements"""
        
        displacement_magnitudes = np.linalg.norm(offsets, axis=2)
        extreme_mask = displacement_magnitudes > max_displacement
        
        if np.any(extreme_mask):
            # Normalize and scale extreme displacements
            for batch_idx in range(offsets.shape[0]):
                for vertex_idx in range(offsets.shape[1]):
                    if extreme_mask[batch_idx, vertex_idx]:
                        offset = offsets[batch_idx, vertex_idx]
                        magnitude = np.linalg.norm(offset)
                        normalized = offset / magnitude
                        offsets[batch_idx, vertex_idx] = normalized * max_displacement
        
        return offsets
    
    def _calculate_quality_metrics(self, offsets):
        """Calculate quality metrics for the prediction"""
        
        displacement_magnitudes = np.linalg.norm(offsets, axis=2)
        
        return inference_pb2.QualityMetrics(
            avg_displacement=float(np.mean(displacement_magnitudes)),
            max_displacement=float(np.max(displacement_magnitudes)),
            extreme_vertex_count=int(np.sum(displacement_magnitudes > 0.08)),  # 8cm threshold
            smoothness_score=0.85  # Placeholder - would calculate actual smoothness
        )
    
    def _get_resource_usage(self):
        """Get current resource usage"""
        
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU usage would require nvidia-ml-py or similar
        gpu_memory = 0.0
        gpu_utilization = 0.0
        
        return inference_pb2.ResourceUsage(
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory.used / 1024 / 1024,
            gpu_memory_usage_mb=gpu_memory,
            gpu_utilization_percent=gpu_utilization
        )

class InferenceServer:
    """
    gRPC server wrapper with lifecycle management
    """
    
    def __init__(self, model_path: str = None, port: int = 50051, max_workers: int = 4):
        self.port = port
        self.max_workers = max_workers
        self.server = None
        self.servicer = InferenceServicer(model_path, max_workers)
        
    def start(self):
        """Start the gRPC server"""
        
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=[
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 300000),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
                ('grpc.max_send_message_length', 100 * 1024 * 1024),     # 100MB
            ]
        )
        
        # Add servicer
        inference_pb2_grpc.add_InferenceServicer_to_server(self.servicer, self.server)
        
        # Bind to localhost only for security
        listen_addr = f'127.0.0.1:{self.port}'
        self.server.add_insecure_port(listen_addr)
        
        # Start server
        self.server.start()
        logging.info(f"Inference server started on {listen_addr}")
        
        return self
    
    def wait_for_termination(self):
        """Wait for server termination"""
        if self.server:
            self.server.wait_for_termination()
    
    def stop(self, grace_period: float = 5.0):
        """Stop the server gracefully"""
        if self.server:
            logging.info("Shutting down inference server...")
            self.server.stop(grace_period)
            logging.info("Server stopped")

def main():
    """Main server entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='VRChat BlendShape Inference Server')
    parser.add_argument('--model', type=str, help='Path to ONNX model file')
    parser.add_argument('--port', type=int, default=50051, help='Server port')
    parser.add_argument('--workers', type=int, default=4, help='Max worker threads')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start server
    server = InferenceServer(args.model, args.port, args.workers)
    
    # Handle shutdown gracefully
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, shutting down...")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        server.start()
        print(f"Server running on port {args.port}. Press Ctrl+C to stop.")
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop()

if __name__ == '__main__':
    main()
```

## Client Implementation for Blender

```python
import grpc
import threading
import time
import logging
import numpy as np
from typing import Optional, Callable, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import bpy

# Generated protobuf classes
import inference_pb2
import inference_pb2_grpc

class InferenceClient:
    """
    Thread-safe gRPC client for Blender integration
    """
    
    def __init__(self, server_address: str = "localhost:50051", timeout: float = 30.0):
        self.server_address = server_address
        self.timeout = timeout
        self.channel = None
        self.stub = None
        self.connected = False
        
        # Thread safety
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Connection management
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
        logging.info(f"InferenceClient initialized for {server_address}")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    def connect(self) -> bool:
        """Establish connection to the inference server"""
        
        with self.lock:
            if self.connected:
                return True
            
            try:
                # Create channel with appropriate options
                options = [
                    ('grpc.keepalive_time_ms', 30000),
                    ('grpc.keepalive_timeout_ms', 5000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),     # 100MB
                ]
                
                self.channel = grpc.insecure_channel(self.server_address, options=options)
                self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
                
                # Test connection
                if self.test_connection():
                    self.connected = True
                    self.connection_attempts = 0
                    logging.info(f"Connected to inference server at {self.server_address}")
                    return True
                else:
                    self.disconnect()
                    return False
                    
            except Exception as e:
                logging.error(f"Connection failed: {e}")
                self.disconnect()
                return False
    
    def disconnect(self):
        """Close connection to the server"""
        
        with self.lock:
            if self.channel:
                try:
                    self.channel.close()
                except:
                    pass
                finally:
                    self.channel = None
                    self.stub = None
                    self.connected = False
    
    def test_connection(self) -> bool:
        """Test if the server is reachable and healthy"""
        
        if not self.stub:
            return False
        
        try:
            request = inference_pb2.HealthCheckRequest(service="InferenceService")
            response = self.stub.HealthCheck(request, timeout=5.0)
            
            return response.status == inference_pb2.HealthCheckResponse.SERVING
            
        except Exception as e:
            logging.warning(f"Health check failed: {e}")
            return False
    
    def predict_vertex_offsets(
        self,
        vertices: np.ndarray,
        expression_type: str,
        expression_name: str,
        intensity: float = 1.0,
        options: Dict[str, Any] = None
    ) -> Optional[np.ndarray]:
        """
        Predict vertex offsets for a given expression
        
        Args:
            vertices: Vertex positions array (N, 3)
            expression_type: 'viseme' or 'emotion'
            expression_name: Name of the expression
            intensity: Expression intensity (0.0-1.0)
            options: Additional processing options
            
        Returns:
            Vertex offsets array (N, 3) or None if failed
        """
        
        if not self.ensure_connected():
            return None
        
        try:
            # Prepare condition
            condition_type = (
                inference_pb2.ExpressionCondition.VISEME 
                if expression_type.lower() == 'viseme' 
                else inference_pb2.ExpressionCondition.EMOTION
            )
            
            condition = inference_pb2.ExpressionCondition(
                type=condition_type,
                name=expression_name,
                intensity=intensity
            )
            
            # Prepare options
            pred_options = inference_pb2.PredictionOptions()
            if options:
                pred_options.apply_smoothing = options.get('apply_smoothing', True)
                pred_options.smoothing_iterations = options.get('smoothing_iterations', 2)
                pred_options.max_displacement = options.get('max_displacement', 0.08)
                pred_options.clamp_extreme_values = options.get('clamp_extreme_values', True)
                pred_options.timeout_seconds = options.get('timeout_seconds', 30)
            
            # Create request
            request = inference_pb2.VertexOffsetRequest(
                vertices=vertices.flatten().tolist(),
                vertex_count=len(vertices),
                condition=condition,
                options=pred_options
            )
            
            # Make request
            response = self.stub.PredictVertexOffsets(request, timeout=self.timeout)
            
            if response.success:
                # Convert back to numpy array
                offsets = np.array(response.offsets, dtype=np.float32)
                offsets = offsets.reshape(len(vertices), 3)
                
                # Log performance info
                if response.metadata:
                    logging.info(
                        f"Prediction completed in {response.metadata.inference_time_ms:.1f}ms "
                        f"(quality score: {response.metadata.quality.smoothness_score:.2f})"
                    )
                
                return offsets
            else:
                logging.error(f"Prediction failed: {response.error_message}")
                return None
                
        except grpc.RpcError as e:
            logging.error(f"gRPC error during prediction: {e}")
            
            # Handle connection issues
            if e.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]:
                self.connected = False
            
            return None
        except Exception as e:
            logging.error(f"Unexpected error during prediction: {e}")
            return None
    
    def predict_batch(
        self,
        requests: List[Dict[str, Any]],
        max_parallel: int = 4,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Optional[np.ndarray]]:
        """
        Batch prediction for multiple expressions
        
        Args:
            requests: List of request dictionaries containing vertices, expression info, etc.
            max_parallel: Maximum parallel requests
            progress_callback: Optional progress callback function
            
        Returns:
            List of vertex offset arrays or None for failed predictions
        """
        
        if not self.ensure_connected():
            return [None] * len(requests)
        
        try:
            # Convert requests to protobuf format
            proto_requests = []
            
            for req in requests:
                vertices = req['vertices']
                
                condition_type = (
                    inference_pb2.ExpressionCondition.VISEME 
                    if req.get('expression_type', '').lower() == 'viseme' 
                    else inference_pb2.ExpressionCondition.EMOTION
                )
                
                condition = inference_pb2.ExpressionCondition(
                    type=condition_type,
                    name=req['expression_name'],
                    intensity=req.get('intensity', 1.0)
                )
                
                options = inference_pb2.PredictionOptions()
                if 'options' in req:
                    opt = req['options']
                    options.apply_smoothing = opt.get('apply_smoothing', True)
                    options.smoothing_iterations = opt.get('smoothing_iterations', 2)
                    options.max_displacement = opt.get('max_displacement', 0.08)
                    options.clamp_extreme_values = opt.get('clamp_extreme_values', True)
                
                proto_request = inference_pb2.VertexOffsetRequest(
                    vertices=vertices.flatten().tolist(),
                    vertex_count=len(vertices),
                    condition=condition,
                    options=options
                )
                
                proto_requests.append(proto_request)
            
            # Create batch request
            batch_request = inference_pb2.BatchPredictionRequest(
                requests=proto_requests,
                max_parallel_requests=max_parallel
            )
            
            # Progress callback wrapper for thread safety
            def safe_progress_callback(progress: float, status: str):
                if progress_callback:
                    # Schedule callback on main thread if we're in Blender
                    try:
                        bpy.app.timers.register(
                            lambda: progress_callback(progress, status) or None,
                            first_interval=0.0
                        )
                    except:
                        # Fallback for non-Blender contexts
                        progress_callback(progress, status)
            
            # Make batch request
            if progress_callback:
                safe_progress_callback(0.0, "Starting batch prediction...")
            
            response = self.stub.PredictBatch(batch_request, timeout=self.timeout * len(requests))
            
            # Process responses
            results = []
            for i, resp in enumerate(response.responses):
                if resp.success:
                    offsets = np.array(resp.offsets, dtype=np.float32)
                    offsets = offsets.reshape(len(requests[i]['vertices']), 3)
                    results.append(offsets)
                else:
                    logging.error(f"Batch prediction {i} failed: {resp.error_message}")
                    results.append(None)
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / len(response.responses)
                    safe_progress_callback(progress, f"Completed {i + 1}/{len(response.responses)}")
            
            if progress_callback:
                safe_progress_callback(1.0, "Batch prediction complete")
            
            logging.info(
                f"Batch prediction completed: {len([r for r in results if r is not None])}/{len(results)} successful"
            )
            
            return results
            
        except grpc.RpcError as e:
            logging.error(f"gRPC error during batch prediction: {e}")
            return [None] * len(requests)
        except Exception as e:
            logging.error(f"Unexpected error during batch prediction: {e}")
            return [None] * len(requests)
    
    def get_performance_stats(self, reset_stats: bool = False) -> Optional[Dict[str, Any]]:
        """Get server performance statistics"""
        
        if not self.ensure_connected():
            return None
        
        try:
            request = inference_pb2.PerformanceStatsRequest(reset_stats=reset_stats)
            response = self.stub.GetPerformanceStats(request, timeout=5.0)
            
            return {
                'total_requests': response.total_requests,
                'avg_inference_time_ms': response.avg_inference_time_ms,
                'max_inference_time_ms': response.max_inference_time_ms,
                'min_inference_time_ms': response.min_inference_time_ms,
                'successful_requests': response.successful_requests,
                'failed_requests': response.failed_requests,
                'uptime_seconds': response.uptime_seconds,
                'resource_usage': {
                    'cpu_usage_percent': response.resource_usage.cpu_usage_percent,
                    'memory_usage_mb': response.resource_usage.memory_usage_mb,
                    'gpu_memory_usage_mb': response.resource_usage.gpu_memory_usage_mb,
                    'gpu_utilization_percent': response.resource_usage.gpu_utilization_percent
                }
            }
            
        except Exception as e:
            logging.error(f"Failed to get performance stats: {e}")
            return None
    
    def ensure_connected(self) -> bool:
        """Ensure connection is active, reconnect if necessary"""
        
        with self.lock:
            if self.connected and self.test_connection():
                return True
            
            if self.connection_attempts < self.max_connection_attempts:
                self.connection_attempts += 1
                logging.info(f"Attempting to reconnect... ({self.connection_attempts}/{self.max_connection_attempts})")
                
                if self.connect():
                    return True
            
            return False

# Blender integration utilities
class BlenderInferenceManager:
    """
    Blender-specific inference management with thread safety
    """
    
    def __init__(self, server_address: str = "localhost:50051"):
        self.client = InferenceClient(server_address)
        self.active_requests = {}
        self.request_counter = 0
        
    def generate_shape_key_async(
        self,
        obj: bpy.types.Object,
        expression_type: str,
        expression_name: str,
        intensity: float = 1.0,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Generate shape key asynchronously
        
        Returns:
            Request ID for tracking
        """
        
        request_id = f"req_{self.request_counter}"
        self.request_counter += 1
        
        def worker():
            try:
                # Get mesh data
                mesh = obj.data
                vertices = np.array([v.co for v in mesh.vertices], dtype=np.float32)
                
                # Make prediction
                offsets = self.client.predict_vertex_offsets(
                    vertices, expression_type, expression_name, intensity
                )
                
                # Schedule shape key creation on main thread
                if offsets is not None:
                    def create_shape_key():
                        self._create_shape_key_from_offsets(obj, expression_name, offsets)
                        if callback:
                            callback(True, f"Shape key '{expression_name}' created successfully")
                        return None
                    
                    bpy.app.timers.register(create_shape_key, first_interval=0.0)
                else:
                    def error_callback():
                        if callback:
                            callback(False, f"Failed to generate '{expression_name}'")
                        return None
                    
                    bpy.app.timers.register(error_callback, first_interval=0.0)
                    
            except Exception as e:
                logging.error(f"Async shape key generation failed: {e}")
                
                def error_callback():
                    if callback:
                        callback(False, str(e))
                    return None
                
                bpy.app.timers.register(error_callback, first_interval=0.0)
            finally:
                # Clean up request tracking
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
        
        # Start worker thread
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        
        self.active_requests[request_id] = thread
        
        return request_id
    
    def _create_shape_key_from_offsets(self, obj, name, offsets):
        """Create shape key from vertex offsets (must be called on main thread)"""
        
        mesh = obj.data
        
        # Ensure shape keys exist
        if not mesh.shape_keys:
            obj.shape_key_add(name='Basis')
        
        # Remove existing shape key
        if name in mesh.shape_keys.key_blocks:
            obj.shape_key_remove(mesh.shape_keys.key_blocks[name])
        
        # Create new shape key
        shape_key = obj.shape_key_add(name=name)
        
        # Apply offsets
        for i, offset in enumerate(offsets):
            if i < len(shape_key.data):
                base_pos = mesh.vertices[i].co
                shape_key.data[i].co = base_pos + offset
```

## Performance Optimization Features

### Connection Pooling

```python
class ConnectionPool:
    """
    Connection pool for managing multiple gRPC channels
    """
    
    def __init__(self, server_address: str, pool_size: int = 3):
        self.server_address = server_address
        self.pool_size = pool_size
        self.connections = []
        self.lock = threading.Lock()
        
        # Initialize pool
        for _ in range(pool_size):
            self.connections.append(self._create_connection())
    
    def _create_connection(self):
        """Create a new gRPC connection"""
        options = [
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
        
        channel = grpc.insecure_channel(self.server_address, options=options)
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        
        return {'channel': channel, 'stub': stub, 'in_use': False}
    
    def get_connection(self):
        """Get an available connection from the pool"""
        with self.lock:
            for conn in self.connections:
                if not conn['in_use']:
                    conn['in_use'] = True
                    return conn
            
            # No available connections, create temporary one
            return self._create_connection()
    
    def release_connection(self, conn):
        """Release a connection back to the pool"""
        with self.lock:
            if conn in self.connections:
                conn['in_use'] = False
```

### Async Request Management

```python
import asyncio
import aiogrpc

class AsyncInferenceClient:
    """
    Async gRPC client for high-throughput scenarios
    """
    
    def __init__(self, server_address: str = "localhost:50051"):
        self.server_address = server_address
        self.channel = None
        self.stub = None
    
    async def __aenter__(self):
        self.channel = aiogrpc.insecure_channel(self.server_address)
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.channel:
            await self.channel.close()
    
    async def predict_vertex_offsets_async(self, vertices, expression_type, expression_name, intensity=1.0):
        """Async vertex offset prediction"""
        
        condition = inference_pb2.ExpressionCondition(
            type=inference_pb2.ExpressionCondition.VISEME if expression_type.lower() == 'viseme' else inference_pb2.ExpressionCondition.EMOTION,
            name=expression_name,
            intensity=intensity
        )
        
        request = inference_pb2.VertexOffsetRequest(
            vertices=vertices.flatten().tolist(),
            vertex_count=len(vertices),
            condition=condition,
            options=inference_pb2.PredictionOptions()
        )
        
        response = await self.stub.PredictVertexOffsets(request)
        
        if response.success:
            offsets = np.array(response.offsets, dtype=np.float32)
            return offsets.reshape(len(vertices), 3)
        else:
            raise Exception(f"Prediction failed: {response.error_message}")
```

## Implementation Timeline

### Week 1: Protocol and Server Foundation
- Protocol buffer definitions
- Basic gRPC server implementation
- Health check and connection management

### Week 2: Client Integration
- Blender client implementation
- Thread safety for bpy integration
- Async request management

### Week 3: Performance Optimization
- Connection pooling
- Batch processing optimization
- Resource monitoring

### Week 4: Production Hardening
- Error handling and recovery
- Logging and monitoring
- Security and deployment

## Performance Targets

- **Latency**: <3 seconds for single prediction
- **Throughput**: >10 concurrent requests
- **Reliability**: 99.5% uptime for localhost deployment
- **Memory**: <100MB additional overhead
- **Thread Safety**: 100% compatibility with Blender's bpy API

This gRPC communication system provides a robust, high-performance foundation for ML inference integration while maintaining the security and reliability required for production VRChat avatar generation workflows.