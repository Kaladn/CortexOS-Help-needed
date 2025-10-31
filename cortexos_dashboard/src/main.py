import os
import sys
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from src.routes.cortexos_api import cortexos_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'cortexos_dashboard_secret_key_2024'

# Enable CORS for all routes
CORS(app)

# Register CortexOS API blueprint
app.register_blueprint(cortexos_bp, url_prefix='/api')

# Mock CortexOS system state for demonstration
class MockCortexOSSystem:
    def __init__(self):
        self.components = {
            'infrastructure': {
                'path_manager': {'status': 'running', 'health': 0.95, 'cpu': 5.2, 'memory': 128},
                'supervisor': {'status': 'running', 'health': 0.98, 'cpu': 3.1, 'memory': 96},
                'cube_storage': {'status': 'running', 'health': 0.92, 'cpu': 8.7, 'memory': 512},
                'contract_manager': {'status': 'running', 'health': 0.89, 'cpu': 4.3, 'memory': 156},
                'sync_manager': {'status': 'running', 'health': 0.94, 'cpu': 6.1, 'memory': 203},
                'neural_fabric': {'status': 'running', 'health': 0.96, 'cpu': 12.4, 'memory': 768}
            },
            'phase1': {
                'neuroengine': {'status': 'running', 'health': 0.93, 'cpu': 15.6, 'memory': 1024},
                'context_engine': {'status': 'running', 'health': 0.91, 'cpu': 11.2, 'memory': 512},
                'neural_gatekeeper': {'status': 'running', 'health': 0.97, 'cpu': 7.8, 'memory': 256}
            },
            'phase2': {
                'resonance_field': {'status': 'running', 'health': 0.88, 'cpu': 18.3, 'memory': 896},
                'resonance_monitor': {'status': 'running', 'health': 0.95, 'cpu': 9.4, 'memory': 384},
                'resonance_reinforcer': {'status': 'running', 'health': 0.92, 'cpu': 13.7, 'memory': 640},
                'topk_sparse_resonance': {'status': 'running', 'health': 0.90, 'cpu': 16.8, 'memory': 728}
            },
            'phase3': {
                'memory_inserter': {'status': 'running', 'health': 0.94, 'cpu': 10.5, 'memory': 512},
                'memory_retriever': {'status': 'running', 'health': 0.96, 'cpu': 8.9, 'memory': 448},
                'memory_consolidator': {'status': 'running', 'health': 0.87, 'cpu': 22.1, 'memory': 1152},
                'cognitive_bridge': {'status': 'running', 'health': 0.93, 'cpu': 14.2, 'memory': 672}
            },
            'phase4': {
                'data_ingestion_engine': {'status': 'running', 'health': 0.91, 'cpu': 19.6, 'memory': 896},
                'stream_processor': {'status': 'running', 'health': 0.89, 'cpu': 25.3, 'memory': 1280},
                'batch_processor': {'status': 'running', 'health': 0.95, 'cpu': 17.4, 'memory': 768},
                'ingestion_validator': {'status': 'running', 'health': 0.98, 'cpu': 6.7, 'memory': 320}
            },
            'phase5': {
                'system_monitor': {'status': 'running', 'health': 0.99, 'cpu': 4.2, 'memory': 192},
                'performance_tracker': {'status': 'running', 'health': 0.97, 'cpu': 7.8, 'memory': 256},
                'health_checker': {'status': 'running', 'health': 0.98, 'cpu': 3.9, 'memory': 128},
                'alert_manager': {'status': 'running', 'health': 0.96, 'cpu': 5.1, 'memory': 164}
            },
            'phase6': {
                'mood_modulator': {'status': 'running', 'health': 0.92, 'cpu': 11.8, 'memory': 448},
                'cognitive_enhancer': {'status': 'running', 'health': 0.94, 'cpu': 16.3, 'memory': 672},
                'neural_optimizer': {'status': 'running', 'health': 0.88, 'cpu': 28.7, 'memory': 1536},
                'adaptive_controller': {'status': 'running', 'health': 0.95, 'cpu': 12.9, 'memory': 512}
            }
        }
        
        self.system_metrics = {
            'overall_health': 0.93,
            'total_cpu_usage': 45.2,
            'total_memory_usage': 12.8,  # GB
            'throughput': 1247.5,  # ops/sec
            'latency': 23.4,  # ms
            'uptime': 86400,  # seconds
            'active_processes': 30,
            'error_count': 2,
            'warning_count': 5
        }
        
        self.recent_activities = [
            {'time': '2024-01-15 14:30:25', 'type': 'optimization', 'component': 'neural_optimizer', 'message': 'Neural optimization completed successfully'},
            {'time': '2024-01-15 14:28:12', 'type': 'memory', 'component': 'memory_consolidator', 'message': 'Memory consolidation cycle finished'},
            {'time': '2024-01-15 14:25:45', 'type': 'alert', 'component': 'health_checker', 'message': 'High CPU usage detected on resonance_field'},
            {'time': '2024-01-15 14:22:18', 'type': 'startup', 'component': 'adaptive_controller', 'message': 'Component restarted successfully'},
            {'time': '2024-01-15 14:20:03', 'type': 'data', 'component': 'stream_processor', 'message': 'Processing 1.2K events/sec'}
        ]

# Global mock system instance
mock_system = MockCortexOSSystem()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
        return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404

# Additional API endpoints for real-time data
@app.route('/api/realtime/metrics')
def get_realtime_metrics():
    """Get real-time system metrics"""
    import random
    
    # Simulate some variation in metrics
    base_metrics = mock_system.system_metrics.copy()
    base_metrics['total_cpu_usage'] += random.uniform(-5, 5)
    base_metrics['total_memory_usage'] += random.uniform(-1, 1)
    base_metrics['throughput'] += random.uniform(-100, 100)
    base_metrics['latency'] += random.uniform(-5, 5)
    base_metrics['timestamp'] = datetime.now().isoformat()
    
    return jsonify(base_metrics)

@app.route('/api/realtime/activities')
def get_recent_activities():
    """Get recent system activities"""
    return jsonify(mock_system.recent_activities)

if __name__ == '__main__':
    print("🧠 Starting CortexOS Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🔧 API endpoints available at: http://localhost:5000/api")
    app.run(host='0.0.0.0', port=5000, debug=True)

