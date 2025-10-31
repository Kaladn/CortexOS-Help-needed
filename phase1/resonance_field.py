"""
phase2/resonance_field.py - CortexOS Resonance Field Monitor
Manages neural resonance patterns and field monitoring across the cognitive space.
"""

import time
import math
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque

# Path placeholders
NEURAL_DATA_DIR = "{PATH_NEURAL_DATA_DIR}"
LOGS_DIR = "{PATH_LOGS_DIR}"
RESONANCE_DATA_DIR = "{PATH_RESONANCE_DATA_DIR}"

class ResonanceField:
    """
    Monitors and manages neural resonance patterns across the CortexOS cognitive space.
    
    Tracks neuron registration, phase clusters, resonance broadcasting, and field
    visualization for optimal neural synchronization and cognitive coherence.
    """
    
    def __init__(self, cortex_cube=None):
        """Initialize the Resonance Field Monitor"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("🌊 Initializing ResonanceFieldMonitor...")
        
        # Core dependencies
        self.cortex_cube = cortex_cube
        
        # Resonance tracking
        self.resonance_buffer = {}
        self.phase_clusters = {}
        self.field_history = deque(maxlen=1000)
        self.resonance_patterns = defaultdict(list)
        
        # Field parameters
        self.field_dimensions = (100, 100, 100)  # Default 3D field size
        self.resonance_threshold = 0.7
        self.cluster_radius = 3
        self.max_history_per_neuron = 100
        self.field_update_interval = 0.1  # seconds
        
        # Agharmonic parameters
        self.base_frequency = 432.0  # Hz
        self.harmonic_ratios = [1.0, 1.5, 2.0, 3.0, 4.0]  # Common harmonic ratios
        self.phase_tolerance = 0.1  # radians
        
        # Field state
        self.active_neurons = set()
        self.last_scan_time = 0
        self.scan_count = 0
        self.field_energy = 0.0
        
        # Performance tracking
        self.broadcast_count = 0
        self.query_count = 0
        self.cluster_count = 0
        
        self.logger.info("✅ ResonanceFieldMonitor initialized")
        
    def set_dependencies(self, **components):
        """Set component dependencies"""
        for name, component in components.items():
            setattr(self, name, component)
            self.logger.debug(f"Dependency set: {name}")
            
    def register_neuron(self, neuron_id: str, x: int, y: int, z: int, 
                       frequency: float = None) -> bool:
        """
        Register a neuron in the resonance field.
        
        Args:
            neuron_id: Unique identifier for the neuron
            x, y, z: 3D coordinates in the field
            frequency: Base resonance frequency (optional)
            
        Returns:
            True if registration successful
        """
        try:
            # Validate coordinates
            if not self._validate_coordinates(x, y, z):
                self.logger.warning(f"Invalid coordinates for neuron {neuron_id}: ({x}, {y}, {z})")
                return False
                
            # Set default frequency if not provided
            if frequency is None:
                frequency = self.base_frequency
                
            # Register neuron
            self.resonance_buffer[neuron_id] = {
                'coords': (x, y, z),
                'frequency': frequency,
                'history': deque(maxlen=self.max_history_per_neuron),
                'last_broadcast': 0,
                'energy_level': 0.0,
                'phase': 0.0,
                'cluster_id': None,
                'registration_time': time.time()
            }
            
            self.active_neurons.add(neuron_id)
            
            self.logger.debug(f"Registered neuron {neuron_id} at ({x}, {y}, {z}) with frequency {frequency}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register neuron {neuron_id}: {e}")
            return False
            
    def unregister_neuron(self, neuron_id: str) -> bool:
        """Unregister a neuron from the resonance field"""
        try:
            if neuron_id in self.resonance_buffer:
                del self.resonance_buffer[neuron_id]
                self.active_neurons.discard(neuron_id)
                self.logger.debug(f"Unregistered neuron {neuron_id}")
                return True
            else:
                self.logger.warning(f"Neuron {neuron_id} not found for unregistration")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to unregister neuron {neuron_id}: {e}")
            return False
            
    def scan_for_phase_clusters(self) -> Dict[str, List[str]]:
        """
        Scan the field for synchronized neuron clusters.
        
        Returns:
            Dictionary mapping cluster IDs to lists of neuron IDs
        """
        try:
            self.last_scan_time = time.time()
            self.scan_count += 1
            
            # Clear existing clusters
            self.phase_clusters.clear()
            
            # Reset cluster assignments
            for neuron_data in self.resonance_buffer.values():
                neuron_data['cluster_id'] = None
                
            # Find clusters based on spatial proximity and phase alignment
            processed_neurons = set()
            cluster_id = 0
            
            for neuron_id, neuron_data in self.resonance_buffer.items():
                if neuron_id in processed_neurons:
                    continue
                    
                # Start new cluster
                cluster_neurons = self._find_cluster_members(neuron_id, processed_neurons)
                
                if len(cluster_neurons) > 1:  # Only create cluster if multiple neurons
                    cluster_key = f"cluster_{cluster_id}"
                    self.phase_clusters[cluster_key] = cluster_neurons
                    
                    # Assign cluster ID to neurons
                    for cluster_neuron_id in cluster_neurons:
                        if cluster_neuron_id in self.resonance_buffer:
                            self.resonance_buffer[cluster_neuron_id]['cluster_id'] = cluster_key
                            
                    cluster_id += 1
                    
                processed_neurons.update(cluster_neurons)
                
            self.cluster_count = len(self.phase_clusters)
            
            self.logger.debug(f"Found {self.cluster_count} phase clusters in field scan")
            return self.phase_clusters.copy()
            
        except Exception as e:
            self.logger.error(f"Phase cluster scan failed: {e}")
            return {}
            
    def _find_cluster_members(self, seed_neuron_id: str, processed_neurons: set) -> List[str]:
        """Find all neurons that should be in the same cluster as the seed neuron"""
        try:
            if seed_neuron_id not in self.resonance_buffer:
                return []
                
            seed_data = self.resonance_buffer[seed_neuron_id]
            seed_coords = seed_data['coords']
            seed_phase = seed_data['phase']
            seed_frequency = seed_data['frequency']
            
            cluster_members = [seed_neuron_id]
            
            # Find nearby neurons with similar phase and frequency
            for neuron_id, neuron_data in self.resonance_buffer.items():
                if neuron_id == seed_neuron_id or neuron_id in processed_neurons:
                    continue
                    
                # Check spatial proximity
                distance = self._calculate_distance(seed_coords, neuron_data['coords'])
                if distance > self.cluster_radius:
                    continue
                    
                # Check frequency compatibility
                freq_ratio = neuron_data['frequency'] / seed_frequency
                if not self._is_harmonic_ratio(freq_ratio):
                    continue
                    
                # Check phase alignment
                phase_diff = abs(neuron_data['phase'] - seed_phase)
                if phase_diff > self.phase_tolerance:
                    continue
                    
                cluster_members.append(neuron_id)
                
            return cluster_members
            
        except Exception as e:
            self.logger.error(f"Failed to find cluster members for {seed_neuron_id}: {e}")
            return [seed_neuron_id] if seed_neuron_id in self.resonance_buffer else []
            
    def broadcast_resonance(self, neuron_id: str, resonance_profile: Dict) -> bool:
        """
        Broadcast resonance from a neuron to the field.
        
        Args:
            neuron_id: ID of the broadcasting neuron
            resonance_profile: Resonance data to broadcast
            
        Returns:
            True if broadcast successful
        """
        try:
            if neuron_id not in self.resonance_buffer:
                self.logger.warning(f"Cannot broadcast from unregistered neuron: {neuron_id}")
                return False
                
            # Validate resonance profile
            if not self._validate_resonance_profile(resonance_profile):
                self.logger.warning(f"Invalid resonance profile from {neuron_id}")
                return False
                
            # Update neuron data
            neuron_data = self.resonance_buffer[neuron_id]
            neuron_data['last_broadcast'] = time.time()
            neuron_data['energy_level'] = resonance_profile.get('energy', 0.0)
            neuron_data['phase'] = resonance_profile.get('phase', 0.0)
            
            # Add to history
            broadcast_record = {
                'timestamp': time.time(),
                'profile': resonance_profile.copy(),
                'field_energy': self.field_energy
            }
            neuron_data['history'].append(broadcast_record)
            
            # Update field energy
            self._update_field_energy()
            
            # Propagate to nearby neurons
            self._propagate_resonance(neuron_id, resonance_profile)
            
            self.broadcast_count += 1
            
            self.logger.debug(f"Broadcasted resonance from {neuron_id}: {resonance_profile}")
            return True
            
        except Exception as e:
            self.logger.error(f"Resonance broadcast failed for {neuron_id}: {e}")
            return False
            
    def query_nearby_resonance(self, x: int, y: int, z: int, 
                             radius: int = 3) -> List[Dict]:
        """
        Query resonance patterns near specified coordinates.
        
        Args:
            x, y, z: Query coordinates
            radius: Search radius
            
        Returns:
            List of nearby neuron resonance data
        """
        try:
            self.query_count += 1
            
            query_coords = (x, y, z)
            nearby_resonance = []
            
            for neuron_id, neuron_data in self.resonance_buffer.items():
                distance = self._calculate_distance(query_coords, neuron_data['coords'])
                
                if distance <= radius:
                    # Calculate resonance strength based on distance
                    strength = max(0.0, 1.0 - (distance / radius))
                    
                    resonance_info = {
                        'neuron_id': neuron_id,
                        'coords': neuron_data['coords'],
                        'distance': distance,
                        'strength': strength,
                        'frequency': neuron_data['frequency'],
                        'phase': neuron_data['phase'],
                        'energy_level': neuron_data['energy_level'],
                        'cluster_id': neuron_data['cluster_id'],
                        'last_broadcast': neuron_data['last_broadcast']
                    }
                    
                    nearby_resonance.append(resonance_info)
                    
            # Sort by strength (closest first)
            nearby_resonance.sort(key=lambda x: x['strength'], reverse=True)
            
            self.logger.debug(f"Found {len(nearby_resonance)} neurons near ({x}, {y}, {z}) within radius {radius}")
            return nearby_resonance
            
        except Exception as e:
            self.logger.error(f"Nearby resonance query failed: {e}")
            return []
            
    def visualize_field(self, z_slice: int) -> Dict:
        """
        Generate visualization data for a specific Z-slice of the field.
        
        Args:
            z_slice: Z-coordinate slice to visualize
            
        Returns:
            Visualization data for the slice
        """
        try:
            if not self._validate_z_slice(z_slice):
                self.logger.warning(f"Invalid z-slice: {z_slice}")
                return {}
                
            # Collect neurons in the specified slice
            slice_neurons = []
            for neuron_id, neuron_data in self.resonance_buffer.items():
                if neuron_data['coords'][2] == z_slice:
                    slice_neurons.append({
                        'id': neuron_id,
                        'x': neuron_data['coords'][0],
                        'y': neuron_data['coords'][1],
                        'frequency': neuron_data['frequency'],
                        'phase': neuron_data['phase'],
                        'energy': neuron_data['energy_level'],
                        'cluster': neuron_data['cluster_id']
                    })
                    
            # Generate field visualization data
            visualization_data = {
                'z_slice': z_slice,
                'timestamp': time.time(),
                'neurons': slice_neurons,
                'field_energy': self.field_energy,
                'cluster_count': len([n for n in slice_neurons if n['cluster']]),
                'total_neurons': len(slice_neurons),
                'avg_frequency': self._calculate_average_frequency(slice_neurons),
                'energy_distribution': self._calculate_energy_distribution(slice_neurons)
            }
            
            self.logger.debug(f"Generated visualization for z-slice {z_slice} with {len(slice_neurons)} neurons")
            return visualization_data
            
        except Exception as e:
            self.logger.error(f"Field visualization failed for z-slice {z_slice}: {e}")
            return {}
            
    def analyze_text(self, text_data: str) -> Dict:
        """
        Analyze text for resonance patterns.
        
        Args:
            text_data: Text to analyze
            
        Returns:
            Resonance analysis results
        """
        try:
            # Basic text resonance analysis
            analysis = {
                'text_length': len(text_data),
                'word_count': len(text_data.split()),
                'resonance_frequency': self._calculate_text_frequency(text_data),
                'harmonic_content': self._analyze_harmonic_content(text_data),
                'phase_alignment': self._calculate_text_phase(text_data),
                'energy_signature': self._calculate_text_energy(text_data)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Text resonance analysis failed: {e}")
            return {'error': str(e)}
            
    # Helper Methods
    def _validate_coordinates(self, x: int, y: int, z: int) -> bool:
        """Validate 3D coordinates within field bounds"""
        return (0 <= x < self.field_dimensions[0] and 
                0 <= y < self.field_dimensions[1] and 
                0 <= z < self.field_dimensions[2])
                
    def _validate_z_slice(self, z_slice: int) -> bool:
        """Validate Z-slice coordinate"""
        return 0 <= z_slice < self.field_dimensions[2]
        
    def _validate_resonance_profile(self, profile: Dict) -> bool:
        """Validate resonance profile structure"""
        required_fields = ['energy', 'phase']
        return all(field in profile for field in required_fields)
        
    def _calculate_distance(self, coords1: Tuple[int, int, int], 
                          coords2: Tuple[int, int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(coords1, coords2)))
        
    def _is_harmonic_ratio(self, ratio: float) -> bool:
        """Check if frequency ratio is a harmonic ratio"""
        for harmonic in self.harmonic_ratios:
            if abs(ratio - harmonic) < 0.1 or abs(ratio - 1/harmonic) < 0.1:
                return True
        return False
        
    def _update_field_energy(self):
        """Update total field energy based on active neurons"""
        try:
            total_energy = sum(neuron_data['energy_level'] 
                             for neuron_data in self.resonance_buffer.values())
            self.field_energy = total_energy / max(1, len(self.resonance_buffer))
            
        except Exception:
            self.field_energy = 0.0
            
    def _propagate_resonance(self, source_neuron_id: str, resonance_profile: Dict):
        """Propagate resonance to nearby neurons"""
        try:
            if source_neuron_id not in self.resonance_buffer:
                return
                
            source_coords = self.resonance_buffer[source_neuron_id]['coords']
            propagation_radius = 5  # Maximum propagation distance
            
            for neuron_id, neuron_data in self.resonance_buffer.items():
                if neuron_id == source_neuron_id:
                    continue
                    
                distance = self._calculate_distance(source_coords, neuron_data['coords'])
                if distance <= propagation_radius:
                    # Calculate influence based on distance
                    influence = max(0.0, 1.0 - (distance / propagation_radius))
                    
                    # Apply resonance influence
                    energy_boost = resonance_profile.get('energy', 0.0) * influence * 0.1
                    neuron_data['energy_level'] += energy_boost
                    
        except Exception as e:
            self.logger.error(f"Resonance propagation failed: {e}")
            
    def _calculate_text_frequency(self, text: str) -> float:
        """Calculate resonance frequency for text"""
        try:
            # Simple frequency calculation based on text characteristics
            word_count = len(text.split())
            char_count = len(text)
            
            if char_count == 0:
                return self.base_frequency
                
            # Frequency based on text density and rhythm
            density = word_count / char_count
            frequency = self.base_frequency * (1.0 + density)
            
            return min(1000.0, max(100.0, frequency))
            
        except Exception:
            return self.base_frequency
            
    def _analyze_harmonic_content(self, text: str) -> List[float]:
        """Analyze harmonic content in text"""
        try:
            # Simple harmonic analysis based on text patterns
            harmonics = []
            
            for ratio in self.harmonic_ratios:
                harmonic_freq = self.base_frequency * ratio
                # Calculate presence of this harmonic in text
                presence = self._calculate_harmonic_presence(text, harmonic_freq)
                harmonics.append(presence)
                
            return harmonics
            
        except Exception:
            return [0.0] * len(self.harmonic_ratios)
            
    def _calculate_harmonic_presence(self, text: str, frequency: float) -> float:
        """Calculate presence of specific harmonic in text"""
        try:
            # Simple calculation based on text rhythm and frequency
            words = text.split()
            if not words:
                return 0.0
                
            # Calculate rhythm score
            avg_word_length = sum(len(word) for word in words) / len(words)
            rhythm_score = 1.0 / (1.0 + abs(avg_word_length - 5.0))  # Optimal around 5 chars
            
            # Frequency alignment score
            freq_alignment = 1.0 / (1.0 + abs(frequency - self.base_frequency) / self.base_frequency)
            
            return rhythm_score * freq_alignment
            
        except Exception:
            return 0.0
            
    def _calculate_text_phase(self, text: str) -> float:
        """Calculate phase alignment for text"""
        try:
            # Simple phase calculation based on text structure
            sentences = text.split('.')
            if len(sentences) <= 1:
                return 0.0
                
            # Phase based on sentence rhythm
            sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
            if not sentence_lengths:
                return 0.0
                
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            
            # Lower variance = better phase alignment
            phase = math.pi * (1.0 - min(1.0, variance / (avg_length ** 2)))
            
            return phase
            
        except Exception:
            return 0.0
            
    def _calculate_text_energy(self, text: str) -> float:
        """Calculate energy signature for text"""
        try:
            # Energy based on text complexity and information density
            words = text.split()
            if not words:
                return 0.0
                
            unique_words = len(set(words))
            diversity = unique_words / len(words)
            
            # Energy increases with diversity and length
            energy = diversity * math.log(1 + len(words)) / 10.0
            
            return min(1.0, energy)
            
        except Exception:
            return 0.0
            
    def _calculate_average_frequency(self, neurons: List[Dict]) -> float:
        """Calculate average frequency for a list of neurons"""
        if not neurons:
            return self.base_frequency
            
        frequencies = [n['frequency'] for n in neurons]
        return sum(frequencies) / len(frequencies)
        
    def _calculate_energy_distribution(self, neurons: List[Dict]) -> Dict:
        """Calculate energy distribution statistics"""
        if not neurons:
            return {'min': 0.0, 'max': 0.0, 'avg': 0.0, 'total': 0.0}
            
        energies = [n['energy'] for n in neurons]
        
        return {
            'min': min(energies),
            'max': max(energies),
            'avg': sum(energies) / len(energies),
            'total': sum(energies)
        }
        
    # Public Interface Methods
    def registered_neuron_ids(self) -> List[str]:
        """Get list of all registered neuron IDs"""
        return list(self.resonance_buffer.keys())
        
    def get_field_stats(self) -> Dict:
        """Get comprehensive field statistics"""
        return {
            'total_neurons': len(self.resonance_buffer),
            'active_neurons': len(self.active_neurons),
            'phase_clusters': len(self.phase_clusters),
            'field_energy': self.field_energy,
            'scan_count': self.scan_count,
            'broadcast_count': self.broadcast_count,
            'query_count': self.query_count,
            'last_scan_time': self.last_scan_time,
            'field_dimensions': self.field_dimensions
        }
        
    def reset_field(self) -> bool:
        """Reset the entire resonance field"""
        try:
            self.resonance_buffer.clear()
            self.phase_clusters.clear()
            self.field_history.clear()
            self.resonance_patterns.clear()
            self.active_neurons.clear()
            
            self.field_energy = 0.0
            self.scan_count = 0
            self.broadcast_count = 0
            self.query_count = 0
            self.cluster_count = 0
            
            self.logger.info("Resonance field reset successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset resonance field: {e}")
            return False


if __name__ == "__main__":
    # Test the CortexOS Resonance Field
    print("🌊 Testing CortexOS Resonance Field...")
    
    field = CortexOSResonanceField()
    
    # Test field startup
    if field.start():
        print("✅ Resonance Field started successfully")
    
    # Test resonance scanning
    test_frequencies = [440.0, 880.0, 1320.0, 660.0, 220.0]
    
    for freq in test_frequencies:
        resonance_data = {
            "frequency": freq,
            "amplitude": 0.8,
            "phase": 0.0,
            "source": "test_neuron"
        }
        field.scan_resonance(resonance_data)
        print(f"✅ Scanned resonance at {freq} Hz")
    
    # Test field analysis
    field_state = field.analyze_field()
    print(f"✅ Field analysis - Energy: {field_state['energy']:.2f}, Stability: {field_state['stability']:.2f}")
    
    # Test resonance broadcasting
    broadcast_data = {
        "frequency": 528.0,  # "Love frequency"
        "amplitude": 1.0,
        "message": "Test broadcast",
        "target_neurons": ["neuron_1", "neuron_2"]
    }
    
    if field.broadcast_resonance(broadcast_data):
        print("✅ Resonance broadcast successful")
    
    # Test field queries
    query_results = field.query_field(frequency_range=(400, 500))
    print(f"✅ Field query returned {len(query_results)} resonances in 400-500 Hz range")
    
    # Test phase clustering
    clusters = field.cluster_phases()
    print(f"✅ Phase clustering found {len(clusters)} clusters")
    
    # Test field visualization data
    viz_data = field.get_visualization_data()
    print(f"✅ Visualization data - {len(viz_data['frequencies'])} frequency points")
    
    # Test field status
    status = field.get_status()
    print(f"✅ Field status: {status['state']}, Scans: {status['scan_count']}, Energy: {status['field_energy']:.2f}")
    
    # Test field reset
    if field.reset_field():
        print("✅ Field reset successful")
    
    # Shutdown
    field.stop()
    print("✅ Resonance Field stopped")
    
    print("🌊 Resonance Field test complete!")

