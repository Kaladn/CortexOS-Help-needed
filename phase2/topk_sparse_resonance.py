"""
phase2/topk_sparse_resonance.py - CortexOS Top-K Sparse Resonance Engine
Prioritizes high-similarity voxel matches with mood-based modulation and phase harmonics.
"""

import time
import threading
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, deque
from datetime import datetime
# phase2/topk_sparse_resonance.py

import logging
from phase1.phase_harmonics import PhaseHarmonics
from phase6.neuromodulation import Neuromodulator

logger = logging.getLogger(__name__)

class TopKSparseResonance:
    """
    Identifies the top-k resonant patterns in a high-dimensional space.
    This class will be more fully implemented as other components come online.
    """
    def __init__(self, config=None):
        logger.info("TopKSparseResonance component initialized.")
        self.config = config if config else {}

    def find_top_k(self, agent_vector, voxel_field, mood="neutral"):
        """
        A placeholder method for this class's main function.
        The core logic is currently in the assess_resonance function below.
        """
        matches, log = assess_resonance(agent_vector, voxel_field, mood, 0)
        return matches


def assess_resonance(agent_vector, voxel_field, mood, time_step):
    """
    Assesses the resonance between a single agent vector and a field of voxel vectors.

    This function is called by SwarmResonance to determine which parts of the
    voxel field an individual agent is "interested" in.

    Args:
        agent_vector (list): The vector of the agent being assessed.
        voxel_field (dict): The environment of voxels to check against.
                            Format: {voxel_id: (vector, time_step)}
        mood (str): The current mood, which affects resonance parameters.
        time_step (int): The current simulation time step.

    Returns:
        tuple: A tuple containing:
            - top_k_matches (list): A list of (voxel_id, similarity_score) tuples.
            - resonance_log (dict): A log of the assessment process.
    """
    # Initialize components needed for calculation
    neuromod = Neuromodulator()
    phase_harmonics = PhaseHarmonics()

    # Get mood-based parameters
    params = neuromod.adjust_resonance_params(mood)
    k = params.get('k', 10)
    threshold = params.get('threshold', 0.8)

    resonance_scores = []
    for voxel_id, (voxel_vector, voxel_time_step) in voxel_field.items():
        # Calculate the time difference for temporal penalty
        time_delta = abs(time_step - voxel_time_step)

        # Use PhaseHarmonics to compute similarity, which includes the temporal penalty
        similarity_result = phase_harmonics.compute_phase_similarity(
            agent_vector,
            voxel_vector,
            time_delta
        )
        final_score = similarity_result['similarity']

        # Only consider matches above the mood-adjusted threshold
        if final_score >= threshold:
            resonance_scores.append((voxel_id, final_score))

    # Sort the results by similarity score in descending order
    resonance_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the top 'k' results
    top_k_matches = resonance_scores[:k]

    # Create a log of the operation
    resonance_log = {
        "activations": [score for _, score in top_k_matches],
        "mood_params": params,
        "input_vector_len": len(agent_vector),
        "field_size": len(voxel_field),
        "matches_found_before_k": len(resonance_scores)
    }

    return top_k_matches, resonance_log
# Path placeholders
NEURAL_DATA_DIR = "{PATH_NEURAL_DATA_DIR}"
LOGS_DIR = "{PATH_LOGS_DIR}"
RESONANCE_DATA_DIR = "{PATH_RESONANCE_DATA_DIR}"

class TopKSparseResonance:
    """
    Implements top-k sparse resonance with mood-modulated parameters and phase harmonics.
    
    Prioritizes high-similarity voxel matches while maintaining computational efficiency
    through sparse activation patterns. Fully compliant with Agharmonic Law tenets.
    """
    
    def __init__(self, default_k: int = 5, default_threshold: float = 0.7, 
                 default_decay: float = 0.05):
        """
        Initialize the Top-K Sparse Resonance Engine.
        
        Args:
            default_k: Default number of top matches to return
            default_threshold: Default similarity threshold
            default_decay: Default decay rate for resonance patterns
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Initializing TopKSparseResonance...")
        
        # Configuration parameters
        self.default_k = default_k
        self.default_threshold = default_threshold
        self.default_decay = default_decay
        
        # Resonance tracking
        self.activation_history = defaultdict(list)
        self.resonance_patterns = {}
        self.voxel_similarities = {}
        self.harmonic_cache = {}
        
        # Performance optimization
        self.sparse_index = {}  # For fast similarity lookups
        self.top_k_cache = {}   # Cache for frequent queries
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Mood and modulation
        self.current_mood = "neutral"
        self.mood_modulation_factors = {
            "excited": {"k_multiplier": 1.5, "threshold_modifier": -0.1},
            "focused": {"k_multiplier": 0.8, "threshold_modifier": 0.1},
            "creative": {"k_multiplier": 2.0, "threshold_modifier": -0.2},
            "analytical": {"k_multiplier": 0.6, "threshold_modifier": 0.2},
            "neutral": {"k_multiplier": 1.0, "threshold_modifier": 0.0}
        }
        
        # Phase harmonics
        self.phase_harmonics = {}
        self.harmonic_frequencies = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # Common harmonic ratios
        self.phase_alignment_tolerance = 0.1
        
        # Temporal management
        self.last_sync = time.time()
        self.last_regulation_time = time.time()
        self.regulation_interval = 300  # 5 minutes
        self.time_step = 0
        
        # Threading and safety
        self.lock = threading.Lock()
        self.active = True
        
        # Fallback and health monitoring
        self.fallback_mode = False
        self.fallback_level = 0
        self.resonance_chain_health = 1.0
        self.error_count = 0
        self.success_count = 0
        
        # Performance metrics
        self.query_count = 0
        self.total_matches_found = 0
        self.avg_similarity_score = 0.0
        self.processing_times = deque(maxlen=100)
        
        # Resonance logging
        self.resonance_log = {
            "activations": deque(maxlen=1000),
            "harmonic_diffs": deque(maxlen=1000),
            "time_steps": deque(maxlen=1000)
        }
        
        # Agharmonic Law parameters
        self.input_frequency_range = [0.6, 1.4]
        self.output_phase_alignment = 0.05
        self.resonance_threshold = 0.8
        self.harmonic_modes = ["sparse", "topk", "phase_aligned"]
        
        # Component references
        self.neuromodulator = None
        self.phase_harmonics_engine = None
        self.sync_manager = None
        
        self.logger.info("✅ TopKSparseResonance initialized")
        
    def set_dependencies(self, **components):
        """Set component dependencies"""
        for name, component in components.items():
            setattr(self, name, component)
            self.logger.debug(f"Dependency set: {name}")
            
    def find_top_k_matches(self, harmonic_vector: Union[List, np.ndarray], 
                          voxel_field: Dict, mood: str = None, 
                          time_step: int = None, k: int = None, 
                          threshold: float = None) -> List[Dict]:
        """
        Find top-k sparse resonance matches for a given harmonic vector.
        
        Args:
            harmonic_vector: Input harmonic vector for matching
            voxel_field: Dictionary of voxel_id -> (vector, timestamp) pairs
            mood: Current mood state for modulation
            time_step: Temporal index for phase harmonics
            k: Number of top matches to return (optional)
            threshold: Similarity threshold (optional)
            
        Returns:
            List of top-k matches with similarity scores and metadata
        """
        try:
            start_time = time.time()
            
            # Validate interface contract
            if not self._validate_interface_contract(
                harmonic_vector, voxel_field, mood, time_step, k, threshold
            ):
                return []
                
            with self.lock:
                # Apply mood modulation
                effective_k, effective_threshold = self._apply_mood_modulation(
                    k or self.default_k, threshold or self.default_threshold, mood
                )
                
                # Update time step
                if time_step is not None:
                    self.time_step = time_step
                else:
                    self.time_step += 1
                    
                # Convert input to numpy array
                query_vector = np.array(harmonic_vector, dtype=np.float32)
                
                # Check cache first
                cache_key = self._generate_cache_key(query_vector, effective_k, effective_threshold)
                if cache_key in self.top_k_cache:
                    self.cache_hits += 1
                    cached_result = self.top_k_cache[cache_key]
                    # Update timestamps and return
                    for match in cached_result:
                        match['query_timestamp'] = time.time()
                    return cached_result
                    
                self.cache_misses += 1
                
                # Calculate similarities
                similarities = []
                for voxel_id, (voxel_vector, voxel_timestamp) in voxel_field.items():
                    try:
                        # Convert voxel vector to numpy array
                        voxel_array = np.array(voxel_vector, dtype=np.float32)
                        
                        # Calculate base similarity
                        similarity = self._calculate_similarity(query_vector, voxel_array)
                        
                        # Apply phase harmonic modulation
                        if time_step is not None:
                            phase_factor = self._calculate_phase_factor(
                                voxel_timestamp, time_step
                            )
                            similarity *= phase_factor
                            
                        # Apply temporal decay
                        current_time = time.time()
                        age_factor = self._calculate_age_factor(voxel_timestamp, current_time)
                        similarity *= age_factor
                        
                        # Only include if above threshold
                        if similarity >= effective_threshold:
                            similarities.append({
                                'voxel_id': voxel_id,
                                'similarity': similarity,
                                'voxel_vector': voxel_vector,
                                'voxel_timestamp': voxel_timestamp,
                                'phase_factor': phase_factor if 'phase_factor' in locals() else 1.0,
                                'age_factor': age_factor,
                                'query_timestamp': current_time
                            })
                            
                    except Exception as e:
                        self.logger.warning(f"Error processing voxel {voxel_id}: {e}")
                        continue
                        
                # Sort by similarity and take top-k
                similarities.sort(key=lambda x: x['similarity'], reverse=True)
                top_k_matches = similarities[:effective_k]
                
                # Update statistics
                self.query_count += 1
                self.total_matches_found += len(top_k_matches)
                if top_k_matches:
                    self.avg_similarity_score = (
                        (self.avg_similarity_score * (self.query_count - 1) + 
                         np.mean([m['similarity'] for m in top_k_matches])) / 
                        self.query_count
                    )
                    
                # Cache result
                self.top_k_cache[cache_key] = top_k_matches.copy()
                
                # Limit cache size
                if len(self.top_k_cache) > 100:
                    # Remove oldest entries
                    oldest_keys = list(self.top_k_cache.keys())[:20]
                    for key in oldest_keys:
                        del self.top_k_cache[key]
                        
                # Record processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Log resonance activity
                self._log_resonance_activity(query_vector, top_k_matches, time_step)
                
                self.success_count += 1
                
                return top_k_matches
                
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Top-k matching failed: {e}")
            return []
            
    def update_voxel_resonance(self, voxel_id: str, resonance_strength: float, 
                             context: Dict = None) -> bool:
        """
        Update resonance strength for a specific voxel.
        
        Args:
            voxel_id: Identifier for the voxel
            resonance_strength: New resonance strength (0.0 to 1.0)
            context: Additional context information
            
        Returns:
            True if update successful
        """
        try:
            with self.lock:
                current_time = time.time()
                
                # Validate resonance strength
                if not (0.0 <= resonance_strength <= 1.0):
                    self.logger.warning(f"Invalid resonance strength: {resonance_strength}")
                    return False
                    
                # Update resonance pattern
                if voxel_id not in self.resonance_patterns:
                    self.resonance_patterns[voxel_id] = {
                        'strength': 0.0,
                        'last_update': current_time,
                        'update_count': 0,
                        'history': deque(maxlen=50)
                    }
                    
                pattern = self.resonance_patterns[voxel_id]
                
                # Record previous strength
                previous_strength = pattern['strength']
                
                # Update strength with momentum
                momentum = 0.1  # Smoothing factor
                pattern['strength'] = (
                    momentum * previous_strength + 
                    (1 - momentum) * resonance_strength
                )
                
                # Update metadata
                pattern['last_update'] = current_time
                pattern['update_count'] += 1
                
                # Record in history
                pattern['history'].append({
                    'timestamp': current_time,
                    'strength': resonance_strength,
                    'context': context or {}
                })
                
                # Update activation history
                self.activation_history[voxel_id].append({
                    'timestamp': current_time,
                    'strength': resonance_strength,
                    'previous_strength': previous_strength,
                    'context': context or {}
                })
                
                # Limit activation history
                if len(self.activation_history[voxel_id]) > 100:
                    self.activation_history[voxel_id] = \
                        self.activation_history[voxel_id][-100:]
                        
                return True
                
        except Exception as e:
            self.logger.error(f"Voxel resonance update failed for {voxel_id}: {e}")
            return False
            
    def apply_sparse_activation(self, activation_threshold: float = 0.5) -> Dict[str, float]:
        """
        Apply sparse activation to resonance patterns.
        
        Args:
            activation_threshold: Threshold for sparse activation
            
        Returns:
            Dictionary of activated voxel patterns
        """
        try:
            with self.lock:
                activated_patterns = {}
                
                for voxel_id, pattern in self.resonance_patterns.items():
                    strength = pattern['strength']
                    
                    # Apply sparse activation
                    if strength >= activation_threshold:
                        # Enhance strong patterns
                        enhanced_strength = min(1.0, strength * 1.2)
                        activated_patterns[voxel_id] = enhanced_strength
                    else:
                        # Suppress weak patterns
                        suppressed_strength = max(0.0, strength * 0.8)
                        self.resonance_patterns[voxel_id]['strength'] = suppressed_strength
                        
                self.logger.debug(f"Sparse activation: {len(activated_patterns)} patterns activated")
                return activated_patterns
                
        except Exception as e:
            self.logger.error(f"Sparse activation failed: {e}")
            return {}
            
    def calculate_harmonic_resonance(self, vector1: np.ndarray, vector2: np.ndarray, 
                                   time_step: int = None) -> float:
        """
        Calculate harmonic resonance between two vectors.
        
        Args:
            vector1: First harmonic vector
            vector2: Second harmonic vector
            time_step: Temporal index for phase calculation
            
        Returns:
            Harmonic resonance score
        """
        try:
            # Base similarity
            base_similarity = self._calculate_similarity(vector1, vector2)
            
            # Calculate harmonic enhancement
            harmonic_factor = 1.0
            
            # Frequency domain analysis
            if len(vector1) >= 4 and len(vector2) >= 4:
                # Extract frequency components (assuming RGBA + intensity format)
                freq1 = np.fft.fft(vector1[:4])
                freq2 = np.fft.fft(vector2[:4])
                
                # Calculate frequency correlation
                freq_correlation = np.abs(np.corrcoef(
                    np.real(freq1), np.real(freq2)
                )[0, 1])
                
                if not np.isnan(freq_correlation):
                    harmonic_factor *= (1.0 + freq_correlation * 0.5)
                    
            # Phase alignment factor
            if time_step is not None:
                phase_factor = self._calculate_phase_alignment(vector1, vector2, time_step)
                harmonic_factor *= phase_factor
                
            # Harmonic ratio analysis
            ratio_factor = self._analyze_harmonic_ratios(vector1, vector2)
            harmonic_factor *= ratio_factor
            
            # Final resonance score
            resonance_score = base_similarity * harmonic_factor
            
            return min(1.0, resonance_score)
            
        except Exception as e:
            self.logger.error(f"Harmonic resonance calculation failed: {e}")
            return 0.0
            
    def get_resonance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resonance statistics"""
        try:
            with self.lock:
                current_time = time.time()
                
                # Basic statistics
                total_patterns = len(self.resonance_patterns)
                active_patterns = sum(1 for p in self.resonance_patterns.values() 
                                    if p['strength'] > 0.1)
                
                # Performance statistics
                avg_processing_time = (np.mean(self.processing_times) 
                                     if self.processing_times else 0.0)
                cache_hit_rate = (self.cache_hits / max(1, self.cache_hits + self.cache_misses))
                
                # Pattern strength statistics
                if self.resonance_patterns:
                    strengths = [p['strength'] for p in self.resonance_patterns.values()]
                    strength_stats = {
                        'min': min(strengths),
                        'max': max(strengths),
                        'mean': np.mean(strengths),
                        'std': np.std(strengths)
                    }
                else:
                    strength_stats = {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
                    
                # Recent activity
                recent_activations = sum(1 for pattern in self.resonance_patterns.values()
                                       if current_time - pattern['last_update'] < 300)
                
                return {
                    'timestamp': current_time,
                    'patterns': {
                        'total': total_patterns,
                        'active': active_patterns,
                        'recent_activations': recent_activations,
                        'strength_stats': strength_stats
                    },
                    'performance': {
                        'query_count': self.query_count,
                        'total_matches_found': self.total_matches_found,
                        'avg_similarity_score': self.avg_similarity_score,
                        'avg_processing_time': avg_processing_time,
                        'cache_hit_rate': cache_hit_rate,
                        'cache_hits': self.cache_hits,
                        'cache_misses': self.cache_misses
                    },
                    'system': {
                        'current_mood': self.current_mood,
                        'time_step': self.time_step,
                        'fallback_mode': self.fallback_mode,
                        'fallback_level': self.fallback_level,
                        'resonance_chain_health': self.resonance_chain_health,
                        'success_count': self.success_count,
                        'error_count': self.error_count
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Statistics generation failed: {e}")
            return {'error': str(e), 'timestamp': time.time()}
            
    def _validate_interface_contract(self, harmonic_vector=None, voxel_field=None, 
                                   mood=None, time_step=None, k=None, threshold=None) -> bool:
        """Validate input parameters against interface contract"""
        try:
            # Validate harmonic_vector if provided
            if harmonic_vector is not None:
                if not isinstance(harmonic_vector, (list, tuple, np.ndarray)):
                    self.logger.warning("harmonic_vector must be a list, tuple, or numpy array")
                    return False
                if len(harmonic_vector) < 4:  # At least [R, G, B, intensity]
                    self.logger.warning("harmonic_vector must have at least 4 elements")
                    return False
                    
            # Validate voxel_field if provided
            if voxel_field is not None:
                if not isinstance(voxel_field, dict):
                    self.logger.warning("voxel_field must be a dictionary")
                    return False
                    
                for voxel_id, voxel_data in voxel_field.items():
                    if not isinstance(voxel_data, (list, tuple)) or len(voxel_data) != 2:
                        self.logger.warning(f"voxel_field[{voxel_id}] must be (vector, timestamp) tuple")
                        return False
                        
                    vector, timestamp = voxel_data
                    if not isinstance(vector, (list, tuple, np.ndarray)):
                        self.logger.warning(f"voxel vector for {voxel_id} must be a list, tuple, or numpy array")
                        return False
                    if not isinstance(timestamp, (int, float)):
                        self.logger.warning(f"timestamp for {voxel_id} must be a number")
                        return False
                        
            # Validate mood if provided
            if mood is not None and not isinstance(mood, str):
                self.logger.warning("mood must be a string")
                return False
                
            # Validate time_step if provided
            if time_step is not None and not isinstance(time_step, int):
                self.logger.warning("time_step must be an integer")
                return False
                
            # Validate k if provided
            if k is not None and (not isinstance(k, int) or k <= 0):
                self.logger.warning("k must be a positive integer")
                return False
                
            # Validate threshold if provided
            if threshold is not None and (not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1)):
                self.logger.warning("threshold must be a float between 0 and 1")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Interface contract validation failed: {e}")
            return False
            
    def _apply_mood_modulation(self, k: int, threshold: float, mood: str = None) -> Tuple[int, float]:
        """Apply mood-based modulation to parameters"""
        try:
            effective_mood = mood or self.current_mood
            
            if effective_mood in self.mood_modulation_factors:
                factors = self.mood_modulation_factors[effective_mood]
                
                # Modulate k
                modulated_k = int(k * factors['k_multiplier'])
                modulated_k = max(1, min(50, modulated_k))  # Reasonable bounds
                
                # Modulate threshold
                modulated_threshold = threshold + factors['threshold_modifier']
                modulated_threshold = max(0.0, min(1.0, modulated_threshold))
                
                return modulated_k, modulated_threshold
            else:
                return k, threshold
                
        except Exception as e:
            self.logger.error(f"Mood modulation failed: {e}")
            return k, threshold
            
    def _calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate similarity between two vectors"""
        try:
            # Ensure vectors are the same length
            min_len = min(len(vector1), len(vector2))
            v1 = vector1[:min_len]
            v2 = vector2[:min_len]
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure similarity is in [0, 1] range
            return max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {e}")
            return 0.0
            
    def _calculate_phase_factor(self, voxel_timestamp: float, time_step: int) -> float:
        """Calculate phase factor for temporal resonance"""
        try:
            # Calculate phase based on timestamp and time step
            phase = (voxel_timestamp + time_step) % (2 * np.pi)
            
            # Find best harmonic alignment
            best_alignment = 0.0
            for harmonic_freq in self.harmonic_frequencies:
                harmonic_phase = (phase * harmonic_freq) % (2 * np.pi)
                alignment = 1.0 - abs(harmonic_phase - np.pi) / np.pi
                best_alignment = max(best_alignment, alignment)
                
            # Convert alignment to factor
            return 0.5 + 0.5 * best_alignment
            
        except Exception:
            return 1.0  # Default factor if calculation fails
            
    def _calculate_age_factor(self, voxel_timestamp: float, current_time: float) -> float:
        """Calculate age-based decay factor"""
        try:
            age = current_time - voxel_timestamp
            
            # Apply exponential decay
            decay_factor = np.exp(-self.default_decay * age)
            
            return max(0.1, decay_factor)  # Minimum factor to prevent complete decay
            
        except Exception:
            return 1.0  # Default factor if calculation fails
            
    def _calculate_phase_alignment(self, vector1: np.ndarray, vector2: np.ndarray, 
                                 time_step: int) -> float:
        """Calculate phase alignment between vectors"""
        try:
            # Simple phase alignment based on vector angle and time step
            if len(vector1) >= 2 and len(vector2) >= 2:
                angle1 = np.arctan2(vector1[1], vector1[0])
                angle2 = np.arctan2(vector2[1], vector2[0])
                
                phase_diff = abs(angle1 - angle2)
                phase_diff = min(phase_diff, 2 * np.pi - phase_diff)  # Wrap around
                
                # Temporal modulation
                temporal_phase = (time_step * 0.1) % (2 * np.pi)
                phase_diff = abs(phase_diff - temporal_phase)
                
                # Convert to alignment factor
                alignment = 1.0 - phase_diff / np.pi
                return max(0.5, alignment)
            else:
                return 1.0
                
        except Exception:
            return 1.0
            
    def _analyze_harmonic_ratios(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Analyze harmonic ratios between vectors"""
        try:
            if len(vector1) < 4 or len(vector2) < 4:
                return 1.0
                
            # Calculate energy in different frequency bands
            energy1 = np.sum(vector1[:4] ** 2)
            energy2 = np.sum(vector2[:4] ** 2)
            
            if energy1 == 0 or energy2 == 0:
                return 1.0
                
            # Calculate ratio
            ratio = energy1 / energy2
            
            # Check if ratio matches harmonic frequencies
            best_match = 0.0
            for harmonic_freq in self.harmonic_frequencies:
                match_score = 1.0 / (1.0 + abs(ratio - harmonic_freq))
                best_match = max(best_match, match_score)
                
            return 0.8 + 0.2 * best_match  # Base factor + harmonic bonus
            
        except Exception:
            return 1.0
            
    def _generate_cache_key(self, query_vector: np.ndarray, k: int, threshold: float) -> str:
        """Generate cache key for query"""
        try:
            # Create hash from vector and parameters
            vector_hash = hash(tuple(query_vector.round(3)))  # Round for cache efficiency
            return f"{vector_hash}_{k}_{threshold:.3f}"
            
        except Exception:
            return f"default_{k}_{threshold:.3f}"
            
    def _log_resonance_activity(self, query_vector: np.ndarray, matches: List[Dict], 
                              time_step: int = None):
        """Log resonance activity for analysis"""
        try:
            current_time = time.time()
            
            # Log activation
            self.resonance_log["activations"].append({
                'timestamp': current_time,
                'query_vector_norm': np.linalg.norm(query_vector),
                'match_count': len(matches),
                'avg_similarity': np.mean([m['similarity'] for m in matches]) if matches else 0.0,
                'time_step': time_step or self.time_step
            })
            
            # Log harmonic differences
            if matches:
                harmonic_diffs = []
                for match in matches:
                    match_vector = np.array(match['voxel_vector'])
                    diff = np.linalg.norm(query_vector - match_vector[:len(query_vector)])
                    harmonic_diffs.append(diff)
                    
                self.resonance_log["harmonic_diffs"].append({
                    'timestamp': current_time,
                    'diffs': harmonic_diffs,
                    'avg_diff': np.mean(harmonic_diffs)
                })
                
            # Log time steps
            self.resonance_log["time_steps"].append({
                'timestamp': current_time,
                'time_step': time_step or self.time_step
            })
            
        except Exception as e:
            self.logger.error(f"Resonance activity logging failed: {e}")
            
    # Agharmonic Law Compliance Methods
    def harmonic_signature(self) -> Dict[str, Any]:
        """Establish frequency compatibility parameters"""
        return {
            "module": "topk_sparse_resonance",
            "input_frequency_range": self.input_frequency_range,
            "output_phase_alignment": self.output_phase_alignment,
            "resonance_threshold": self.resonance_threshold,
            "harmonic_modes": self.harmonic_modes,
            "compatible_modules": ["neuroengine", "resonance_field", "chord_resonator"],
            "harmonic_frequencies": self.harmonic_frequencies
        }
        
    def interface_contract(self) -> Dict[str, List[str]]:
        """Define interface contract for external modules"""
        return {
            "inputs": ["harmonic_vector", "voxel_field", "mood", "time_step", "k", "threshold"],
            "outputs": ["top_k_matches", "similarity_scores", "resonance_patterns"],
            "methods": [
                "find_top_k_matches",
                "update_voxel_resonance",
                "apply_sparse_activation",
                "calculate_harmonic_resonance"
            ]
        }
        
    def cognitive_energy_flow(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signals based on sparse activation patterns"""
        try:
            if signal.size == 0:
                return np.array([])
                
            # Apply sparse activation to signal
            sparse_threshold = 0.3
            sparse_mask = signal > sparse_threshold
            
            # Enhance strong signals, suppress weak ones
            enhanced_signal = np.where(
                sparse_mask,
                signal * 1.2,  # Enhance strong signals
                signal * 0.5   # Suppress weak signals
            )
            
            # Normalize to [0.05, 0.95] range
            signal_min = enhanced_signal.min()
            signal_max = enhanced_signal.max()
            
            if signal_max - signal_min == 0:
                return np.full_like(enhanced_signal, 0.5)
                
            normalized = (enhanced_signal - signal_min) / (signal_max - signal_min)
            return normalized * 0.9 + 0.05
            
        except Exception as e:
            self.logger.error(f"Cognitive energy flow failed: {e}")
            return signal
            
    def sync_clock(self, global_clock: Any = None) -> bool:
        """Connect to master temporal framework"""
        try:
            if global_clock and hasattr(global_clock, 'get_time'):
                synchronized_time = global_clock.get_time()
                self.last_sync = synchronized_time
                return True
            else:
                self.last_sync = time.time()
                return True
                
        except Exception as e:
            self.logger.error(f"Clock synchronization failed: {e}")
            return False
            
    def self_regulate(self) -> Dict[str, Any]:
        """Perform self-regulation and internal monitoring"""
        try:
            current_time = time.time()
            
            # Check if regulation is needed
            if current_time - self.last_regulation_time < self.regulation_interval:
                return {'status': 'regulation_not_needed'}
                
            regulation_actions = []
            
            # Clean up old cache entries
            if len(self.top_k_cache) > 50:
                # Remove oldest 25% of entries
                remove_count = len(self.top_k_cache) // 4
                oldest_keys = list(self.top_k_cache.keys())[:remove_count]
                for key in oldest_keys:
                    del self.top_k_cache[key]
                regulation_actions.append(f'cleaned_cache_{remove_count}_entries')
                
            # Apply temporal decay to resonance patterns
            decayed_count = 0
            patterns_to_remove = []
            
            for voxel_id, pattern in self.resonance_patterns.items():
                age = current_time - pattern['last_update']
                decay_amount = self.default_decay * age
                
                pattern['strength'] -= decay_amount
                
                if pattern['strength'] <= 0.01:
                    patterns_to_remove.append(voxel_id)
                else:
                    decayed_count += 1
                    
            # Remove weak patterns
            for voxel_id in patterns_to_remove:
                del self.resonance_patterns[voxel_id]
                
            if decayed_count > 0 or patterns_to_remove:
                regulation_actions.append(f'decayed_{decayed_count}_patterns_removed_{len(patterns_to_remove)}')
                
            # Update health metrics
            total_ops = self.success_count + self.error_count
            if total_ops > 0:
                self.resonance_chain_health = self.success_count / total_ops
            else:
                self.resonance_chain_health = 1.0
                
            self.last_regulation_time = current_time
            
            return {
                'status': 'regulation_completed',
                'actions': regulation_actions,
                'health': self.resonance_chain_health,
                'timestamp': current_time
            }
            
        except Exception as e:
            self.logger.error(f"Self-regulation failed: {e}")
            return {'status': 'regulation_failed', 'error': str(e)}
            
    def graceful_fallback(self) -> Dict[str, Any]:
        """Implement graceful fallback for sparse resonance failures"""
        try:
            self.fallback_mode = True
            self.fallback_level += 1
            
            fallback_actions = []
            
            if self.fallback_level == 1:
                # Level 1: Reduce complexity
                self.default_k = max(1, self.default_k // 2)
                self.default_threshold += 0.1
                fallback_actions.append('reduced_complexity')
                
            elif self.fallback_level == 2:
                # Level 2: Disable caching
                self.top_k_cache.clear()
                fallback_actions.append('disabled_caching')
                
            elif self.fallback_level >= 3:
                # Level 3: Minimal operation
                self.default_k = 1
                self.default_threshold = 0.8
                self.harmonic_frequencies = [1.0]  # Single frequency
                fallback_actions.append('minimal_operation_mode')
                
            return {
                'status': 'fallback_activated',
                'level': self.fallback_level,
                'actions': fallback_actions,
                'reduced_capabilities': [
                    'complex_harmonic_analysis',
                    'advanced_phase_calculations',
                    'large_k_queries'
                ],
                'maintained_capabilities': [
                    'basic_similarity_matching',
                    'simple_resonance_patterns',
                    'sparse_activation'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Graceful fallback failed: {e}")
            return {'status': 'fallback_failed', 'error': str(e)}
            
    def resonance_chain_validator(self, chain_data: Dict = None) -> bool:
        """Validate resonance chain integrity"""
        try:
            if chain_data is None:
                # Validate internal state
                if not self.resonance_patterns:
                    return True  # Empty state is valid
                    
                # Check pattern validity
                for voxel_id, pattern in self.resonance_patterns.items():
                    if not (0.0 <= pattern['strength'] <= 1.0):
                        return False
                    if pattern['last_update'] > time.time():
                        return False  # Future timestamp
                        
                return True
            else:
                # Validate external chain data
                required_fields = ['patterns', 'similarities']
                return all(field in chain_data for field in required_fields)
                
        except Exception as e:
            self.logger.error(f"Resonance chain validation failed: {e}")
            return False
            
    # Public Interface Methods
    def set_mood(self, mood: str) -> bool:
        """Set current mood for modulation"""
        try:
            if mood in self.mood_modulation_factors:
                self.current_mood = mood
                self.logger.debug(f"Mood set to: {mood}")
                return True
            else:
                self.logger.warning(f"Unknown mood: {mood}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to set mood: {e}")
            return False
            
    def clear_cache(self) -> bool:
        """Clear all cached data"""
        try:
            with self.lock:
                self.top_k_cache.clear()
                self.harmonic_cache.clear()
                self.cache_hits = 0
                self.cache_misses = 0
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False
            
    def reset_system(self) -> bool:
        """Reset the entire sparse resonance system"""
        try:
            with self.lock:
                self.activation_history.clear()
                self.resonance_patterns.clear()
                self.voxel_similarities.clear()
                self.top_k_cache.clear()
                self.harmonic_cache.clear()
                
                self.cache_hits = 0
                self.cache_misses = 0
                self.query_count = 0
                self.total_matches_found = 0
                self.avg_similarity_score = 0.0
                self.processing_times.clear()
                
                self.success_count = 0
                self.error_count = 0
                self.fallback_mode = False
                self.fallback_level = 0
                self.resonance_chain_health = 1.0
                
                self.time_step = 0
                self.current_mood = "neutral"
                
                # Clear logs
                for log_list in self.resonance_log.values():
                    log_list.clear()
                    
                self.logger.info("Sparse resonance system reset successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to reset system: {e}")
            return False


if __name__ == "__main__":
    # Test the CortexOS Top-K Sparse Resonance Engine
    print("🎯 Testing CortexOS Top-K Sparse Resonance Engine...")
    
    engine = CortexOSTopKSparseResonance()
    
    # Test engine startup
    if engine.start():
        print("✅ Top-K Sparse Resonance Engine started successfully")
    
    # Test sparse resonance processing with different moods
    test_moods = ["excited", "focused", "creative", "analytical", "neutral"]
    
    for mood in test_moods:
        engine.set_mood(mood)
        
        # Test data for each mood
        test_data = {
            "input_vector": [0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6],
            "context": f"test_context_{mood}",
            "priority": 1,
            "source": "test_neuron"
        }
        
        result = engine.process_sparse_resonance(test_data)
        if result:
            print(f"✅ Processed sparse resonance in {mood} mood - Top activations: {len(result['top_k_activations'])}")
    
    # Test similarity matching
    query_vector = [0.2, 0.7, 0.4, 0.8, 0.1, 0.6, 0.3, 0.9]
    similarities = engine.find_similar_patterns(query_vector, k=3)
    print(f"✅ Found {len(similarities)} similar patterns")
    
    # Test harmonic analysis
    harmonic_data = {
        "frequencies": [440.0, 880.0, 1320.0],
        "amplitudes": [0.8, 0.6, 0.4],
        "phases": [0.0, 0.1, 0.2]
    }
    
    harmonic_result = engine.analyze_harmonic_resonance(harmonic_data)
    if harmonic_result:
        print(f"✅ Harmonic analysis - Fundamental: {harmonic_result['fundamental_frequency']} Hz")
    
    # Test cache performance
    cache_stats = engine.get_cache_statistics()
    print(f"✅ Cache stats - Hits: {cache_stats['cache_hits']}, Misses: {cache_stats['cache_misses']}, Hit rate: {cache_stats['hit_rate']:.2f}")
    
    # Test optimization
    optimization_result = engine.optimize_sparse_processing()
    print(f"✅ Sparse processing optimization: {'Success' if optimization_result else 'Failed'}")
    
    # Test performance metrics
    performance = engine.get_performance_metrics()
    print(f"✅ Performance - Processed: {performance['total_processed']}, Success rate: {performance['success_rate']:.2f}")
    
    # Test mood influence analysis
    mood_analysis = engine.analyze_mood_influence()
    print(f"✅ Mood analysis - Current: {mood_analysis['current_mood']}, Influence: {mood_analysis['mood_influence']:.2f}")
    
    # Test engine status
    status = engine.get_status()
    print(f"✅ Engine status: {status['state']}, Active patterns: {status['active_patterns']}, Health: {status['system_health']:.2f}")
    
    # Test system reset
    if engine.reset_system():
        print("✅ Sparse resonance system reset successful")
    
    # Shutdown
    engine.stop()
    print("✅ Top-K Sparse Resonance Engine stopped")
    
    print("🎯 Top-K Sparse Resonance Engine test complete!")

