#!/usr/bin/env python3
"""
CortexOS Phase 6: Mood Modulator
Advanced cognitive mood and emotional state modulation system
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from enum import Enum
import random
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MoodState(Enum):
    """Primary mood states"""
    CALM = "calm"
    FOCUSED = "focused"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    ENERGETIC = "energetic"
    CONTEMPLATIVE = "contemplative"
    ALERT = "alert"
    RELAXED = "relaxed"

class EmotionalTone(Enum):
    """Emotional tone modifiers"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    OPTIMISTIC = "optimistic"
    CAUTIOUS = "cautious"
    CONFIDENT = "confident"
    CURIOUS = "curious"
    DETERMINED = "determined"

class ModulationTrigger(Enum):
    """Triggers for mood modulation"""
    PERFORMANCE_BASED = "performance_based"
    TIME_BASED = "time_based"
    CONTEXT_BASED = "context_based"
    USER_REQUESTED = "user_requested"
    ADAPTIVE = "adaptive"
    EMERGENCY = "emergency"

class ModulationIntensity(Enum):
    """Intensity levels for modulation"""
    SUBTLE = "subtle"
    MODERATE = "moderate"
    STRONG = "strong"
    INTENSE = "intense"

@dataclass
class MoodProfile:
    """Comprehensive mood profile"""
    profile_id: str
    name: str
    primary_mood: MoodState
    emotional_tone: EmotionalTone
    intensity: float  # 0.0 to 1.0
    stability: float  # 0.0 to 1.0 (resistance to change)
    adaptability: float  # 0.0 to 1.0 (willingness to change)
    cognitive_bias: Dict[str, float] = field(default_factory=dict)
    behavioral_modifiers: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ModulationRequest:
    """Request for mood modulation"""
    request_id: str
    target_mood: MoodState
    target_tone: EmotionalTone
    trigger: ModulationTrigger
    intensity: ModulationIntensity
    duration_minutes: Optional[int] = None
    reason: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1-10, 10 being highest
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ModulationResult:
    """Result of mood modulation"""
    result_id: str
    request_id: str
    success: bool
    previous_profile: MoodProfile
    new_profile: MoodProfile
    modulation_strength: float
    side_effects: List[str] = field(default_factory=list)
    duration_actual: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CognitiveState:
    """Current cognitive state"""
    attention_level: float  # 0.0 to 1.0
    processing_speed: float  # 0.0 to 1.0
    memory_efficiency: float  # 0.0 to 1.0
    creativity_index: float  # 0.0 to 1.0
    analytical_depth: float  # 0.0 to 1.0
    emotional_stability: float  # 0.0 to 1.0
    stress_level: float  # 0.0 to 1.0
    fatigue_level: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)

class MoodStateEngine:
    """Core mood state management engine"""
    
    def __init__(self):
        self.mood_transitions = self._initialize_mood_transitions()
        self.mood_effects = self._initialize_mood_effects()
        self.tone_modifiers = self._initialize_tone_modifiers()
    
    def _initialize_mood_transitions(self) -> Dict[MoodState, Dict[MoodState, float]]:
        """Initialize mood transition probabilities"""
        transitions = {}
        
        # Define transition probabilities between mood states
        transitions[MoodState.CALM] = {
            MoodState.FOCUSED: 0.8,
            MoodState.CONTEMPLATIVE: 0.9,
            MoodState.RELAXED: 0.9,
            MoodState.CREATIVE: 0.6,
            MoodState.ANALYTICAL: 0.7,
            MoodState.ENERGETIC: 0.4,
            MoodState.ALERT: 0.5
        }
        
        transitions[MoodState.FOCUSED] = {
            MoodState.ANALYTICAL: 0.9,
            MoodState.ALERT: 0.8,
            MoodState.CALM: 0.7,
            MoodState.ENERGETIC: 0.6,
            MoodState.CREATIVE: 0.5,
            MoodState.CONTEMPLATIVE: 0.6,
            MoodState.RELAXED: 0.4
        }
        
        transitions[MoodState.CREATIVE] = {
            MoodState.ENERGETIC: 0.8,
            MoodState.CONTEMPLATIVE: 0.7,
            MoodState.FOCUSED: 0.6,
            MoodState.CALM: 0.7,
            MoodState.ANALYTICAL: 0.4,
            MoodState.ALERT: 0.5,
            MoodState.RELAXED: 0.6
        }
        
        transitions[MoodState.ANALYTICAL] = {
            MoodState.FOCUSED: 0.9,
            MoodState.CONTEMPLATIVE: 0.8,
            MoodState.ALERT: 0.7,
            MoodState.CALM: 0.6,
            MoodState.CREATIVE: 0.5,
            MoodState.ENERGETIC: 0.4,
            MoodState.RELAXED: 0.3
        }
        
        transitions[MoodState.ENERGETIC] = {
            MoodState.ALERT: 0.8,
            MoodState.CREATIVE: 0.7,
            MoodState.FOCUSED: 0.6,
            MoodState.ANALYTICAL: 0.5,
            MoodState.CALM: 0.4,
            MoodState.CONTEMPLATIVE: 0.3,
            MoodState.RELAXED: 0.2
        }
        
        transitions[MoodState.CONTEMPLATIVE] = {
            MoodState.CALM: 0.8,
            MoodState.CREATIVE: 0.7,
            MoodState.ANALYTICAL: 0.8,
            MoodState.FOCUSED: 0.6,
            MoodState.RELAXED: 0.7,
            MoodState.ENERGETIC: 0.3,
            MoodState.ALERT: 0.4
        }
        
        transitions[MoodState.ALERT] = {
            MoodState.FOCUSED: 0.8,
            MoodState.ENERGETIC: 0.7,
            MoodState.ANALYTICAL: 0.7,
            MoodState.CREATIVE: 0.5,
            MoodState.CALM: 0.5,
            MoodState.CONTEMPLATIVE: 0.4,
            MoodState.RELAXED: 0.3
        }
        
        transitions[MoodState.RELAXED] = {
            MoodState.CALM: 0.9,
            MoodState.CONTEMPLATIVE: 0.8,
            MoodState.CREATIVE: 0.6,
            MoodState.FOCUSED: 0.4,
            MoodState.ANALYTICAL: 0.3,
            MoodState.ENERGETIC: 0.2,
            MoodState.ALERT: 0.2
        }
        
        return transitions
    
    def _initialize_mood_effects(self) -> Dict[MoodState, Dict[str, float]]:
        """Initialize mood effects on cognitive functions"""
        effects = {}
        
        effects[MoodState.CALM] = {
            'attention_level': 0.7,
            'processing_speed': 0.6,
            'memory_efficiency': 0.8,
            'creativity_index': 0.6,
            'analytical_depth': 0.7,
            'emotional_stability': 0.9,
            'stress_level': 0.2,
            'fatigue_level': 0.3
        }
        
        effects[MoodState.FOCUSED] = {
            'attention_level': 0.9,
            'processing_speed': 0.8,
            'memory_efficiency': 0.8,
            'creativity_index': 0.5,
            'analytical_depth': 0.9,
            'emotional_stability': 0.7,
            'stress_level': 0.4,
            'fatigue_level': 0.5
        }
        
        effects[MoodState.CREATIVE] = {
            'attention_level': 0.6,
            'processing_speed': 0.7,
            'memory_efficiency': 0.6,
            'creativity_index': 0.9,
            'analytical_depth': 0.5,
            'emotional_stability': 0.6,
            'stress_level': 0.3,
            'fatigue_level': 0.4
        }
        
        effects[MoodState.ANALYTICAL] = {
            'attention_level': 0.8,
            'processing_speed': 0.7,
            'memory_efficiency': 0.9,
            'creativity_index': 0.4,
            'analytical_depth': 0.9,
            'emotional_stability': 0.8,
            'stress_level': 0.5,
            'fatigue_level': 0.6
        }
        
        effects[MoodState.ENERGETIC] = {
            'attention_level': 0.8,
            'processing_speed': 0.9,
            'memory_efficiency': 0.7,
            'creativity_index': 0.8,
            'analytical_depth': 0.6,
            'emotional_stability': 0.5,
            'stress_level': 0.6,
            'fatigue_level': 0.2
        }
        
        effects[MoodState.CONTEMPLATIVE] = {
            'attention_level': 0.7,
            'processing_speed': 0.5,
            'memory_efficiency': 0.8,
            'creativity_index': 0.7,
            'analytical_depth': 0.8,
            'emotional_stability': 0.8,
            'stress_level': 0.3,
            'fatigue_level': 0.4
        }
        
        effects[MoodState.ALERT] = {
            'attention_level': 0.9,
            'processing_speed': 0.8,
            'memory_efficiency': 0.7,
            'creativity_index': 0.6,
            'analytical_depth': 0.8,
            'emotional_stability': 0.6,
            'stress_level': 0.7,
            'fatigue_level': 0.5
        }
        
        effects[MoodState.RELAXED] = {
            'attention_level': 0.5,
            'processing_speed': 0.4,
            'memory_efficiency': 0.6,
            'creativity_index': 0.7,
            'analytical_depth': 0.4,
            'emotional_stability': 0.9,
            'stress_level': 0.1,
            'fatigue_level': 0.2
        }
        
        return effects
    
    def _initialize_tone_modifiers(self) -> Dict[EmotionalTone, Dict[str, float]]:
        """Initialize emotional tone modifiers"""
        modifiers = {}
        
        modifiers[EmotionalTone.POSITIVE] = {
            'creativity_index': 1.2,
            'emotional_stability': 1.1,
            'stress_level': 0.8,
            'processing_speed': 1.1
        }
        
        modifiers[EmotionalTone.NEUTRAL] = {
            'attention_level': 1.0,
            'processing_speed': 1.0,
            'analytical_depth': 1.0,
            'emotional_stability': 1.0
        }
        
        modifiers[EmotionalTone.NEGATIVE] = {
            'analytical_depth': 1.1,
            'attention_level': 0.9,
            'stress_level': 1.3,
            'emotional_stability': 0.8
        }
        
        modifiers[EmotionalTone.OPTIMISTIC] = {
            'creativity_index': 1.3,
            'processing_speed': 1.1,
            'stress_level': 0.7,
            'fatigue_level': 0.8
        }
        
        modifiers[EmotionalTone.CAUTIOUS] = {
            'analytical_depth': 1.2,
            'attention_level': 1.1,
            'processing_speed': 0.9,
            'stress_level': 1.1
        }
        
        modifiers[EmotionalTone.CONFIDENT] = {
            'processing_speed': 1.2,
            'creativity_index': 1.1,
            'emotional_stability': 1.2,
            'stress_level': 0.7
        }
        
        modifiers[EmotionalTone.CURIOUS] = {
            'attention_level': 1.2,
            'creativity_index': 1.2,
            'analytical_depth': 1.1,
            'fatigue_level': 0.8
        }
        
        modifiers[EmotionalTone.DETERMINED] = {
            'attention_level': 1.3,
            'processing_speed': 1.1,
            'analytical_depth': 1.1,
            'stress_level': 1.0
        }
        
        return modifiers
    
    def calculate_transition_probability(self, current_mood: MoodState, target_mood: MoodState, 
                                       current_stability: float) -> float:
        """Calculate probability of successful mood transition"""
        if current_mood == target_mood:
            return 1.0
        
        base_probability = self.mood_transitions.get(current_mood, {}).get(target_mood, 0.3)
        stability_factor = 1.0 - current_stability
        
        return base_probability * stability_factor
    
    def apply_mood_effects(self, mood: MoodState, tone: EmotionalTone, 
                          base_cognitive_state: CognitiveState) -> CognitiveState:
        """Apply mood and tone effects to cognitive state"""
        mood_effects = self.mood_effects.get(mood, {})
        tone_modifiers = self.tone_modifiers.get(tone, {})
        
        new_state = CognitiveState()
        
        # Apply mood effects
        new_state.attention_level = mood_effects.get('attention_level', 0.5)
        new_state.processing_speed = mood_effects.get('processing_speed', 0.5)
        new_state.memory_efficiency = mood_effects.get('memory_efficiency', 0.5)
        new_state.creativity_index = mood_effects.get('creativity_index', 0.5)
        new_state.analytical_depth = mood_effects.get('analytical_depth', 0.5)
        new_state.emotional_stability = mood_effects.get('emotional_stability', 0.5)
        new_state.stress_level = mood_effects.get('stress_level', 0.5)
        new_state.fatigue_level = mood_effects.get('fatigue_level', 0.5)
        
        # Apply tone modifiers
        for attribute, modifier in tone_modifiers.items():
            if hasattr(new_state, attribute):
                current_value = getattr(new_state, attribute)
                modified_value = min(1.0, max(0.0, current_value * modifier))
                setattr(new_state, attribute, modified_value)
        
        return new_state

class AdaptiveMoodController:
    """Adaptive mood control system"""
    
    def __init__(self):
        self.learning_rate = 0.1
        self.adaptation_history = deque(maxlen=1000)
        self.performance_correlations = defaultdict(list)
        self.optimal_moods = {}
    
    def learn_from_performance(self, mood_profile: MoodProfile, performance_metrics: Dict[str, float]):
        """Learn optimal moods from performance feedback"""
        try:
            # Record performance correlation
            correlation_entry = {
                'mood': mood_profile.primary_mood,
                'tone': mood_profile.emotional_tone,
                'intensity': mood_profile.intensity,
                'performance': performance_metrics,
                'timestamp': datetime.now()
            }
            
            self.adaptation_history.append(correlation_entry)
            
            # Update performance correlations
            mood_key = f"{mood_profile.primary_mood.value}_{mood_profile.emotional_tone.value}"
            self.performance_correlations[mood_key].append(performance_metrics)
            
            # Calculate optimal mood for different tasks
            self._update_optimal_moods()
            
        except Exception as e:
            logger.error(f"Error learning from performance: {e}")
    
    def _update_optimal_moods(self):
        """Update optimal mood recommendations based on learning"""
        try:
            for mood_key, performance_list in self.performance_correlations.items():
                if len(performance_list) >= 5:  # Minimum samples for learning
                    # Calculate average performance for this mood
                    avg_performance = {}
                    for metric in performance_list[0].keys():
                        values = [p[metric] for p in performance_list if metric in p]
                        avg_performance[metric] = sum(values) / len(values) if values else 0.0
                    
                    self.optimal_moods[mood_key] = avg_performance
            
        except Exception as e:
            logger.error(f"Error updating optimal moods: {e}")
    
    def recommend_mood_for_task(self, task_type: str, required_capabilities: List[str]) -> Tuple[MoodState, EmotionalTone]:
        """Recommend optimal mood for specific task"""
        try:
            best_mood = MoodState.FOCUSED
            best_tone = EmotionalTone.NEUTRAL
            best_score = 0.0
            
            # Evaluate each learned mood combination
            for mood_key, performance in self.optimal_moods.items():
                mood_str, tone_str = mood_key.split('_')
                
                # Calculate score based on required capabilities
                score = 0.0
                for capability in required_capabilities:
                    if capability in performance:
                        score += performance[capability]
                
                if score > best_score:
                    best_score = score
                    best_mood = MoodState(mood_str)
                    best_tone = EmotionalTone(tone_str)
            
            return best_mood, best_tone
            
        except Exception as e:
            logger.error(f"Error recommending mood for task: {e}")
            return MoodState.FOCUSED, EmotionalTone.NEUTRAL
    
    def adapt_modulation_strategy(self, recent_results: List[ModulationResult]) -> Dict[str, float]:
        """Adapt modulation strategy based on recent results"""
        try:
            if not recent_results:
                return {}
            
            # Analyze success rates
            successful_modulations = [r for r in recent_results if r.success]
            success_rate = len(successful_modulations) / len(recent_results)
            
            # Analyze modulation strengths
            avg_strength = sum(r.modulation_strength for r in successful_modulations) / len(successful_modulations) if successful_modulations else 0.5
            
            # Adapt strategy
            adaptations = {
                'success_rate': success_rate,
                'recommended_strength': avg_strength,
                'learning_confidence': min(len(recent_results) / 100.0, 1.0)
            }
            
            return adaptations
            
        except Exception as e:
            logger.error(f"Error adapting modulation strategy: {e}")
            return {}

class MoodModulator:
    """Advanced cognitive mood and emotional state modulation system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.mood_engine = MoodStateEngine()
        self.adaptive_controller = AdaptiveMoodController()
        
        # Current state
        self.current_profile = MoodProfile(
            profile_id="default",
            name="Default Profile",
            primary_mood=MoodState.CALM,
            emotional_tone=EmotionalTone.NEUTRAL,
            intensity=0.5,
            stability=0.7,
            adaptability=0.6
        )
        
        self.current_cognitive_state = CognitiveState()
        
        # Modulation history and queue
        self.modulation_history = deque(maxlen=1000)
        self.modulation_queue = asyncio.Queue()
        self.active_modulations = {}
        
        # Configuration
        self.enable_adaptive_learning = self.config.get('enable_adaptive_learning', True)
        self.modulation_interval = self.config.get('modulation_interval', 10)  # seconds
        self.max_concurrent_modulations = self.config.get('max_concurrent_modulations', 3)
        self.safety_limits = self.config.get('safety_limits', {
            'max_intensity': 0.9,
            'min_stability': 0.1,
            'max_stress_level': 0.8
        })
        
        # State
        self.running = False
        self.modulation_task = None
        self.monitoring_task = None
        
        logger.info("Mood Modulator initialized")
    
    async def start(self):
        """Start mood modulation system"""
        try:
            self.running = True
            
            # Start modulation processing task
            self.modulation_task = asyncio.create_task(self._modulation_loop())
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Initialize cognitive state
            self.current_cognitive_state = self.mood_engine.apply_mood_effects(
                self.current_profile.primary_mood,
                self.current_profile.emotional_tone,
                self.current_cognitive_state
            )
            
            logger.info("Mood Modulator started")
            
        except Exception as e:
            logger.error(f"Error starting Mood Modulator: {e}")
            raise
    
    async def stop(self):
        """Stop mood modulation system"""
        try:
            self.running = False
            
            # Cancel tasks
            if self.modulation_task:
                self.modulation_task.cancel()
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            # Wait for tasks to complete
            tasks = [t for t in [self.modulation_task, self.monitoring_task] if t]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.info("Mood Modulator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Mood Modulator: {e}")
    
    async def request_modulation(self, request: ModulationRequest) -> str:
        """Request mood modulation"""
        try:
            # Validate request
            if not self._validate_modulation_request(request):
                raise ValueError("Invalid modulation request")
            
            # Add to queue
            await self.modulation_queue.put(request)
            
            logger.info(f"Modulation requested: {request.request_id} - {request.target_mood.value}")
            return request.request_id
            
        except Exception as e:
            logger.error(f"Error requesting modulation: {e}")
            raise
    
    def _validate_modulation_request(self, request: ModulationRequest) -> bool:
        """Validate modulation request against safety limits"""
        try:
            # Check intensity limits
            intensity_map = {
                ModulationIntensity.SUBTLE: 0.3,
                ModulationIntensity.MODERATE: 0.5,
                ModulationIntensity.STRONG: 0.7,
                ModulationIntensity.INTENSE: 0.9
            }
            
            requested_intensity = intensity_map.get(request.intensity, 0.5)
            if requested_intensity > self.safety_limits.get('max_intensity', 0.9):
                logger.warning(f"Modulation intensity too high: {requested_intensity}")
                return False
            
            # Check for conflicting active modulations
            if len(self.active_modulations) >= self.max_concurrent_modulations:
                logger.warning("Too many concurrent modulations")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating modulation request: {e}")
            return False
    
    async def _modulation_loop(self):
        """Main modulation processing loop"""
        logger.info("Modulation loop started")
        
        while self.running:
            try:
                # Process modulation requests
                try:
                    request = await asyncio.wait_for(self.modulation_queue.get(), timeout=1.0)
                    await self._process_modulation_request(request)
                except asyncio.TimeoutError:
                    pass
                
                # Update active modulations
                await self._update_active_modulations()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in modulation loop: {e}")
                await asyncio.sleep(1)
        
        logger.info("Modulation loop stopped")
    
    async def _process_modulation_request(self, request: ModulationRequest):
        """Process individual modulation request"""
        try:
            # Calculate transition probability
            transition_prob = self.mood_engine.calculate_transition_probability(
                self.current_profile.primary_mood,
                request.target_mood,
                self.current_profile.stability
            )
            
            # Apply modulation intensity
            intensity_map = {
                ModulationIntensity.SUBTLE: 0.3,
                ModulationIntensity.MODERATE: 0.5,
                ModulationIntensity.STRONG: 0.7,
                ModulationIntensity.INTENSE: 0.9
            }
            
            modulation_strength = intensity_map.get(request.intensity, 0.5) * transition_prob
            
            # Check if modulation succeeds
            success_threshold = 0.3 + (self.current_profile.adaptability * 0.4)
            success = modulation_strength >= success_threshold
            
            if success:
                # Create new mood profile
                new_profile = MoodProfile(
                    profile_id=f"modulated_{int(time.time())}",
                    name=f"Modulated to {request.target_mood.value}",
                    primary_mood=request.target_mood,
                    emotional_tone=request.target_tone,
                    intensity=min(1.0, self.current_profile.intensity + modulation_strength * 0.5),
                    stability=max(0.1, self.current_profile.stability - modulation_strength * 0.2),
                    adaptability=self.current_profile.adaptability
                )
                
                # Apply mood effects to cognitive state
                new_cognitive_state = self.mood_engine.apply_mood_effects(
                    new_profile.primary_mood,
                    new_profile.emotional_tone,
                    self.current_cognitive_state
                )
                
                # Create modulation result
                result = ModulationResult(
                    result_id=f"result_{request.request_id}",
                    request_id=request.request_id,
                    success=True,
                    previous_profile=self.current_profile,
                    new_profile=new_profile,
                    modulation_strength=modulation_strength
                )
                
                # Update current state
                self.current_profile = new_profile
                self.current_cognitive_state = new_cognitive_state
                
                # Add to active modulations if duration specified
                if request.duration_minutes:
                    self.active_modulations[request.request_id] = {
                        'request': request,
                        'result': result,
                        'start_time': time.time(),
                        'duration': request.duration_minutes * 60
                    }
                
                logger.info(f"Modulation successful: {request.request_id}")
                
            else:
                # Modulation failed
                result = ModulationResult(
                    result_id=f"result_{request.request_id}",
                    request_id=request.request_id,
                    success=False,
                    previous_profile=self.current_profile,
                    new_profile=self.current_profile,
                    modulation_strength=modulation_strength,
                    side_effects=["Insufficient transition probability"]
                )
                
                logger.warning(f"Modulation failed: {request.request_id}")
            
            # Store result
            self.modulation_history.append(result)
            
        except Exception as e:
            logger.error(f"Error processing modulation request: {e}")
    
    async def _update_active_modulations(self):
        """Update and expire active modulations"""
        try:
            current_time = time.time()
            expired_modulations = []
            
            for request_id, modulation in self.active_modulations.items():
                elapsed_time = current_time - modulation['start_time']
                
                if elapsed_time >= modulation['duration']:
                    expired_modulations.append(request_id)
            
            # Expire modulations
            for request_id in expired_modulations:
                modulation = self.active_modulations[request_id]
                
                # Revert to previous state (simplified)
                previous_profile = modulation['result'].previous_profile
                
                # Create reversion profile (gradual return)
                reversion_profile = MoodProfile(
                    profile_id=f"reversion_{int(time.time())}",
                    name="Reversion Profile",
                    primary_mood=previous_profile.primary_mood,
                    emotional_tone=previous_profile.emotional_tone,
                    intensity=previous_profile.intensity * 0.8 + self.current_profile.intensity * 0.2,
                    stability=previous_profile.stability,
                    adaptability=previous_profile.adaptability
                )
                
                self.current_profile = reversion_profile
                
                # Update cognitive state
                self.current_cognitive_state = self.mood_engine.apply_mood_effects(
                    reversion_profile.primary_mood,
                    reversion_profile.emotional_tone,
                    self.current_cognitive_state
                )
                
                del self.active_modulations[request_id]
                logger.info(f"Modulation expired and reverted: {request_id}")
            
        except Exception as e:
            logger.error(f"Error updating active modulations: {e}")
    
    async def _monitoring_loop(self):
        """Monitoring and adaptive learning loop"""
        logger.info("Monitoring loop started")
        
        while self.running:
            try:
                # Perform adaptive learning if enabled
                if self.enable_adaptive_learning:
                    await self._perform_adaptive_learning()
                
                # Monitor cognitive state health
                await self._monitor_cognitive_health()
                
                await asyncio.sleep(self.modulation_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
        
        logger.info("Monitoring loop stopped")
    
    async def _perform_adaptive_learning(self):
        """Perform adaptive learning from recent modulations"""
        try:
            # Get recent modulation results
            recent_results = list(self.modulation_history)[-10:] if self.modulation_history else []
            
            if recent_results:
                # Analyze performance (simplified)
                performance_metrics = {
                    'attention_level': self.current_cognitive_state.attention_level,
                    'processing_speed': self.current_cognitive_state.processing_speed,
                    'creativity_index': self.current_cognitive_state.creativity_index,
                    'analytical_depth': self.current_cognitive_state.analytical_depth,
                    'emotional_stability': self.current_cognitive_state.emotional_stability
                }
                
                # Learn from performance
                self.adaptive_controller.learn_from_performance(self.current_profile, performance_metrics)
                
                # Adapt modulation strategy
                adaptations = self.adaptive_controller.adapt_modulation_strategy(recent_results)
                
                if adaptations:
                    logger.debug(f"Adaptive learning: {adaptations}")
            
        except Exception as e:
            logger.error(f"Error in adaptive learning: {e}")
    
    async def _monitor_cognitive_health(self):
        """Monitor cognitive state for health issues"""
        try:
            # Check for unhealthy states
            warnings = []
            
            if self.current_cognitive_state.stress_level > self.safety_limits.get('max_stress_level', 0.8):
                warnings.append("High stress level detected")
            
            if self.current_cognitive_state.fatigue_level > 0.8:
                warnings.append("High fatigue level detected")
            
            if self.current_cognitive_state.emotional_stability < 0.3:
                warnings.append("Low emotional stability detected")
            
            if warnings:
                logger.warning(f"Cognitive health warnings: {warnings}")
                
                # Auto-suggest calming modulation
                if self.current_profile.primary_mood != MoodState.CALM:
                    calming_request = ModulationRequest(
                        request_id=f"auto_calm_{int(time.time())}",
                        target_mood=MoodState.CALM,
                        target_tone=EmotionalTone.POSITIVE,
                        trigger=ModulationTrigger.EMERGENCY,
                        intensity=ModulationIntensity.MODERATE,
                        duration_minutes=30,
                        reason="Automatic health intervention"
                    )
                    
                    await self.request_modulation(calming_request)
            
        except Exception as e:
            logger.error(f"Error monitoring cognitive health: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current mood and cognitive state"""
        return {
            'mood_profile': {
                'primary_mood': self.current_profile.primary_mood.value,
                'emotional_tone': self.current_profile.emotional_tone.value,
                'intensity': self.current_profile.intensity,
                'stability': self.current_profile.stability,
                'adaptability': self.current_profile.adaptability
            },
            'cognitive_state': {
                'attention_level': self.current_cognitive_state.attention_level,
                'processing_speed': self.current_cognitive_state.processing_speed,
                'memory_efficiency': self.current_cognitive_state.memory_efficiency,
                'creativity_index': self.current_cognitive_state.creativity_index,
                'analytical_depth': self.current_cognitive_state.analytical_depth,
                'emotional_stability': self.current_cognitive_state.emotional_stability,
                'stress_level': self.current_cognitive_state.stress_level,
                'fatigue_level': self.current_cognitive_state.fatigue_level
            },
            'active_modulations': len(self.active_modulations),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_modulation_history(self, hours: int = 24) -> List[ModulationResult]:
        """Get modulation history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [result for result in self.modulation_history if result.timestamp >= cutoff_time]
    
    def recommend_mood_for_task(self, task_type: str, capabilities: List[str]) -> Tuple[MoodState, EmotionalTone]:
        """Get mood recommendation for specific task"""
        return self.adaptive_controller.recommend_mood_for_task(task_type, capabilities)
    
    def get_status(self) -> Dict[str, Any]:
        """Get mood modulator status"""
        return {
            'running': self.running,
            'current_mood': self.current_profile.primary_mood.value,
            'current_tone': self.current_profile.emotional_tone.value,
            'modulation_queue_size': self.modulation_queue.qsize(),
            'active_modulations': len(self.active_modulations),
            'modulation_history_size': len(self.modulation_history),
            'enable_adaptive_learning': self.enable_adaptive_learning,
            'learned_mood_patterns': len(self.adaptive_controller.optimal_moods),
            'cognitive_health_score': (
                self.current_cognitive_state.emotional_stability * 0.3 +
                (1.0 - self.current_cognitive_state.stress_level) * 0.3 +
                (1.0 - self.current_cognitive_state.fatigue_level) * 0.2 +
                self.current_cognitive_state.attention_level * 0.2
            )
        }

# Test and demonstration
async def test_mood_modulator():
    """Test the mood modulator system"""
    print("🧠 Testing CortexOS Mood Modulator...")
    
    # Create configuration
    config = {
        'enable_adaptive_learning': True,
        'modulation_interval': 5,  # 5 seconds for testing
        'max_concurrent_modulations': 2,
        'safety_limits': {
            'max_intensity': 0.8,
            'min_stability': 0.2,
            'max_stress_level': 0.7
        }
    }
    
    # Initialize mood modulator
    modulator = MoodModulator(config)
    
    try:
        # Start modulator
        await modulator.start()
        print("✅ Mood Modulator started")
        
        # Display initial state
        initial_state = modulator.get_current_state()
        print(f"\n🧠 Initial State:")
        print(f"   Mood: {initial_state['mood_profile']['primary_mood']}")
        print(f"   Tone: {initial_state['mood_profile']['emotional_tone']}")
        print(f"   Attention: {initial_state['cognitive_state']['attention_level']:.3f}")
        print(f"   Creativity: {initial_state['cognitive_state']['creativity_index']:.3f}")
        print(f"   Stress: {initial_state['cognitive_state']['stress_level']:.3f}")
        
        # Request mood modulations
        print(f"\n🎯 Requesting mood modulations...")
        
        # Request creative mood
        creative_request = ModulationRequest(
            request_id="test_creative_001",
            target_mood=MoodState.CREATIVE,
            target_tone=EmotionalTone.OPTIMISTIC,
            trigger=ModulationTrigger.USER_REQUESTED,
            intensity=ModulationIntensity.MODERATE,
            duration_minutes=2,  # 2 minutes for testing
            reason="Testing creative modulation"
        )
        
        await modulator.request_modulation(creative_request)
        
        # Wait for modulation to process
        await asyncio.sleep(3)
        
        # Check state after creative modulation
        creative_state = modulator.get_current_state()
        print(f"\n🎨 After Creative Modulation:")
        print(f"   Mood: {creative_state['mood_profile']['primary_mood']}")
        print(f"   Tone: {creative_state['mood_profile']['emotional_tone']}")
        print(f"   Creativity: {creative_state['cognitive_state']['creativity_index']:.3f}")
        print(f"   Processing Speed: {creative_state['cognitive_state']['processing_speed']:.3f}")
        
        # Request analytical mood
        analytical_request = ModulationRequest(
            request_id="test_analytical_001",
            target_mood=MoodState.ANALYTICAL,
            target_tone=EmotionalTone.CAUTIOUS,
            trigger=ModulationTrigger.PERFORMANCE_BASED,
            intensity=ModulationIntensity.STRONG,
            duration_minutes=1,
            reason="Testing analytical modulation"
        )
        
        await modulator.request_modulation(analytical_request)
        
        # Wait for modulation
        await asyncio.sleep(3)
        
        # Check state after analytical modulation
        analytical_state = modulator.get_current_state()
        print(f"\n🔬 After Analytical Modulation:")
        print(f"   Mood: {analytical_state['mood_profile']['primary_mood']}")
        print(f"   Tone: {analytical_state['mood_profile']['emotional_tone']}")
        print(f"   Analytical Depth: {analytical_state['cognitive_state']['analytical_depth']:.3f}")
        print(f"   Memory Efficiency: {analytical_state['cognitive_state']['memory_efficiency']:.3f}")
        
        # Test mood recommendation
        print(f"\n💡 Testing mood recommendations...")
        recommended_mood, recommended_tone = modulator.recommend_mood_for_task(
            "creative_writing", 
            ["creativity_index", "processing_speed"]
        )
        print(f"   Recommended for creative writing: {recommended_mood.value} + {recommended_tone.value}")
        
        # Wait for modulations to expire
        print(f"\n⏳ Waiting for modulations to expire...")
        await asyncio.sleep(70)  # Wait for expiration
        
        # Check final state
        final_state = modulator.get_current_state()
        print(f"\n🔄 After Modulation Expiration:")
        print(f"   Mood: {final_state['mood_profile']['primary_mood']}")
        print(f"   Active Modulations: {final_state['active_modulations']}")
        
        # Display modulation history
        history = modulator.get_modulation_history(1)
        print(f"\n📚 Modulation History: {len(history)} modulations")
        for result in history:
            status = "✅" if result.success else "❌"
            print(f"   {status} {result.request_id}: {result.modulation_strength:.3f} strength")
        
        # Display system status
        print(f"\n🔧 System Status:")
        status = modulator.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        print("\n✅ Mood Modulator test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        await modulator.stop()

if __name__ == "__main__":
    asyncio.run(test_mood_modulator())

