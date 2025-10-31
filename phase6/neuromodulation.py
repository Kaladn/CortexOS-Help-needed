"""
Neuromodulation module for CortexOS.
Dynamically adjusts resonance parameters based on mood states.
"""
class Neuromodulator:
    """
    Manages mood states to modulate resonance behavior in CortexOS.
    Adjusts k, threshold, decay, phase_step, and base_frequency for TopKSparseResonance.
    """
    def __init__(self):
        self.mood_configs = {
            "neutral": {"k": 10, "threshold": 0.85, "decay": 0.20, "phase_step": 0.1, "base_frequency": 1.0},
            "excited": {"k": 15, "threshold": 0.80, "decay": 0.10, "phase_step": 0.05, "base_frequency": 2.0},
            "focused": {"k": 5, "threshold": 0.90, "decay": 0.50, "phase_step": 0.01, "base_frequency": 1.5},
            "cautious": {"k": 8, "threshold": 0.95, "decay": 0.30, "phase_step": 0.02, "base_frequency": 1.2},
            "curious": {"k": 20, "threshold": 0.75, "decay": 0.05, "phase_step": 0.15, "base_frequency": 0.8},
            "alert": {"k": 12, "threshold": 0.88, "decay": 0.15, "phase_step": 0.05, "base_frequency": 1.3},
            "fatigued": {"k": 6, "threshold": 0.70, "decay": 0.05, "phase_step": 0.3, "base_frequency": 0.5},
            "paranoid": {"k": 4, "threshold": 0.98, "decay": 0.40, "phase_step": 0.01, "base_frequency": 1.8},
            "imaginative": {"k": 25, "threshold": 0.70, "decay": 0.03, "phase_step": 0.25, "base_frequency": 0.7},
            "reflective": {"k": 8, "threshold": 0.82, "decay": 0.25, "phase_step": 0.15, "base_frequency": 0.9},
            "impulsive": {"k": 18, "threshold": 0.65, "decay": 0.10, "phase_step": 0.08, "base_frequency": 1.4},
            "dreamy": {"k": 16, "threshold": 0.65, "decay": 0.01, "phase_step": 0.5, "base_frequency": 0.3},
            "aggressive": {"k": 12, "threshold": 0.80, "decay": 0.05, "phase_step": 0.06, "base_frequency": 1.6},
            "calm": {"k": 7, "threshold": 0.88, "decay": 0.35, "phase_step": 0.2, "base_frequency": 0.6},
            "anxious": {"k": 14, "threshold": 0.92, "decay": 0.50, "phase_step": 0.03, "base_frequency": 2.0},
            "playful": {"k": 13, "threshold": 0.78, "decay": 0.08, "phase_step": 0.1, "base_frequency": 1.1}
        }
    def adjust_resonance_params(self, mood):
        """
        Get resonance parameters for a given mood state.
        Args:
            mood (str): Mood state (e.g., 'excited', 'curious', 'paranoid')
        Returns:
            dict: Resonance parameters {'k', 'threshold', 'decay', 'phase_step', 'base_frequency'}
        Raises:
            ValueError: If mood is invalid
        """
        if mood not in self.mood_configs:
            raise ValueError(f"Invalid mood state: {mood}")
        return self.mood_configs[mood]
    def update_mood_config(self, mood, k=None, threshold=None, decay=None, phase_step=None, base_frequency=None):
        """
        Update parameters for a mood state.
        Args:
            mood (str): Mood state to update
            k (int, optional): Number of top resonances
            threshold (float, optional): Similarity threshold
            decay (float, optional): Decay rate for voxel strength
            phase_step (float, optional): Phase increment for temporal coding
            base_frequency (float, optional): Base frequency for phase oscillations
        """
        if mood not in self.mood_configs:
            self.mood_configs[mood] = {
                "k": 10, "threshold": 0.85, "decay": 0.2, "phase_step": 0.1, "base_frequency": 1.0
            }
        if k is not None:
            self.mood_configs[mood]["k"] = k
        if threshold is not None:
            self.mood_configs[mood]["threshold"] = threshold
        if decay is not None:
            self.mood_configs[mood]["decay"] = decay
        if phase_step is not None:
            self.mood_configs[mood]["phase_step"] = phase_step
        if base_frequency is not None:
            self.mood_configs[mood]["base_frequency"] = base_frequency
