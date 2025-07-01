# app/services/traffic_controller.py
import asyncio
import time
from typing import Dict, List
import logging
from app.config.settings import settings

logger = logging.getLogger(__name__)

class TrafficController:
    def __init__(self):
        self.intersections = {}  # Store intersection configurations
        self.current_signals = {}  # Current signal states
        self.signal_timers = {}  # Timer management
        self.emergency_override = {}  # Emergency vehicle priority
        
    def initialize_intersection(self, intersection_id: str, config: Dict):
        """
        Initialize intersection with lanes and signal configuration
        """
        self.intersections[intersection_id] = {
            'lanes': config.get('lanes', ['north', 'south', 'east', 'west']),
            'signal_groups': config.get('signal_groups', {
                'ns': ['north', 'south'],
                'ew': ['east', 'west']
            }),
            'min_green_time': config.get('min_green_time', settings.MIN_GREEN_TIME),
            'max_green_time': config.get('max_green_time', settings.MAX_GREEN_TIME),
            'yellow_time': config.get('yellow_time', settings.YELLOW_TIME)
        }
        
        # Initialize signal state
        self.current_signals[intersection_id] = {
            'active_group': 'ns',
            'state': 'green',
            'start_time': time.time(),
            'duration': settings.MIN_GREEN_TIME
        }
        
        logger.info(f"Initialized intersection: {intersection_id}")
    
    async def optimize_signal_timing(self, intersection_id: str, 
                                   density_data: Dict) -> Dict:
        """
        Optimize signal timing based on real-time traffic density
        """
        if intersection_id not in self.intersections:
            return {'error': 'Intersection not found'}
        
        intersection = self.intersections[intersection_id]
        current_signal = self.current_signals[intersection_id]
        
        # Calculate optimal timing based on density
        lane_densities = self._calculate_lane_densities(density_data, intersection)
        optimal_timings = self._calculate_optimal_timings(lane_densities, intersection)
        
        # Check if signal change is needed
        current_time = time.time()
        elapsed_time = current_time - current_signal['start_time']
        
        should_change = False
        
        # Check minimum time constraint
        if elapsed_time >= intersection['min_green_time']:
            # Check if opposite direction has significantly higher density
            active_group = current_signal['active_group']
            inactive_groups = [g for g in intersection['signal_groups'].keys() if g != active_group]
            
            for inactive_group in inactive_groups:
                if optimal_timings[inactive_group] > optimal_timings[active_group] * 1.5:
                    should_change = True
                    break
        
        # Check maximum time constraint
        if elapsed_time >= intersection['max_green_time']:
            should_change = True
        
        if should_change:
            await self._change_signal(intersection_id, optimal_timings)
        
        return {
            'intersection_id': intersection_id,
            'current_state': current_signal,
            'optimal_timings': optimal_timings,
            'changed': should_change
        }
    
    def _calculate_lane_densities(self, density_data: Dict, intersection: Dict) -> Dict:
        """
        Calculate density for each lane direction
        """
        lane_densities = {}
        
        # This is a simplified calculation - in practice, you'd need
        # camera positioning and lane detection to map grid density to lanes
        total_density = density_data.get('total_density', 0)
        num_lanes = len(intersection['lanes'])
        
        # For now, distribute density evenly (can be enhanced with lane detection)
        base_density = total_density / num_lanes if num_lanes > 0 else 0
        
        for lane in intersection['lanes']:
            # Add some randomness to simulate real conditions
            import random
            lane_densities[lane] = base_density * (0.8 + 0.4 * random.random())
        
        return lane_densities
    
    def _calculate_optimal_timings(self, lane_densities: Dict, intersection: Dict) -> Dict:
        """
        Calculate optimal signal timings for each signal group
        """
        signal_timings = {}
        
        for group_name, lanes in intersection['signal_groups'].items():
            group_density = sum(lane_densities.get(lane, 0) for lane in lanes)
            
            # Calculate timing based on density (Webster's formula adaptation)
            base_time = intersection['min_green_time']
            density_factor = min(group_density / 10.0, 3.0)  # Cap the factor
            optimal_time = base_time + (density_factor * 20)
            
            # Ensure within bounds
            optimal_time = max(intersection['min_green_time'], 
                             min(optimal_time, intersection['max_green_time']))
            
            signal_timings[group_name] = optimal_time
        
        return signal_timings
    
    async def _change_signal(self, intersection_id: str, optimal_timings: Dict):
        """
        Change signal state with proper yellow phase
        """
        current_signal = self.current_signals[intersection_id]
        intersection = self.intersections[intersection_id]
        
        # Start yellow phase
        current_signal['state'] = 'yellow'
        current_signal['start_time'] = time.time()
        current_signal['duration'] = intersection['yellow_time']
        
        logger.info(f"Signal changed to YELLOW at intersection {intersection_id}")
        
        # Wait for yellow phase
        await asyncio.sleep(intersection['yellow_time'])
        
        # Switch to next group
        current_group = current_signal['active_group']
        signal_groups = list(intersection['signal_groups'].keys())
        current_index = signal_groups.index(current_group)
        next_group = signal_groups[(current_index + 1) % len(signal_groups)]
        
        # Set green for next group
        current_signal['active_group'] = next_group
        current_signal['state'] = 'green'
        current_signal['start_time'] = time.time()
        current_signal['duration'] = optimal_timings.get(next_group, intersection['min_green_time'])
        
        logger.info(f"Signal changed to GREEN for group {next_group} at intersection {intersection_id}")
    
    async def handle_emergency_vehicle(self, intersection_id: str, 
                                     emergency_direction: str) -> Dict:
        """
        Handle emergency vehicle priority - critical for Indian traffic
        """
        if intersection_id not in self.intersections:
            return {'error': 'Intersection not found'}
        
        intersection = self.intersections[intersection_id]
        current_signal = self.current_signals[intersection_id]
        
        # Find which signal group contains the emergency direction
        emergency_group = None
        for group_name, lanes in intersection['signal_groups'].items():
            if emergency_direction in lanes:
                emergency_group = group_name
                break
        
        if not emergency_group:
            return {'error': 'Invalid emergency direction'}
        
        # If emergency group is not currently active, switch immediately
        if current_signal['active_group'] != emergency_group:
            logger.warning(f"EMERGENCY OVERRIDE: Switching to {emergency_group} at {intersection_id}")
            
            # Set emergency override
            self.emergency_override[intersection_id] = {
                'active': True,
                'group': emergency_group,
                'start_time': time.time()
            }
            
            # Immediate yellow phase
            current_signal['state'] = 'yellow'
            current_signal['start_time'] = time.time()
            current_signal['duration'] = 2  # Shorter yellow for emergency
            
            await asyncio.sleep(2)
            
            # Switch to emergency group
            current_signal['active_group'] = emergency_group
            current_signal['state'] = 'green'
            current_signal['start_time'] = time.time()
            current_signal['duration'] = 60  # Extended green for emergency
        
        return {
            'intersection_id': intersection_id,
            'emergency_override': True,
            'active_group': emergency_group
        }
    
    def get_signal_status(self, intersection_id: str) -> Dict:
        """
        Get current signal status
        """
        if intersection_id not in self.current_signals:
            return {'error': 'Intersection not found'}
        
        current_signal = self.current_signals[intersection_id]
        elapsed_time = time.time() - current_signal['start_time']
        remaining_time = max(0, current_signal['duration'] - elapsed_time)
        
        return {
            'intersection_id': intersection_id,
            'active_group': current_signal['active_group'],
            'state': current_signal['state'],
            'elapsed_time': elapsed_time,
            'remaining_time': remaining_time,
            'emergency_active': intersection_id in self.emergency_override
        }
