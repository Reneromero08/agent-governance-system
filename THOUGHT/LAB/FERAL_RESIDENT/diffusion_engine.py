"""
Feral Resident Diffusion Engine (A.2.1)

Semantic navigation via pure geometry.
Navigates the vector space using E (Born rule) and geometric operations.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import hashlib
import json
from dataclasses import dataclass, field

# Add imports path
FERAL_PATH = Path(__file__).parent
CAPABILITY_PATH = FERAL_PATH.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"

if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))
if str(FERAL_PATH) not in sys.path:
    sys.path.insert(0, str(FERAL_PATH))

from geometric_reasoner import GeometricReasoner, GeometricState, GeometricOperations
from vector_store import VectorStore
from resident_db import VectorRecord


@dataclass
class NavigationStep:
    """A single step in semantic navigation"""
    depth: int
    current_Df: float
    E_values: List[float]
    neighbors: List[Tuple[str, float]]  # (vector_id, E)
    state_hash: str


@dataclass
class NavigationResult:
    """Result of a navigation"""
    path: List[NavigationStep]
    final_state: GeometricState
    start_hash: str
    end_hash: str
    total_depth: int
    E_evolution: List[float]
    Df_evolution: List[float]
    navigation_hash: str


class SemanticDiffusion:
    """
    Semantic navigation through vector space via pure geometry.

    Implements iterative diffusion that:
    1. Finds neighbors using E (Born rule)
    2. Projects onto neighbor context (Q44)
    3. Composes for next iteration
    4. Tracks evolution metrics

    Usage:
        diffusion = SemanticDiffusion(vector_store)

        # Navigate from query
        result = diffusion.navigate(query_state, depth=5, k=10)

        # Navigate between concepts
        path = diffusion.path_between("hot", "cold", steps=5)

        # Explore neighborhood
        neighbors = diffusion.explore(query_state, radius=0.5)
    """

    def __init__(self, store: VectorStore):
        """
        Initialize diffusion engine.

        Args:
            store: VectorStore for persistence and neighbor lookup
        """
        self.store = store
        self.reasoner = store.reasoner

        # Navigation stats
        self.stats = {
            'navigations': 0,
            'total_steps': 0,
            'avg_E_per_step': 0.0
        }

    def navigate(
        self,
        query_state: GeometricState,
        depth: int = 5,
        k: int = 10,
        min_E_threshold: float = 0.1
    ) -> NavigationResult:
        """
        Navigate semantic space starting from query.

        Each iteration:
        1. Find k nearest neighbors by E (Born rule)
        2. Project query onto neighbor subspace
        3. Superpose with current for next iteration

        Args:
            query_state: Starting state
            depth: Number of navigation steps
            k: Neighbors per step
            min_E_threshold: Minimum E to consider neighbor

        Returns:
            NavigationResult with path and final state
        """
        path = []
        current = query_state
        start_hash = current.receipt()['vector_hash']
        E_evolution = []
        Df_evolution = [current.Df]

        for d in range(depth):
            # Find neighbors using E (Born rule)
            neighbors = self.store.find_nearest(current, k=k)

            # Filter by threshold
            filtered_neighbors = [
                (rec, E) for rec, E in neighbors
                if E > min_E_threshold
            ]

            if not filtered_neighbors:
                # No more neighbors above threshold, stop
                break

            # Extract E values
            E_values = [E for _, E in filtered_neighbors]
            E_evolution.append(np.mean(E_values))

            # Build neighbor states for projection
            neighbor_states = []
            neighbor_info = []
            for record, E in filtered_neighbors:
                state = GeometricState(
                    vector=record.vector,
                    operation_history=[]
                )
                neighbor_states.append(state)
                neighbor_info.append((record.vector_id, E))

            # Record step
            step = NavigationStep(
                depth=d,
                current_Df=current.Df,
                E_values=E_values,
                neighbors=neighbor_info,
                state_hash=current.receipt()['vector_hash']
            )
            path.append(step)

            # Project onto neighbor context (Q44 Born rule)
            if neighbor_states:
                projected = self.reasoner.project(current, neighbor_states)

                # Compose for next iteration (weighted blend)
                # More weight to projected, less to current
                current = self.reasoner.superpose(projected, current)

            Df_evolution.append(current.Df)

        # Build result
        end_hash = current.receipt()['vector_hash']

        # Hash the navigation for provenance
        nav_data = json.dumps({
            'start': start_hash,
            'depth': len(path),
            'k': k,
            'path_hashes': [s.state_hash for s in path]
        }, sort_keys=True)
        navigation_hash = hashlib.sha256(nav_data.encode()).hexdigest()[:16]

        # Update stats
        self.stats['navigations'] += 1
        self.stats['total_steps'] += len(path)
        if E_evolution:
            self.stats['avg_E_per_step'] = np.mean(E_evolution)

        return NavigationResult(
            path=path,
            final_state=current,
            start_hash=start_hash,
            end_hash=end_hash,
            total_depth=len(path),
            E_evolution=E_evolution,
            Df_evolution=Df_evolution,
            navigation_hash=navigation_hash
        )

    def navigate_from_text(
        self,
        text: str,
        depth: int = 5,
        k: int = 10
    ) -> NavigationResult:
        """
        Navigate starting from text query.

        Args:
            text: Starting text
            depth: Navigation depth
            k: Neighbors per step

        Returns:
            NavigationResult
        """
        query_state = self.store.embed(text, store=False)
        return self.navigate(query_state, depth=depth, k=k)

    def path_between(
        self,
        start_text: str,
        end_text: str,
        steps: int = 5
    ) -> List[Dict]:
        """
        Find path between two concepts via geodesic interpolation.

        Args:
            start_text: Starting concept
            end_text: Ending concept
            steps: Number of interpolation steps

        Returns:
            List of waypoints with E values and neighbors
        """
        start_state = self.store.embed(start_text, store=False)
        end_state = self.store.embed(end_text, store=False)

        path = []

        for i in range(steps + 1):
            t = i / steps

            # Interpolate along geodesic
            waypoint = self.reasoner.interpolate(start_state, end_state, t)

            # Find neighbors at this point
            neighbors = self.store.find_nearest(waypoint, k=5)

            # Decode to text if we have corpus
            path.append({
                't': t,
                'Df': waypoint.Df,
                'distance_from_start': waypoint.distance_to(start_state),
                'distance_to_end': waypoint.distance_to(end_state),
                'E_with_start': waypoint.E_with(start_state),
                'E_with_end': waypoint.E_with(end_state),
                'neighbors': [(rec.vector_id, E) for rec, E in neighbors[:3]],
                'state_hash': waypoint.receipt()['vector_hash']
            })

        return path

    def explore(
        self,
        center: GeometricState,
        radius: float = 0.5,
        samples: int = 100
    ) -> Dict:
        """
        Explore neighborhood around a state.

        Samples random directions and finds structure.

        Args:
            center: Center state
            radius: Search radius in radians
            samples: Number of random samples

        Returns:
            Dict with neighborhood statistics
        """
        # Sample random directions
        dim = len(center.vector)
        E_values = []
        Df_values = []
        neighbor_counts = []

        for _ in range(samples):
            # Random perturbation
            noise = np.random.randn(dim).astype(np.float32)
            noise = noise / np.linalg.norm(noise)

            # Interpolate by radius
            perturbed = self.reasoner.interpolate(
                center,
                GeometricState(vector=noise, operation_history=[]),
                radius / np.pi  # Convert radius to t
            )

            # Measure properties
            E_values.append(perturbed.E_with(center))
            Df_values.append(perturbed.Df)

            # Count neighbors above threshold
            neighbors = self.store.find_nearest(perturbed, k=10)
            count = sum(1 for _, E in neighbors if E > 0.3)
            neighbor_counts.append(count)

        return {
            'center_hash': center.receipt()['vector_hash'],
            'center_Df': center.Df,
            'radius': radius,
            'samples': samples,
            'E_stats': {
                'mean': float(np.mean(E_values)),
                'std': float(np.std(E_values)),
                'min': float(np.min(E_values)),
                'max': float(np.max(E_values))
            },
            'Df_stats': {
                'mean': float(np.mean(Df_values)),
                'std': float(np.std(Df_values)),
                'min': float(np.min(Df_values)),
                'max': float(np.max(Df_values))
            },
            'neighbor_density': {
                'mean': float(np.mean(neighbor_counts)),
                'max': int(np.max(neighbor_counts))
            }
        }

    def contextual_walk(
        self,
        start: GeometricState,
        context: List[GeometricState],
        steps: int = 10
    ) -> List[Dict]:
        """
        Walk through space while staying in context.

        At each step, projects onto context to stay relevant.

        Args:
            start: Starting state
            context: Context states to stay near
            steps: Number of steps

        Returns:
            List of walk steps
        """
        path = []
        current = start

        for i in range(steps):
            # Project onto context
            projected = self.reasoner.project(current, context)

            # Find neighbors
            neighbors = self.store.find_nearest(projected, k=5)

            # Compute context relevance
            context_E = [projected.E_with(c) for c in context]

            path.append({
                'step': i,
                'Df': projected.Df,
                'context_E_mean': float(np.mean(context_E)),
                'context_E_max': float(np.max(context_E)),
                'neighbors': [(rec.vector_id, E) for rec, E in neighbors[:3]],
                'state_hash': projected.receipt()['vector_hash']
            })

            # Move toward highest E neighbor
            if neighbors:
                best_neighbor, best_E = neighbors[0]
                target = GeometricState(
                    vector=best_neighbor.vector,
                    operation_history=[]
                )
                # Small step toward best neighbor
                current = self.reasoner.interpolate(projected, target, 0.3)
            else:
                current = projected

        return path

    def resonance_map(
        self,
        query: GeometricState,
        depth_limit: int = 3
    ) -> Dict:
        """
        Build a resonance map showing E-structure around query.

        Returns tree of neighbors with their sub-neighbors.

        Args:
            query: Query state
            depth_limit: How deep to explore

        Returns:
            Nested dict of neighbors
        """
        def explore_node(state: GeometricState, depth: int) -> Dict:
            if depth >= depth_limit:
                return {'hash': state.receipt()['vector_hash'], 'Df': state.Df, 'children': []}

            neighbors = self.store.find_nearest(state, k=5, exclude_hashes=[state.receipt()['vector_hash']])

            children = []
            for record, E in neighbors[:3]:  # Limit branching
                child_state = GeometricState(
                    vector=record.vector,
                    operation_history=[]
                )
                children.append({
                    'E': E,
                    'vector_id': record.vector_id,
                    'subtree': explore_node(child_state, depth + 1)
                })

            return {
                'hash': state.receipt()['vector_hash'],
                'Df': state.Df,
                'children': children
            }

        return explore_node(query, 0)

    def get_stats(self) -> Dict:
        """Get diffusion statistics"""
        return {
            **self.stats,
            'store_metrics': self.store.get_metrics()
        }


# ============================================================================
# Testing
# ============================================================================

def example_usage():
    """Demonstrate diffusion engine"""
    import tempfile
    import os

    print("=== SemanticDiffusion Example ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_diffusion.db")
        store = VectorStore(db_path)
        diffusion = SemanticDiffusion(store)

        # Populate some vectors
        concepts = [
            "authentication methods",
            "OAuth protocol",
            "JWT tokens",
            "session management",
            "password hashing",
            "two factor authentication",
            "API keys",
            "bearer tokens",
            "refresh tokens",
            "access control"
        ]

        print("Populating vector space...")
        for concept in concepts:
            store.embed(concept)

        # Navigate from a query
        print("\nNavigating from 'security tokens'...")
        result = diffusion.navigate_from_text("security tokens", depth=3, k=5)

        print(f"Navigation depth: {result.total_depth}")
        print(f"Df evolution: {[f'{d:.1f}' for d in result.Df_evolution]}")
        print(f"E evolution: {[f'{e:.3f}' for e in result.E_evolution]}")

        # Path between concepts
        print("\nPath from 'login' to 'logout'...")
        path = diffusion.path_between("login", "logout", steps=3)
        for waypoint in path:
            print(f"  t={waypoint['t']:.1f}: E_start={waypoint['E_with_start']:.3f}, E_end={waypoint['E_with_end']:.3f}")

        # Explore neighborhood
        print("\nExploring neighborhood...")
        query = store.embed("authentication", store=False)
        exploration = diffusion.explore(query, radius=0.3, samples=20)
        print(f"E stats: mean={exploration['E_stats']['mean']:.3f}, std={exploration['E_stats']['std']:.3f}")

        # Stats
        print(f"\nDiffusion stats: {diffusion.stats}")

        store.close()
        print("\nDone!")


if __name__ == "__main__":
    example_usage()
