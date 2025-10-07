"""Tests for MAP-Elites archive."""

import pytest
import numpy as np
from fibonacci_engine.core.map_elites import MAPElites, Candidate


def test_candidate_creation():
    """Test Candidate dataclass."""
    params = np.array([1.0, 2.0, 3.0])
    candidate = Candidate(
        params=params,
        fitness=0.8,
        descriptor=[0.5, 0.5],
        generation=1,
    )
    
    assert candidate.fitness == 0.8
    assert len(candidate.descriptor) == 2
    assert candidate.generation == 1
    assert candidate.metadata == {}


def test_map_elites_init():
    """Test MAP-Elites initialization."""
    archive = MAPElites(grid_size=(10, 10))
    
    assert archive.grid_size == (10, 10)
    assert archive.n_dims == 2
    assert len(archive.archive) == 0
    assert archive.best_global is None


def test_descriptor_to_index():
    """Test descriptor to grid index conversion."""
    archive = MAPElites(grid_size=(10, 10))
    
    # Test center
    idx = archive.descriptor_to_index([0.5, 0.5])
    assert idx == (5, 5)
    
    # Test corners
    idx_00 = archive.descriptor_to_index([0.0, 0.0])
    assert idx_00 == (0, 0)
    
    idx_11 = archive.descriptor_to_index([1.0, 1.0])
    assert idx_11 == (9, 9)
    
    # Test out of bounds (should clamp)
    idx_oob = archive.descriptor_to_index([1.5, -0.5])
    assert 0 <= idx_oob[0] < 10
    assert 0 <= idx_oob[1] < 10


def test_add_candidate():
    """Test adding candidates to archive."""
    archive = MAPElites(grid_size=(5, 5))
    
    # Add first candidate
    c1 = Candidate(
        params=np.array([1.0]),
        fitness=0.5,
        descriptor=[0.5, 0.5],
        generation=1,
    )
    
    was_added = archive.add(c1)
    assert was_added
    assert len(archive.archive) == 1
    assert archive.best_global == c1
    
    # Add better candidate to same niche
    c2 = Candidate(
        params=np.array([2.0]),
        fitness=0.7,
        descriptor=[0.5, 0.5],
        generation=2,
    )
    
    was_added = archive.add(c2)
    assert was_added
    assert len(archive.archive) == 1  # Same niche
    assert archive.best_global == c2
    
    # Add worse candidate to same niche
    c3 = Candidate(
        params=np.array([3.0]),
        fitness=0.3,
        descriptor=[0.5, 0.5],
        generation=3,
    )
    
    was_added = archive.add(c3)
    assert not was_added
    assert len(archive.archive) == 1
    
    # Add candidate to different niche
    c4 = Candidate(
        params=np.array([4.0]),
        fitness=0.6,
        descriptor=[0.1, 0.9],
        generation=4,
    )
    
    was_added = archive.add(c4)
    assert was_added
    assert len(archive.archive) == 2


def test_sample():
    """Test sampling from archive."""
    archive = MAPElites(grid_size=(10, 10))
    
    # Add multiple candidates
    for i in range(10):
        c = Candidate(
            params=np.array([float(i)]),
            fitness=i * 0.1,
            descriptor=[i * 0.1, (10-i) * 0.1],
            generation=1,
        )
        archive.add(c)
    
    # Sample uniform
    samples = archive.sample(5, method="uniform")
    assert len(samples) == 5
    assert all(isinstance(s, Candidate) for s in samples)
    
    # Sample fitness-weighted
    samples_fitness = archive.sample(5, method="fitness")
    assert len(samples_fitness) == 5


def test_statistics():
    """Test archive statistics."""
    archive = MAPElites(grid_size=(10, 10))
    
    # Empty archive
    stats = archive.get_statistics()
    assert stats["coverage"] == 0.0
    assert stats["n_elites"] == 0
    assert stats["best_fitness"] is None
    
    # Add candidates
    for i in range(5):
        c = Candidate(
            params=np.array([float(i)]),
            fitness=i * 0.2,
            descriptor=[i * 0.2, 0.5],
            generation=1,
        )
        archive.add(c)
    
    stats = archive.get_statistics()
    assert stats["n_elites"] == 5
    assert stats["coverage"] > 0.0
    assert stats["best_fitness"] == 0.8
    assert "mean_fitness" in stats
    assert "std_fitness" in stats


def test_save_load(tmp_path):
    """Test saving and loading archive."""
    archive = MAPElites(grid_size=(5, 5))
    
    # Add candidates
    for i in range(3):
        c = Candidate(
            params=np.array([float(i)]),
            fitness=i * 0.3,
            descriptor=[i * 0.3, 0.5],
            generation=1,
        )
        archive.add(c)
    
    # Save
    filepath = tmp_path / "archive.json"
    archive.save(str(filepath))
    
    # Load
    loaded = MAPElites.load(str(filepath))
    
    assert loaded.grid_size == archive.grid_size
    assert len(loaded.archive) == len(archive.archive)
    assert loaded.best_global.fitness == archive.best_global.fitness


def test_clear():
    """Test clearing archive."""
    archive = MAPElites(grid_size=(5, 5))
    
    # Add candidates
    for i in range(5):
        c = Candidate(
            params=np.array([float(i)]),
            fitness=i * 0.2,
            descriptor=[i * 0.2, 0.5],
            generation=1,
        )
        archive.add(c)
    
    assert len(archive.archive) > 0
    
    # Clear
    archive.clear()
    
    assert len(archive.archive) == 0
    assert archive.best_global is None
    assert archive.coverage == 0.0
