"""Tests for WORM Ledger."""

import pytest
import time
from fibonacci_engine.core.worm_ledger import WormLedger, LedgerEntry


def test_ledger_initialization():
    """Test ledger initialization with genesis block."""
    ledger = WormLedger()
    
    assert len(ledger) == 1
    assert ledger.chain[0].event_type == "genesis"
    assert ledger.chain[0].prev_hash == "0" * 64
    assert ledger.verify()


def test_append_entry():
    """Test appending entries to ledger."""
    ledger = WormLedger()
    
    # Append first entry
    entry1 = ledger.append("test_event", {"key": "value", "number": 42})
    
    assert len(ledger) == 2
    assert entry1.event_type == "test_event"
    assert entry1.data["key"] == "value"
    assert entry1.prev_hash == ledger.chain[0].hash
    
    # Append second entry
    entry2 = ledger.append("another_event", {"data": [1, 2, 3]})
    
    assert len(ledger) == 3
    assert entry2.prev_hash == entry1.hash


def test_hash_computation():
    """Test that hashes are computed correctly."""
    ledger = WormLedger()
    
    entry = ledger.append("test", {"value": 123})
    
    # Recompute hash and verify it matches
    computed_hash = entry.compute_hash()
    assert computed_hash == entry.hash
    
    # Verify full chain
    assert ledger.verify()


def test_chain_integrity():
    """Test that chain integrity is maintained."""
    ledger = WormLedger()
    
    # Add multiple entries
    for i in range(10):
        ledger.append(f"event_{i}", {"index": i})
    
    # Verify integrity
    assert ledger.verify()
    
    # Tamper with an entry (should break verification)
    ledger.chain[5].data["index"] = 999
    assert not ledger.verify()


def test_get_entries():
    """Test querying entries."""
    ledger = WormLedger()
    
    # Add entries of different types
    ledger.append("type_a", {"value": 1})
    ledger.append("type_b", {"value": 2})
    ledger.append("type_a", {"value": 3})
    ledger.append("type_b", {"value": 4})
    
    # Get all entries
    all_entries = ledger.get_entries()
    assert len(all_entries) == 5  # Including genesis
    
    # Get entries by type
    type_a_entries = ledger.get_entries(event_type="type_a")
    assert len(type_a_entries) == 2
    
    # Get entries by range
    range_entries = ledger.get_entries(start_index=2, end_index=4)
    assert len(range_entries) == 2


def test_statistics():
    """Test ledger statistics."""
    ledger = WormLedger()
    
    # Empty stats (only genesis)
    stats = ledger.get_statistics()
    assert stats["total_entries"] == 1
    assert stats["is_valid"]
    
    # Add entries
    for i in range(5):
        ledger.append("generation", {"gen": i})
    
    for i in range(3):
        ledger.append("elite_added", {"elite": i})
    
    stats = ledger.get_statistics()
    assert stats["total_entries"] == 9  # 1 genesis + 5 + 3
    assert stats["event_types"]["generation"] == 5
    assert stats["event_types"]["elite_added"] == 3


def test_save_load(tmp_path):
    """Test saving and loading ledger."""
    ledger = WormLedger()
    
    # Add entries
    for i in range(5):
        ledger.append("test", {"value": i})
    
    # Save
    filepath = tmp_path / "ledger.json"
    ledger.save(str(filepath))
    
    # Load
    loaded = WormLedger.load(str(filepath))
    
    assert len(loaded) == len(ledger)
    assert loaded.verify()
    
    # Check data integrity
    for i in range(len(ledger)):
        assert loaded.chain[i].hash == ledger.chain[i].hash


def test_invalid_data():
    """Test that non-JSON-serializable data is rejected."""
    ledger = WormLedger()
    
    # This should raise an error
    with pytest.raises(ValueError):
        ledger.append("test", {"func": lambda x: x})


def test_immutability():
    """Test that ledger is write-once (entries can't be modified)."""
    ledger = WormLedger()
    
    ledger.append("event1", {"value": 1})
    ledger.append("event2", {"value": 2})
    
    assert ledger.verify()
    
    # Modifying an entry should break verification
    original_hash = ledger.chain[1].hash
    ledger.chain[1].data["value"] = 999
    
    # Hash is still the same (not recomputed)
    assert ledger.chain[1].hash == original_hash
    
    # But verification should fail
    assert not ledger.verify()
