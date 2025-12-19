#!/usr/bin/env python3
"""
Quick test script for OpenPipe setup.
Tests basic functionality without requiring Docker or GPU.
"""

import sys
import os

def test_imports():
    """Test if required packages can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import datasets
        print("  ✅ datasets")
    except ImportError as e:
        print(f"  ❌ datasets: {e}")
        return False
    
    try:
        from pydantic import BaseModel
        print("  ✅ pydantic")
    except ImportError as e:
        print(f"  ❌ pydantic: {e}")
        return False
    
    # Optional imports (not critical for basic test)
    try:
        import art
        print("  ✅ openpipe-art")
    except ImportError:
        print("  ⚠️  openpipe-art (not installed - expected if not in Docker)")
    
    try:
        import wandb
        print("  ✅ wandb")
    except ImportError:
        print("  ⚠️  wandb (not installed - expected if not in Docker)")
    
    return True


def test_dataset_loading():
    """Test loading the financial dataset."""
    print("\n📊 Testing dataset loading...")
    
    try:
        from datasets import load_dataset
        
        print("  Loading virattt/financial-qa-10K...")
        dataset = load_dataset('virattt/financial-qa-10K', split='train')
        
        print(f"  ✅ Dataset loaded: {len(dataset)} examples")
        print(f"  ✅ Columns: {dataset.column_names}")
        
        # Show sample
        sample = dataset[0]
        print(f"\n  📝 Sample entry:")
        print(f"     Question: {sample.get('question', 'N/A')[:100]}...")
        print(f"     Answer: {sample.get('answer', sample.get('response', 'N/A'))[:100]}...")
        
        return True
    except Exception as e:
        print(f"  ❌ Error loading dataset: {e}")
        return False


def test_reward_function():
    """Test if reward function can be imported."""
    print("\n🎯 Testing reward function...")
    
    # Add parent directory to path to import from modal/
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    modal_dir = os.path.join(parent_dir, 'modal')
    sys.path.insert(0, modal_dir)
    
    try:
        from reward import compute_reward
        print("  ✅ Reward function imported from modal/reward.py")
        
        # Test reward calculation
        test_reward = compute_reward(
            "What is 2+2?",
            "The answer is 4",
            "4"
        )
        print(f"  ✅ Test reward calculated: {test_reward:.3f}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error with reward function: {e}")
        return False


def test_finetune_script_syntax():
    """Test if finetune_job.py has valid syntax."""
    print("\n📄 Testing finetune_job.py syntax...")
    
    try:
        import py_compile
        script_path = os.path.join(os.path.dirname(__file__), 'finetune_job.py')
        py_compile.compile(script_path, doraise=True)
        print("  ✅ finetune_job.py syntax is valid")
        return True
    except Exception as e:
        print(f"  ❌ Syntax error in finetune_job.py: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("OpenPipe Setup Quick Test")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Dataset Loading", test_dataset_loading()))
    results.append(("Reward Function", test_reward_function()))
    results.append(("Script Syntax", test_finetune_script_syntax()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\n📋 Next steps:")
        print("   1. Start Docker Desktop")
        print("   2. Run: cd openpipe && docker compose build")
        print("   3. Run: docker compose up")
    else:
        print("\n⚠️  Some tests failed. Please install missing dependencies.")
        print("   Run: pip install datasets pydantic")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
