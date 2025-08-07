#!/usr/bin/env python3
"""
Test runner for Risk Atlas project.
Runs all tests in the tests directory.
"""
import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test(test_file):
    """Run a single test file."""
    logger.info(f"Running {test_file}...")
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            logger.info(f"✓ {test_file} passed")
            print(result.stdout)
            return True
        else:
            logger.error(f"✗ {test_file} failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"Error running {test_file}: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting Risk Atlas test suite...")
    print("=" * 60)
    
    # Get all test files
    test_dir = os.path.dirname(__file__)
    test_files = [f for f in os.listdir(test_dir) 
                  if f.startswith('test_') and f.endswith('.py')]
    
    if not test_files:
        logger.warning("No test files found")
        return 1
    
    # Run tests
    passed = 0
    failed = 0
    
    for test_file in sorted(test_files):
        test_path = os.path.join(test_dir, test_file)
        if run_test(test_path):
            passed += 1
        else:
            failed += 1
        print("-" * 60)
    
    # Summary
    print("=" * 60)
    logger.info(f"Test Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"❌ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(main()) 