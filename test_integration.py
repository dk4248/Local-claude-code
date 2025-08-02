#!/usr/bin/env python3
"""
Test script for the integrated AI-Augmented UNIX Kernel with code generation
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AI-Augmented-UNIX-Kernel'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from summarizer import CodeSummarizer, BatchProcessor
        print("✅ Summarizer imports successful")
    except ImportError as e:
        print(f"❌ Summarizer import failed: {e}")
        return False
    
    try:
        from coder import CoderAgent, TaskType
        print("✅ Coder imports successful")
    except ImportError as e:
        print(f"❌ Coder import failed: {e}")
        return False
    
    try:
        from shell_ai.code_assistant import CodeAssistant, CodeCommandHandler
        print("✅ Code assistant imports successful")
    except ImportError as e:
        print(f"❌ Code assistant import failed: {e}")
        return False
    
    try:
        from shell_ai.assistant import ShellAIAssistant
        print("✅ Shell AI assistant imports successful")
    except ImportError as e:
        print(f"❌ Shell AI assistant import failed: {e}")
        return False
    
    return True

def test_code_assistant_initialization():
    """Test CodeAssistant initialization"""
    print("\nTesting CodeAssistant initialization...")
    try:
        from shell_ai.code_assistant import CodeAssistant
        
        # Create code assistant
        assistant = CodeAssistant(model="qwen3-coder")
        print("✅ CodeAssistant created successfully")
        
        # Check directories were created
        if assistant.summaries_dir.exists():
            print(f"✅ Summaries directory created: {assistant.summaries_dir}")
        else:
            print("❌ Summaries directory not created")
            return False
        
        if assistant.generated_code_dir.exists():
            print(f"✅ Generated code directory created: {assistant.generated_code_dir}")
        else:
            print("❌ Generated code directory not created")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error initializing CodeAssistant: {e}")
        return False

def test_sample_summarization():
    """Test summarization on a small sample file"""
    print("\nTesting sample file summarization...")
    try:
        from summarizer import CodeSummarizer
        
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def calculate_sum(a: int, b: int) -> int:
    '''Calculate the sum of two numbers'''
    return a + b

def main():
    result = calculate_sum(5, 3)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
""")
            temp_file = f.name
        
        # Initialize summarizer (without checking model)
        summarizer = CodeSummarizer(model="qwen3-coder")
        
        # Create summary prompt
        with open(temp_file, 'r') as f:
            code = f.read()
        
        prompt = summarizer.create_summary_prompt(code, temp_file)
        
        if "calculate_sum" in prompt and "main" in prompt:
            print("✅ Summary prompt created successfully")
            result = True
        else:
            print("❌ Summary prompt incorrect")
            result = False
        
        # Clean up
        os.unlink(temp_file)
        return result
        
    except Exception as e:
        print(f"❌ Error in summarization test: {e}")
        return False

def test_code_handler():
    """Test CodeCommandHandler"""
    print("\nTesting CodeCommandHandler...")
    try:
        from shell_ai.code_assistant import CodeCommandHandler
        
        handler = CodeCommandHandler(model="qwen3-coder")
        print("✅ CodeCommandHandler created successfully")
        
        # Check commands are registered
        if 'code' in handler.commands:
            print("✅ 'code' command registered")
        else:
            print("❌ 'code' command not registered")
            return False
        
        if 'analyze' in handler.commands:
            print("✅ 'analyze' command registered")
        else:
            print("❌ 'analyze' command not registered")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error testing CodeCommandHandler: {e}")
        return False

def test_integration_with_shell_ai():
    """Test that Shell AI properly integrates code features"""
    print("\nTesting Shell AI integration...")
    try:
        from shell_ai.assistant import ShellAIAssistant
        from shell_ai.config import Config
        
        # Create config
        config = Config()
        config.set('provider', 'ollama')
        config.set('ollama.model', 'qwen3-coder')
        
        # Create assistant
        assistant = ShellAIAssistant(config)
        print("✅ ShellAIAssistant created with code handler")
        
        # Check code handler exists
        if hasattr(assistant, 'code_handler'):
            print("✅ Code handler integrated into Shell AI")
        else:
            print("❌ Code handler not found in Shell AI")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error testing Shell AI integration: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing AI-Augmented UNIX Kernel Integration")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("CodeAssistant Initialization", test_code_assistant_initialization),
        ("Sample Summarization", test_sample_summarization),
        ("Code Handler", test_code_handler),
        ("Shell AI Integration", test_integration_with_shell_ai)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n### {name}")
        print("-" * 40)
        success = test_func()
        results.append((name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Integration successful!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())