#!/usr/bin/env python3
"""
Batch Code Summarizer using Ollama
This script processes multiple code files and generates detailed summaries
using the Ollama qwen3-coder model.
"""

import os
import sys
import json
import logging
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    import ollama
    from ollama import AsyncClient
except ImportError:
    print("Error: ollama-python library not installed.")
    print("Please install it using: pip install ollama")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class CodeSummarizer:
    """Handles code summarization using Ollama"""
    
    def __init__(self, model: str = "qwen3-coder", host: str = "http://localhost:11434"):
        """
        Initialize the summarizer with model and host configuration
        
        Args:
            model: The Ollama model to use (default: qwen3-coder)
            host: The Ollama server URL (default: http://localhost:11434)
        """
        self.model = model
        self.host = host
        self.client = ollama.Client(host=host)
        self.async_client = AsyncClient(host=host)
        
        # Check if model is available
        self._check_model()
    
    def _check_model(self):
        """Check if the model is available, pull it if necessary."""
        try:
            resp = self.client.list()
            # Handle different response formats from ollama
            if isinstance(resp, dict) and "models" in resp:
                entries = resp["models"]
            elif hasattr(resp, "models"):
                entries = resp.models
            elif isinstance(resp, list):
                entries = resp
            else:
                # If we can't determine the format, try to pull anyway
                logger.warning(f"Unexpected response format from ollama.list()")
                entries = []

            # Extract names whether each entry is a dict or a string
            model_names = []
            for m in entries:
                if isinstance(m, dict):
                    name = m.get("name", "")
                    if name:
                        model_names.append(name.split(":")[0])  # Get base name
                elif isinstance(m, str):
                    model_names.append(m.split(":")[0])
                elif hasattr(m, "name"):
                    model_names.append(m.name.split(":")[0])

            base = self.model.split(":", 1)[0]
            if not any(base == name for name in model_names):
                logger.info(f"Model {self.model} not found locally. Pulling…")
                self.client.pull(self.model)
                logger.info(f"Successfully pulled {self.model}")
        except Exception as e:
            logger.error(f"Error checking/pulling model: {e}")
            # Don't raise here, let it fail later if model truly unavailable
            logger.warning("Continuing anyway - model might be available")
    
    def create_summary_prompt(self, code: str, file_path: str) -> str:
        """
        Create a detailed prompt for code summarization
        
        Args:
            code: The code content to summarize
            file_path: Path to the file being summarized
        
        Returns:
            Formatted prompt string
        """
        # Truncate very long code files
        if len(code) > 20000:
            code = code[:20000] + "\n... (truncated)"
            
        prompt = f"""You are an expert code analyzer. Please analyze the following code file and provide a comprehensive summary.

File: {file_path}

Code:
```
{code}
```

Please provide a detailed summary that includes:

1. **Purpose and Overview**: What is the main purpose of this file? What role does it play in the larger codebase?

2. **Key Components**:
   - List and explain all important classes, functions, or modules
   - For each major function/class, describe:
     - Its purpose
     - Input parameters and return values
     - Key logic or algorithms used
     - Any side effects or external dependencies

3. **Dependencies and Imports**: What external libraries or internal modules does this file depend on?

4. **Data Flow**: How does data flow through this file? What are the main inputs and outputs?

5. **Integration Points**: How does this file integrate with other parts of the codebase? What interfaces does it expose or consume?

6. **Code Patterns and Architecture**: What design patterns or architectural decisions are evident in this code?

7. **Important Code Snippets**: Include relevant code snippets for critical functions or complex logic, with explanations.

8. **Potential Issues or Improvements**: Are there any obvious issues, TODOs, or areas for improvement?

9. **Database/API Interactions**: If applicable, describe any database queries, API calls, or external service interactions.

10. **Configuration and Environment**: Does this file use any configuration values or environment variables?

Please format your response in a clear, structured manner that would help another developer quickly understand this file's role and implementation details."""
        
        return prompt
    
    def summarize_file_sync(self, file_path: str) -> Tuple[str, str, Optional[str]]:
        """
        Synchronously summarize a single file
        
        Args:
            file_path: Path to the file to summarize
            
        Returns:
            Tuple of (file_path, summary, error)
        """
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            # Skip empty files
            if not code.strip():
                return file_path, "Empty file - no code to summarize.", None
            
            # Skip binary files
            if '\0' in code:
                return file_path, "Binary file - skipped.", None
            
            # Create the prompt
            prompt = self.create_summary_prompt(code, file_path)
            
            # Generate summary
            logger.info(f"Generating summary for: {file_path}")
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,  # Lower temperature for more consistent summaries
                    "num_predict": 2048,  # Maximum tokens for response
                }
            )
            
            summary = response['response']
            return file_path, summary, None
            
        except UnicodeDecodeError:
            return file_path, "Binary or non-text file - skipped.", None
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logger.error(error_msg)
            return file_path, "", error_msg
    
    async def summarize_file_async(self, file_path: str) -> Tuple[str, str, Optional[str]]:
        """
        Asynchronously summarize a single file
        
        Args:
            file_path: Path to the file to summarize
            
        Returns:
            Tuple of (file_path, summary, error)
        """
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            # Skip empty files
            if not code.strip():
                return file_path, "Empty file - no code to summarize.", None
            
            # Skip binary files
            if '\0' in code:
                return file_path, "Binary file - skipped.", None
            
            # Create the prompt
            prompt = self.create_summary_prompt(code, file_path)
            
            # Generate summary
            logger.info(f"Generating summary for: {file_path}")
            response = await self.async_client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 2048,
                }
            )
            
            summary = response['response']
            return file_path, summary, None
            
        except UnicodeDecodeError:
            return file_path, "Binary or non-text file - skipped.", None
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logger.error(error_msg)
            return file_path, "", error_msg

class BatchProcessor:
    """Handles batch processing of multiple files"""
    
    def __init__(self, summarizer: CodeSummarizer, output_dir: str = "./summaries"):
        """
        Initialize the batch processor
        
        Args:
            summarizer: CodeSummarizer instance
            output_dir: Directory to save summaries
        """
        self.summarizer = summarizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"batch_{timestamp}"
        self.session_dir.mkdir(exist_ok=True)
    
    def save_summary(self, file_path: str, summary: str, error: Optional[str] = None):
        """
        Save summary to a text file
        
        Args:
            file_path: Original file path
            summary: Generated summary
            error: Error message if any
        """
        # Create output filename based on input file
        input_path = Path(file_path)
        # Create a safe filename by replacing path separators
        safe_name = str(input_path).replace(os.sep, '_').replace(':', '')
        output_filename = f"{safe_name}_summary.txt"
        output_path = self.session_dir / output_filename
        
        # Write summary
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"File: {file_path}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.summarizer.model}\n")
            f.write("=" * 80 + "\n\n")
            
            if error:
                f.write(f"ERROR: {error}\n")
            else:
                f.write(summary)
        
        logger.debug(f"Saved summary to: {output_path}")
    
    def process_batch_sync(self, file_paths: List[str], max_workers: int = 4):
        """
        Process files synchronously with thread pool
        
        Args:
            file_paths: List of file paths to process
            max_workers: Maximum number of concurrent workers
        """
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.summarizer.summarize_file_sync, fp): fp 
                for fp in file_paths
            }
            
            # Process completed tasks with progress
            completed = 0
            total = len(file_paths)
            
            for future in as_completed(future_to_file):
                file_path, summary, error = future.result()
                results.append((file_path, summary, error))
                self.save_summary(file_path, summary, error)
                
                completed += 1
                if completed % 5 == 0 or completed == total:
                    logger.info(f"Progress: {completed}/{total} files processed")
        
        # Save batch report
        self._save_batch_report(results, time.time() - start_time)
    
    async def process_batch_async(self, file_paths: List[str], max_concurrent: int = 4):
        """
        Process files asynchronously
        
        Args:
            file_paths: List of file paths to process
            max_concurrent: Maximum number of concurrent requests
        """
        start_time = time.time()
        results = []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.summarizer.summarize_file_async(file_path)
        
        # Process all files
        tasks = [process_with_semaphore(fp) for fp in file_paths]
        
        completed = 0
        total = len(file_paths)
        
        for coro in asyncio.as_completed(tasks):
            file_path, summary, error = await coro
            results.append((file_path, summary, error))
            self.save_summary(file_path, summary, error)
            
            completed += 1
            if completed % 5 == 0 or completed == total:
                logger.info(f"Progress: {completed}/{total} files processed")
        
        # Save batch report
        self._save_batch_report(results, time.time() - start_time)
    
    def _save_batch_report(self, results: List[Tuple[str, str, Optional[str]]], duration: float):
        """Save a summary report of the batch processing"""
        report_path = self.session_dir / "batch_report.json"
        
        successful = sum(1 for _, _, error in results if error is None)
        failed = len(results) - successful
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": self.summarizer.model,
            "total_files": len(results),
            "successful": successful,
            "failed": failed,
            "duration_seconds": round(duration, 2),
            "files_per_second": round(len(results) / duration, 2) if duration > 0 else 0,
            "session_directory": str(self.session_dir),
            "files": [
                {
                    "path": path,
                    "status": "success" if error is None else "failed",
                    "error": error,
                    "summary_file": f"{str(Path(path)).replace(os.sep, '_').replace(':', '')}_summary.txt"
                }
                for path, _, error in results
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nBatch processing complete!")
        logger.info(f"Total files: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Summaries saved to: {self.session_dir}")

def find_code_files(paths: List[str], extensions: Optional[List[str]] = None, 
                   exclude_dirs: Optional[List[str]] = None) -> List[str]:
    """
    Find all code files from given paths (files or directories)
    
    Args:
        paths: List of file paths or directory paths
        extensions: List of file extensions to include (e.g., ['.py', '.js'])
        exclude_dirs: List of directory names to exclude
    
    Returns:
        List of file paths
    """
    if extensions is None:
        # Default code file extensions
        extensions = [
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', 
            '.h', '.hpp', '.cs', '.rb', '.go', '.rs', '.php', '.swift',
            '.kt', '.scala', '.r', '.m', '.mm', '.lua', '.pl', '.sh',
            '.dart', '.elm', '.ex', '.exs', '.clj', '.cljs', '.ml', '.fs'
        ]
    
    if exclude_dirs is None:
        exclude_dirs = [
            'node_modules', '__pycache__', '.git', '.svn', 'build', 
            'dist', 'target', 'bin', 'obj', '.idea', '.vscode',
            'venv', 'env', '.env', 'vendor', 'packages', '.pytest_cache',
            '.mypy_cache', 'coverage', '.coverage', 'htmlcov'
        ]
    
    file_paths = []
    
    for path in paths:
        path_obj = Path(path)
        
        if path_obj.is_file():
            # Single file
            if any(path_obj.suffix.lower() == ext for ext in extensions):
                file_paths.append(str(path_obj))
        elif path_obj.is_dir():
            # Directory - recursively find files
            for file_path in path_obj.rglob('*'):
                # Skip excluded directories
                if any(excluded in file_path.parts for excluded in exclude_dirs):
                    continue
                
                # Check if it's a file with the right extension
                if file_path.is_file() and any(file_path.suffix.lower() == ext for ext in extensions):
                    file_paths.append(str(file_path))
    
    # Remove duplicates and sort
    file_paths = sorted(list(set(file_paths)))
    
    return file_paths

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Batch code summarizer using Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize specific files
  python summarizer.py file1.py file2.js file3.java
  
  # Summarize all Python files in a directory
  python summarizer.py /path/to/project --extensions .py
  
  # Use a different model
  python summarizer.py /path/to/code --model mistral
  
  # Adjust concurrency
  python summarizer.py /path/to/code --concurrent 8
  
  # Exclude certain directories
  python summarizer.py . --exclude-dirs tests docs
        """
    )
    
    parser.add_argument(
        'paths',
        nargs='+',
        help='File paths or directories to process'
    )
    
    parser.add_argument(
        '--model',
        default='qwen3-coder',
        help='Ollama model to use (default: qwen3-coder)'
    )
    
    parser.add_argument(
        '--host',
        default='http://localhost:11434',
        help='Ollama server URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--output',
        default='./summaries',
        help='Output directory for summaries (default: ./summaries)'
    )
    
    parser.add_argument(
        '--extensions',
        nargs='+',
        help='File extensions to process (e.g., .py .js .java)'
    )
    
    parser.add_argument(
        '--exclude-dirs',
        nargs='+',
        help='Directory names to exclude (e.g., tests docs __pycache__)'
    )
    
    parser.add_argument(
        '--concurrent',
        type=int,
        default=4,
        help='Number of concurrent requests (default: 4)'
    )
    
    parser.add_argument(
        '--use-async',
        dest='use_async',
        action='store_true',
        help='Use async processing (default: sync with thread pool)'
    )
    
    args = parser.parse_args()
    
    try:
        # Find all code files
        file_paths = find_code_files(args.paths, args.extensions, args.exclude_dirs)
        
        if not file_paths:
            logger.error("No code files found to process")
            return
        
        logger.info(f"Found {len(file_paths)} files to process")
        
        # Show first few files
        if len(file_paths) <= 10:
            for fp in file_paths:
                logger.info(f"  - {fp}")
        else:
            for fp in file_paths[:5]:
                logger.info(f"  - {fp}")
            logger.info(f"  ... and {len(file_paths) - 5} more files")
        
        # Initialize summarizer and processor
        summarizer = CodeSummarizer(model=args.model, host=args.host)
        processor = BatchProcessor(summarizer, output_dir=args.output)
        
        # Process files
        if args.use_async:
            # Async processing
            asyncio.run(processor.process_batch_async(file_paths, args.concurrent))
        else:
            # Sync processing with thread pool
            processor.process_batch_sync(file_paths, args.concurrent)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()