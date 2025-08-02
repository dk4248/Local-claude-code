#!/usr/bin/env python3
"""
Intelligent Coder Agent
Uses code summaries to write contextually-aware code that integrates with existing codebase
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

try:
    import ollama
except ImportError:
    print("Error: ollama-python library not installed.")
    print("Please install it using: pip install ollama")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of coding tasks the agent can handle"""
    NEW_FEATURE = "new_feature"
    MODIFY_EXISTING = "modify_existing"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    ADD_TESTS = "add_tests"
    DOCUMENTATION = "documentation"
    API_ENDPOINT = "api_endpoint"
    DATABASE_SCHEMA = "database_schema"


@dataclass
class CodeContext:
    """Represents context from a code summary"""
    file_path: str
    summary: str
    relevance_score: float = 0.0
    
    @property
    def file_name(self) -> str:
        return Path(self.file_path).name
    
    @property
    def module_path(self) -> str:
        """Extract module path for imports"""
        parts = Path(self.file_path).parts
        if 'src' in parts:
            idx = parts.index('src')
            return '.'.join(parts[idx+1:]).replace('.py', '')
        return self.file_name.replace('.py', '')


class CoderAgent:
    """Main coder agent that generates code using summaries"""
    
    def __init__(self, 
                 summary_paths: List[str],
                 model: str = "qwen3-coder",
                 output_dir: str = "./generated_code"):
        """
        Initialize the coder agent
        
        Args:
            summary_paths: List of paths to summary txt files
            model: Ollama model to use
            output_dir: Directory to save generated code
        """
        self.model = model
        self.client = ollama.Client()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load all summaries
        self.contexts = self._load_summaries(summary_paths)
        logger.info(f"Loaded {len(self.contexts)} code summaries")
        
        # Extract codebase patterns
        self.codebase_info = self._analyze_codebase()
    
    def _load_summaries(self, summary_paths: List[str]) -> List[CodeContext]:
        """Load and parse summary files"""
        contexts = []
        
        for path in summary_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract original file path from summary
                lines = content.split('\n')
                file_path = None
                
                for line in lines:
                    if line.startswith("File: "):
                        file_path = line.replace("File: ", "").strip()
                        break
                
                if file_path:
                    contexts.append(CodeContext(
                        file_path=file_path,
                        summary=content
                    ))
                else:
                    logger.warning(f"Could not extract file path from {path}")
                    
            except Exception as e:
                logger.error(f"Error loading summary {path}: {e}")
        
        return contexts
    
    def _analyze_codebase(self) -> Dict:
        """Analyze summaries to extract codebase patterns"""
        info = {
            "languages": set(),
            "frameworks": set(),
            "patterns": set(),
            "dependencies": set(),
            "file_structure": {},
            "naming_conventions": set(),
            "common_imports": []
        }
        
        for context in self.contexts:
            # Detect language
            ext = Path(context.file_path).suffix
            if ext:
                info["languages"].add(ext)
            
            # Extract frameworks and libraries
            import_patterns = [
                r'import\s+(\w+)',
                r'from\s+(\w+)',
                r'require\([\'"](\w+)[\'"]\)',
                r'include\s+[<"](\w+)',
                r'using\s+(\w+)'
            ]
            
            for pattern in import_patterns:
                imports = re.findall(pattern, context.summary)
                for imp in imports:
                    if imp in ['django', 'flask', 'fastapi', 'express', 'react', 'vue', 'angular', 'spring', 'rails']:
                        info["frameworks"].add(imp)
                    info["dependencies"].add(imp)
            
            # Detect patterns
            pattern_checks = [
                ("inheritance", r'class\s+\w+\s*\([^)]+\)'),
                ("async", r'async\s+def|async\s+function|Promise'),
                ("decorators", r'@\w+'),
                ("dependency_injection", r'__init__.*\(.*:.*\)'),
                ("singleton", r'_instance|getInstance'),
                ("factory", r'create\w+|factory'),
                ("observer", r'subscribe|notify|observe'),
            ]
            
            for pattern_name, pattern_regex in pattern_checks:
                if re.search(pattern_regex, context.summary):
                    info["patterns"].add(pattern_name)
            
            # Detect naming conventions
            if re.search(r'def\s+[a-z_]+', context.summary):
                info["naming_conventions"].add("snake_case_functions")
            if re.search(r'class\s+[A-Z][a-zA-Z]+', context.summary):
                info["naming_conventions"].add("PascalCase_classes")
            if re.search(r'const\s+[A-Z_]+\s*=', context.summary):
                info["naming_conventions"].add("UPPER_CASE_constants")
            
            # Build file structure
            parts = Path(context.file_path).parts
            current = info["file_structure"]
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Convert sets to lists for JSON serialization
        info["languages"] = list(info["languages"])
        info["frameworks"] = list(info["frameworks"])
        info["patterns"] = list(info["patterns"])
        info["naming_conventions"] = list(info["naming_conventions"])
        info["common_imports"] = list(info["dependencies"])[:20]  # Top 20 imports
        info["dependencies"] = list(info["dependencies"])
        
        return info
    
    def _calculate_relevance(self, context: CodeContext, task: str, keywords: List[str]) -> float:
        """Calculate relevance score for a context based on the task"""
        score = 0.0
        
        # Check task mentions
        task_lower = task.lower()
        summary_lower = context.summary.lower()
        file_name_lower = context.file_name.lower()
        
        # File name relevance
        for keyword in keywords:
            if keyword.lower() in file_name_lower:
                score += 3.0
        
        # Direct task mentions
        task_words = [w for w in re.findall(r'\b\w+\b', task_lower) if len(w) > 3]
        for word in task_words:
            if word in file_name_lower:
                score += 2.0
            if word in summary_lower:
                score += 0.5 * summary_lower.count(word)
        
        # Check for related functionality
        relevance_patterns = [
            ("database", ["database", "sql", "query", "model", "schema"]),
            ("api", ["api", "endpoint", "route", "rest", "graphql"]),
            ("auth", ["auth", "login", "user", "permission", "token"]),
            ("test", ["test", "spec", "mock", "assert", "expect"]),
            ("ui", ["component", "render", "view", "template", "style"]),
            ("config", ["config", "settings", "environment", "options"]),
            ("util", ["util", "helper", "common", "shared", "lib"]),
        ]
        
        for category, terms in relevance_patterns:
            if any(term in task_lower for term in terms):
                if any(term in summary_lower for term in terms):
                    score += 2.0
        
        # Path-based relevance
        path_parts = Path(context.file_path).parts
        for part in path_parts:
            if any(keyword.lower() in part.lower() for keyword in keywords):
                score += 1.5
        
        return score
    
    def _select_relevant_contexts(self, 
                                 task: str, 
                                 max_contexts: int = 5) -> List[CodeContext]:
        """Select the most relevant contexts for the task"""
        # Extract keywords from task
        keywords = re.findall(r'\b[a-zA-Z_]+\b', task)
        keywords = [k for k in keywords if len(k) > 3]  # Filter short words
        
        # Calculate relevance for each context
        for context in self.contexts:
            context.relevance_score = self._calculate_relevance(context, task, keywords)
        
        # Sort by relevance and return top contexts
        sorted_contexts = sorted(self.contexts, 
                               key=lambda x: x.relevance_score, 
                               reverse=True)
        
        # Always include at least some context
        relevant = [c for c in sorted_contexts if c.relevance_score > 0][:max_contexts]
        
        if not relevant and sorted_contexts:
            # If no relevant contexts found, include top 2 anyway
            relevant = sorted_contexts[:2]
        
        logger.info(f"Selected {len(relevant)} relevant contexts for the task")
        for ctx in relevant:
            logger.debug(f"  - {ctx.file_path} (score: {ctx.relevance_score:.2f})")
        
        return relevant
    
    def _detect_task_type(self, instruction: str) -> TaskType:
        """Detect the type of coding task from instruction"""
        instruction_lower = instruction.lower()
        
        # Check for test-related keywords first
        if any(word in instruction_lower for word in ['test', 'spec', 'unit test', 'integration test']):
            return TaskType.ADD_TESTS
        
        # API endpoints
        if any(word in instruction_lower for word in ['api', 'endpoint', 'route', 'rest', 'graphql']):
            return TaskType.API_ENDPOINT
        
        # Database
        if any(word in instruction_lower for word in ['database', 'schema', 'migration', 'model']):
            return TaskType.DATABASE_SCHEMA
        
        # Documentation
        if any(word in instruction_lower for word in ['document', 'docs', 'comment', 'readme']):
            return TaskType.DOCUMENTATION
        
        # Bug fixes
        if any(word in instruction_lower for word in ['fix', 'bug', 'issue', 'error', 'problem']):
            return TaskType.BUG_FIX
        
        # Modifications
        if any(word in instruction_lower for word in ['modify', 'update', 'change', 'edit']):
            return TaskType.MODIFY_EXISTING
        
        # Refactoring
        if any(word in instruction_lower for word in ['refactor', 'improve', 'optimize', 'clean']):
            return TaskType.REFACTOR
        
        # Default to new feature
        return TaskType.NEW_FEATURE
    
    def _create_code_prompt(self, 
                          instruction: str, 
                          contexts: List[CodeContext],
                          task_type: TaskType) -> str:
        """Create the prompt for code generation"""
        # Format context summaries
        context_text = ""
        if contexts:
            context_text = "\n\n".join([
                f"=== {ctx.file_path} (Relevance: {ctx.relevance_score:.1f}) ===\n{ctx.summary[:1500]}..."
                if len(ctx.summary) > 1500 else 
                f"=== {ctx.file_path} (Relevance: {ctx.relevance_score:.1f}) ===\n{ctx.summary}"
                for ctx in contexts
            ])
        
        # Build codebase overview
        codebase_overview = f"""
CODEBASE INFORMATION:
- Languages: {', '.join(self.codebase_info['languages']) if self.codebase_info['languages'] else 'Unknown'}
- Frameworks: {', '.join(self.codebase_info['frameworks']) if self.codebase_info['frameworks'] else 'Standard library'}
- Design Patterns: {', '.join(self.codebase_info['patterns']) if self.codebase_info['patterns'] else 'None detected'}
- Naming Conventions: {', '.join(self.codebase_info['naming_conventions']) if self.codebase_info['naming_conventions'] else 'Standard'}
- Common Imports: {', '.join(self.codebase_info['common_imports'][:10])}
"""
        
        # Task-specific instructions
        task_instructions = {
            TaskType.NEW_FEATURE: """
Generate complete, production-ready code for the new feature.
Include:
1. All necessary imports (prefer existing project imports when possible)
2. Complete implementation with error handling
3. Integration points with existing code
4. Configuration setup if needed
5. Usage examples and inline documentation
6. Follow the project's structure and patterns
""",
            TaskType.API_ENDPOINT: """
Create a complete API endpoint implementation including:
1. Route/endpoint definition following project patterns
2. Request validation and parsing
3. Business logic implementation
4. Error handling with appropriate status codes
5. Response formatting consistent with other endpoints
6. Authentication/authorization if used in project
7. API documentation/comments
8. Example requests/responses
""",
            TaskType.BUG_FIX: """
Provide the bug fix including:
1. The corrected code with clear markers for changes
2. Explanation of what was causing the bug
3. Why the fix works
4. Any additional changes needed in related files
5. Test cases to verify the fix
6. Steps to prevent similar issues
""",
            TaskType.REFACTOR: """
Refactor the code with:
1. Improved structure following project patterns
2. Better performance optimizations
3. Maintained backward compatibility
4. Clear explanation of all changes
5. Migration steps if needed
6. Updated tests if applicable
""",
            TaskType.ADD_TESTS: """
Create comprehensive tests including:
1. Unit tests for individual functions/methods
2. Integration tests if applicable
3. Edge cases and error scenarios
4. Test fixtures/mocks matching project style
5. Clear test descriptions
6. Setup and teardown methods
7. Coverage of happy path and error cases
""",
            TaskType.DATABASE_SCHEMA: """
Provide database implementation including:
1. Schema definition in project's style
2. Migrations or setup scripts
3. Model/Entity classes
4. Relationships and constraints
5. Indexes for performance
6. Seed data if helpful
7. Example queries
""",
            TaskType.DOCUMENTATION: """
Create documentation including:
1. Clear, comprehensive documentation
2. Code examples
3. API references if applicable
4. Setup/installation instructions
5. Common use cases
6. Troubleshooting section
7. Follow project's documentation style
""",
            TaskType.MODIFY_EXISTING: """
Modify the existing code with:
1. Clear indication of what changes to make
2. The modified code sections
3. Explanation of changes
4. Impact on other parts of the system
5. Any necessary updates to tests or docs
6. Backward compatibility considerations
"""
        }
        
        prompt = f"""You are an expert developer working on an existing codebase. 
Your task is to write code that perfectly integrates with the existing project.

{codebase_overview}

{'RELEVANT CODE CONTEXT:' if contexts else 'NO SPECIFIC CONTEXT AVAILABLE - USING GENERAL BEST PRACTICES'}
{context_text if context_text else 'Generate code following standard conventions for the detected language.'}

TASK TYPE: {task_type.value}
INSTRUCTION: {instruction}

{task_instructions.get(task_type, task_instructions[TaskType.NEW_FEATURE])}

IMPORTANT REQUIREMENTS:
- Match the exact coding style from the existing codebase
- Use consistent naming conventions
- Import from existing modules when possible
- Follow the project's file organization
- Include comprehensive error handling
- Add helpful comments and documentation
- Ensure the code is production-ready
- If multiple files are needed, clearly separate them

Generate the complete code:"""

        return prompt
    
    def generate_code(self, instruction: str, output_file: Optional[str] = None) -> Dict:
        """
        Generate code based on instruction and summaries
        
        Args:
            instruction: What code to write
            output_file: Optional specific output filename
            
        Returns:
            Dict with generated code and metadata
        """
        logger.info(f"Generating code for: {instruction[:100]}...")
        
        # Detect task type
        task_type = self._detect_task_type(instruction)
        logger.info(f"Detected task type: {task_type.value}")
        
        # Select relevant contexts
        contexts = self._select_relevant_contexts(instruction)
        
        # Create prompt
        prompt = self._create_code_prompt(instruction, contexts, task_type)
        
        # Generate code
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "num_predict": 4096,
                }
            )
            
            generated_code = response['response']
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise
        
        # Parse and save the code
        result = self._parse_and_save_code(generated_code, instruction, output_file)
        
        # Add metadata
        result['metadata'] = {
            'task_type': task_type.value,
            'contexts_used': [ctx.file_path for ctx in contexts],
            'relevance_scores': {ctx.file_path: ctx.relevance_score for ctx in contexts},
            'instruction': instruction,
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'codebase_languages': self.codebase_info['languages'],
            'codebase_frameworks': self.codebase_info['frameworks']
        }
        
        return result
    
    def _parse_and_save_code(self, 
                           generated_code: str, 
                           instruction: str,
                           output_file: Optional[str]) -> Dict:
        """Parse the generated code and save to files"""
        result = {
            'files': [],
            'main_file': None,
            'raw_output': generated_code
        }
        
        # Extract code blocks with language hints
        code_pattern = r'```(?:(\w+)\n)?(.*?)```'
        code_blocks = re.findall(code_pattern, generated_code, re.DOTALL)
        
        # Also check for file markers like "# filename: example.py"
        file_sections = re.split(r'#\s*(?:filename|file):\s*(.+)\n', generated_code)
        
        saved_files = []
        
        if code_blocks:
            for i, (lang, code) in enumerate(code_blocks):
                code = code.strip()
                if not code:
                    continue
                
                # Determine filename
                filename = None
                
                # Check for filename comment at the start of code
                filename_match = re.match(r'(?://|#|--)\s*(?:filename|file):\s*(.+)', code)
                if filename_match:
                    filename = filename_match.group(1).strip()
                    # Remove the filename comment from code
                    code = '\n'.join(code.split('\n')[1:])
                
                if not filename:
                    if output_file and i == 0:
                        filename = output_file
                    else:
                        # Generate filename based on content and language
                        if 'class' in code:
                            class_match = re.search(r'class\s+(\w+)', code)
                            if class_match:
                                filename = f"{class_match.group(1).lower()}"
                        elif 'def' in code or 'function' in code:
                            func_match = re.search(r'(?:def|function)\s+(\w+)', code)
                            if func_match:
                                filename = f"{func_match.group(1)}"
                        else:
                            filename = f"generated_{i+1}"
                        
                        # Add extension based on language
                        if lang:
                            ext_map = {
                                'python': '.py', 'py': '.py',
                                'javascript': '.js', 'js': '.js', 
                                'typescript': '.ts', 'ts': '.ts',
                                'java': '.java', 'cpp': '.cpp', 
                                'c': '.c', 'csharp': '.cs', 'cs': '.cs',
                                'ruby': '.rb', 'rb': '.rb',
                                'go': '.go', 'rust': '.rs', 'rs': '.rs',
                                'php': '.php', 'swift': '.swift',
                                'kotlin': '.kt', 'kt': '.kt',
                                'html': '.html', 'css': '.css',
                                'sql': '.sql', 'json': '.json',
                                'yaml': '.yaml', 'yml': '.yml',
                                'xml': '.xml', 'markdown': '.md', 'md': '.md'
                            }
                            ext = ext_map.get(lang.lower(), f'.{lang}')
                            filename += ext
                        else:
                            # Try to detect from imports/syntax
                            if 'import' in code and 'from' in code:
                                filename += '.py'
                            elif 'const' in code or 'require(' in code:
                                filename += '.js'
                            elif '#include' in code:
                                filename += '.cpp'
                            else:
                                filename += '.txt'
                
                # Save file
                file_path = self.output_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                saved_files.append(str(file_path))
                if i == 0:
                    result['main_file'] = str(file_path)
                    
                logger.info(f"Saved generated code to: {file_path}")
        else:
            # No code blocks found, treat entire output as code
            if output_file:
                filename = output_file
            else:
                # Detect language from content
                if 'import' in generated_code and 'from' in generated_code:
                    ext = '.py'
                elif 'const' in generated_code or 'require(' in generated_code:
                    ext = '.js'
                elif '#include' in generated_code:
                    ext = '.cpp'
                else:
                    ext = '.txt'
                filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            
            file_path = self.output_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(generated_code)
            
            saved_files.append(str(file_path))
            result['main_file'] = str(file_path)
            logger.info(f"Saved generated code to: {file_path}")
        
        result['files'] = saved_files
        
        # Save metadata
        self._save_generation_metadata(result, instruction)
        
        return result
    
    def _save_generation_metadata(self, result: Dict, instruction: str):
        """Save metadata about the generation"""
        metadata_path = self.output_dir / "generation_log.json"
        
        # Load existing log if exists
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                log = json.load(f)
        else:
            log = []
        
        # Add this generation
        entry = {
            'timestamp': datetime.now().isoformat(),
            'instruction': instruction,
            'files_generated': result['files'],
            'main_file': result.get('main_file'),
            'metadata': result.get('metadata', {}),
            'contexts_used': len(result.get('metadata', {}).get('contexts_used', []))
        }
        
        log.append(entry)
        
        # Keep only last 100 entries
        if len(log) > 100:
            log = log[-100:]
        
        with open(metadata_path, 'w') as f:
            json.dump(log, f, indent=2)
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n🤖 Coder Agent Interactive Mode")
        print("=" * 50)
        print(f"Loaded {len(self.contexts)} code summaries")
        print(f"Using model: {self.model}")
        print(f"Detected languages: {', '.join(self.codebase_info['languages'])}")
        print(f"Detected frameworks: {', '.join(self.codebase_info['frameworks']) if self.codebase_info['frameworks'] else 'None'}")
        print("\nType 'help' for commands or 'exit' to quit")
        print("=" * 50)
        
        while True:
            try:
                instruction = input("\n💡 What code should I write? > ").strip()
                
                if not instruction:
                    continue
                    
                if instruction.lower() == 'exit':
                    break
                    
                if instruction.lower() == 'help':
                    print("""
Commands:
- Type any coding instruction to generate code
- 'list' - Show loaded summaries
- 'analyze' - Show codebase analysis
- 'patterns' - Show detected patterns and conventions
- 'exit' - Quit

Examples:
- "Create a REST API endpoint for user registration"
- "Fix the database connection timeout issue"
- "Add unit tests for the authentication module"
- "Refactor the data processing pipeline for better performance"
- "Create a React component for user profile display"
- "Implement caching for the product API"
                    """)
                    continue
                
                if instruction.lower() == 'list':
                    print("\nLoaded summaries:")
                    for i, ctx in enumerate(self.contexts[:15]):
                        print(f"{i+1}. {ctx.file_path}")
                    if len(self.contexts) > 15:
                        print(f"... and {len(self.contexts) - 15} more")
                    continue
                
                if instruction.lower() == 'analyze':
                    print("\nCodebase Analysis:")
                    print(json.dumps(self.codebase_info, indent=2, default=str))
                    continue
                
                if instruction.lower() == 'patterns':
                    print("\nDetected Patterns and Conventions:")
                    print(f"Design Patterns: {', '.join(self.codebase_info['patterns'])}")
                    print(f"Naming Conventions: {', '.join(self.codebase_info['naming_conventions'])}")
                    print(f"Common Dependencies: {', '.join(self.codebase_info['common_imports'][:15])}")
                    continue
                
                # Generate code
                print("\n🔄 Generating code...")
                result = self.generate_code(instruction)
                
                print(f"\n✅ Code generated successfully!")
                print(f"Files created: {', '.join(result['files'])}")
                
                # Show preview
                if result['main_file']:
                    with open(result['main_file'], 'r') as f:
                        preview = f.read()[:800]
                    print(f"\nPreview of {Path(result['main_file']).name}:")
                    print("-" * 50)
                    print(preview)
                    if len(preview) == 800:
                        print("... (truncated)")
                    print("-" * 50)
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n❌ Error: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Intelligent Coder Agent - Generates code using codebase summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python coder.py summary1.txt summary2.txt summary3.txt
  
  # Single instruction mode
  python coder.py summaries/*.txt --instruction "Create user authentication API"
  
  # With specific output file
  python coder.py summaries/*.txt -i "Fix database connection" -o db_fix.py
  
  # Using different model
  python coder.py summaries/*.txt --model codellama
        """
    )
    
    parser.add_argument(
        'summaries',
        nargs='+',
        help='Paths to summary text files'
    )
    
    parser.add_argument(
        '-i', '--instruction',
        help='Coding instruction (if not provided, runs in interactive mode)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output filename for generated code'
    )
    
    parser.add_argument(
        '--model',
        default='qwen3-coder',
        help='Ollama model to use (default: qwen3-coder)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./generated_code',
        help='Directory for generated code (default: ./generated_code)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize coder agent
        agent = CoderAgent(
            summary_paths=args.summaries,
            model=args.model,
            output_dir=args.output_dir
        )
        
        if args.instruction:
            # Single instruction mode
            result = agent.generate_code(args.instruction, args.output)
            print(f"\n✅ Successfully generated code!")
            print(f"Files created: {', '.join(result['files'])}")
            if result['main_file']:
                print(f"Main file: {result['main_file']}")
        else:
            # Interactive mode
            agent.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()