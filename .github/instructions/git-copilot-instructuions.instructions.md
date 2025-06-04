---
applyTo: '**'
---

# GitHub Copilot Instructions

## Project Development Guidelines
always use pwershell syntax 
### Code Integrity
- ALWAYS preserve existing functionality when adding new features
- TEST each implementation thoroughly before proceeding to the next task
- DO NOT alter the project's established navigation patterns

### Structure & Organization
- MAINTAIN the existing project structure and organization
- AVOID creating unnecessary files that complicate the project hierarchy
- KEEP related functionality grouped in appropriate modules

### Code Quality
- WRITE clean, readable code with descriptive variable and function names
- USE proper indentation (4 spaces for Python, 2 spaces for JavaScript/HTML)
- FOLLOW language-specific best practices and style guides
- INCLUDE appropriate error handling in all functions

### Documentation
- DOCUMENT all new implementations immediately after completion
- ADD docstrings to all functions and classes
- UPDATE README files when adding significant features
- PLACE detailed implementation instructions in the doc/ folder

### Environment & Execution
- ALWAYS use Conda environment named `pygpu` for all development, testing, and execution
- ALWAYS use PowerShell syntax for command examples
- ALWAYS run scripts yourself to verify functionality and not ask the user to run the scripts
- DETECT and activate the `pygpu` Conda environment before running Python scripts
- USE the following pattern for environment activation and command execution:
    ```powershell
    # If conda environment 'pygpu' exists
    conda activate pygpu
    # Run command
    ```
- DO NOT use venv or any other environment manager; only use Conda `pygpu` from now on
- ENSURE all dependency installation and script execution is done within the `pygpu` environment

### Integration Guidelines
- WRITE clear, step-by-step instructions for module integration
- INCLUDE prerequisites, dependencies, and potential conflicts
- DOCUMENT API endpoints and data structures
- SPECIFY testing procedures for integrated components