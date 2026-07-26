```markdown
# capsule_brain_v2_milestone11_repaired Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches you the core development patterns, coding conventions, and workflows used in the `capsule_brain_v2_milestone11_repaired` repository. The codebase is written in Python and focuses on modular service development, persistence, and maintainable architecture. It emphasizes clean code, artifact management, and robust testing, making it suitable for scalable backend systems.

## Coding Conventions

- **File Naming:**  
  Use `snake_case` for Python files and modules.  
  *Example:*  
  ```
  models.py
  repository.py
  service.py
  runner.py
  test_semantic_search.py
  ```

- **Import Style:**  
  Use relative imports within modules.  
  *Example:*  
  ```python
  from .repository import SemanticSearchRepository
  from .models import WorkflowModel
  ```

- **Export Style:**  
  Use named exports (explicitly define what is exported).  
  *Example:*  
  ```python
  __all__ = ["SemanticSearchService", "SemanticSearchRepository"]
  ```

- **Commit Messages:**  
  Mixed types, often prefixed with `chore`.  
  *Example:*  
  ```
  chore: update semantic search service and add persistence layer
  ```

## Workflows

### Add New Core Service with Persistence and Tests
**Trigger:** When adding a new major feature or subsystem (e.g., semantic search, workflow engine) with persistence and test coverage.  
**Command:** `/new-core-service`

1. **Define or update models:**  
   Create or modify `models.py` in the relevant module.
   ```python
   # src/capsule_brain/semantic_search/models.py
   class SemanticSearchResult(BaseModel):
       id: int
       content: str
   ```
2. **Implement repository logic:**  
   Add repository code in `repository.py` (e.g., using SQLite).
   ```python
   # src/capsule_brain/semantic_search/repository.py
   class SemanticSearchRepository:
       def save(self, result: SemanticSearchResult):
           # Save logic here
   ```
3. **Implement service logic:**  
   Add business logic in `service.py` or `runner.py`.
   ```python
   # src/capsule_brain/semantic_search/service.py
   class SemanticSearchService:
       def search(self, query: str):
           # Search logic here
   ```
4. **Wire up the new service:**  
   Update `runtime/bootstrap.py` to register the service and modify `configs/v2_runtime.yaml` as needed.
5. **Update module initialization:**  
   Ensure `__init__.py` and supporting files are present.
6. **Write unit tests:**  
   Add or update tests in `tests/unit/`, following the pattern `test_<feature>.py`.
   ```python
   # tests/unit/test_semantic_search.py
   def test_search_returns_results():
       ...
   ```
7. **Update documentation:**  
   Document the new feature in `MILESTONE_12.md` or similar tracking files.

### Remove Build Artifacts and Generated Files
**Trigger:** When cleaning up the repository by removing tracked build artifacts and generated runtime files.  
**Command:** `/cleanup-artifacts`

1. **Identify tracked artifacts:**  
   Find `.pyc` and `.sqlite*` files tracked by git.
   ```bash
   git ls-files | grep -E '\.pyc$|\.sqlite'
   ```
2. **Remove from version control:**  
   Remove these files.
   ```bash
   git rm --cached src/**/__pycache__/*.pyc
   git rm --cached data/*.sqlite*
   git rm --cached tests/unit/__pycache__/*.pyc
   ```
3. **Update .gitignore:**  
   Ensure patterns for these files are present in `.gitignore`.
   ```
   __pycache__/
   *.pyc
   *.sqlite*
   ```

## Testing Patterns

- **Testing Framework:**  
  Not explicitly detected, but tests follow standard Python unit test conventions.

- **Test File Naming:**  
  Use `test_<feature>.py` in `tests/unit/`.
  *Example:*  
  ```
  tests/unit/test_semantic_search.py
  ```

- **Test Example:**  
  ```python
  def test_feature_behavior():
      # Arrange
      # Act
      # Assert
      assert ...
  ```

## Commands

| Command            | Purpose                                                        |
|--------------------|----------------------------------------------------------------|
| /new-core-service  | Scaffold a new core service with persistence and tests         |
| /cleanup-artifacts | Remove tracked build artifacts and generated files from git    |
```