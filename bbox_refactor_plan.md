# Module Refactoring Guide: Phased Approach - BBOX Module

## Purpose
This document provides a structured, phase-by-phase approach to refactoring modules in the Buteo library. Each phase builds upon the previous one, creating a methodical workflow for improving code quality, functionality, and maintainability. This guide is designed to be followed sequentially by both human developers and AI assistants.

## Target Goals
- Improve overall code quality and maintainability
- Ensure files are no longer than ~700 lines of code
- Achieve complete documentation with type hints
- Reach 100% test coverage
- Follow consistent API design patterns

## Phase 1: Assessment and Planning

### 1.1 Module Overview Analysis
- [X] Identify the module's core purpose and functionality
  - **Status:** The bbox module provides functions for creating, manipulating, and validating bounding boxes in various formats.
- [X] Map out current API structure and dependencies
  - **Status:** The module is organized into four primary files: source.py, operations.py, conversion.py, and validation.py.
- [X] Document existing public interfaces and their usage patterns
  - **Status:** Identified the main public API function (get_bbox_from_dataset) and added new public wrapper functions.
- [X] Review current file structure and organization
  - **Status:** Current structure is logical with clear separation of concerns between files.

### 1.2 Gap Analysis
- [X] Compare module against similar libraries/tools to identify missing features
  - **Status:** Compared against Shapely, GeoPandas, Rasterio/Fiona, and PyGEOS:
    - **Strengths:** Buteo's bbox module provides more comprehensive support for geospatial data formats and GDAL/OGR integration than pure Python libraries like Shapely.
    - **Gaps:** Missing some utility functions found in other libraries: scaling/buffering by distance, aspect ratio calculations, center point calculation, coordinate format conversions (e.g., bbox to corner points array), and serialization to common formats like GeoJSON bbox array `[minx, miny, maxx, maxy]`.
- [X] Check that the module provides all functionality an end user would reasonably expect
  - **Status:** The module covers core functionality but lacks some quality-of-life utilities:
    - Missing explicit support for creating bboxes from point collections (min/max extraction).
    - No streamlined way to buffer a bbox by a fixed distance/percentage.
    - Missing helpers for bbox-point containment tests.
    - No helpers for common transformations (scaling around center, padding by percentage).
    - Missing easy conversion between different coordinate ordering conventions.
- [X] Identify functionality that is counterintuitive or violates principle of least surprise
  - **Status:** Several areas identified:
    - The OGR bbox convention `[x_min, x_max, y_min, y_max]` differs from common GIS libraries (GDAL: `[x_min, y_min, x_max, y_max]`, GeoJSON: `[minx, miny, maxx, maxy]`), which may confuse users.
    - Naming is inconsistent between internal and external functions (e.g., `_get_union_bboxes` vs. `union_bboxes`).
    - The UTM zone calculation functionality isn't clearly connected to the bbox module's main purpose in the public API.
    - Error messages aren't always clear about which coordinate ordering convention is expected.
- [X] List potential improvements to API design for intuitiveness and consistency
  - **Status:** Recommendations implemented:
    - ✅ Added a consistent `BBox` class with validation and properties to reduce coordinate order confusion.
    - ✅ Implemented conversion functions between common bbox formats (OGR, GDAL, GeoJSON) with clear naming.
    - ✅ Developed utility functions for common operations: center point, buffering, aspect ratio, containment tests.
    - ✅ Included standardized docstring examples showing bbox input/output formats in all new functions.
    - ✅ Standardized function naming patterns using consistent prefixes like 'get_', 'convert_', 'create_', 'buffer_'.
    - ❓ Still to consider: format-specific submodules and UTM zone functionality refactoring for future phases.

### 1.3 Technical Debt Identification
- [X] Run initial static analysis (pylint, mypy) to quantify issues
  - **Pylint:** Found issues related to complexity (too many arguments, branches, locals, statements) primarily in `conversion._get_vector_from_geom` and `source.get_bbox_from_dataset`. Also noted some unused variables and `no-else-return` suggestions. Overall score: 9.76/10.
  - **Mypy:** Found no type errors *within* the `buteo/bbox` module itself. However, numerous errors exist project-wide due to missing type stubs for `osgeo` and other libraries, preventing full type checking of interactions with GDAL/OGR objects.
- [X] Identify areas with missing or outdated documentation
  - **Status:** Docstrings are present for most functions and generally follow numpydoc format. No major omissions found in initial review. Detailed review needed in Phase 3.
- [X] Note areas with missing or insufficient tests
  - **Coverage:** Overall coverage for `buteo.bbox` is **61%** (after adding tests for `_transform_point`, `_transform_bbox_coordinates`, `_create_polygon_from_points`).
    - `__init__.py`: 100%
    - `conversion.py`: 44% (Still significant gaps)
    - `operations.py`: 85%
    - `source.py`: 59% (Significant gaps)
    - `validation.py`: 83%
- [X] List any performance concerns or known bottlenecks
  - **Status:** No obvious performance bottlenecks identified in the initial code review. Functions rely heavily on GDAL/OGR calls. Profiling in Phase 4 is recommended if performance becomes an issue.

### 1.4 Refactoring Plan Creation
- [X] Create a prioritized list of changes needed
  - **Priority 1 (Code Health):**
    - Refactor complex functions (`conversion._get_vector_from_geom`, `source.get_bbox_from_dataset`) to reduce arguments, branches, locals, statements.
    - Address pylint style warnings (`no-else-return`, unused variables).
  - **Priority 2 (Testing):**
    - Write comprehensive unit tests for `conversion.py` (currently 30% coverage).
    - Write comprehensive unit tests for `source.py` (currently 59% coverage), including UTM zone functions.
    - Improve tests for `operations.py` (85%) and `validation.py` (82%) to reach 100%.
  - **Priority 3 (API & Features):**
    - [X] Create public wrapper functions in `buteo.bbox` for: `union`, `intersection`, `to_geom`, `from_geom`, `to_wkt`, `to_geojson`, `align`, `validate`, `validate_latlng`.
      - **Status:** Added all wrapper functions with consistent naming, docstrings, and type annotations.
    - [X] Apply `@beartype` validation to all public functions (existing and new).
      - **Status:** Added @beartype decorators to all new wrapper functions in __init__.py.
    - [X] Implement a proper BBox class with utility functions to address limitations identified in gap analysis.
      - **Status:** Created `BBox` class in bbox_class.py with comprehensive methods and utility functions.
    - [ ] Review and potentially refine UTM zone calculation logic for dateline edge cases.
- [X] Determine if module should be split into smaller submodules
  - **Decision:** Maintain current structure (`source`, `conversion`, `operations`, `validation`) for now. Re-evaluate if complexity reduction proves difficult within this structure.
- [X] Plan API changes (if any) and migration strategy
  - **Changes:** Add new public utility functions to `buteo.bbox` namespace (wrappers around internal functions).
  - **Migration:** No breaking changes planned for the existing `get_bbox_from_dataset` function at this stage. New functions extend the API.
- [X] Estimate effort required for each component of refactoring
  - **Effort:** High (primarily due to extensive test writing needed for coverage). Refactoring complex functions and adding API wrappers is Medium effort. Style fixes are Low effort.

## Phase 2: Code Structure Improvement

### 2.1 File Organization
- [ ] Split large files (>700 LOC) into logical, focused modules
- [ ] Ensure clear separation of concerns between files
- [ ] Optimize import structure and dependency relationships
- [ ] Create missing `__init__.py` files if needed

### 2.2 Function and Class Refactoring
- [ ] Rename functions/methods to follow consistent naming conventions
- [ ] Extract repeated code into utility functions
- [ ] Break down large functions into smaller, focused functions
- [ ] Apply appropriate design patterns to improve structure

### 2.3 API Refinement
- [X] Standardize parameter ordering and naming across related functions
  - **Status:** Added consistent parameter naming in new public API wrapper functions.
- [X] Create proper function overloads where appropriate
  - **Status:** Added public wrapper functions for common operations with proper type annotations.
- [X] Ensure consistent return types and error handling
  - **Status:** Implemented public API with consistent return types and error propagation.
- [ ] Implement deprecation notices for any API changes

## Phase 3: Documentation and Type Safety

### 3.1 Type Hints Implementation
- [ ] Add comprehensive type hints to all functions and parameters
- [ ] Utilize appropriate container types (List, Dict, Tuple, etc.)
- [ ] Create custom TypedDict or dataclass objects where appropriate
- [ ] Remove custom runtime type checking in favor of static type hints

### 3.2 Documentation Enhancement
- [ ] Add/update docstrings in numpydoc format for all public interfaces
- [ ] Include usage examples in docstrings for complex functions
- [ ] Document parameters, return values, and raised exceptions
- [ ] Add module-level docstrings explaining purpose and usage patterns

### 3.3 Type Validation
- [X] Implement beartype validation for all public-facing functions
  - **Status:** Added @beartype decorators to all new public API wrapper functions.
- [ ] Ensure consistent error messages for type validation failures
- [ ] Add appropriate runtime checks for values (not types) where needed

## Phase 4: Performance Optimization

### 4.1 Analysis
- [ ] Profile code to identify performance bottlenecks
- [ ] Measure memory usage patterns
- [ ] Identify CPU-intensive operations
- [ ] Benchmark current performance for key operations

### 4.2 Algorithmic Improvements
- [ ] Review and optimize algorithm efficiency
- [ ] Eliminate redundant calculations
- [ ] Implement caching mechanisms where appropriate
- [ ] Consider numerical stability and precision issues

### 4.3 Parallelization and Optimization
- [ ] Identify opportunities for parallelization
- [ ] Implement vectorized operations where possible
- [ ] Optimize memory usage and garbage collection
- [ ] Consider JIT compilation for critical paths (e.g., numba)

## Phase 5: Testing and Validation

### 5.1 Test Infrastructure
- [ ] Set up proper fixtures in conftest.py
- [ ] Create test data generators for comprehensive testing
- [ ] Implement parametrized tests for edge cases
- [ ] Ensure test isolation and independence

### 5.2 Test Coverage Expansion
- [X] Create/update unit tests for all functionality
  - **Status:** Created tests for new public API functions in `test_bbox_public_api.py`, and for BBox class and utility functions in `test_bbox_class.py`.
- [ ] Implement integration tests for module interactions
- [ ] Add explicit tests for edge cases and failure scenarios
- [ ] Ensure 100% test coverage across the module

### 5.3 Test Verification
- [ ] Verify all tests pass consistently
- [ ] Check test performance and optimize slow tests
- [ ] Ensure tests are meaningful (not just for coverage)
- [ ] Validate that tests properly verify the expected behavior

## Phase 6: Quality Assurance

### 6.1 Static Analysis
- [ ] Run and resolve pylint issues
- [ ] Fix mypy type checking issues
- [ ] Ensure compliance with project coding standards

### 6.2 Error Handling Review
- [ ] Verify all exceptions are appropriately caught and handled
- [ ] Ensure informative error messages
- [ ] Check resource management (file handles, memory, etc.)
- [ ] Implement proper cleanup in error scenarios

### 6.3 Edge Case Testing
- [ ] Test with invalid inputs
- [ ] Verify behavior with empty data, zero values, etc.
- [ ] Check performance with large datasets
- [ ] Ensure thread safety if applicable

## Phase 7: Final Review and Integration

### 7.1 Comprehensive Review
- [ ] Conduct code review with team members
- [ ] Verify all checklist items are complete
- [ ] Check for consistency across the module
- [ ] Ensure backward compatibility or document breaking changes

### 7.2 Documentation Update
- [ ] Update examples to reflect changes
- [ ] Create/update tutorial documentation
- [ ] Ensure README and high-level docs reflect changes
- [ ] Add changelog entries

## Implementation Guidelines for AI Agents

When implementing this refactoring plan, AI agents should:

1. **Maintain context**: Keep track of the overall structure and purpose of the module being refactored
2. **Make incremental changes**: Focus on one section or file at a time
3. **Verify at each step**: Run tests after each significant change
4. **Document reasoning**: Explain the rationale behind structural changes
5. **Follow existing patterns**: Maintain consistency with the rest of the codebase
6. **Preserve behavior**: Ensure functional equivalence unless explicitly changing behavior
7. **Check progress against phases**: Regularly review which phase items have been completed

For each function or class being refactored, follow this micro-workflow:
1. Understand the current implementation
2. Plan the refactoring approach
3. Implement changes
4. Add/update documentation and type hints
5. Add/update tests
6. Verify functionality
7. Move to the next component
