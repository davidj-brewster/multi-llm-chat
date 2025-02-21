# Memory Usage Analysis and Optimization Plan

## Current Issues

### 1. SpaCy Model Loading
- **Problem**: Multiple redundant instances of large 'en_core_web_trf' transformer model
- **Cause**: Each model client creates its own AdaptiveInstructionManager which creates its own ContextAnalyzer
- **Impact**: ~400MB per instance, multiplied by number of clients

### 2. Inefficient Client Management
- **Problem**: All possible clients initialized at startup
- **Cause**: ConversationManager creates every client in __init__ regardless of use
- **Impact**: Unnecessary memory usage for unused clients
- **Additional Issue**: Duplicate OpenAIClient class definition causing double instantiation

### 3. Memory-Intensive Analysis
- **Problem**: Large memory usage in context analysis
- **Causes**:
  * TfidfVectorizer with max_features=12000 for each analyzer
  * Conversation histories copied multiple times
  * No cleanup of processed data
  * Large instruction templates stored redundantly

## Proposed Solutions

### 1. Singleton Pattern for SpaCy and Analysis Tools
```python
class ContextAnalyzerSingleton:
    _instance = None
    _nlp = None
    _vectorizer = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ContextAnalyzer()
        return cls._instance

    @classmethod
    def get_nlp(cls):
        if cls._nlp is None:
            cls._nlp = spacy.load('en_core_web_trf')
        return cls._nlp
```

### 2. Lazy Client Initialization
```python
class ConversationManager:
    def __init__(self):
        self.model_map = {}  # Empty initially
        self._initialized_clients = set()

    def get_client(self, model_name):
        if model_name not in self._initialized_clients:
            self.model_map[model_name] = self._create_client(model_name)
            self._initialized_clients.add(model_name)
        return self.model_map[model_name]
```

### 3. Memory-Efficient Analysis
- Implement conversation history truncation
- Add cleanup methods for processed data
- Share instruction templates across instances
- Optimize TfidfVectorizer parameters

## Implementation Plan

1. Create Singleton Classes:
   - ContextAnalyzerSingleton
   - InstructionManagerSingleton
   - SpacyModelSingleton

2. Refactor Client Management:
   - Remove duplicate OpenAIClient class
   - Implement lazy initialization
   - Add client cleanup methods

3. Optimize Analysis:
   - Reduce TfidfVectorizer max_features
   - Implement conversation history limits
   - Add data cleanup methods
   - Share templates via singleton

4. Add Memory Monitoring:
   - Implement memory usage logging
   - Add warning thresholds
   - Create cleanup triggers

## Expected Improvements

- Reduce initial memory footprint by ~80%
- Eliminate redundant model loading
- More efficient resource utilization
- Better memory scaling with conversation length

## Next Steps

1. Switch to Code mode to implement these changes
2. Start with the Singleton implementations
3. Modify client initialization system
4. Add memory monitoring
5. Test with various conversation scenarios