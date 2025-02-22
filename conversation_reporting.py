"""Reporting and analysis module for AI Battle framework"""
import json
import yaml
from typing import Dict, List, Optional
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from config_integration import DiscussionConfig, ModelConfig, FileConfig, TimeoutConfig
MODEL_CONFIG = ConfigDict(
    extra='allow',  # Allow additional fields
    arbitrary_types_allowed=True,
    #protected_namespaces=('model_', ),
)

class MessageAnalysis(BaseModel):
    model_config = MODEL_CONFIG
    """Detailed analysis of a single message"""
    content: str
    nlp_analysis: NLPAnalysis
    tokens: int
    sentences: int
    avg_sentence_length: float
    readability_score: float
    response_type: str  # e.g., "explanation", "question", "challenge", "agreement"
    discourse_level: str  # e.g., "surface", "analytical", "evaluative", "synthetic"

class ConversationSummary(BaseModel):
    model_config = MODEL_CONFIG
    """Structured summary of conversation context"""
    key_points: List[str] = Field(default_factory=list)
    main_topics: Dict[str, float] = Field(default_factory=dict)  # Topic -> importance score
    argument_chain: List[Dict[str, str]] = Field(default_factory=list)  # Logical flow of arguments
    unresolved_questions: List[str] = Field(default_factory=list)
    context_length: int = 0  # Original context length in tokens
    summary_length: int = 0  # Summarized length in tokens

class AssertionEvidence(BaseModel):
    model_config = MODEL_CONFIG
    """Evidence supporting an assertion"""
    assertion: str
    sources: List[Dict[str, str]]  # List of {url: str, excerpt: str}
    confidence: float
    verification_method: str

class ConversationMetrics(BaseModel):
    model_config = MODEL_CONFIG
    """Metrics for conversation evaluation"""
    coherence: float
    relevance: float
    depth: float
    engagement: float
    reasoning: float
    knowledge: float
    goal_progress: float
    strategy_effectiveness: float
    grounded_assertions: int = 0
    unverified_claims: int = 0

class ParticipantMetrics(BaseModel):
    model_config = MODEL_CONFIG
    """Metrics for individual participant evaluation"""
    response_quality: float
    knowledge_accuracy: float
    reasoning_depth: float
    engagement_level: float
    strategy_adherence: float
    adaptation: float
    factual_accuracy: float
    citation_quality: float
    message_analyses: List[MessageAnalysis] = Field(default_factory=list)

class AssessmentSchema(BaseModel):
    model_config = MODEL_CONFIG

    class ParticipantRating(BaseModel):
        model_config = ConfigDict(extra='allow')
        coherence: float
        engagement: float
        reasoning_depth: float
        response_relevance: float

    class ConversationQuality(BaseModel):
        model_config = ConfigDict(extra='allow')
        flow_coherence: float
        topic_depth: float
        knowledge_exchange: float
        goal_progress: float

    participant_ratings: Dict[str, Dict[str, float]]
    conversation_quality: Dict[str, float]
    assertions: List[str]
    key_insights: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)

class ArbiterResult(BaseModel):
    """Complete evaluation result"""
    model_config = ConfigDict(extra='allow')
    winner: str
    conversation_metrics: Dict[str, ConversationMetrics]
    participant_metrics: Dict[str, Dict[str, ParticipantMetrics]]
    key_insights: List[str]
    improvement_suggestions: List[str]
    strategy_analysis: Dict[str, float]
    grounded_assertions: Dict[str, Dict[str, AssertionEvidence]]
    execution_timestamp: str
    conversation_ids: Dict[str, str]


class ConversationMetrics:
    model_config = ConfigDict(extra='allow')
    """Tracks and analyzes conversation quality metrics"""
    def __init__(self):
        self.model_config = ConfigDict(extra='allow')
        self.turn_metrics = []
        self.overall_metrics = {
            "semantic_coherence": 0.0,
            "topic_relevance": 0.0,
            "engagement_level": 0.0,
            "reasoning_depth": 0.0,
            "knowledge_integration": 0.0
        }
        self.model_metrics = {}
        self.strategy_effectiveness = {}

    def add_turn_metrics(self, turn_number: int, metrics: Dict) -> None:
        """Add metrics for a single conversation turn"""
        self.turn_metrics.append({
            "turn": turn_number,
            "metrics": metrics
        })
        
        # Update overall metrics
        for key in self.overall_metrics:
            if key in metrics:
                self.overall_metrics[key] = (
                    (self.overall_metrics[key] * (turn_number - 1) + metrics[key]) / turn_number
                )

    def add_model_metrics(self, model_name: str, metrics: Dict) -> None:
        """Add metrics for a specific model's performance"""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = []
        self.model_metrics[model_name].append(metrics)

    def add_strategy_effectiveness(self, strategy: str, effectiveness: float) -> None:
        """Track effectiveness of conversation strategies"""
        if strategy not in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy] = []
        self.strategy_effectiveness[strategy].append(effectiveness)

    def get_summary(self) -> Dict:
        """Get summary of all metrics"""
        return {
            "overall_metrics": self.overall_metrics,
            "model_metrics": {
                model: {
                    "average": sum(m.values()) / len(m) for m in metrics
                }
                for model, metrics in self.model_metrics.items()
            },
            "strategy_effectiveness": {
                strategy: sum(scores) / len(scores)
                for strategy, scores in self.strategy_effectiveness.items()
            },
            "turn_progression": [
                {
                    "turn": m["turn"],
                    "average_metrics": sum(m["metrics"].values()) / len(m["metrics"])
                }
                for m in self.turn_metrics
            ]
        }

class StrategyTracker:
    model_config = ConfigDict(extra='allow')
    """Tracks conversation strategies and their outcomes"""
    def __init__(self):
        self.model_config = ConfigDict(extra='allow')
        self.strategies = []
        self.adaptations = []
        self.effectiveness = {}

    def add_strategy(self, turn: int, model: str, strategy: str, context: str) -> None:
        """Record a strategy being used"""
        self.strategies.append({
            "turn": turn,
            "model": model,
            "strategy": strategy,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })

    def add_adaptation(self, turn: int, reason: str, old_strategy: str, new_strategy: str) -> None:
        """Record a strategy adaptation"""
        self.adaptations.append({
            "turn": turn,
            "reason": reason,
            "old_strategy": old_strategy,
            "new_strategy": new_strategy,
            "timestamp": datetime.now().isoformat()
        })

    def update_effectiveness(self, strategy: str, score: float) -> None:
        """Update effectiveness score for a strategy"""
        if strategy not in self.effectiveness:
            self.effectiveness[strategy] = []
        self.effectiveness[strategy].append(score)

    def get_summary(self) -> Dict:
        """Get summary of strategy usage and effectiveness"""
        return {
            "strategy_usage": {
                strategy: len([s for s in self.strategies if s["strategy"] == strategy])
                for strategy in set(s["strategy"] for s in self.strategies)
            },
            "adaptations": len(self.adaptations),
            "effectiveness": {
                strategy: sum(scores) / len(scores)
                for strategy, scores in self.effectiveness.items()
            }
        }

def format_config_text(config: DiscussionConfig) -> str:
    """Format configuration as readable text"""
    yaml_dict = {
        "discussion": asdict(config)
    }
    return yaml.dump(yaml_dict, default_flow_style=False, sort_keys=False)

def format_config_html(config: DiscussionConfig) -> str:
    """Format configuration as HTML"""
    html = [
        '<div class="config-section">',
        '<h2>Discussion Configuration</h2>',
        f'<div class="config-item"><strong>Turns:</strong> {config.turns}</div>',
        f'<div class="config-item"><strong>Goal:</strong> {config.goal}</div>',
        '<h3>Models</h3>'
    ]

    for name, model in config.models.items():
        html.extend([
            '<div class="model-config">',
            f'<h4>{name}</h4>',
            f'<div class="model-item"><strong>Type:</strong> {model.type}</div>',
            f'<div class="model-item"><strong>Role:</strong> {model.role}</div>'
        ])
        if model.persona:
            html.extend([
                '<div class="model-item"><strong>Persona:</strong>',
                f'<pre>{model.persona}</pre>',
                '</div>'
            ])
        html.append('</div>')

    if config.input_file:
        html.extend([
            '<h3>Input File</h3>',
            '<div class="file-config">',
            f'<div class="file-item"><strong>Path:</strong> {config.input_file.path}</div>',
            f'<div class="file-item"><strong>Type:</strong> {config.input_file.type}</div>'
        ])
        if config.input_file.max_resolution:
            html.append(f'<div class="file-item"><strong>Max Resolution:</strong> {config.input_file.max_resolution}</div>')
        html.append('</div>')

    if config.timeouts:
        html.extend([
            '<h3>Timeouts</h3>',
            '<div class="timeout-config">',
            f'<div class="timeout-item"><strong>Request:</strong> {config.timeouts.request}s</div>',
            f'<div class="timeout-item"><strong>Retry Count:</strong> {config.timeouts.retry_count}</div>',
            f'<div class="timeout-item"><strong>Notify On:</strong> {", ".join(config.timeouts.notify_on)}</div>',
            '</div>'
        ])

    html.append('</div>')
    return '\n'.join(html)

def generate_conversation_report(
    config: DiscussionConfig,
    conversation: List[Dict],
    metrics: ConversationMetrics,
    strategy_tracker: StrategyTracker,
    output_path: str
) -> None:
    """Generate comprehensive HTML report of the conversation"""
    
    # Load HTML template
    template_path = Path("templates/conversation_report.html")
    if template_path.exists():
        template = template_path.read_text()
    else:
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>AI Battle Conversation Report</title>
            <style>
                body {
                    font-family: system-ui, -apple-system, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .section {
                    margin: 30px 0;
                    padding: 20px;
                    border: 1px solid #e5e7eb;
                    border-radius: 8px;
                }
                .config-section { background: #f8f9fa; }
                .metrics-section { background: #f0f7ff; }
                .strategy-section { background: #f0fff4; }
                .conversation-section { background: #fff; }
                .model-config, .file-config, .timeout-config {
                    margin: 10px 0;
                    padding: 10px;
                    background: #fff;
                    border-radius: 4px;
                }
                .chart {
                    width: 100%;
                    height: 300px;
                    margin: 20px 0;
                }
                pre {
                    background: #f3f4f6;
                    padding: 10px;
                    border-radius: 4px;
                    overflow-x: auto;
                }
                .thinking {
                    background: #f0f7ff;
                    border-left: 4px solid #3b82f6;
                    padding: 10px;
                    margin: 10px 0;
                }
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>AI Battle Conversation Report</h1>
            
            <!-- Configuration -->
            <div class="section config-section">
                {config_html}
            </div>
            
            <!-- Metrics -->
            <div class="section metrics-section">
                <h2>Conversation Metrics</h2>
                <div id="metrics-chart" class="chart"></div>
                <div id="model-metrics-chart" class="chart"></div>
                <pre>{metrics_json}</pre>
            </div>
            
            <!-- Strategies -->
            <div class="section strategy-section">
                <h2>Conversation Strategies</h2>
                <div id="strategy-chart" class="chart"></div>
                <pre>{strategy_json}</pre>
            </div>
            
            <!-- Conversation -->
            <div class="section conversation-section">
                <h2>Conversation Transcript</h2>
                {conversation_html}
            </div>
            
            <script>
                // Metrics Chart
                const metrics = {metrics_js};
                Plotly.newPlot('metrics-chart', [{
                    x: Object.keys(metrics.overall_metrics),
                    y: Object.values(metrics.overall_metrics),
                    type: 'bar',
                    name: 'Overall Metrics'
                }]);
                
                // Model Metrics Chart
                const modelMetrics = {model_metrics_js};
                Plotly.newPlot('model-metrics-chart', 
                    Object.entries(modelMetrics).map(([model, data]) => ({
                        x: Object.keys(data),
                        y: Object.values(data),
                        type: 'bar',
                        name: model
                    }))
                );
                
                // Strategy Chart
                const strategies = {strategy_js};
                Plotly.newPlot('strategy-chart', [{
                    x: Object.keys(strategies.strategy_usage),
                    y: Object.values(strategies.strategy_usage),
                    type: 'bar',
                    name: 'Strategy Usage'
                }]);
            </script>
        </body>
        </html>
        """

    # Format conversation transcript
    conversation_html = []
    for msg in conversation:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # Extract thinking tags
        thinking_content = ""
        if "<thinking>" in content:
            import re
            thinking_parts = re.findall(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
            if thinking_parts:
                thinking_content = '<div class="thinking">' + '<br>'.join(thinking_parts) + '</div>'
                content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)

        conversation_html.append(f"""
            <div class="message {role}">
                <div class="header">{role}</div>
                {thinking_content}
                <div class="content">{content}</div>
            </div>
        """)

    # Get metrics and strategy summaries
    metrics_summary = metrics.get_summary()
    strategy_summary = strategy_tracker.get_summary()

    # Render template
    html = template.format(
        config_html=format_config_html(config),
        metrics_json=json.dumps(metrics_summary, indent=2),
        strategy_json=json.dumps(strategy_summary, indent=2),
        conversation_html='\n'.join(conversation_html),
        metrics_js=json.dumps(metrics_summary),
        model_metrics_js=json.dumps(metrics_summary["model_metrics"]),
        strategy_js=json.dumps(strategy_summary)
    )

    # Write report
    with open(output_path, 'w') as f:
        f.write(html)

def print_conversation_summary(
    config: DiscussionConfig,
    metrics: ConversationMetrics,
    strategy_tracker: StrategyTracker
) -> None:
    """Print summary of conversation to console"""
    print("\n=== Conversation Summary ===\n")
    
    print("Configuration:")
    print(format_config_text(config))
    
    print("\nMetrics:")
    metrics_summary = metrics.get_summary()
    print(json.dumps(metrics_summary, indent=2))
    
    print("\nStrategies:")
    strategy_summary = strategy_tracker.get_summary()
    print(json.dumps(strategy_summary, indent=2))