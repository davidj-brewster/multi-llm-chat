"""Report generation for arbiter and metrics analysis."""
import json
import datetime
import logging
from typing import Dict, Any, List

from arbiter_v4 import VisualizationGenerator
from metrics_analyzer import analyze_conversations

logger = logging.getLogger(__name__)


async def save_arbiter_report(report: Dict[str, Any]) -> None:
    """Save arbiter analysis report with visualization support."""
    try:
        # If report is a string, we're just passing through the Gemini report
        if isinstance(report, str):
            logger.debug("Using pre-generated arbiter report from Gemini")
            # The report is already saved by the ground_assertions method in arbiter_v4.py
            return

        # Only proceed if we have a report dict with metrics to visualize
        try:
            with open("templates/arbiter_report.html") as f:
                template = f.read()

            # Generate dummy data for the report if needed
            dummy_metrics = {
                "ai_ai": {"depth_score": 0.7, "topic_coherence": 0.8, "assertion_density": 0.6,
                          "question_answer_ratio": 0.5, "avg_complexity": 0.75},
                "human_ai": {"depth_score": 0.6, "topic_coherence": 0.7, "assertion_density": 0.5,
                             "question_answer_ratio": 0.6, "avg_complexity": 0.7}
            }

            dummy_flow = {
                "ai_ai": {"nodes": [{"id": 0, "role": "user", "preview": "Sample", "metrics": {}}],
                          "edges": []},
                "human_ai": {"nodes": [{"id": 0, "role": "user", "preview": "Sample", "metrics": {}}],
                             "edges": []}
            }

            # Generate visualizations if metrics are available
            viz_generator = VisualizationGenerator()
            metrics_chart = ""
            timeline_chart = ""
            if report.get("metrics", {}).get("conversation_quality"):
                metrics_chart = viz_generator.generate_metrics_chart(report["metrics"])
                timeline_chart = viz_generator.generate_timeline(report.get("flow", {}))

            # Format report content with safe defaults
            report_content = template % {
                "report_content": report.get("content", "No content available"),
                "metrics_data": json.dumps(report.get("metrics", dummy_metrics)),
                "flow_data": json.dumps(report.get("flow", dummy_flow)),
                "metrics_chart": metrics_chart,
                "timeline_chart": timeline_chart,
                "winner": report.get("winner", "No clear winner determined"),
            }

            # Save report with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"arbiter_visualization_{timestamp}.html"

            with open(filename, "w") as f:
                f.write(report_content)

            logger.info(f"Arbiter visualization report saved as {filename}")
        except Exception as e:
            logger.warning(f"Failed to generate visualization report: {e}")
            # Not a critical error since we already have the main report

    except Exception as e:
        logger.error(f"Failed to save arbiter report: {e}")

    except Exception as e:
        logger.error(f"Failed to save arbiter report: {e}")


async def save_metrics_report(
    ai_ai_conversation: List[Dict[str, str]],
    human_ai_conversation: List[Dict[str, str]],
) -> None:
    """Save metrics analysis report."""
    try:
        if ai_ai_conversation and human_ai_conversation:
            try:
                analysis_data = analyze_conversations(
                    ai_ai_conversation, human_ai_conversation
                )
                logger.debug("Metrics report generated successfully")

                # Save the metrics report to a file
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                metrics_filename = f"metrics_report_{timestamp}.html"

                # Create a basic HTML representation
                html_content = f"""
                <html>
                <head>
                    <title>Conversation Metrics Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2 {{ color: #333; }}
                        .metrics-container {{ display: flex; }}
                        .metrics-section {{ flex: 1; padding: 15px; margin: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    <h1>Conversation Metrics Report</h1>
                    <div class="metrics-container">
                        <div class="metrics-section">
                            <h2>AI-AI Conversation Metrics</h2>
                            <table>
                                <tr><th>Metric</th><th>Value</th></tr>
                                <tr><td>Total Messages</td><td>{analysis_data['metrics']['ai-ai']['total_messages']}</td></tr>
                                <tr><td>Average Message Length</td><td>{analysis_data['metrics']['ai-ai']['avg_message_length']:.2f}</td></tr>
                                <tr><td>Topic Coherence</td><td>{analysis_data['metrics']['ai-ai']['topic_coherence']:.2f}</td></tr>
                                <tr><td>Turn Taking Balance</td><td>{analysis_data['metrics']['ai-ai']['turn_taking_balance']:.2f}</td></tr>
                                <tr><td>Average Complexity</td><td>{analysis_data['metrics']['ai-ai']['avg_complexity']:.2f}</td></tr>
                            </table>
                        </div>
                        <div class="metrics-section">
                            <h2>Human-AI Conversation Metrics</h2>
                            <table>
                                <tr><th>Metric</th><th>Value</th></tr>
                                <tr><td>Total Messages</td><td>{analysis_data['metrics']['human-ai']['total_messages']}</td></tr>
                                <tr><td>Average Message Length</td><td>{analysis_data['metrics']['human-ai']['avg_message_length']:.2f}</td></tr>
                                <tr><td>Topic Coherence</td><td>{analysis_data['metrics']['human-ai']['topic_coherence']:.2f}</td></tr>
                                <tr><td>Turn Taking Balance</td><td>{analysis_data['metrics']['human-ai']['turn_taking_balance']:.2f}</td></tr>
                                <tr><td>Average Complexity</td><td>{analysis_data['metrics']['human-ai']['avg_complexity']:.2f}</td></tr>
                            </table>
                        </div>
                    </div>
                </body>
                </html>
                """

                with open(metrics_filename, "w") as f:
                    f.write(html_content)

                logger.info(f"Metrics report saved successfully as {metrics_filename}")

            except ValueError as e:
                if "Negative values in data" in str(e):
                    logger.error(f"Failed to generate metrics report due to distance calculation error: {e}")
                    # Create a simplified metrics report that doesn't depend on the problematic clustering
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    metrics_filename = f"metrics_report_basic_{timestamp}.html"

                    # Calculate basic metrics that don't depend on complex clustering
                    ai_ai_msg_count = len(ai_ai_conversation)
                    human_ai_msg_count = len(human_ai_conversation)

                    ai_ai_avg_length = sum(len(msg.get('content', '')) for msg in ai_ai_conversation) / max(1, ai_ai_msg_count)
                    human_ai_avg_length = sum(len(msg.get('content', '')) for msg in human_ai_conversation) / max(1, human_ai_msg_count)

                    html_content = f"""
                    <html>
                    <head>
                        <title>Basic Conversation Metrics Report</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            h1, h2 {{ color: #333; }}
                            .metrics-container {{ display: flex; }}
                            .metrics-section {{ flex: 1; padding: 15px; margin: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                            table {{ border-collapse: collapse; width: 100%; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                            th {{ background-color: #f2f2f2; }}
                            .error {{ color: red; padding: 10px; background-color: #ffeeee; border-radius: 5px; margin-bottom: 20px; }}
                        </style>
                    </head>
                    <body>
                        <h1>Basic Conversation Metrics Report</h1>
                        <div class="error">
                            <p>Note: Advanced metrics calculation failed with error: "{str(e)}"</p>
                            <p>This is a simplified report with basic metrics only.</p>
                        </div>
                        <div class="metrics-container">
                            <div class="metrics-section">
                                <h2>AI-AI Conversation Basic Metrics</h2>
                                <table>
                                    <tr><th>Metric</th><th>Value</th></tr>
                                    <tr><td>Total Messages</td><td>{ai_ai_msg_count}</td></tr>
                                    <tr><td>Average Message Length</td><td>{ai_ai_avg_length:.2f}</td></tr>
                                </table>
                            </div>
                            <div class="metrics-section">
                                <h2>Human-AI Conversation Basic Metrics</h2>
                                <table>
                                    <tr><th>Metric</th><th>Value</th></tr>
                                    <tr><td>Total Messages</td><td>{human_ai_msg_count}</td></tr>
                                    <tr><td>Average Message Length</td><td>{human_ai_avg_length:.2f}</td></tr>
                                </table>
                            </div>
                        </div>
                    </body>
                    </html>
                    """

                    with open(metrics_filename, "w") as f:
                        f.write(html_content)

                    logger.debug(f"Basic metrics report saved as {metrics_filename}")
                else:
                    # For other value errors, rethrow
                    raise
        else:
            logger.warning("Skipping metrics report - empty conversations")
    except Exception as e:
        logger.error(f"Failed to generate metrics report: {e}")
        # Create an error report file
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            error_filename = f"metrics_error_{timestamp}.html"

            html_content = f"""
            <html>
            <head>
                <title>Metrics Report Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #d33; }}
                    .error {{ padding: 15px; background-color: #ffeeee; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>Error Generating Metrics Report</h1>
                <div class="error">
                    <p><strong>Error:</strong> {str(e)}</p>
                    <p>The system encountered an error while generating the metrics report.</p>
                    <p>This does not affect the arbiter report or the conversation outputs.</p>
                </div>
            </body>
            </html>
            """

            with open(error_filename, "w") as f:
                f.write(html_content)

            logger.debug(f"Error report saved as {error_filename}")
        except Exception as inner_e:
            logger.error(f"Failed to save error report: {inner_e}")
