"""OSeer — OS Seer.

An MCP server that lets an agent *dry-run* a terminal command (or another MCP
tool call) before executing it. OSeer gathers read-only facts about the real
terminal environment, asks the Qwen-AgentWorld world model to predict what the
command would do, adds a fast static safety assessment, and returns a structured
prediction — so agents avoid wrong, inefficient, or destructive commands.
"""

__version__ = "0.1.0"
