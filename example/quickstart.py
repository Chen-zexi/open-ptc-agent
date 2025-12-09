"""
PTC Agent SDK Quickstart

This example shows how to create and run a PTC agent programmatically
with MCP server configuration for additional tool capabilities.

REQUIRED ENVIRONMENT VARIABLES:
- DAYTONA_API_KEY: Your Daytona API key (get from https://app.daytona.io)
- ANTHROPIC_API_KEY: Your Anthropic API key (or other LLM provider key)

OPTIONAL (for MCP servers):
- TAVILY_API_KEY: For web search capabilities

Usage:
    python example/quickstart.py
"""

import asyncio
import os
from pathlib import Path

# Load .env file from project root
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

from langchain_anthropic import ChatAnthropic

from ptc_agent import AgentConfig, PTCAgent
from ptc_agent.config import MCPServerConfig
from ptc_agent.core import SessionManager


async def main():
    # =================================================================
    # CONFIGURATION
    # Create your LLM instance and pass it to AgentConfig.create()
    # =================================================================

    # Pass any langchain chat model instance
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")

    # Create config with LLM and optional MCP servers
    config = AgentConfig.create(
        llm=llm,
        # Optional: MCP servers for additional tools
        mcp_servers=[
            MCPServerConfig(
                name="tavily",
                description="Web search capabilities via Tavily",
                instruction="Use for searching the web for current information",
                command="npx",
                args=["-y", "tavily-mcp@latest"],
                env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
            ),
            # Add more MCP servers as needed:
            # MCPServerConfig(
            #     name="alpha-vantage",
            #     description="Financial market data via Alpha Vantage",
            #     instruction="Use for fetching stock prices, forex rates, and financial data",
            #     command="npx",
            #     args=["-y", "@anthropic/mcp-server-alpha-vantage"],
            #     env={"ALPHA_VANTAGE_API_KEY": os.getenv("ALPHA_VANTAGE_API_KEY", "")},
            # ),
        ],
        # Optional: Override defaults
        log_level="INFO",
        max_execution_time=300,  # 5 minutes
        # Optional: Enable subagents for complex tasks
        subagents_enabled=["research", "general-purpose"],
    )

    # Validate that required API keys are set (only checks DAYTONA_API_KEY)
    config.validate_api_keys()

    # Create session (initializes Daytona sandbox)
    session = SessionManager.get_session("quickstart", config.to_core_config())

    try:
        print("Initializing sandbox...")
        await session.initialize()
        print(f"Sandbox ready: {session.sandbox.sandbox_id}")

        # Create the PTC agent
        ptc_agent = PTCAgent(config)
        agent = ptc_agent.create_agent(
            sandbox=session.sandbox,
            mcp_registry=session.mcp_registry,
        )

        # Run a simple task
        print("\nRunning task: Find news article and filter based on the keyword")
        result = await agent.ainvoke({
            "messages": [
                {"role": "user", "content": "Find all the news articles about Nvidia that mention TPU in last 2 days."}
            ]
        })

        # Print the result
        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        if result.get("messages"):
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                print(last_message.content)
            else:
                print(last_message)

    finally:
        # Clean up the session
        print("\nCleaning up...")
        await SessionManager.cleanup_session("quickstart")
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
