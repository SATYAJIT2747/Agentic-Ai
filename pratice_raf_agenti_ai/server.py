from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("Calculator Server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio")