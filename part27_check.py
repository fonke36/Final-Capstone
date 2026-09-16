from crewai.tools import BaseTool


class TestTool(BaseTool):
    name: str = "test_tool"
    description: str = "A simple test tool."

    def _run(self, query: str) -> str:
        return f"Tool received: {query}"


tool = TestTool()

print("=" * 50)
print("PART 2.7 - CREWAI TOOL CHECK")
print("=" * 50)

print("CrewAI tool class:", type(tool).__name__)
print("Tool name:", tool.name)
print("Tool description:", tool.description)
print("Tool arguments schema:", tool.args_schema)

print("\nDirect tool test:")
print(tool.run(query="Hello CrewAI"))

print("\nCrewAI tool interface check passed.")