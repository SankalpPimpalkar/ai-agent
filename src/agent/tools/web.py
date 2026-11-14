from ollama import Tool

web_search_spec = Tool(
    type="function",
    function=Tool.Function(
        name="web_search",
        description="""
Search the live web for current real-world information.
Use this ONLY when the user explicitly asks you to search the web or lookup online information.
Never use web_search for local coding, text processing, data parsing, or math tasks.
""".strip(),
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "query": {
                    "type": "string",
                    "description": "Search query. Keep it short and specific.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                },
            },
            required=["query"],
        ),
    ),
)

web_fetch_spec = Tool(
    type="function",
    function=Tool.Function(
        name="web_fetch",
        description="""
Fetch the contents of an external web page.
Use this ONLY when the user explicitly asks you to browse, fetch, or read a web URL.
NEVER fetch URLs that are part of sample input text, mock data, or markdown links to convert.
""".strip(),
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "url": {
                    "type": "string",
                    "description": "Full URL including https://",
                },
            },
            required=["url"],
        ),
    ),
)

