from bmtools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = 'klarna',  'https://www.klarna.com/'
tool_name, tool_config = load_single_tools(tool_name, tool_url)
print(tool_name, tool_config)
stqa =  STQuestionAnswerer()

agent = stqa.load_tools(tool_name, tool_config)
agent("{Your Question}")