from bmtools.agent.tools_controller import load_valid_tools, MTQuestionAnswerer
from bmtools.models.cpmbee_model import CpmBeeLLM
tools_mappings = {
    'weather': "http://127.0.0.1:8079/tools/weather/",
    'wolframalpha': "http://127.0.0.1:8079/tools/wolframalpha/",
}

tools = load_valid_tools(tools_mappings)
print(tools)
# SET config_path and ckpt_path
config_path = "/home/cc/workspace/ximu/workspace/yh/Models/cpm-bee/config.json"
ckpt_path = "/home/cc/workspace/ximu/workspace/yh/Models/cpm-bee/pytorch_model.bin"
llm =CpmBeeLLM(config_path = config_path,  ckpt_path = ckpt_path, device="cuda")
print("llm load successful")
qa =  MTQuestionAnswerer(llm, all_tools=tools)

agent = qa.build_runner()

agent("what is the weather in Beijing right now?")