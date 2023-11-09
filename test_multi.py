from bmtools.agent.tools_controller import load_valid_tools, MTQuestionAnswerer
from bmtools.models.cpmbee_model import CpmBeeLLM
tools_mappings = {
    'weather': "http://127.0.0.1:8079/tools/weather/",
    "douban-film": "http://127.0.0.1:8079/tools/douban-film/",
    #'wolframalpha': "http://127.0.0.1:8079/tools/wolframalpha/",
}

tools = load_valid_tools(tools_mappings)
print(tools)
# SET config_path and ckpt_path
#config_path = "/home/cc/workspace/ximu/workspace/yh/Models/cpm-bee/config.json"
#ckpt_path = "/home/cc/workspace/ximu/workspace/yh/Models/cpm-bee/pytorch_model.bin"
config_path = "/home/cc/workspace/ximu/workspace/yh/Models/5b/cpm-bee-5b.json"
ckpt_path = "/home/cc/workspace/ximu/workspace/yh/Models/5b/cpm-bee-5b-ckpt.pt"
llm =CpmBeeLLM(config_path = config_path,  ckpt_path = ckpt_path, device="cuda")
qa =  MTQuestionAnswerer(llm, all_tools=tools)

agent = qa.build_runner()

agent("北京现在的天气如何？")