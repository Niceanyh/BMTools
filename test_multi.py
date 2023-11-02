from bmtools.agent.tools_controller import load_valid_tools, MTQuestionAnswerer
from bmtools.models.cpmbee_model import CpmBeeLLM
tools_mappings = {
    'weather': "http://127.0.0.1:8079/tools/weather/",
    'file_operation': "http://127.0.0.1:8079/tools/file_operation/",
}

tools = load_valid_tools(tools_mappings)
# SET config_path and ckpt_path
config_path = "../Models/config.json"
ckpt_path = "../Models/pytorch_model.bin"
llm =CpmBeeLLM(config_path = config_path,  ckpt_path = ckpt_path, device="cuda")

qa =  MTQuestionAnswerer(llm, all_tools=tools)

agent = qa.build_runner()

agent("what is the weather in Beijing right now?")