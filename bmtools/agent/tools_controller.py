from langchain.llms import OpenAI
from langchain import OpenAI, LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor
import importlib
import json
import os
import requests
import yaml
from bmtools.agent.apitool import Tool
from bmtools.agent.singletool import STQuestionAnswerer
from bmtools.agent.executor import Executor, AgentExecutorWithTranslation
from bmtools import get_logger
from bmtools.models.customllm import CustomLLM

logger = get_logger(__name__)

def load_valid_tools(tools_mappings):
    tools_to_config = {}
    for key in tools_mappings:
        get_url = tools_mappings[key]+".well-known/ai-plugin.json"
        
        response = requests.get(get_url)

        if response.status_code == 200:
            tools_to_config[key] = response.json()
        else:
            logger.warning("Load tool {} error, status code {}".format(key, response.status_code))

    return tools_to_config
    


class MTQuestionAnswerer:
    """Use multiple tools to answer a question. Basically pass a natural question to 
    """
    def __init__(self, base_llm, all_tools, stream_output=False):
        """_summary_

        Args:
            base_llm (CpmBeeLLM): _Base LM_
            all_tools (_type_): _tools to be used_
            stream_output (bool, optional): _description_. Defaults to False.
        """
        # Add CPM-Bee to MTQuestionAnswerer
        self.llm = base_llm
        self.llm_model = base_llm.model_name
        logger.info("Using {}".format(self.llm_model))
        self.stream_output = stream_output
        self.load_tools(all_tools)

    def load_tools(self, all_tools):
        logger.info("All tools: {}".format(all_tools))
        self.all_tools_map = {}
        self.tools_pool = []
        for name in all_tools:
            meta_info = all_tools[name]

            question_answer = STQuestionAnswerer(self.openai_api_key, stream_output=self.stream_output, llm=self.llm_model)
            subagent = question_answer.load_tools(name, meta_info, prompt_type="react-with-tool-description", return_intermediate_steps=False)
            tool_logo_md = f'<img src="{meta_info["logo_url"]}" width="32" height="32" style="display:inline-block">'
            for tool in subagent.tools:
                tool.tool_logo_md = tool_logo_md
            tool = Tool(
                name=meta_info['name_for_model'],
                description=meta_info['description_for_model'].replace("{", "{{").replace("}", "}}"),
                func=subagent,
            )
            tool.tool_logo_md = tool_logo_md
            self.tools_pool.append(tool)
        
    def build_runner(self, ):
        from langchain.vectorstores import FAISS
        from langchain.docstore import InMemoryDocstore
        from langchain.embeddings import OpenAIEmbeddings
        embeddings_model = OpenAIEmbeddings()
        import faiss
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        
        from .autogptmulti.agent import AutoGPT
        from langchain.chat_models import ChatOpenAI
        agent_executor = AutoGPT.from_llm_and_tools(
            ai_name="Tom",
            ai_role="Assistant",
            tools=self.tools_pool,
            llm=ChatOpenAI(temperature=0),
            memory=vectorstore.as_retriever()
        )
        '''
        # 可以修改prompt来让模型表现更好，也可以修改tool的doc
        prefix = """Answer the following questions as best you can. In this level, you are calling the tools in natural language format, since the tools are actually an intelligent agent like you, but they expert only in one area. Several things to remember. (1) Remember to follow the format of passing natural language as the Action Input. (2) DO NOT use your imagination, only use concrete information given by the tools. (3) If the observation contains images or urls which has useful information, YOU MUST INCLUDE ALL USEFUL IMAGES and links in your Answer and Final Answers using format ![img](url). BUT DO NOT provide any imaginary links. (4) The information in your Final Answer should include ALL the informations returned by the tools. (5) If a user's query is a language other than English, please translate it to English without tools, and translate it back to the source language in Final Answer. You have access to the following tools (Only use these tools we provide you):"""
        suffix = """\nBegin! Remember to . \nQuestion: {input}\n{agent_scratchpad}"""

    
        prompt = ZeroShotAgent.create_prompt(
            self.tools_pool,
            prefix=prefix,
            suffix=suffix, 
            input_variables=["input", "agent_scratchpad"]
        )
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        logger.info("Full Prompt Template:\n {}".format(prompt.template))
        tool_names = [tool.name for tool in self.tools_pool]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        if self.stream_output:
            agent_executor = Executor.from_agent_and_tools(agent=agent, tools=self.tools_pool, verbose=True, return_intermediate_steps=True)
        else:
            agent_executor = AgentExecutorWithTranslation.from_agent_and_tools(agent=agent, tools=self.tools_pool, verbose=True, return_intermediate_steps=True)
        '''
        return agent_executor



if __name__ == "__main__":
    tools_mappings = {
        "klarna": "https://www.klarna.com/",
        "chemical-prop": "http://127.0.0.1:8079/tools/chemical-prop/",
        "wolframalpha": "http://127.0.0.1:8079/tools/wolframalpha/",
        "weather": "http://127.0.0.1:8079/tools/weather/",
    }

    tools = load_valid_tools(tools_mappings)
    from bmtools.models.cpmbee_model import CpmBeeLLM
    # SET config_path and ckpt_path
    config_path = ""
    ckpt_path = ""
    llm =CpmBeeLLM(config_path = config_path,  ckpt_path = ckpt_path, device="cuda")

    qa =  MTQuestionAnswerer(llm, all_tools=tools)

    agent = qa.build_runner()
    
    agent("How many carbon elements are there in CH3COOH? How many people are there in China?")
    
