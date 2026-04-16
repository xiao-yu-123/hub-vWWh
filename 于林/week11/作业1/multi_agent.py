import os

os.environ['OPENAI_API_KEY'] = 'sk-9bf45d961ac64f75a3b6a64c7fd08817'
os.environ['OPENAI_BASE_URL'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from agents import Agent,Runner, trace, RawResponsesStreamEvent
from agents import set_default_openai_api, set_tracing_disabled, TResponseInputItem
from openai.types.responses import ResponseTextDeltaEvent, ResponseContentPartDoneEvent
set_default_openai_api('chat_completions')
set_tracing_disabled(True)

senticly_agent = Agent(
    model='qwen-plus',
    name='senticly',
    instructions="""你是情感分类专家小A，对用户输入文本进行情感分类
    情感分类标签有：正面 负面 中性
    回答问题的时候先告诉我你是谁，再回答情感分类标签
    """
)

infoExtra_agent = Agent(
    model='qwen-plus',
    name='infoExtra',
    instructions="""你是信息抽取专家小B，对用户输入文本进行信息抽取
    实体标签有：code Src startDate_dateOrig film endLoc_city artistRole location_country location_area author startLoc_city season dishNamet media datetime_date episode teleOperator questionWord receiver ingredient name startDate_time startDate_date location_province endLoc_poi artist dynasty area location_poi relIssue Dest content keyword target startLoc_area tvchannel type song queryField awayName headNum homeName decade payment popularity tag startLoc_poi date startLoc_province endLoc_province location_city absIssue utensil scoreDescr dishName endLoc_area resolution yesterday timeDescr category subfocus theatre datetime_time
    回答问题的时候先告诉我你是谁，再回答抽取标签"""
)
tiage_agent = Agent(
    model='qwen-plus',
    name='tiage',
    instructions='Handoff to the appropriate agent based on the language of the request.',
    handoffs=[senticly_agent, infoExtra_agent],
)

async def main():
    conversation_id = str(uuid.uuid4().hex[:16])

    query = input('你好，我可以帮你进行情感分类和信息抽取，你还有什么问题？')
    agent = tiage_agent
    # list[TResponseInputItem] 表示一个消息列表，可以包含多条交替的用户消息和助手消息，用来维持多轮对话的上下文。
    inputs : list[TResponseInputItem] = [{"content": query, 'role': 'user'}]
    while True:
        with trace('Routing example', group_id=conversation_id):
            result = Runner.run_streamed(
                agent,
                input=inputs
            )

            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end='', flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        inputs = result.to_input_list()
        print("\n")
        user_msg = input("Enter a message")
        inputs.append({'content':user_msg, 'role':'user'})
        agent = result.current_agent



if __name__ == '__main__':
    asyncio.run(main())



