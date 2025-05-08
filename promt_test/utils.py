import time
import asyncio

async def send_chat_completion(messages, model,client, max_tokens=256, temperature=0.7, top_p=0.9, top_k=50):
    extra_body={
      "top_k": top_k,
      "top_p": top_p,
      "temperature": temperature}
#     token_limit =4 *3000
#     for idx, message in enumerate(messages):
#         message_length = len(message['content'])
#         if message_length > token_limit:
#             print(f"Сообщение {idx} превышает лимит, обрезаем...")
#             message['content'] = message['content'][:token_limit]
#             messages[idx] = message
#     print(messages)
    completion = await client.chat.completions.create(model=model,
                                                      messages=messages,
                                                      max_tokens=max_tokens,
                                                      seed=42,
                                                      extra_body=extra_body)
    return completion.choices[0].message.content.strip()

async def send_async_requests(prompts_messages, model,client,**kwargs):
    tasks = [send_chat_completion(msgs, model,client,**kwargs) for msgs in prompts_messages]
    responses = await asyncio.gather(*tasks)
    return responses