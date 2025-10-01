from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder


chat_template=ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history=[]

with open('chat_hisotry.txt') as f:
    chat_history.extend(f.readline())

print(chat_history)

chat_template.invoke({
    'chat_hisotry':chat_history,
    'query':'where is my refund'
})