from app.memory.conversation_memory import ConversationMemory


memory = ConversationMemory()

# First exchange
memory.add_user_message(
    "Who is the founder of Hidden Mind Solutions?"
)

memory.add_assistant_message(
    "Himanshu Sanadhya is the founder of Hidden Mind Solutions."
)

# Second exchange
memory.add_user_message(
    "What services does the company provide?"
)

memory.add_assistant_message(
    "Hidden Mind Solutions provides software development, "
    "web development, AI solutions, and other IT services."
)


# Display conversation history
print("\nConversation History")
print("=" * 60)

for message in memory.get_history():
    print(f"{message['role'].upper()}: {message['content']}")