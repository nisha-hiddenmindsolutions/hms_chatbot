class ConversationMemory:
    """
    Stores conversation history for a chatbot session.
    """

    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str):
        """Add a user message to the conversation history."""
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_assistant_message(self, message: str):
        """Add an assistant message to the conversation history."""
        self.messages.append({
            "role": "assistant",
            "content": message
        })

    def get_history(self) -> list[dict]:
        """Return the conversation history."""
        return self.messages